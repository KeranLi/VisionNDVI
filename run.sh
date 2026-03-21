#!/bin/bash
# =============================================================================
# VisionNDVI 统一运行脚本
# 用法: ./run.sh [命令] [选项]
# =============================================================================

set -e

# 默认配置
DATASET_DIR="${DATASET_DIR:-./datasets/AWI-CM-1-1-MR/}"
BASE_MODEL="${BASE_MODEL:-./checkpoints/AWI_prediction_model.pth}"
ADAPTER="${ADAPTER:-./checkpoints/adapter_v3/DeepMultiTimeAdapter_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
LOG_DIR="${LOG_DIR:-./logs}"

# 创建必要目录
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# 辅助函数
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_file() {
    if [ ! -f "$1" ]; then
        log_error "文件不存在: $1"
        exit 1
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        log_error "目录不存在: $1"
        exit 1
    fi
}

# =============================================================================
# 推理命令
# =============================================================================

cmd_inference() {
    local mode="${1:-rolling}"
    shift 2>/dev/null || true
    
    # 解析参数
    local start_date="201501"
    local end_date="202012"
    local labeled_end_date=""
    local online_finetune="--online_finetune"
    local background=false
    local viz_formats="png"
    local save_residuals=""
    local mask_path=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --start|-s) start_date="$2"; shift 2 ;;
            --end|-e) end_date="$2"; shift 2 ;;
            --labeled-end) labeled_end_date="$2"; shift 2 ;;
            --no-finetune) online_finetune="--no_online_finetune"; shift ;;
            --bg|--background) background=true; shift ;;
            --viz-formats) viz_formats="$2"; shift 2 ;;
            --save-residuals) save_residuals="--save_residuals"; shift ;;
            --mask-path) mask_path="--mask_path '$2'"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    log_info "推理模式: $mode"
    log_info "时间范围: $start_date - $end_date"
    
    if [ "$mode" = "nofinetune" ]; then
        check_file "$BASE_MODEL"
        check_file "$ADAPTER"
        check_dir "$DATASET_DIR"
        
        local out_dir="$OUTPUT_DIR/no_finetune_${start_date}_${end_date}"
        local log_file="$LOG_DIR/inference_nofinetune_$(date +%Y%m%d_%H%M%S).log"
        
        local cmd="python inference_no_finetune.py \
            --base_model '$BASE_MODEL' \
            --adapter '$ADAPTER' \
            --dataset_dir '$DATASET_DIR' \
            --start_date '$start_date' \
            --end_date '$end_date' \
            --output_dir '$out_dir'"
        
        if [ "$background" = true ]; then
            log_info "后台运行，日志: $log_file"
            eval "$cmd > '$log_file' 2>&1 &"
            echo $! > "$LOG_DIR/inference.pid"
            log_success "任务已启动 (PID: $!)"
        else
            eval "$cmd"
        fi
        
    else
        # rolling 模式
        check_file "$BASE_MODEL"
        check_file "$ADAPTER"
        check_dir "$DATASET_DIR"
        
        local out_dir="$OUTPUT_DIR/rolling_${start_date}_${end_date}"
        local log_file="$LOG_DIR/inference_rolling_$(date +%Y%m%d_%H%M%S).log"
        
        local cmd="python inference_rolling.py \
            --base_model '$BASE_MODEL' \
            --adapter '$ADAPTER' \
            --dataset_dir '$DATASET_DIR' \
            --start_date '$start_date' \
            --end_date '$end_date' \
            $online_finetune \
            --viz_formats '$viz_formats' \
            --save_residuals \
            $mask_path \
            --output_dir '$out_dir'"
        
        [ -n "$labeled_end_date" ] && cmd="$cmd --labeled_end_date '$labeled_end_date'"
        
        if [ "$background" = true ]; then
            log_info "后台运行，日志: $log_file"
            eval "$cmd > '$log_file' 2>&1 &"
            echo $! > "$LOG_DIR/inference.pid"
            log_success "任务已启动 (PID: $!)"
        else
            eval "$cmd"
        fi
    fi
}

# =============================================================================
# 批量推理命令
# =============================================================================

cmd_batch() {
    local start_date="${1:-201501}"
    local end_date="${2:-202012}"
    
    local models=("AWI" "BCC" "CMCC_CM2_HR4" "CMCC_CM2_SR5" "CMCC_ESM2" "FIO" "MPI")
    
    log_info "批量推理: $start_date - $end_date"
    
    for model in "${models[@]}"; do
        local model_path="./checkpoints/${model}_prediction_model.pth"
        local data_dir="./datasets/${model}-CM-1-1-MR/"
        
        if [ ! -f "$model_path" ]; then
            log_warn "跳过 $model: 模型不存在"
            continue
        fi
        
        log_info "[$model] 启动推理..."
        
        BASE_MODEL="$model_path" \
        DATASET_DIR="$data_dir" \
        ./run.sh inference nofinetune --start "$start_date" --end "$end_date" --bg
        
        # 等待当前任务完成
        if [ -f "$LOG_DIR/inference.pid" ]; then
            local pid=$(cat "$LOG_DIR/inference.pid")
            log_info "等待任务完成 (PID: $pid)..."
            while ps -p "$pid" > /dev/null 2>&1; do
                sleep 5
            done
        fi
        log_success "[$model] 完成"
    done
    
    log_success "所有模型推理完成"
}

# =============================================================================
# 训练命令
# =============================================================================

cmd_train() {
    local mode="${1:-adapter}"
    shift 2>/dev/null || true
    
    local background=false
    local epochs=5
    local lr=2e-3
    local start_date="198201"
    local end_date="201412"
    local adapter_type="DeepMultiTimeAdapter"
    local window_size=3
    local iterations=50
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --bg|--background) background=true; shift ;;
            --epochs|-e) epochs="$2"; shift 2 ;;
            --lr) lr="$2"; shift 2 ;;
            --start-date|-s) start_date="$2"; shift 2 ;;
            --end-date|-e) end_date="$2"; shift 2 ;;
            --adapter-type|-t) adapter_type="$2"; shift 2 ;;
            --window-size|-w) window_size="$2"; shift 2 ;;
            --iterations|-i) iterations="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    if [ "$mode" = "adapter" ]; then
        check_file "$BASE_MODEL"
        check_dir "$DATASET_DIR"
        
        local out_dir="./checkpoints/adapter"
        local log_file="$LOG_DIR/train_adapter_$(date +%Y%m%d_%H%M%S).log"
        
        log_info "Adapter 预训练"
        log_info "类型: $adapter_type"
        log_info "时间: $start_date - $end_date"
        log_info "Epochs: $epochs, LR: $lr, Iterations: $iterations"
        
        local cmd="python train_adapter.py \
            --base_model '$BASE_MODEL' \
            --dataset_dir '$DATASET_DIR' \
            --output_dir '$out_dir' \
            --adapter_type '$adapter_type' \
            --window_size $window_size \
            --start_date '$start_date' \
            --end_date '$end_date' \
            --epochs $epochs \
            --lr $lr \
            --iterations_per_sample $iterations"
        
        if [ "$background" = true ]; then
            log_info "后台运行，日志: $log_file"
            eval "$cmd > '$log_file' 2>&1 &"
            echo $! > "$LOG_DIR/train.pid"
            log_success "训练已启动 (PID: $!)"
            echo "查看日志: tail -f $log_file"
        else
            eval "$cmd"
        fi
    fi
}

# =============================================================================
# 任务管理命令
# =============================================================================

cmd_status() {
    log_info "任务状态"
    echo "========================================"
    
    local found=0
    for pid_file in "$LOG_DIR"/*.pid; do
        [ -f "$pid_file" ] || continue
        
        local pid=$(cat "$pid_file")
        local name=$(basename "$pid_file" .pid)
        
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "[$name] ${GREEN}运行中${NC} - PID: $pid"
            local gpu_mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep "^$pid," | cut -d',' -f2)
            [ -n "$gpu_mem" ] && echo "  GPU: ${gpu_mem}MiB"
            found=1
        else
            echo -e "[$name] ${YELLOW}已结束${NC} - PID: $pid"
            rm -f "$pid_file"
        fi
    done
    
    [ $found -eq 0 ] && echo "没有正在运行的任务"
    echo "========================================"
}

cmd_stop() {
    local target="${1:-all}"
    
    log_info "停止任务: $target"
    
    for pid_file in "$LOG_DIR"/*.pid; do
        [ -f "$pid_file" ] || continue
        
        local pid=$(cat "$pid_file")
        local name=$(basename "$pid_file" .pid)
        
        if [ "$target" = "all" ] || [ "$target" = "$name" ]; then
            if ps -p "$pid" > /dev/null 2>&1; then
                log_info "停止 [$name] (PID: $pid)..."
                kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
            fi
            rm -f "$pid_file"
        fi
    done
    
    log_success "已停止"
}

cmd_logs() {
    local target="${1:-}"
    
    if [ -n "$target" ]; then
        local log_file=$(ls -t "$LOG_DIR"/*"$target"*.log 2>/dev/null | head -1)
        if [ -n "$log_file" ]; then
            log_info "查看日志: $log_file"
            tail -f "$log_file"
        else
            log_error "未找到日志文件"
        fi
    else
        local log_file=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
        if [ -n "$log_file" ]; then
            log_info "查看最新日志: $log_file"
            tail -f "$log_file"
        else
            log_error "未找到日志文件"
        fi
    fi
}

# =============================================================================
# 结果查看
# =============================================================================

cmd_results() {
    log_info "推理结果汇总"
    echo "========================================"
    
    for result_dir in "$OUTPUT_DIR"/*/; do
        [ -d "$result_dir" ] || continue
        
        local name=$(basename "$result_dir")
        if [ -f "$result_dir/metrics.csv" ]; then
            local last_line=$(tail -1 "$result_dir/metrics.csv" 2>/dev/null)
            echo -e "[$name] ${GREEN}✓${NC}"
            echo "  路径: $result_dir"
            echo "  最新指标: $last_line"
            echo ""
        else
            echo -e "[$name] ${YELLOW}无指标文件${NC}"
        fi
    done
    echo "========================================"
}

# =============================================================================
# 清理命令
# =============================================================================

cmd_clean() {
    log_info "清理旧文件..."
    
    # 清理7天前的日志
    find "$LOG_DIR" -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
    find "$LOG_DIR" -name "*.pid" -type f -mtime +1 -delete 2>/dev/null || true
    
    log_success "清理完成"
}

# =============================================================================
# 帮助信息
# =============================================================================

cmd_help() {
    cat << 'EOF'
VisionNDVI 统一运行脚本

用法: ./run.sh [命令] [选项]

命令:
  inference [模式] [选项]   运行推理
    模式:
      rolling              滚动推理（默认，支持在线微调）
      nofinetune           无微调推理
    选项:
      -s, --start DATE     开始日期 (默认: 201501)
      -e, --end DATE       结束日期 (默认: 202012)
      --labeled-end DATE   有标签数据截止日期
      --no-finetune        禁用在线微调
      --bg, --background   后台运行
      --viz-formats        可视化格式 (默认: png, 可用: png,jpg,svg,pdf)
      --save-residuals     保存残差分布图
      --mask-path          指定mask文件路径 (用于无mask的数据集)

  batch [开始] [结束]       批量运行所有模型

  train [模式] [选项]       训练模型
    模式:
      adapter              训练 Adapter（默认）
    选项:
      --bg, --background   后台运行
      --epochs, -e N       训练轮数 (默认: 5)
      --lr LR              学习率 (默认: 2e-3)
      --start-date, -s     开始日期 (默认: 198201)
      --end-date, -e       结束日期 (默认: 201412)
      --adapter-type, -t   Adapter类型 (默认: DeepMultiTimeAdapter)
      --window-size, -w    历史窗口大小 (默认: 3)
      --iterations, -i     每样本迭代次数 (默认: 50)

  status (s)               查看任务状态
  stop [任务名|all]        停止任务
  logs [任务名]            查看日志
  results (r)              查看结果汇总
  clean (c)                清理旧日志
  help (h)                 显示帮助

环境变量:
  DATASET_DIR              数据集目录 (默认: ./datasets/AWI-CM-1-1-MR/)
  BASE_MODEL               基础模型路径 (默认: ./checkpoints/AWI_prediction_model.pth)
  ADAPTER                  Adapter路径 (默认: ./checkpoints/adapter_v3/DeepMultiTimeAdapter_best.pth)
  OUTPUT_DIR               输出目录 (默认: ./results)

示例:
  # 滚动推理（有标签期间在线微调）
  ./run.sh inference rolling --start 201501 --end 205012 --labeled-end 202012

  # 无微调推理（后台运行）
  ./run.sh inference nofinetune --start 202101 --end 205012 --bg

  # 批量推理所有模型
  ./run.sh batch 201501 202012

  # 训练 Adapter
  ./run.sh train adapter --epochs 10 --bg

  # 查看状态并查看日志
  ./run.sh status
  ./run.sh logs inference
EOF
}

# =============================================================================
# 主入口
# =============================================================================

main() {
    local cmd="${1:-help}"
    shift 2>/dev/null || true
    
    case "$cmd" in
        inference|pred|predict)
            cmd_inference "$@"
            ;;
        batch)
            cmd_batch "$@"
            ;;
        train)
            cmd_train "$@"
            ;;
        status|s)
            cmd_status
            ;;
        stop|kill|k)
            cmd_stop "$@"
            ;;
        logs|log|l)
            cmd_logs "$@"
            ;;
        results|r)
            cmd_results
            ;;
        clean|c)
            cmd_clean
            ;;
        help|h|--help|-h)
            cmd_help
            ;;
        *)
            log_error "未知命令: $cmd"
            echo "使用 './run.sh help' 查看帮助"
            exit 1
            ;;
    esac
}

main "$@"
