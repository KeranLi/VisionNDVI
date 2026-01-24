import os
import numpy as np
import rioxarray as rxr
import pandas as pd
import sys
import re
import glob

# 获取输入的参数
src_path = sys.argv[1]  # 当前 .tif 文件路径
output_dir = sys.argv[2]  # 输出目录

# ================== 仅改这里 ==================
INPUT_DIR = "E:/scenarioMIP_output/future_resampled_this"
OUTPUT_DIR = output_dir  # 使用从批处理脚本传递的 OUTPUT_DIR
START_YYYYMM = 201501
END_YYYYMM = 210012
# ============================================

def yyyymm_to_datetime(yyyymm_int):
    """198201 → Timestamp('1982-01-01')"""
    y, m = divmod(yyyymm_int, 100)
    return pd.Timestamp(year=y, month=m, day=1)

START_DATE = yyyymm_to_datetime(START_YYYYMM)
END_DATE = yyyymm_to_datetime(END_YYYYMM)

def sanitize_var(name):
    """ 从文件名中提取变量名，去掉无关的部分 """
    return name.split('_')[1]  # 从 'evspsbl_AWI-CM-1-1-MR_ssp126_2015-01.tif' 提取 'AWI-CM-1-1-MR'

def parse_times(da, src_path):
    """
    返回 list[(index, YYYYMM_int)]，只保留在 [START_DATE, END_DATE] 内的
    支持 'time' 坐标 或 'band' 维
    解析文件名中的日期
    """
    # 从文件路径中提取日期（例如 '2015-01' -> 201501）
    match = re.search(r"(\d{4})-(\d{2})\.tif", os.path.basename(src_path))
    if not match:
        raise ValueError(f"无法从文件名提取日期: {src_path}")
    
    year, month = map(int, match.groups())
    file_yyyymm = year * 100 + month
    idx_yyyymm = [(0, file_yyyymm)] if START_DATE <= pd.Timestamp(year, month, 1) <= END_DATE else []
    
    return idx_yyyymm

def process_one_tif(src_path):
    da = rxr.open_rasterio(src_path, masked=True)
    idx_yyyymm = parse_times(da, src_path)
    if not idx_yyyymm:
        return 0

    var_name = sanitize_var(os.path.basename(src_path))
    rel_dir = os.path.relpath(os.path.dirname(src_path), INPUT_DIR)
    out_dir = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    count_written = 0
    for idx, yyyymm in idx_yyyymm:
        out_file = os.path.join(out_dir, f"{yyyymm}.npy")
        if os.path.exists(out_file):
            continue
        arr = da.isel(time=idx).values.astype(np.float32) if 'time' in da.coords \
              else da.isel(band=idx).values.astype(np.float32)
        np.save(out_file, arr)
        count_written += 1
    return count_written

def main():
    # 处理传递进来的单个 .tif 文件
    total_written = process_one_tif(src_path)
    print(f"✅ 文件 {src_path} 转换完成，共写出 {total_written} 个 .npy")

if __name__ == "__main__":
    main()
