import os
import numpy as np
import rioxarray as rxr
import pandas as pd
import re
import glob

# ================== 设置输入和输出目录 ==================
INPUT_DIR = "E:/scenarioMIP_output/future_resampled_this"  # 你的输入目录
OUTPUT_DIR = "E:/npy"  # 你的输出目录
START_YYYYMM = 201501  # 起始年月（含）
END_YYYYMM = 210012  # 结束年月（含）

# ================== 辅助函数 ==================
def yyyymm_to_datetime(yyyymm_int):
    """将年月格式转换为 pandas Timestamp"""
    y, m = divmod(yyyymm_int, 100)
    return pd.Timestamp(year=y, month=m, day=1)

START_DATE = yyyymm_to_datetime(START_YYYYMM)
END_DATE = yyyymm_to_datetime(END_YYYYMM)

def sanitize_var(name):
    """从文件名中提取变量名，去掉无关的部分"""
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
    """处理单个 .tif 文件并转换为 .npy 文件"""
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
    """主函数：遍历所有 .tif 文件并进行转换"""
    tif_files = glob.glob(os.path.join(INPUT_DIR, "**/*.tif*"), recursive=True)
    if not tif_files:
        print("❌ 未发现 .tif/.tiff 文件")
        return

    total_written = 0
    for fp in tif_files:
        total_written += process_one_tif(fp)

    print(f"✅ 全部完成！共写出 {total_written} 个 .npy 文件")

if __name__ == "__main__":
    main()
