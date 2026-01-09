#!/usr/bin/env python3
"""
tif2npy.py
按指定年月范围把多波段 GeoTIFF 拆成单波段 .npy
用法：python tif2npy.py
"""

import os, re, glob, numpy as np, rioxarray as rxr, pandas as pd
from tqdm import tqdm

# ================== 仅改这里 ==================
INPUT_DIR   = "./dataset/Temperature/"
OUTPUT_DIR  = "./dataset/Temperature/"
START_YYYYMM = 198201   # 起始年月（含）
END_YYYYMM   = 201412   # 结束年月（含）
# ============================================

def yyyymm_to_datetime(yyyymm_int):
    """198201 → Timestamp('1982-01-01')"""
    y, m = divmod(yyyymm_int, 100)
    return pd.Timestamp(year=y, month=m, day=1)

START_DATE = yyyymm_to_datetime(START_YYYYMM)
END_DATE   = yyyymm_to_datetime(END_YYYYMM)

def sanitize_var(name):
    return name.split('_')[0]

def parse_times(da):
    """
    返回 list[(index, YYYYMM_int)]，只保留在 [START_DATE, END_DATE] 内的
    支持 'time' 坐标 或 'band' 维
    """
    if 'time' in da.coords:
        times = pd.to_datetime(da['time'].values)
        idx_yyyymm = [(i, t.year*100 + t.month) for i, t in enumerate(times)
                      if START_DATE <= t <= END_DATE]
    elif 'band' in da.dims and da.sizes['band'] > 1:
        # 没有时间坐标，按 band1..bandN 依次映射到连续月份
        # 假设 band1=START_DATE, band2=START_DATE+1month ...
        total_bands = da.sizes['band']
        base = pd.date_range(START_DATE, periods=total_bands, freq='MS')
        idx_yyyymm = [(i, d.year*100 + d.month) for i, d in enumerate(base)
                      if d <= END_DATE]
    else:
        # 单波段
        idx_yyyymm = [(0, START_YYYYMM)] if START_DATE == END_DATE else []
    return idx_yyyymm

def process_one_tif(src_path):
    da = rxr.open_rasterio(src_path, masked=True)
    idx_yyyymm = parse_times(da)
    if not idx_yyyymm:
        return 0

    var_name = sanitize_var(os.path.basename(src_path))
    rel_dir  = os.path.relpath(os.path.dirname(src_path), INPUT_DIR)
    out_dir  = os.path.join(OUTPUT_DIR, rel_dir)
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
    tif_files = glob.glob(os.path.join(INPUT_DIR, "**/*.tif*"), recursive=True)
    if not tif_files:
        print("❌ 未发现 .tif/.tiff")
        return

    total_written = 0
    for fp in tqdm(tif_files, desc="Scanning"):
        da = rxr.open_rasterio(fp, masked=True)
        idx_yyyymm = parse_times(da)
        expected = (END_YYYYMM // 100 - START_YYYYMM // 100) * 12 \
                 + (END_YYYYMM % 100 - START_YYYYMM % 100) + 1
        actual   = len(idx_yyyymm)
        print(f"{os.path.basename(fp):<40} 期望 {expected} 月  实际 {actual} 月")
        total_written += process_one_tif(fp)

    print(f"✅ 全部完成！共写出 {total_written} 个 .npy")

if __name__ == "__main__":
    main()