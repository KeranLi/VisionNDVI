#!/usr/bin/env python3
"""
make_sov.py
逐月把 rlds.npy + rsds.npy → SOV.npy
用法:  python make_sov.py
"""

import os
import glob
import numpy as np

# ========= 只改这里 =========
ROOT_RLDS = "/root/autodl-tmp/dataset/rlds"
ROOT_RSDS = "/root/autodl-tmp/dataset/rsds"
ROOT_SOV  = "/root/autodl-tmp/dataset/SOV"
# 如需限制时间区间，可在此添加 START_YYYYMM / END_YYYYMM
# ============================

def yyyymm_list():
    """生成 198201 … 201412 的字符串列表，按需可改"""
    from datetime import datetime, timedelta
    d = datetime(1982, 1, 1)
    months = []
    while d <= datetime(2014, 12, 1):
        months.append(d.strftime("%Y%m"))
        d += timedelta(days=32)
        d = d.replace(day=1)
    return months

def iter_month_paths(root):
    """返回 dict {yyyymm: full_path}"""
    pattern = os.path.join(root, "**", "*.npy")
    files = glob.glob(pattern, recursive=True)
    mapping = {}
    for fp in files:
        yyyymm = os.path.basename(fp).split('.')[0]   # 201401.npy → 201401
        mapping[yyyymm] = fp
    return mapping

def make_one(ym, rlds_path, rsds_path):
    sov_dir = os.path.join(
        ROOT_SOV,
        os.path.relpath(os.path.dirname(rlds_path), ROOT_RLDS)
    )
    os.makedirs(sov_dir, exist_ok=True)
    out_file = os.path.join(sov_dir, f"{ym}.npy")
    if os.path.exists(out_file):
        return  # 断点续跑

    rlds = np.load(rlds_path, mmap_mode='r')
    rsds = np.load(rsds_path, mmap_mode='r')

    if rlds.shape != rsds.shape:
        raise ValueError(f"Shape mismatch {rlds.shape} vs {rsds.shape} for {ym}")

    sov = (rlds.astype(np.float32) + rsds.astype(np.float32))
    np.save(out_file, sov)

def main():
    rlds_map = iter_month_paths(ROOT_RLDS)
    rsds_map = iter_month_paths(ROOT_RSDS)

    common_months = sorted(set(rlds_map) & set(rsds_map))
    if not common_months:
        print("❌ 找不到任何公共月份")
        return

    print(f"发现 {len(common_months)} 个公共月份，开始生成 SOV …")
    for ym in common_months:
        make_one(ym, rlds_map[ym], rsds_map[ym])
    print("✅ SOV 全部生成完毕！")

if __name__ == "__main__":
    main()