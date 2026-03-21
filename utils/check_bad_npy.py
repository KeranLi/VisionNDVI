import os
import numpy as np
from collections import defaultdict

# 1. 配置你的目录和期望 shape
ROOT_DIR   = './dataset'          # 数据集根目录
TARGET_SHAPE = (2160, 4320)       # 期望的二维 shape
CATEGORIES = ['Evapotranspiration', 'Precipitation', 'SOV',
              'Temperature', 'NDVI_Monthly']

# 2. 遍历函数
def walk_npy(root):
    """返回 dict[category] -> list of absolute paths"""
    dd = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith('.npy'):
                cat = os.path.basename(dirpath)
                dd[cat].append(os.path.join(dirpath, f))
    return dd

# 3. 检查单个文件
def check_one(path):
    """
    返回 (is_ok, reason)
    is_ok : bool
    reason: str or None
    """
    if os.path.getsize(path) == 0:
        return False, 'empty_file'
    try:
        arr = np.load(path, mmap_mode='r')   # 只映射，不占内存
    except Exception as e:
        return False, f'load_error:{e}'
    if arr.shape != TARGET_SHAPE:
        return False, f'bad_shape:{arr.shape}'
    return True, None

# 4. 主流程
if __name__ == '__main__':
    all_files = walk_npy(ROOT_DIR)
    bad_list  = []          # (path, reason)
    ok_list   = []

    for cat in CATEGORIES:
        for p in all_files.get(cat, []):
            ok, reason = check_one(p)
            if ok:
                ok_list.append(p)
            else:
                bad_list.append((p, reason))

    # 5. 报告
    print('==========  Bad Files  ==========')
    for p, reason in bad_list:
        print(f'{reason:15}  {p}')
    print(f'\nTotal bad  files: {len(bad_list)}')
    print(f'Total good files: {len(ok_list)}')

    # 6. 可选：把黑名单写成 txt，方便二次处理
    with open('bad_npy_files.txt', 'w') as f:
        for p, reason in bad_list:
            f.write(f'{reason}\t{p}\n')