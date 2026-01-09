import os
import rasterio
import numpy as np
import pandas as pd

# 获取所有TIF文件的路径
def get_tif_files(directory):
    tif_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tif"):
                tif_files.append(os.path.join(root, file))
    return tif_files

# 读取TIF文件为数组
def read_tif(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # 读取第一波段
        data = np.nan_to_num(data)  # 替换NaN值
    return data

# 统计TIF文件的基本信息
def calculate_statistics(data, file_path):
    stats = {
        'file': os.path.basename(file_path),
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'nan_count': np.isnan(data).sum(),
        'inf_count': np.isinf(data).sum(),
        'shape': data.shape
    }
    return stats

# 检查并统计TIF文件
def check_tif_files(tif_files, output_file='tif_statistics.csv'):
    stats_list = []
    for file_path in tif_files:
        try:
            data = read_tif(file_path)
            stats = calculate_statistics(data, file_path)
            stats_list.append(stats)
            print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # 保存统计结果为CSV文件
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_file, index=False)
    print(f"统计信息已保存至 {output_file}")

# 主函数
def main(directory):
    tif_files = get_tif_files(directory)
    print(f"发现 {len(tif_files)} 个 TIF 文件")
    check_tif_files(tif_files)

if __name__ == "__main__":
    dataset_dir = "./dataset"  # 替换为你的数据集路径
    main(dataset_dir)
