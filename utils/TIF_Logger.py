import os
import numpy as np
import rasterio
import pandas as pd


# 读取TIF文件并定位异常值的位置
def read_tif_and_find_exceptions(file_path, value_threshold=1e15):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # 读取第一波段

    # 处理 NaN 和 Inf 的位置
    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)

    # 处理极端大值的位置
    extreme_value_mask = data > value_threshold

    # 获取异常值的索引位置
    nan_indices = np.argwhere(nan_mask)
    inf_indices = np.argwhere(inf_mask)
    extreme_value_indices = np.argwhere(extreme_value_mask)

    return nan_indices, inf_indices, extreme_value_indices


# 遍历所有TIF文件，生成异常值位置的报告
def generate_exception_report(tif_files, output_csv='exception_report.csv', value_threshold=1e10):
    exceptions = []

    for file_path in tif_files:
        print(f"Processing file: {file_path}")
        nan_indices, inf_indices, extreme_value_indices = read_tif_and_find_exceptions(file_path, value_threshold)

        # 记录异常位置信息
        for row, col in nan_indices:
            exceptions.append([os.path.basename(file_path), 'NaN', row, col])
        for row, col in inf_indices:
            exceptions.append([os.path.basename(file_path), 'Inf', row, col])
        for row, col in extreme_value_indices:
            exceptions.append([os.path.basename(file_path), 'Extreme Value', row, col])

    # 将异常信息保存为CSV
    df = pd.DataFrame(exceptions, columns=['File', 'Exception Type', 'Row', 'Column'])
    df.to_csv(output_csv, index=False)
    print(f"异常值报告已生成: {output_csv}")


# 获取所有TIF文件的路径，忽略NDVI_Monthly文件夹
def get_tif_files(directory):
    tif_files = []
    for root, dirs, files in os.walk(directory):
        if 'NDVI_Monthly' not in root:  # 忽略 NDVI_Monthly 文件夹
            for file in files:
                if file.endswith(".tif"):
                    tif_files.append(os.path.join(root, file))
    return tif_files


# 主函数
def main(dataset_dir, output_csv='exception_report.csv', value_threshold=1e10):
    tif_files = get_tif_files(dataset_dir)
    print(f"发现 {len(tif_files)} 个 TIF 文件（不包括 NDVI_Monthly 文件夹）")
    generate_exception_report(tif_files, output_csv, value_threshold)


if __name__ == "__main__":
    dataset_dir = "./dataset"  # 替换为你的数据集路径
    main(dataset_dir, output_csv="exception_report.csv", value_threshold=1e10)
