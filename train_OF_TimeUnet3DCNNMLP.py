import os
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import csv
import gc
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
import numpy as np

gc.collect()
torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = True

# 初始化分布式模式
def init_distributed_mode(local_rank, world_size):
    # 设置环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # 主节点的 IP 地址
    os.environ['MASTER_PORT'] = '29500'  # 通信的端口号
    
    # 其余的初始化逻辑
    torch.distributed.init_process_group(
        backend='nccl',  # 使用 NCCL 后端，适用于 GPU
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)  # 设置每个进程使用的 GPU
    print(f"Rank {local_rank}/{world_size} initialized.")

# 清理和释放 GPU 内存
def cleanup():
    dist.destroy_process_group()

# 设置随机种子
def set_random_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 设置随机种子
set_random_seeds(42)

def check_file_sequence(directory, prefix):
    files = sorted([f for f in os.listdir(directory) if f.startswith(prefix)])
    print(f"Found files for {prefix}:")
    for f in files:
        print(f)
    return len(files)

#check_file_sequence('./dataset/Evapotranspiration', 'evspsbl_5m_his')
#check_file_sequence('./dataset/NDVI_Monthly', 'NDVI_Processed')
#check_file_sequence('./dataset/Precipitation', 'pr_5m_his')
#check_file_sequence('./dataset/SOV', 'SOV_5m_his')
#check_file_sequence('./dataset/Temperature', 'tas_5m_his')

# 获取所有NPY文件的路径
def get_npy_files(directory):
    npy_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".npy"):
                category = os.path.basename(root)  # 获取子文件夹名作为类别
                if category not in npy_files:
                    npy_files[category] = []
                npy_files[category].append(os.path.join(root, file))
    return npy_files

def filter_files_by_date(tif_files, date_format="%Y%m", cutoff_date="201412"):
    filtered_files = {}
    cutoff_datetime = datetime.strptime(cutoff_date, date_format)
    for category, files in tif_files.items():
        filtered_files[category] = []
        for file in files:
            try:
                # 从文件名中提取日期
                date_str = os.path.basename(file).split("_")[-1].split(".")[0]
                file_datetime = datetime.strptime(date_str, date_format)
                if file_datetime <= cutoff_datetime:
                    filtered_files[category].append(file)
            except ValueError:
                # 如果文件名中没有日期信息，直接跳过
                filtered_files[category].append(file)

    return filtered_files

def read_npy(file_path, category):
    data = np.load(file_path)
    data = np.nan_to_num(data, nan=-100.0)  # 替换NaN值为-100

    # 根据类别处理空值和非空值
    if category == 'elevation':
        data[data == -32768] = -100  # 将原始空值替换为-100
        data = data / 10.0  # 非空值除以10
    elif category == 'slope':
        data[data == 3.4e+38] = -100  # 将原始空值替换为-100
    elif category in ['NDVI_Monthly']:
        data[data == 0] = -100  # 将原始空值替换为-100
        data = data * 1000.0  # 非空值乘以1000
    elif category in ['Evapotranspiration', 'Precipitation']:
        data[data == 1e+20] = -100  # 将原始空值替换为-100
        data = data * 1000000.0  # 非空值乘以1000000
    elif category in ['SOV', 'Temperature']:
        data[data == 1e+20] = -100  # 将原始空值替换为-100
        
    # Check for NaNs or extreme values after processing
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"Warning: NaNs or Infs found in {file_path} after processing")
    
    return data

# 处理并加载所有特征和标签文件
def load_and_process_npy_files(npy_files):
    processed_feature_files = {}
    processed_label_files = {}

    for category, files in npy_files.items():
        if category == 'NDVI_Monthly':  # 标签文件
            processed_label_files[category] = files  # Store file paths, not loaded arrays
        else:  # 特征文件
            processed_feature_files[category] = files  # Store file paths, not loaded arrays

    return processed_feature_files, processed_label_files

class NDVIDataset(Dataset):
    def __init__(self, feature_files, label_files, slope, elevation, mask_path, time_window=3):
        self.feature_files = feature_files
        self.label_files = label_files
        self.slope_path = slope
        self.elevation_path = elevation
        self.mask = torch.tensor(np.load(mask_path), dtype=torch.float32)
        self.time_window = time_window  # 设定时间窗口大小

        # 排除静态特征文件，计算动态特征的最小长度
        dynamic_feature_files = {category: files for category, files in self.feature_files.items() if category not in ['dataset']}
        self.min_len = min(len(label_files['NDVI_Monthly']), *[len(v) for v in dynamic_feature_files.values()])
        print(f"Min_len: {self.min_len}, Time window: {self.time_window}")

    def __len__(self):
        # 返回数据集大小，考虑时间窗口的限制
        return self.min_len - self.time_window

    def __getitem__(self, idx):
        current_idx = idx + self.time_window // 2
    
        # 加载标签数据
        label_path = self.label_files['NDVI_Monthly'][current_idx]
        labels = np.load(label_path)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
        # 加载时间序列特征
        features_tensors = []
        for category in self.feature_files:
            if category == 'dataset':  # 跳过静态特征
                continue
            feature_data = []
            for i in range(self.time_window):
                idx_with_offset = current_idx + i - self.time_window // 2
                feature_data.append(np.load(self.feature_files[category][idx_with_offset]))
    
            feature_tensor = torch.stack([torch.tensor(f, dtype=torch.float32) for f in feature_data], dim=0)
            features_tensors.append(feature_tensor)
    
        # 加载静态特征（slope 和 elevation），并扩展维度
        slope_data = np.load(self.slope_path)
        elevation_data = np.load(self.elevation_path)
    
        # 通过 unsqueeze 扩展静态特征的维度，使其变为 [time_window, height, width]
        slope_tensor = torch.tensor(slope_data, dtype=torch.float32).unsqueeze(0).repeat(self.time_window, 1, 1)
        elevation_tensor = torch.tensor(elevation_data, dtype=torch.float32).unsqueeze(0).repeat(self.time_window, 1, 1)
    
        # 将扩展后的静态特征添加到特征列表中
        features_tensors.append(slope_tensor)
        features_tensors.append(elevation_tensor)
    
        # 堆叠所有特征并应用掩码
        stacked_features = torch.stack(features_tensors, dim=0) * self.mask
        return stacked_features, labels_tensor

class UNetEnhancedCNNModel(nn.Module):
    def __init__(self, time_window=3):
        super(UNetEnhancedCNNModel, self).__init__()

        self.time_window = time_window  # 时间窗口

        # Encoding path (downsampling) with 3D convolution
        self.enc1 = nn.Conv3d(6, 16, kernel_size=(3, 3, 3), padding=1)
        self.enc2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.enc3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.enc4 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)

        # Decoding path (upsampling) with 3D convolution
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=1)
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=1)
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=1)

        # Final output layer to map to a single output channel
        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):

        # 合并时间窗口和通道维度
        x = x.view(x.size(0), self.time_window, 6, x.size(3), x.size(4))  # (batch_size, time_window, channels, height, width)
    
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, channels, time_window, height, width)
    
        # Encoding path with 3D convolutions
        enc1 = F.leaky_relu(self.enc1(x), negative_slope=0.01)
    
        enc2 = F.leaky_relu(self.enc2(F.max_pool3d(enc1, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=1)), negative_slope=0.01)
    
        enc3 = F.leaky_relu(self.enc3(F.max_pool3d(enc2, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=1)), negative_slope=0.01)
    
        enc4 = F.leaky_relu(self.enc4(F.max_pool3d(enc3, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=1)), negative_slope=0.01)
    
        # Decoding path with skip connections and size adjustment
        dec3 = F.leaky_relu(self.dec3(torch.cat([F.interpolate(self.upconv3(enc4), size=enc3.shape[2:]), enc3], dim=1)), negative_slope=0.01)
    
        dec2 = F.leaky_relu(self.dec2(torch.cat([F.interpolate(self.upconv2(dec3), size=enc2.shape[2:]), enc2], dim=1)), negative_slope=0.01)
    
        dec1 = F.leaky_relu(self.dec1(torch.cat([F.interpolate(self.upconv1(dec2), size=enc1.shape[2:]), enc1], dim=1)), negative_slope=0.01)

        # Final output layer to map to a single output channel
        outputs = self.final(dec1)
    
        # 在时间维度上求平均，得到 [batch_size, 1, height, width]
        outputs = torch.mean(outputs, dim=2)  # 在时间维度上进行平均
    
        # 去除单一的通道维度（如果有的话）
        return outputs.squeeze(1)  # (batch_size, height, width)
        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_losses_to_csv(epoch_losses, filename="losses.csv"):
    """
    Save training, validation, and test losses to a CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Test Loss"])
        writer.writerows(epoch_losses)

def plot_prediction_vs_actual(predictions, actuals, title, epoch):
    """
    Plot predictions vs actual values and save in both SVG and TIFF formats.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(predictions, cmap='viridis', vmin=0, vmax=1)  # Set colorbar range [0, 1]
    plt.title('Predictions')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(actuals, cmap='viridis', vmin=0, vmax=1)  # Set colorbar range [0, 1]
    plt.title('Actuals')
    plt.colorbar()

    plt.suptitle(f'{title} - Epoch {epoch}')
    
    # Save both SVG and TIFF formats
    plt.savefig(f'predictions_vs_actuals_epoch_{epoch}.svg', format='svg')
    plt.savefig(f'predictions_vs_actuals_epoch_{epoch}.tiff', format='tiff', dpi=300)
    plt.close()

# 定义模型训练和评估函数
def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10, accumulation_steps=16, local_rank=0, device=None):
    model = model.to(device)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision
    epoch_losses = []  # List to store losses for each epoch

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Clear gradients at the start of each epoch

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Mixed precision training
            with autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)

            # Scale loss and accumulate gradients
            scaler.scale(loss).backward()

            # Perform gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)  # Apply gradients to update weights
                scaler.update()  # Update the scaler for mixed precision
                optimizer.zero_grad()  # Reset gradients after update

                torch.cuda.empty_cache()  # Clear cache after updating weights
                
            train_loss += loss.item()

        # Compute average training loss
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions, val_actuals = [], []

        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(val_loader):
                features, labels = features.to(device), labels.to(device)
                with autocast():
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

                if batch_idx < 1:  # Save the first batch for visualization
                    val_predictions.append(outputs.cpu().numpy())
                    val_actuals.append(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_actuals = np.concatenate(val_actuals, axis=0)

        if epoch % 2 == 1:  # Save visualizations every two epochs
            plot_prediction_vs_actual(val_predictions[0], val_actuals[0], f'Epoch {epoch + 1}', epoch + 1)

        # Test phase
        test_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                with autocast():
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        # Log epoch losses
        epoch_losses.append([epoch + 1, train_loss, val_loss, test_loss])

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Val Loss = {val_loss}, Test Loss = {test_loss}")

    # Save all losses to a CSV file
    save_losses_to_csv(epoch_losses)

# 分布式训练的主入口函数
def main(local_rank, world_size):
    init_distributed_mode(local_rank, world_size)

    # 获取设备
    device = torch.device(f'cuda:{local_rank}')

    # 定义数据集文件夹路径
    dataset_dir = "./dataset"

    # 获取所有特征文件和标签文件
    npy_files = get_npy_files(dataset_dir)

    # 过滤掉201412之前的文件
    filtered_npy_files = filter_files_by_date(npy_files)

    # 处理并加载所有的特征和标签文件
    feature_files, label_files = load_and_process_npy_files(filtered_npy_files)

    # 读取elevation和slope数据的路径
    slope = "./dataset/slope.npy"
    elevation = "./dataset/elevation.npy"
    mask_path = "./dataset/mask.npy"

    # 创建数据集对象
    dataset = NDVIDataset(feature_files, label_files, elevation, slope, mask_path, time_window=12)

    # 获取总数据集大小
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")

    # 手动划分数据集
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # 划分数据集
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))

    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)

    # 使用 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=4, pin_memory=True)

    # 设置模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEnhancedCNNModel(time_window=12).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 设置损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # 开始训练
    train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=200, local_rank=local_rank, device=device)

    # 保存模型
    if local_rank == 0:  # 只在主进程保存模型
        torch.save(model.state_dict(), "ndvi_prediction_model.pth")

    cleanup()

# 启动分布式训练
def run():
    world_size = torch.cuda.device_count()  # 获取GPU数量
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    run()