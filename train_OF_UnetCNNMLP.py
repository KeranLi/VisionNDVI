import os
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import csv
torch.backends.cudnn.benchmark = True

TARGET_SHAPE = (2154, 4320)
CATEGORIES = ['Evapotranspiration', 'Precipitation', 'SOV', 'Temperature']

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

def filter_files_by_date(file_dict, date_format="%Y%m", cutoff_date="201412"):
    cutoff = datetime.strptime(cutoff_date, date_format)
    filtered = {}
    for category, files in file_dict.items():
        filtered[category] = []
        for fp in files:
            # 取文件名最后 6 位数字（198201）
            date_str = os.path.basename(fp).split('.')[0][-6:]
            try:
                if datetime.strptime(date_str, date_format) <= cutoff:
                    filtered[category].append(fp)
            except ValueError:
                continue
    return filtered

#def read_npy(file_path, category):
    #data = np.load(file_path)

    # 先统一处理 NoData / NaN / inf
    #data = np.nan_to_num(data, nan=-100.0)
    #data[data == -9999.0] = -100.0      # 合作方给的 NoData
    #data[np.isinf(data)] = -100.0

    # 再按类别做值域修正
    #if category == 'elevation':
        #data[data == -32768] = -100
        #data = data / 100.0
    #elif category == 'slope':
        #data[data == 3.4e+38] = -100
    #elif category == 'NDVI_Monthly':
        #data[data == 0] = -100
        #data = data * 1000.0
    #elif category in ['Evapotranspiration', 'Precipitation']:
        #data[data == 1e+20] = -100
        #data = data * 1e2          # 原来是 1e6，容易 overflow，改小
    #elif category in ['SOV', 'Temperature']:
        #data[data == 1e+20] = -100

    # 兜底再清理一次
    #data = np.nan_to_num(data, nan=-100.0, posinf=1e3, neginf=-1e3)
    #return data

def denormalize(tensor, cat, stats):
    """
    tensor: 0-1 的预测结果（或标签）
    cat   : 对应的变量名
    stats : dataset.stats
    return: 原始物理量
    """
    d_min = stats[cat]['min']
    d_max = stats[cat]['max']
    return tensor * (d_max - d_min) + d_min

def read_npy(file_path, category, stats_dict, normalize=True):
    """
    normalize=True  → 特征做 0-1 归一化
    normalize=False → 标签保持原始值
    """

    if os.path.getsize(file_path) == 0:
        print(f"[Warning] Empty file skipped: {file_path}")
        return np.zeros((2160, 4320), dtype=np.float32)
    
    data = np.load(file_path)
    data = np.nan_to_num(data, nan=-100.0)
    data[data == -9999.0] = -100.0
    data[np.isinf(data)] = -100.0
    data[data < -999] = -100.0

    if normalize:
        valid = data != -100
        if valid.any():
            d_min = float(data[valid].min())
            d_max = float(data[valid].max())
        else:
            d_min, d_max = 0.0, 1.0
        # 缓存 min / max
        if stats_dict[category]['min'] is None:
            stats_dict[category]['min'] = d_min
            stats_dict[category]['max'] = d_max
        # 归一化
        data[valid] = (data[valid] - d_min) / (d_max - d_min + 1e-8)
        data[~valid] = 0.0
    else:
        # 标签：只清 NoData，不做归一化
        data[data == -100] = 0.0   # 或者保留 -100 做掩膜，随你需求
    return data.astype(np.float32)
    
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

# 处理并加载所有特征和标签文件
class NDVIDataset(Dataset):
    def __init__(self, feature_files, label_files, slope, elevation, mask_path):
        self.feature_files = {
            'Evapotranspiration': feature_files.get('Evapotranspiration', []),
            'Precipitation': feature_files.get('Precipitation', []),
            'SOV': feature_files.get('SOV', []),
            'Temperature': feature_files.get('Temperature', []),
        }
        self.label_files = label_files
        self.slope_path = slope
        self.elevation_path = elevation
        self.mask = torch.tensor(np.load(mask_path), dtype=torch.float32)
        self.target_shape = TARGET_SHAPE
        self.stats = {}          # 新增
        for cat in CATEGORIES + ['elevation', 'slope', 'NDVI_Monthly']:
            self.stats[cat] = {'min': None, 'max': None}

    def __len__(self):
        return min(len(self.label_files['NDVI_Monthly']), *[len(v) for v in self.feature_files.values()])

    def __getitem__(self, idx):
        # 1. 标签
        label_path = self.label_files['NDVI_Monthly'][idx]
        labels = read_npy(label_path, 'NDVI_Monthly', self.stats, normalize=True) #不归一化
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        labels_tensor = F.interpolate(
            labels_tensor.unsqueeze(0).unsqueeze(0),
            size=self.target_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0) * self.mask
    
        # 2. 特征
        features_tensors = []
        for cat in CATEGORIES:
            feature_path = self.feature_files[cat][idx]
            feature = read_npy(feature_path, cat, self.stats, normalize=True)
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            feature_tensor = F.interpolate(
                feature_tensor.unsqueeze(0).unsqueeze(0),
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            features_tensors.append(feature_tensor)
    
        # 3. elevation & slope
        elev = read_npy(self.elevation_path, 'elevation', self.stats)
        slop = read_npy(self.slope_path,   'slope', self.stats)
        elev_tensor = torch.tensor(elev, dtype=torch.float32)
        slop_tensor = torch.tensor(slop, dtype=torch.float32)
    
        for t in (elev_tensor, slop_tensor):
            t = F.interpolate(
                t.unsqueeze(0).unsqueeze(0),
                size=self.target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            features_tensors.append(t)

        #for i, ft in enumerate(features_tensors):
            #print(f"Feature {i} min/max/NaN:", ft.min().item(), ft.max().item(), torch.isnan(ft).any().item())

        stacked_features = torch.stack(features_tensors, dim=0) * self.mask

        return stacked_features, labels_tensor

    def visualize_features(self, idx):
        features, labels = self.__getitem__(idx)  # features: [C, H, W]
        features = features.unsqueeze(0)  # 添加 batch 维度：[1, C, H, W]
    
        plt.figure(figsize=(15, 10))
        num_features = features.shape[1]
    
        for i in range(num_features):
            plt.subplot(2, (num_features + 1) // 2, i + 1)
            plt.imshow(features[0, i].numpy(), cmap='viridis')
            plt.title(f"Feature {i}")
            plt.colorbar()
    
        plt.suptitle(f'Feature Visualization - Sample {idx}')
        plt.savefig(f'feature_visualization_sample_{idx}.png')
        plt.close()
    
        # 可视化堆叠特征（可选）
        plt.figure(figsize=(10, 5))
        stacked_img = torch.mean(features[0], dim=0)  # 平均所有通道
        plt.imshow(stacked_img.numpy(), cmap='viridis')
        plt.title('Mean of All Features')
        plt.colorbar()
        plt.savefig(f'stacked_features_sample_{idx}.png')
        plt.close()

# 定义数据集文件夹
dataset_dir = "./dataset"

# 获取所有特征文件和标签文件
npy_files = get_npy_files(dataset_dir)

# 过滤掉201412之前的文件
filtered_npy_files = filter_files_by_date(npy_files)
for c, lst in filtered_npy_files.items():
    print(f"{c}: {len(lst)} files")

# 处理并加载所有的特征和标签文件
feature_files, label_files = load_and_process_npy_files(filtered_npy_files)

# 读取elevation和slope数据
slope = "./dataset/slope.npy"  # Store only the file path, not the loaded data
elevation = "./dataset/elevation.npy"  # Store only the file path, not the loaded data
mask_path = "./dataset/mask.npy"  # 确保 mask 文件路径正确

# 创建数据集
dataset = NDVIDataset(feature_files, label_files, elevation, slope, mask_path)
dataset.visualize_features(0)  # 可视化索引为0的样本特征

# 计算训练、验证和测试集合的大小
total_size = len(dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

# 确保总和为数据集大小
assert train_size + val_size + test_size == total_size

# 使用random_split来划分数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, worker_init_fn=seed_worker)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, worker_init_fn=seed_worker)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, worker_init_fn=seed_worker)

#for _, labels in train_loader:
    #print("Labels:", labels.min().item(), labels.max().item(), torch.isnan(labels).any().item())
    #break

class UNetEnhancedCNNModel(nn.Module):
    def __init__(self):
        super(UNetEnhancedCNNModel, self).__init__()
        
        # Feature weights for input channels
        self.feature_weights = nn.Parameter(torch.ones(6))  # Initialize weights with 1

        # Encoding path (downsampling)
        self.enc1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Decoding path (upsampling)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # Output layer to map to the required single output channel
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
        # MLP layers to apply across feature dimension
        self.fc1 = nn.Linear(1, 64)  # Ensure to match the flattened input size
        self.fc2 = nn.Linear(64, 1)  # Project back to original feature size

    def forward(self, x):
        #print("Input min/max:", x.min().item(), x.max().item(), torch.isnan(x).any().item())
        
        # Normalize feature weights and perform weighted sum across channels
        #normalized_weights = F.softmax(self.feature_weights, dim=0)
        #x = torch.sum(x * normalized_weights.view(-1, 1, 1), dim=0, keepdim=True)
        normalized_weights = F.softmax(self.feature_weights, dim=0)
        x = x * normalized_weights.view(1, -1, 1, 1)  # 通道加权，不压缩维度
        # x 仍然是 [B, 6, H, W]
        #print(f"Weighted sum shape: {x.shape}")  # Debugging info
        
        # Encoding path
        enc1 = F.leaky_relu(self.enc1(x), negative_slope=0.01)
        #print(f"enc1 shape: {enc1.shape}")
        enc2 = F.leaky_relu(self.enc2(F.max_pool2d(enc1, 2)), negative_slope=0.01)
        #print(f"enc2 shape: {enc2.shape}")
        enc3 = F.leaky_relu(self.enc3(F.max_pool2d(enc2, 2)), negative_slope=0.01)
        #print(f"enc3 shape: {enc3.shape}")
        enc4 = F.leaky_relu(self.enc4(F.max_pool2d(enc3, 2)), negative_slope=0.01)
        #print(f"enc4 shape: {enc4.shape}")

        # Decoding path with skip connections and size adjustment
        dec3 = F.leaky_relu(self.dec3(torch.cat([F.interpolate(self.upconv3(enc4), size=enc3.shape[2:], mode='bilinear', align_corners=False), enc3], dim=1)), negative_slope=0.01)
        #print(f"dec3 shape: {dec3.shape}")
        dec2 = F.leaky_relu(self.dec2(torch.cat([F.interpolate(self.upconv2(dec3), size=enc2.shape[2:], mode='bilinear', align_corners=False), enc2], dim=1)), negative_slope=0.01)
        #print(f"dec2 shape: {dec2.shape}")
        dec1 = F.leaky_relu(self.dec1(torch.cat([F.interpolate(self.upconv1(dec2), size=enc1.shape[2:], mode='bilinear', align_corners=False), enc1], dim=1)), negative_slope=0.01)
        #print(f"dec1 shape: {dec1.shape}")

        # Final output layer
        outputs = self.final(dec1)  # Shape (batch_size, 1, H, W)
        #print(f"Outputs shape before MLP: {outputs.shape}")
        
        # Apply MLP channel-wise without affecting spatial resolution
        batch_size, channels, h, w = outputs.shape
        outputs = outputs.permute(0, 2, 3, 1).contiguous()  # Rearrange to (batch_size, H, W, C)
        outputs_flat = outputs.view(batch_size * h * w, -1) # Flatten spatial dimensions while keeping channels

        # Print shapes for debugging
        #print(f"Outputs shape after MLP: {outputs.shape}")
        
        # Apply the MLP to each spatial position independently
        mlp_out = torch.tanh(self.fc1(outputs_flat))        # MLP layer 1 with Tanh
        mlp_out = torch.tanh(self.fc2(mlp_out))             # MLP layer 2 with Tanh
        outputs = mlp_out.view(batch_size, h, w, channels)  # Reshape to (batch_size, H, W, C)
        outputs = outputs.permute(0, 3, 1, 2).contiguous()  # Restore to (batch_size, C, H, W)
        
        return outputs.squeeze(1)
   
# Step 1: Check for multiple GPUs, move model to the appropriate device(s)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    device = torch.device("cuda:0")  # Primary device where model will reside
    model = UNetEnhancedCNNModel().to(device)  # Move model to primary GPU
    model = nn.DataParallel(model, device_ids=[0, 1])  # Use nn.DataParallel with specified GPUs
else:
    print("Using a single GPU or CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEnhancedCNNModel().to(device)  # Move model to GPU or CPU as needed

# Step 2: Set up criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scaler = GradScaler()  # For scaling gradients in mixed precision
accumulation_steps = 4

# Initialize lists to store the first ten predictions and labels
predictions_list = []
labels_list = []

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
    # Remove batch dimension from predictions if it exists
    predictions = predictions.squeeze(0)  # Remove batch dimension if necessary
    print(f"Shapes of predictions: {predictions.shape}")
    plt.imshow(predictions, cmap='viridis', vmin=0, vmax=1)  # Set colorbar range [0, 1]
    plt.title('Predictions')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    print(f"Shapes of actuals: {actuals.shape}")
    plt.imshow(actuals, cmap='viridis', vmin=0, vmax=1)  # Set colorbar range [0, 1]
    plt.title('Actuals')
    plt.colorbar()

    plt.suptitle(f'{title} - Epoch {epoch}')
    
    # Save both SVG and TIFF formats
    plt.savefig(f'predictions_vs_actuals_epoch_{epoch}.svg', format='svg')
    plt.savefig(f'predictions_vs_actuals_epoch_{epoch}.tiff', format='tiff', dpi=300)
    plt.close()

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=10):
    """
    Train and evaluate the model, logging losses and visualizing predictions.
    """
    scaler = GradScaler()  # Initialize GradScaler for mixed precision
    epoch_losses = []  # List to store losses for each epoch

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision training            
            with autocast():
                outputs = model(features)
                outputs = outputs.unsqueeze(0)  # Add batch dimension if necessary
                #print("Outputs:", outputs.min().item(), outputs.max().item(), torch.isnan(outputs).any().item())
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

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
                    # Ensure outputs is in the correct shape (N, C, H, W) before interpolation
                    outputs = outputs.unsqueeze(0)  # Add batch dimension if necessary
                    #outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
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
                    # Ensure outputs is in the correct shape (N, C, H, W) before interpolation
                    outputs = outputs.unsqueeze(0)  # Add batch dimension if necessary
                    #outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
                    loss = criterion(outputs, labels)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        # Log epoch losses
        epoch_losses.append([epoch + 1, train_loss, val_loss, test_loss])

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss}, Val Loss = {val_loss}, Test Loss = {test_loss}")

    # Save all losses to a CSV file
    save_losses_to_csv(epoch_losses)

# 调用训练和评估函数
train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=80)

# 保存模型
torch.save(model.state_dict(), "ndvi_prediction_model.pth")