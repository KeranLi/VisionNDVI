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
    def __init__(self, feature_files, label_files, slope, elevation, mask_path):
        # Store paths only, not loaded arrays
        self.feature_files = {
            'Evapotranspiration': feature_files.get('Evapotranspiration', []),
            'Precipitation': feature_files.get('Precipitation', []),
            'SOV': feature_files.get('SOV', []),
            'Temperature': feature_files.get('Temperature', []),
        }
        self.label_files = label_files
        self.slope_path = slope  # Store path instead of loading slope
        self.elevation_path = elevation  # Store path instead of loading elevation
        self.mask = torch.tensor(np.load(mask_path), dtype=torch.float32)

        # Determine minimum length across all feature categories and labels
        self.min_len = min(len(label_files['NDVI_Monthly']), *[len(v) for v in self.feature_files.values()])

    def __len__(self):
        return self.min_len

    def __getitem__(self, idx):
        # Load label data
        label_path = self.label_files['NDVI_Monthly'][idx]
        labels = np.load(label_path)
        
        # Convert labels to PyTorch tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32) * self.mask / 1000.0
        
        # Load and add each feature category
        features_tensors = []
        for category in self.feature_files:
            feature_path = self.feature_files[category][idx]
            feature = np.load(feature_path)
            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            features_tensors.append(feature_tensor)
            del feature  # Free memory

        # Load slope and elevation data and add to features_tensors
        slope_data = np.load(self.slope_path)
        elevation_data = np.load(self.elevation_path)
        slope_tensor = torch.tensor(slope_data, dtype=torch.float32)
        elevation_tensor = torch.tensor(elevation_data, dtype=torch.float32)
        
        features_tensors.append(slope_tensor)
        features_tensors.append(elevation_tensor)

        # Stack features along a new dimension for weighted combination in the model
        stacked_features = torch.stack(features_tensors, dim=0) * self.mask

        # 删除 features_tensors 列表以释放内存
        del features_tensors
        
        # 打印 stacked_features 的形状，确保有6个通道
        #print(f"stacked_features shape: {stacked_features.shape}")
        
        return stacked_features, labels_tensor

# 定义数据集文件夹
dataset_dir = "./dataset"

# 获取所有特征文件和标签文件
npy_files = get_npy_files(dataset_dir)

# 过滤掉201412之前的文件
filtered_npy_files = filter_files_by_date(npy_files)

# 处理并加载所有的特征和标签文件
feature_files, label_files = load_and_process_npy_files(filtered_npy_files)

# 读取elevation和slope数据
slope = "./dataset/slope.npy"  # Store only the file path, not the loaded data
elevation = "./dataset/elevation.npy"  # Store only the file path, not the loaded data
mask_path = "./dataset/mask.npy"  # 确保 mask 文件路径正确

# 创建数据集
dataset = NDVIDataset(feature_files, label_files, elevation, slope, mask_path)

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

class SimpleCNNModel(nn.Module):
    def __init__(self):
        super(SimpleCNNModel, self).__init__()
        
        # Define a set of learnable weights for the six features
        self.feature_weights = nn.Parameter(torch.ones(6))  # Initialize weights with 1
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1)  # 输入通道数改为6
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1)  # 1 output channel for final spatial feature extraction

        # Global Average Pooling to reduce dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((16, 16))  # 将输出缩小到16x16

        # MLP layers after global pooling
        self.fc1 = nn.Linear(16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 16 * 16)  # Output size matches pooled feature map

        # Upsample layer to restore to original size
        self.upsample = nn.Upsample(size=(2154, 4320), mode='bilinear', align_corners=False)  # 上采样到原始大小

        # 使用固定的初始化方法
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalize feature weights to ensure they sum to 1 (softmax or other normalization)
        normalized_weights = F.softmax(self.feature_weights, dim=0)
        
        # Weighted sum of features along the first dimension
        weighted_features = torch.sum(x * normalized_weights.view(-1, 1, 1), dim=0, keepdim=True)
        
        # Apply convolution layers
        x = F.leaky_relu(self.conv1(weighted_features), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)  # Output shape: (batch_size, 1, H, W)

        # Apply Global Average Pooling
        x = self.global_avg_pool(x).view(x.size(0), -1)  # Flatten after pooling

        # Apply MLP layers
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.fc3(x)

        # Reshape back to match the pooled dimensions and upsample
        x = x.view(-1, 1, 16, 16)
        x = self.upsample(x)  # Upsample to (batch_size, 1, 2154, 4320)
        outputs = x.squeeze(1)
        
        return outputs  # Final output matches (batch_size, 1, 2154, 4320)

# Step 1: Check for multiple GPUs, move model to the appropriate device(s)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    device = torch.device("cuda:0")  # Primary device where model will reside
    model = SimpleCNNModel().to(device)  # Move model to primary GPU
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use nn.DataParallel with specified GPUs
else:
    print("Using a single GPU or CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNNModel().to(device)  # Move model to GPU or CPU as needed

# Step 2: Set up criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

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

# 调用训练和评估函数
train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=70)

# 保存模型
torch.save(model.state_dict(), "ndvi_prediction_model.pth")