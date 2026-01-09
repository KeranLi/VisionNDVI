import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import csv

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

# 获取所有TIF文件的路径
def get_tif_files(directory):
    tif_files = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tif"):
                category = os.path.basename(root)  # 获取子文件夹名作为类别
                if category not in tif_files:
                    tif_files[category] = []
                tif_files[category].append(os.path.join(root, file))
    return tif_files

# 从TIF文件中加载数据
def load_tif(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()  # Assuming the TIF file has multiple bands (time steps)
    return data

# 处理并加载所有特征和标签文件
def load_and_process_tif_files(tif_files):
    processed_feature_files = {}
    processed_label_files = {}

    # 假设 'NDVI_output' 是标签文件夹
    label_category = 'NDVI_output'
    for category, files in tif_files.items():
        if category == label_category:  # 标签文件
            processed_label_files[category] = files
        else:  # 特征文件
            processed_feature_files[category] = files

    # 检查是否正确填充了 label_files
    if not processed_label_files:
        raise ValueError(f"No label files found in the dataset. Please check the label category: {label_category}")

    return processed_feature_files, processed_label_files

class NDVIDataset(Dataset):
    def __init__(self, feature_files, label_files, mask_path):
        # Store paths only, not loaded arrays
        self.feature_files = {
            'evspsbl': feature_files.get('evspsbl', []),
            'pr': feature_files.get('pr', []),
            'rsds': feature_files.get('rsds', []),
            'tas': feature_files.get('tas', []),
            'rlds': feature_files.get('rlds', []),  # 添加 rlds 作为特征文件
        }
        self.label_files = label_files
        self.mask = torch.tensor(load_tif(mask_path), dtype=torch.float32)

        # Load all feature and label data into memory
        self.features_data = {}
        self.labels_data = []

        for category, files in self.feature_files.items():
            self.features_data[category] = load_tif(files[0])  # Assuming each category has only one file

        label_category = list(label_files.keys())[0]
        for file in label_files[label_category]:
            self.labels_data.append(load_tif(file))  # Load each label file

        # Concatenate all label data along the time dimension
        self.labels_data = np.concatenate(self.labels_data, axis=0)

        # Determine the number of time steps
        self.num_time_steps = self.labels_data.shape[0]

    def __len__(self):
        return self.num_time_steps

    def __getitem__(self, idx):
        # Extract the time step data from each feature and label
        features_tensors = []
        for category, data in self.features_data.items():
            feature_tensor = torch.tensor(data[idx, :, :], dtype=torch.float32)
            features_tensors.append(feature_tensor)

        labels_tensor = torch.tensor(self.labels_data[idx, :, :], dtype=torch.float32) * self.mask / 1000.0

        # Stack features along a new dimension for weighted combination in the model
        stacked_features = torch.stack(features_tensors, dim=0) * self.mask

        return stacked_features, labels_tensor

# 定义数据集文件夹
dataset_dir = "./datasets"

# 获取所有特征文件和标签文件
tif_files = get_tif_files(dataset_dir)

# 处理并加载所有的特征和标签文件
feature_files, label_files = load_and_process_tif_files(tif_files)

# 读取mask数据
mask_path = "./datasets/Spatial/globalmask.tif"  # 确保 mask 文件路径正确

# 创建数据集
dataset = NDVIDataset(feature_files, label_files, mask_path)

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

    def forward(self, x):
        # Normalize feature weights and perform weighted sum across channels
        normalized_weights = F.softmax(self.feature_weights, dim=0)
        x = torch.sum(x * normalized_weights.view(-1, 1, 1), dim=0, keepdim=True)

        # Encoding path
        enc1 = F.leaky_relu(self.enc1(x), negative_slope=0.01)
        enc2 = F.leaky_relu(self.enc2(F.max_pool2d(enc1, 2)), negative_slope=0.01)
        enc3 = F.leaky_relu(self.enc3(F.max_pool2d(enc2, 2)), negative_slope=0.01)
        enc4 = F.leaky_relu(self.enc4(F.max_pool2d(enc3, 2)), negative_slope=0.01)

        # Decoding path with skip connections and size adjustment
        dec3 = F.leaky_relu(self.dec3(torch.cat([F.interpolate(self.upconv3(enc4), size=enc3.shape[2:]), enc3], dim=1)), negative_slope=0.01)
        dec2 = F.leaky_relu(self.dec2(torch.cat([F.interpolate(self.upconv2(dec3), size=enc2.shape[2:]), enc2], dim=1)), negative_slope=0.01)
        dec1 = F.leaky_relu(self.dec1(torch.cat([F.interpolate(self.upconv1(dec2), size=enc1.shape[2:]), enc1], dim=1)), negative_slope=0.01)

        # Final output layer
        outputs = self.final(dec1)
        outputs = outputs.squeeze(1)
        return outputs


# Step 1: Check for multiple GPUs, move model to the appropriate device(s)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    device = torch.device("cuda:0")  # Primary device where model will reside
    model = UNetEnhancedCNNModel().to(device)  # Move model to primary GPU
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use nn.DataParallel with specified GPUs
else:
    print("Using a single GPU or CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEnhancedCNNModel().to(device)  # Move model to GPU or CPU as needed

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
train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=200)

# 保存模型
torch.save(model.state_dict(), "ndvi_prediction_model.pth")
