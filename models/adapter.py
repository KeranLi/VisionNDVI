import torch
import torch.nn as nn
import torch.nn.functional as F

class InferenceAdapter(nn.Module):
    def __init__(self, input_shape):
        super(InferenceAdapter, self).__init__()
        # Adjust the input dimension to match the flattened input size
        self.fc1 = nn.Linear(input_shape, 128)  # input_shape should match the flattened input size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, input_shape)  # Output dimension is same as input dimension
    
    def forward(self, x):
        # Ensure the input is in float32
        x = x.to(torch.float32)
        
        # Flatten the input
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class FineTuningAdapter(nn.Module):
    def __init__(self, input_size, output_size=512):
        super(FineTuningAdapter, self).__init__()
        # Input size should match the flattened grid size (900 for 30x30 grid)
        self.fc1 = nn.Linear(input_size, 64, bias=False)
        self.fc2 = nn.Linear(64, 256, bias=False)
        self.fc3 = nn.Linear(256, 512, bias=False)
        self.fc4 = nn.Linear(512, input_size, bias=False)  # Adjust output size to match input size

    def forward(self, x):
        # Ensure the input tensor is on the same device as the model's parameters
        x = x.to(self.fc1.weight.device)  # Ensure input tensor is on the same device as the model's parameters

        # Flatten the input tensor to (batch_size, channels * height * width)
        batch_size, channels, height, width = x.size()
        x = x.reshape(batch_size, -1)  # Use reshape to flatten to a 2D tensor: (batch_size, channels * height * width)

        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = torch.relu(self.fc3(x))  # Apply ReLU activation
        x = self.fc4(x)  # Apply second fully connected layer
        
        # Reshape back to the original grid size
        x = x.reshape(batch_size, channels, height, width)  # Reshape back to (batch_size, channels, height, width)
        
        return x


class ResFineTuningAdapter(nn.Module):
    def __init__(self, input_size, output_size=512, dropout_prob=0.5, weight_decay=1e-4):
        super(ResFineTuningAdapter, self).__init__()
        
        # First fully connected layer (fc1)
        self.fc1 = nn.Linear(input_size, 128, bias=False)
        self.fc2 = nn.Linear(128, 256, bias=False)
        self.fc3 = nn.Linear(256, 512, bias=False)
        self.fc4 = nn.Linear(512, input_size, bias=False)  # Output size adjusted to match input size

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_prob)

        # Residual connections
        self.residual = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x):
        # Ensure the input tensor is on the same device as the model's parameters
        x = x.to(self.fc1.weight.device)  # Ensure input tensor is on the same device as the model's parameters

        # Flatten the input tensor to (batch_size, channels * height * width)
        batch_size, channels, height, width = x.size()
        x_flat = x.reshape(batch_size, -1)  # Flatten to (batch_size, channels * height * width)

        # First fully connected layer with ReLU and dropout
        x = torch.relu(self.fc1(x_flat))
        x = self.dropout(x)  # Dropout layer for regularization
        
        # Second fully connected layer with ReLU and dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        # Third fully connected layer with ReLU and dropout
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)

        # Final layer to match the input size
        x = self.fc4(x)

        # Residual connection (skip connection)
        residual = self.residual(x_flat)
        x = x + residual  # Add residual to enhance gradient flow

        # Reshape back to the original grid size
        x = x.reshape(batch_size, channels, height, width)
        
        return x

class ConvResAdapter(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super(ConvResAdapter, self).__init__()
        
        # 卷积层：保持空间分辨率 (Padding=1, Kernel=3)
        # 这样输入是 30x30，输出依然是 30x30
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        
        # 最后一层将通道数还原回 in_channels (通常是 1)
        self.conv4 = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 如果你担心过拟合时数值爆炸，可以加入 BatchNorm
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        # 记录原始输入，用于残差连接 (跳跃连接)
        identity = x 
        
        # 卷积前向计算
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.conv4(out)
        
        # 数值对齐的核心：UNet的粗略预测 + Adapter学习到的细节修正
        # 形状完全一致: [Batch, 1, 30, 30] + [Batch, 1, 30, 30]
        return out + identity

class ResidualBlock(nn.Module):
    """
    一个标准的残差块，帮助深层网络更好收敛
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.relu(out)

class TimeSpaceAdapter(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64): # 增加基础通道数到 64
        super(TimeSpaceAdapter, self).__init__()
        
        # 1. 初始特征投影：从 2 通道 (当前预测 + 上月残差) 映射到高维空间
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 堆叠多个残差块 (Deep Layers)
        # 增加深度到 3 个残差块（共 6 层卷积），显著提升非线性表达能力
        self.res_layers = nn.Sequential(
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels)
        )
        
        # 3. 输出头：映射回单通道的 Delta (修正值)
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1)
        )

    def forward(self, current_pred, last_residual=None):
        if last_residual is None:
            last_residual = torch.zeros_like(current_pred)
            
        # 拼接当前预测和上月残差
        x = torch.cat([current_pred, last_residual], dim=1) 
        
        identity = current_pred
        
        # 特征提取
        feat = self.input_conv(x)
        feat = self.res_layers(feat)
        
        # 计算残差修正值
        delta = self.output_conv(feat)
        
        # 最终输出 = 原预测 + 学习到的增量
        return identity + delta

