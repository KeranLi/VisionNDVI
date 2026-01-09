# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adapter import InferenceAdapter  # Import InferenceAdapter

class UNetEnhancedCNNModel(nn.Module):
    """UNet-based model for NDVI prediction"""
    def __init__(self, num_input_channels=6):
        super(UNetEnhancedCNNModel, self).__init__()
        
        self.feature_weights = nn.Parameter(torch.ones(num_input_channels))
        
        # Encoding path
        self.enc1 = nn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Decoding path
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Output layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
        # MLP layers
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Apply feature weights
        normalized_weights = F.softmax(self.feature_weights, dim=0)
        x = x * normalized_weights.view(1, -1, 1, 1)
        
        # Encoding path
        enc1 = F.leaky_relu(self.enc1(x), negative_slope=0.01)
        enc2 = F.leaky_relu(self.enc2(F.max_pool2d(enc1, 2)), negative_slope=0.01)
        enc3 = F.leaky_relu(self.enc3(F.max_pool2d(enc2, 2)), negative_slope=0.01)
        enc4 = F.leaky_relu(self.enc4(F.max_pool2d(enc3, 2)), negative_slope=0.01)
        
        # Decoding path with skip connections
        dec3 = F.leaky_relu(self.dec3(torch.cat([
            F.interpolate(self.upconv3(enc4), size=enc3.shape[2:], mode='bilinear', align_corners=False),
            enc3
        ], dim=1)), negative_slope=0.01)
        
        dec2 = F.leaky_relu(self.dec2(torch.cat([
            F.interpolate(self.upconv2(dec3), size=enc2.shape[2:], mode='bilinear', align_corners=False),
            enc2
        ], dim=1)), negative_slope=0.01)
        
        dec1 = F.leaky_relu(self.dec1(torch.cat([
            F.interpolate(self.upconv1(dec2), size=enc1.shape[2:], mode='bilinear', align_corners=False),
            enc1
        ], dim=1)), negative_slope=0.01)
        
        # Final output
        outputs = self.final(dec1)
        
        # Apply MLP channel-wise
        batch_size, channels, h, w = outputs.shape
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        outputs_flat = outputs.view(batch_size * h * w, -1)
        
        mlp_out = torch.tanh(self.fc1(outputs_flat))
        mlp_out = torch.tanh(self.fc2(mlp_out))
        
        outputs = mlp_out.view(batch_size, h, w, channels)
        outputs = outputs.permute(0, 3, 1, 2).contiguous()
        
        return outputs.squeeze(1)

def load_model(checkpoint_path, device, use_dataparallel=False):
    """Load model from checkpoint with DataParallel handling"""
    model = UNetEnhancedCNNModel()
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel state dict
    if 'module.' in list(state_dict.keys())[0]:
        model = nn.DataParallel(model)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model

def load_adapter(input_shape, device):
    """Load and return the InferenceAdapter"""
    adapter = InferenceAdapter(input_shape).to(device)
    return adapter