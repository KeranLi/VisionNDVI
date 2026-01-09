import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim  # 使用torchmetrics的SSIM
import torch.nn.functional as F

# 逐栅格均方误差 (MSE) 加整体均方误差
class CombinedMSELoss(nn.Module):
    def __init__(self):
        super(CombinedMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, labels):
        per_pixel_mse = self.mse(outputs, labels)
        global_mse = self.mse(outputs.mean(dim=(1, 2, 3)), labels.mean(dim=(1, 2, 3)))
        return per_pixel_mse + global_mse

# 逐栅格 L1 损失加整体均方误差
class CombinedL1MSELoss(nn.Module):
    def __init__(self):
        super(CombinedL1MSELoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, labels):
        per_pixel_l1 = self.l1(outputs, labels)
        global_mse = self.mse(outputs.mean(dim=(1, 2, 3)), labels.mean(dim=(1, 2, 3)))
        return per_pixel_l1 + global_mse

# 结构相似性（SSIM）与逐栅格均方误差的结合
class CombinedSSIMMSELoss(nn.Module):
    def __init__(self):
        super(CombinedSSIMMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, labels):
        per_pixel_mse = self.mse(outputs, labels)
        ssim_loss = 1 - ssim(outputs, labels)  # 1 - SSIM 得到损失
        return per_pixel_mse + ssim_loss

# 逐栅格损失加平滑正则项
class CombinedMSEGradientLoss(nn.Module):
    def __init__(self):
        super(CombinedMSEGradientLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, outputs, labels):
        def gradient_loss(preds):
            dy = torch.abs(preds[:, :, 1:, :] - preds[:, :, :-1, :])
            dx = torch.abs(preds[:, :, :, 1:] - preds[:, :, :, :-1])
            return torch.mean(dx) + torch.mean(dy)

        per_pixel_mse = self.mse(outputs, labels)
        smoothness_loss = gradient_loss(outputs)
        return per_pixel_mse + 0.1 * smoothness_loss
