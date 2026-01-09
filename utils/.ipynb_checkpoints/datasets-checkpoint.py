# utils/datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from .helpers import read_npy, seed_worker, TARGET_SHAPE, CATEGORIES

class NDVIDataset(Dataset):
    def __init__(self, feature_files, label_files, slope_path, elevation_path, mask_path, stats, mode='train'):
        self.feature_files = {cat: feature_files.get(cat, []) for cat in CATEGORIES}
        self.label_files = label_files
        self.slope_path = slope_path
        self.elevation_path = elevation_path
        self.mask = torch.tensor(np.load(mask_path), dtype=torch.float32)
        self.stats = stats
        self.target_shape = TARGET_SHAPE
        self.mode = mode
        
        # Align file counts
        if mode == 'train':
            self.num_samples = min(len(self.label_files.get('NDVI_Monthly', [])),
                                  *[len(v) for v in self.feature_files.values()])
        else:
            # Inference mode: use feature files count
            self.num_samples = min(*[len(v) for v in self.feature_files.values()])
        
        print(f"Dataset mode: {mode}, samples: {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        features_tensors = []
        
        # 1. Load dynamic features
        for cat in CATEGORIES:
            if idx >= len(self.feature_files[cat]):
                # Handle mismatched file counts
                feature_tensor = torch.zeros(self.target_shape, dtype=torch.float32)
            else:
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
        
        # 2. Load static features
        elev = read_npy(self.elevation_path, 'elevation', self.stats)
        slop = read_npy(self.slope_path, 'slope', self.stats)
        
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
        
        # Stack features
        stacked_features = torch.stack(features_tensors, dim=0) * self.mask
        
        # 3. Load label if in training mode
        if self.mode == 'train':
            if 'NDVI_Monthly' in self.label_files and idx < len(self.label_files['NDVI_Monthly']):
                label_path = self.label_files['NDVI_Monthly'][idx]
                label = read_npy(label_path, 'NDVI_Monthly', self.stats, normalize=True)
                label_tensor = torch.tensor(label, dtype=torch.float32)
                label_tensor = F.interpolate(
                    label_tensor.unsqueeze(0).unsqueeze(0),
                    size=self.target_shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0) * self.mask
            else:
                label_tensor = torch.zeros(self.target_shape, dtype=torch.float32)
            
            return stacked_features, label_tensor

        else:
            # Inference mode: return file path for tracking
            if 'NDVI_Monthly' in self.label_files and idx < len(self.label_files['NDVI_Monthly']):
                file_path = self.label_files['NDVI_Monthly'][idx]
                
                # ================ 添加这4行：加载真实标签 ================
                label = read_npy(file_path, 'NDVI_Monthly', self.stats, normalize=True)
                label_tensor = torch.tensor(label, dtype=torch.float32)
                # 确保形状匹配
                if label_tensor.shape != self.target_shape:
                    label_tensor = F.interpolate(
                        label_tensor.unsqueeze(0).unsqueeze(0),
                        size=self.target_shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                label_tensor = label_tensor * self.mask
                # =========================================================
                
            else:
                file_path = f"sample_{idx}"
                label_tensor = torch.zeros(self.target_shape, dtype=torch.float32)
            
            return stacked_features, label_tensor, file_path  # 现在返回真实标签！