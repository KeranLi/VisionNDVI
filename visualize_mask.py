import numpy as np
import matplotlib.pyplot as plt
import os

def check_mask_alignment(mask_path, sample_ndvi_path):
    # 1. 加载数据
    mask = np.load(mask_path)
    # 尝试加载一个真实的 NDVI 样本作为参照
    sample_ndvi = np.load(sample_ndvi_path)
    
    print(f"Mask shape: {mask.shape}, Range: [{mask.min()}, {mask.max()}]")
    print(f"NDVI shape: {sample_ndvi.shape}")

    # 2. 绘图对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左图：展示 Mask 本身 (1为陆地, 0为海洋)
    im1 = axes[0].imshow(mask, cmap='gray')
    axes[0].set_title("Ocean-Land Mask (npy)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 中图：展示 NDVI 样本
    # 假设 NDVI 数据在 [0, 1] 或已标准化
    im2 = axes[1].imshow(sample_ndvi, cmap='YlGn') 
    axes[1].set_title("Sample NDVI Data")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 右图：叠加检查 (Overlay)
    # 如果 Mask 和 NDVI 完全对齐，陆地应该严丝合缝地重叠
    axes[2].imshow(sample_ndvi, cmap='Greens', alpha=0.7)
    # 将海洋部分标红，检查是否误伤了陆地
    overlay_mask = np.where(mask == 0, 1, np.nan) 
    axes[2].imshow(overlay_mask, cmap='Reds', alpha=0.5)
    axes[2].set_title("Alignment Check (Red=Ocean)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    save_path = "mask_check.png"
    plt.savefig(save_path, dpi=200)
    print(f"Visualization saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 请根据你的实际路径修改
    MASK_FILE = r'datasets/AWI-CM-1-1-MR/mask.npy'
    # 找一个具体的 NDVI 原始文件做对比
    SAMPLE_FILE = r'datasets/AWI-CM-1-1-MR/NDVI_Monthly/NDVI_Processed_198201.npy' 
    
    if os.path.exists(MASK_FILE) and os.path.exists(SAMPLE_FILE):
        check_mask_alignment(MASK_FILE, SAMPLE_FILE)
    else:
        print("Path Error: Please check your MASK_FILE and SAMPLE_FILE paths.")