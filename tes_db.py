import torch
from torch.utils.data import DataLoader
from datautil.ycb_dataset import PoseDataset
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

root_path = 'dataset/ycb/YCB_Video_Dataset'
mode = 'test'                              # or 'train'
num_points = 1000                          # Number of points to sample from object mask
add_noise = False                          # For clean test, set False
noise_trans = 0.0                          # No translation noise
refine = False

def create_ycb_dataset():
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    if not os.path.exists('dataset/ycb'):
        os.makedirs('dataset/ycb')
    if not os.path.exists('dataset/ycb/YCB_Video_Dataset'):
        os.makedirs('dataset/ycb/YCB_Video_Dataset')

dataset = PoseDataset(
    mode=mode,
    num_pt=num_points,
    add_noise=add_noise,
    root=root_path,
    noise_trans=noise_trans,
    refine=refine
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

sample = next(iter(dataloader))

print("\n[INFO] Fetched one sample from YCB PoseDataset:")
print("rgb shape:", sample['rgb'].shape)
print("depth shape:", sample['depth'].shape)
print("label shape:", sample['label'].shape)
print("objs_id:", sample['objs_id'])
print("bbx:", sample['bbx'].shape)
print("RTs:", sample['RTs'].shape)
print("centermaps:", sample['centermaps'].shape)

# -------- OPTIONAL VISUALIZATION --------
# Unnormalize and show RGB + label
unnorm = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

rgb = sample['rgb_full'][0]  # (3, H, W)
rgb = unnorm(rgb).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
label = sample['label'][0].cpu().numpy()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(label, cmap='jet')
plt.title("Label Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
