# --- 1. IMPORT POSECNN + BACKBONE ---
from pose_cnn import PoseCNN  # make sure pose_cnn.py is in your PYTHONPATH
import torchvision.models as models
import numpy as np

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

# --- 2. MODEL CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone_name = 'vgg16'  # or 'resnet18', 'efficientnet', etc.
num_classes = 21
input_dim = 512

# --- 3. LOAD PRETRAINED BACKBONE ---
if backbone_name == 'vgg16':
    backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
elif backbone_name == 'resnet18':
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Add more if needed...

# --- 4. BUILD PoseCNN MODEL ---
# Use the point cloud models from your dataset
num_classes = len(dataset.cld)
num_points = dataset.num_pt_mesh_large  # or num_pt_mesh_small
models_pcd_np = []

for cid in range(1, num_classes + 1):
    model = dataset.cld[cid]
    if model.shape[0] >= num_points:
        sampled = model[:num_points]
    else:
        sampled = np.pad(model, ((0, num_points - model.shape[0]), (0, 0)), mode='wrap')
    models_pcd_np.append(sampled)

models_pcd = torch.from_numpy(np.array(models_pcd_np)).float()

cam_intrinsic = np.array([
    [dataset.cam_fx_1, 0, dataset.cam_cx_1],
    [0, dataset.cam_fy_1, dataset.cam_cy_1],
    [0, 0, 1]
])

posecnn_model = PoseCNN(
    backbone_name=backbone_name,
    pretrained_backbone=backbone,
    input_dim=input_dim,
    models_pcd=models_pcd,
    cam_intrinsic=cam_intrinsic
).to(DEVICE)

# --- 5. LOAD WEIGHTS (if available) ---
# posecnn_model.load_state_dict(torch.load("path_to_your_trained_posecnn.pth"))

posecnn_model.eval()

# --- 6. PREPARE SAMPLE FOR INFERENCE ---
sample = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in sample.items()}

with torch.no_grad():
    output_dict, segmentation = posecnn_model(sample)

print("\n[INFO] PoseCNN inference complete.")
print("Predicted object poses per batch index (output_dict):")
for batch_id in output_dict:
    for class_id in output_dict[batch_id]:
        print(f"Batch {batch_id}, Class {class_id}:")
        print(output_dict[batch_id][class_id])  # 4x4 transformation matrix
