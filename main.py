import os
import sys
import time
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.models as models

from torch.utils.data import DataLoader

import multiprocessing
from pose_cnn import FeatureExtraction, SegmentationBranch
from pose_cnn import PoseCNN
from tqdm import tqdm

from p4_helper import *
from rob599 import reset_seed
from rob599.grad import rel_error
from rob599.PROPSPoseDataset import PROPSPoseDataset

reset_seed(0)

# for plotting
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["font.size"] = 16
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

if torch.cuda.is_available():
    print("Good to go!")
    DEVICE = torch.device("cuda")
else:
    print("Please set GPU via Edit -> Notebook Settings.")
    DEVICE = torch.device("cpu")

# Set a few constants related to data loading.
NUM_CLASSES = 10
BATCH_SIZE = 4
NUM_WORKERS = multiprocessing.cpu_count()

train_dataset = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset',split='train')
test_dataset = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset',split='val')


vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
posecnn_model = PoseCNN(pretrained_backbone = vgg16,
                       models_pcd = torch.tensor(train_dataset.models_pcd).to(DEVICE, dtype=torch.float32),
                       cam_intrinsic = train_dataset.cam_intrinsic).to(DEVICE)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.Adam(posecnn_model.parameters(), lr=0.001,
                                 betas=(0.9, 0.999))