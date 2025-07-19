import torch
from torch.utils.data import DataLoader

from rob599.PROPSPoseDataset import PROPSPoseDataset


train_set = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset',split='train')
test_set = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset',split='val')

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)