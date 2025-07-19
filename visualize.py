import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from rob599.PROPSPoseDataset import PROPSPoseDataset


train_set = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset',split='train')
test_set = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset',split='val')

from rob599 import reset_seed, visualize_dataset

reset_seed(0)

grid_vis = visualize_dataset(test_set,alpha = 0.25)
plt.axis('off')
plt.imshow(grid_vis)
plt.show()

