import torch
from torch.utils.data import DataLoader
from YCB_Dataset import PoseDataset as YCBPoseDataset
import os

num_objects = 21 #number of object classes in the dataset
num_points = 1000 #number of points on the input pointcloud
outf = './trained_models/ycb' #folder to save trained models
repeat_epoch = 1 #number of repeat times for one epoch training

dataset_root = ''
noise_train = 0.03
refine_start = True

def create_ycb_dataset():
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    if not os.path.exists('dataset/ycb'):
        os.makedirs('dataset/ycb')
    if not os.path.exists('dataset/ycb/YCB_Video_Dataset'):
        os.makedirs('dataset/ycb/YCB_Video_Dataset')

create_ycb_dataset()

# train_set = YCBPoseDataset('train', num_points, True, dataset_root, noise_trans, refine_start)

# train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
