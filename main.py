import os
import sys
import time
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader

import multiprocessing
from pose_cnn import FeatureExtraction, SegmentationBranch
from pose_cnn import PoseCNN, eval
from tqdm import tqdm

from p4_helper import *
from rob599 import reset_seed
from rob599.grad import rel_error
from rob599.PROPSPoseDataset import PROPSPoseDataset
from metrics import compute_add, quaternion_to_rotation_matrix, compute_adds, compute_auc
from filters import *

def get_backbone(name: str):
    name = name.lower()
    if name == "vgg16":
        return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif name == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif name == "efficientnet":
        return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif name == "vit":
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        return vit
    elif name == "swin":
        return models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
    elif name == "convnext":
        return models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

def main(args, io):


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
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = multiprocessing.cpu_count()

    train_dataset = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset',split='train')
    test_dataset = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset',split='val')

    backbone_name = args.backbone_name
    backbone_model = get_backbone(backbone_name)

    input_dim = 512

    if backbone_name == 'resnet18':
        input_dim = 128
    
    posecnn_model = PoseCNN(pretrained_backbone = backbone_model,
                            input_dim = input_dim,
                           models_pcd = torch.tensor(train_dataset.models_pcd).to(DEVICE, dtype=torch.float32),
                           cam_intrinsic = train_dataset.cam_intrinsic).to(DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(posecnn_model.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999))



    loss_history = []
    log_period = 5
    _iter = 0

    epochs = args.epochs
    st_time = time.time()

    posecnn_model.train()
    for epoch in range(epochs):
        train_loss = []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1} from {epochs}'):
            for item in batch:
                batch[item] = batch[item].to(DEVICE)
            loss_dict = posecnn_model(batch)
            optimizer.zero_grad()
            total_loss = 0

            for loss in loss_dict:
                total_loss += loss_dict[loss]


            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())
        
            if _iter % log_period == 0:
                loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
                for key, value in loss_dict.items():
                    loss_str += f"[{key}: {value:.3f}]"

                # print(loss_str)
                loss_history.append(total_loss.item())
            _iter += 1
        
        print('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                      ', ' + 'Epoch %02d' % epoch + ', ' + 'Training finished' + f' , with mean training loss {np.array(train_loss).mean()}'))

    # torch.save(posecnn_model.state_dict(), os.path.join(args.model_path, "posecnn_model.pth"))
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch_last.pth')
    torch.save(posecnn_model.state_dict(), save_file)
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.savefig("loss_curve_"+args.exp_name+".png")

def inference(args, device):
    
    test_dataset = PROPSPoseDataset(root='dataset/PROPS-Pose-Dataset', split='val')
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)  # use batch_size=1 for easier ADD matching

    backbone_name = args.backbone_name
    backbone_model = get_backbone(backbone_name)

    input_dim = 512

    if args.backbone_name == 'resnet18':
        input_dim = 128
    
    posecnn_model = PoseCNN(
        pretrained_backbone=backbone_model,
        input_dim=input_dim,
        models_pcd=torch.tensor(test_dataset.models_pcd).to(device, dtype=torch.float32),
        cam_intrinsic=test_dataset.cam_intrinsic
    ).to(device)

    posecnn_model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    posecnn_model.eval()

    K = test_dataset.cam_intrinsic  # Camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    add_scores = []
    add_s_scores = []

    for batch in tqdm(test_loader, desc="Evaluating ADD"):
        rgb = batch['rgb'].to(device)
        input_dict = {'rgb': rgb}
        pred_pose_dict, _ = posecnn_model(input_dict)

        # Loop over detected objects
        batch_id = 0  # since batch size = 1
        if batch_id not in pred_pose_dict:
            continue

        for cls_id, pred_RT in pred_pose_dict[batch_id].items():
            gt_RT = batch['RTs'][0][cls_id - 1].cpu().numpy()  # (4,4)
            model_points = test_dataset.models_pcd[cls_id - 1]  # (N, 3)

            R_gt = gt_RT[:3, :3]
            t_gt = gt_RT[:3, 3]
            R_pred = pred_RT[:3, :3]
            t_pred = pred_RT[:3, 3]

            ## ICP Evaluation
            if args.filter == 'icp':
                # Get real depth points for the object
                depth = batch['depth'][0][0].cpu().numpy() / 1000.0 # shape: (H, W)
                mask = batch['label'][0][cls_id].cpu().numpy().astype(bool)

                ys, xs = np.where(mask & (depth > 0))

                if len(xs) < 20:
                    continue  # skip unreliable masks

                zs = depth[ys, xs]
                xs_real = (xs - cx) * zs / fx
                ys_real = (ys - cy) * zs / fy
                depth_points = np.stack((xs_real, ys_real, zs), axis=-1)

                transformed_model = (R_pred @ model_points.T).T + t_pred  # (N, 3)

                delta_RT, fitness = refine_pose_with_icp(transformed_model, depth_points, np.eye(4), threshold=0.05)
                
                if fitness < 0.1:
                    continue

                pred_RT = delta_RT @ pred_RT
                R_pred = pred_RT[:3, :3]
                t_pred = pred_RT[:3, 3]
            elif args.filter == 'gf':
                depth = batch['depth'][0][0].cpu().numpy() / 1000.0 # shape: (H, W)
                mask = batch['label'][0][cls_id].cpu().numpy().astype(bool)

                ys, xs = np.where(mask & (depth > 0))

                if len(xs) < 20:
                    continue  # skip unreliable masks

                zs = depth[ys, xs]
                xs_real = (xs - cx) * zs / fx
                ys_real = (ys - cy) * zs / fy
                depth_points = np.stack((xs_real, ys_real, zs), axis=-1)

                transformed_model = (R_pred @ model_points.T).T + t_pred  # (N, 3)

                # delta_RT, fitness = refine_pose_with_guided_filter(transformed_model, depth_points, np.eye(4), radius=0.05)
                delta_RT, fitness = refine_pose_with_guided_filter_improved(transformed_model, depth_points, np.eye(4), radius=0.05)
                if fitness < 0.1:
                    continue

                pred_RT = delta_RT @ pred_RT
                R_pred = pred_RT[:3, :3]
                t_pred = pred_RT[:3, 3]

            add = compute_add(R_gt, t_gt, R_pred, t_pred, model_points)
            adds = compute_adds(R_gt, t_gt, R_pred, t_pred, model_points)
            add_scores.append(add)
            add_s_scores.append(adds)

    valid_add_scores = [s for s in add_scores if np.isfinite(s)]
    valid_adds_scores = [s for s in add_s_scores if np.isfinite(s)]

    if valid_add_scores:
        # print(add_scores)
        mean_add = np.mean(valid_add_scores)
        auc_add = compute_auc(valid_add_scores)

        print(f"\nMean ADD over test set: {mean_add:.4f} meters")
        print(f"AUC-ADD (0-10cm): {auc_add:.4f}")
    else:
        print("\n No valid predictions to compute ADD.")

    if valid_adds_scores:
        # print(add_s_scores)
        mean_adds = np.mean(valid_adds_scores)
        auc_adds = compute_auc(valid_adds_scores)
        print(f"\nMean ADD-S over test set: {mean_adds:.4f} meters")
        print(f"AUC-ADD-S (0-10cm): {auc_adds:.4f}")
    else:
        print("\n No valid predictions to compute ADD.")


    # --- Optional Visualization ---
    if args.visualize:
        num_samples = 5
        fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
        for i in range(num_samples):
            out = eval(posecnn_model, test_loader, device)
            axs[i].imshow(out)
            axs[i].axis('off')
            axs[i].set_title(f'Sample {i+1}')
        plt.tight_layout()
        plt.show()


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Pose CNN')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='posecnn', metavar='N',
                        choices=['posecnn'],
                        help='Model to use, [posecnn]')
    parser.add_argument('--backbone_name', type=str, default='vgg16', metavar='N',
                        choices=['vgg16','resnet18','resnet50','efficientnet','vit','swin','convnext'],
                        help='Model to use, [resnet18]')
    parser.add_argument('--dataset_name', type=str, default='propspose', metavar='N',
                        choices=['propspose', 'ycb'],
                        help='Dataset name to test, [modelnet40svm, scanobjectnnsvm]')
    parser.add_argument('--filter', type=str, default='none', metavar='N',
                        choices=['none', 'icp','gf'],
                        help='Dataset name to test, [modelnet40svm, scanobjectnnsvm]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--visualize', action='store_true', help='Visualization')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight Decay')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    _init_()

    device = torch.device(f"cuda:{args.gpu}")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    reset_seed(args.seed)

    if args.eval:
        inference(args, device)
    else:
        main(args, io)
    