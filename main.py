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

    reset_seed(args.seed)

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


    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    posecnn_model = PoseCNN(pretrained_backbone = vgg16,
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
    for epoch in tqdm(range(epochs)):
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

                print(loss_str)
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
    plt.savefig("posecnn.png")


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Pose CNN')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='posecnn', metavar='N',
                        choices=['posecnn'],
                        help='Model to use, [posecnn]')
    parser.add_argument('--dataset_name', type=str, default='modelnet40svm', metavar='N',
                        choices=['modelnet40svm', 'scanobjectnnsvm'],
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
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight Decay')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
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

    torch.manual_seed(args.seed)

    main(args, io)
    