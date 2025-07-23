"""
Implements the PoseCNN network architecture in PyTorch.
"""
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision.models as models
from torchvision.ops import RoIPool
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models.vgg import VGG

import numpy as np
import random
import statistics
import time
from typing import Dict, List, Callable, Optional

from rob599 import quaternion_to_matrix
from p4_helper import HoughVoting, _LABEL2MASK_THRESHOL, loss_cross_entropy, loss_Rotation, IOUselection


def hello_pose_cnn():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from pose_cnn.py!")

def get_last_conv_out_channels(layers):
    # Recursively search for last Conv2d
    for layer in reversed(layers):
        if isinstance(layer, nn.Conv2d):
            return layer.out_channels
        elif isinstance(layer, nn.Sequential):
            result = get_last_conv_out_channels(list(layer.children()))
            if result is not None:
                return result
        elif hasattr(layer, 'children'):
            result = get_last_conv_out_channels(list(layer.children()))
            if result is not None:
                return result
    return None


class FeatureExtraction(nn.Module):
    """
    Feature Embedding Module for PoseCNN. Using pretrained VGG16 network as backbone.
    """    
    def __init__(self, pretrained_model, backbone_name):
        super(FeatureExtraction, self).__init__()
        # embedding_layers = list(pretrained_model.features)[:30]
        # ## Embedding Module from begining till the first output feature map
        # self.embedding1 = nn.Sequential(*embedding_layers[:23])
        # ## Embedding Module from the first output feature map till the second output feature map
        # self.embedding2 = nn.Sequential(*embedding_layers[23:])

        # for i in [0, 2, 5, 7, 10, 12, 14]:
        #     self.embedding1[i].weight.requires_grad = False
        #     self.embedding1[i].bias.requires_grad = False
        self.backbone_name = backbone_name
        if backbone_name == 'vgg16':
            # VGG-like (e.g., VGG16)
            embedding_layers = list(pretrained_model.features)[:30]
            self.embedding1 = nn.Sequential(*embedding_layers[:23])
            self.embedding2 = nn.Sequential(*embedding_layers[23:])

            # Freeze early layers
            for i in [0, 2, 5, 7, 10, 12, 14]:
                self.embedding1[i].weight.requires_grad = False
                self.embedding1[i].bias.requires_grad = False

        elif backbone_name == 'resnet18' or backbone_name == 'resnet50':
            self.embedding1 = nn.Sequential(
                pretrained_model.conv1,
                pretrained_model.bn1,
                pretrained_model.relu,
                pretrained_model.maxpool,
                pretrained_model.layer1,  # 64
                pretrained_model.layer2,  # 128
            )

            resnet_out_channels = pretrained_model.layer4[-1].conv1.in_channels  # 512 (resnet18) or 2048 (resnet50)

            if resnet_out_channels == 2048:
                self.embedding2 = nn.Sequential(
                    pretrained_model.layer3,
                    pretrained_model.layer4,
                    nn.Conv2d(resnet_out_channels, 512, kernel_size=1)  # force to 512
                )
            elif resnet_out_channels == 512:
                self.embedding2 = nn.Sequential(
                    pretrained_model.layer3,
                    pretrained_model.layer4,
                    nn.Conv2d(resnet_out_channels, 128, kernel_size=1)  # force to 512
                )

        elif backbone_name == 'efficientnet':
            blocks = list(pretrained_model.features.children())

            # Manual ResNet-like split for EfficientNet-B0
            embedding1_blocks = blocks[:6]   # Up to block index 5 (usually ends with 112 channels)
            embedding2_blocks = blocks[6:]   # From block 6 onward (ends with 1280 channels)


            # Set EfficientNet feature extractors
            self.embedding1 = nn.Sequential(*embedding1_blocks)
            self.embedding2 = nn.Sequential(*embedding2_blocks)

            # Get actual output channels from each stage
            out_channels1 = get_last_conv_out_channels(embedding1_blocks)
            out_channels2 = get_last_conv_out_channels(embedding2_blocks)

            # print("out_channels1", out_channels1)  # Expect 112
            # print("out_channels2", out_channels2)  # Expect 1280

            # Projection layers (after embedding stages)
            self.embedding1_proj = nn.Conv2d(out_channels1, 512, kernel_size=1)
            self.embedding2_proj = nn.Conv2d(out_channels2, 512, kernel_size=1)
        elif backbone_name == 'vit':
            self.vit = pretrained_model
            # Get the input dim to the classification head (usually 768)
            in_dim = self.vit.heads[0].in_features

            # Replace classification head so vit(x) returns CLS token embedding
            self.vit.heads = nn.Identity()

            # Project to 512 for PoseCNN compatibility
            self.proj = nn.Conv2d(in_dim, 512, kernel_size=1)
            self.embedding2 = nn.Identity()

        elif backbone_name == 'swin':
            self.backbone = pretrained_model  # torchvision.models.swin_*
            
            in_dim = self.backbone.head.in_features

            # Remove the classification head so it outputs features
            self.backbone.head = nn.Identity()
            
            
            print("IN DIM",in_dim)
            self.proj = nn.Conv2d(in_dim, 512, kernel_size=1)

            self.embedding2 = nn.Identity()

        elif backbone_name == 'convnext':
            blocks = list(pretrained_model.features.children())
            split_point = len(blocks) // 2
            self.embedding1 = nn.Sequential(*blocks[:split_point], nn.Conv2d(384, 512, 1))
            self.embedding2 = nn.Sequential(*blocks[split_point:], nn.Conv2d(768, 512, 1))

        else:
            raise ValueError("Unsupported backbone architecture: {}".format(pretrained_model.__class__.__name__))
    
    def forward(self, datadict):
        x = datadict['rgb']
        """
        feature1: [bs, 512, H/8, W/8]
        feature2: [bs, 512, H/16, W/16]
        """
        if self.backbone_name == 'vit':
            B = x.size(0)
            x = nn.functional.interpolate(x, (224, 224))     # Resize to ViT input
            # x = self.vit._process_input(x)                   # Normalized input

            with torch.no_grad():
                x = self.vit.conv_proj(x)                    # [B, 768, 14, 14]
                x = x.flatten(2).transpose(1, 2)             # [B, 196, 768]
                cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)  # [B, 1, 768]
                x = torch.cat((cls_token, x), dim=1)              # [B, 197, 768]
                x = self.vit.encoder(x)                      # Encoder already adds cls token & pos embedding

            # Discard CLS token (first token), keep spatial tokens
            x = x[:, 1:, :]                                  # [B, 196, 768]
            x = x.permute(0, 2, 1).reshape(B, 768, 14, 14)    # [B, 768, 14, 14]

            # Project down if needed (e.g., to [B, 512, 14, 14])
            feature1 = self.proj(x)
            feature1 = nn.functional.interpolate(feature1, size=(30, 40), mode='bilinear')

            feature2 = torch.zeros_like(feature1)

            return feature1, feature2
        elif self.backbone_name == 'swin':  # for swin
            B = x.size(0)
            
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            # print("X before backbone feature:",x.shape)
            with torch.no_grad():
                x = self.backbone.features(x)  # Output: [B, C, H/32, W/32] (e.g. [B, 768, 7, 7])

            print("X before permute:",x.shape)
            x = x.permute(0, 3, 1, 2).contiguous()


            # Upsample to match [B, 512, H/8, W/8] as needed

            # print("X before proj:",x.shape)
            x = self.proj(x)  # [B, 512, h, w]
            x = nn.functional.interpolate(x, size=(30, 40), mode='bilinear', align_corners=False)

            feature1 = x
            feature2 = torch.zeros_like(feature1)

            return feature1, feature2
        elif hasattr(self, 'proj') and hasattr(self, 'backbone'):
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                # Forward through the entire Swin model up to the final head input
                x = self.backbone.features(x)  # [B, H, W, C]
                # print("Backbone output:", x.shape)
                x = x.permute(0, 3, 1, 2) # [B, C, H, W]
                x = x.mean(dim=[2, 3])         # Global Average Pooling → [B, C]

            x = self.proj(x)  # → [B, 512]

            feature1 = x.view(x.size(0), 512, 1, 1).expand(-1, 512, 30, 40)
            feature2 = torch.zeros_like(feature1)
            return feature1, feature2
        elif self.backbone_name == 'efficientnet':
            feat1 = self.embedding1(x)
            feat2 = self.embedding2(feat1)
            feat1_proj = self.embedding1_proj(feat1)
            feat2_proj = self.embedding2_proj(feat2)

            return feat1_proj, feat2_proj

        else:
            print("Resnet:",x.shape)
            feature1 = self.embedding1(x)
            feature2 = self.embedding2(feature1)
            return feature1, feature2



class SegmentationBranch(nn.Module):
    """
    Instance Segmentation Module for PoseCNN. 
    """    
    def __init__(self, num_classes = 10, input_dim=512, hidden_layer_dim = 64):
        super(SegmentationBranch, self).__init__()

        ######################################################################
        # TODO: Initialize instance segmentation branch layers for PoseCNN.  #
        #                                                                    #
        # 1) Both feature1 and feature2 should be passed through a 1x1 conv  #
        # + ReLU layer (seperate layer for each feature).                    #
        #                                                                    #
        # 2) Next, intermediate features from feature1 should be upsampled   #
        # to match spatial resolution of features2.                          #
        #                                                                    #
        # 3) Intermediate features should be added, element-wise.            #
        #                                                                    #
        # 4) Final probability map generated by 1x1 conv+ReLU -> softmax     #
        #                                                                    #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        #                                                                    #
        # Note: num_classes passed as input does not include the background  #
        # our desired probability map should be over classses and background #
        # Input channels will be 512, hidden_layer_dim gives channels for    #
        # each embedding layer in this network.                              #
        ######################################################################
        # Replace "pass" statement with your code

        self.num_classes = num_classes



        self.conv1 = nn.Conv2d(input_dim, hidden_layer_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.relu1 = nn.ReLU()


        self.conv2 = nn.Conv2d(input_dim, hidden_layer_dim, 1)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.relu2 = nn.ReLU()


        self.conv3 = nn.Conv2d(hidden_layer_dim,  self.num_classes+1, 1)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        self.relu3 = nn.ReLU()





        self.softmax = nn.Softmax(dim=1)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            probability: Segmentation map of probability for each class at each pixel.
                probability size: (B,num_classes+1,H,W)
            segmentation: Segmentation map of class id's with highest prob at each pixel.
                segmentation size: (B,H,W)
            bbx: Bounding boxs detected from the segmentation. Can be extracted 
                from the predicted segmentation map using self.label2bbx(segmentation).
                bbx size: (N,6) with (batch_ids, x1, y1, x2, y2, cls)
        """
        probability = None
        segmentation = None
        bbx = None
        
        ######################################################################
        # TODO: Implement forward pass of instance segmentation branch.      #
        ######################################################################
        # Replace "pass" statement with your code

        x1 = self.relu1(self.conv1(feature1))
        x2 = self.relu2(self.conv2(feature2))
        # print("x1:", x1.shape)
        # print("x2:", x2.shape)
        x2_up = nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        temp = x2_up + x1

        # temp = nn.functional.interpolate(x2, scale_factor=2) + x1

        up_sample = nn.functional.interpolate(temp, size=(480,640), mode='bilinear') # up_sample =(N,64,480,640)
        x3 = self.conv3(up_sample)   # x3 =(N,11,480,640)

        probability = self.softmax(x3)
        segmentation = torch.argmax(probability,dim=1) #  (B,H,W)
        bbx = self.label2bbx(segmentation)

        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return probability, segmentation, bbx
    
    def label2bbx(self, label):
        bbx = []
        bs, H, W = label.shape
        device = label.device
        label_repeat = label.view(bs, 1, H, W).repeat(1, self.num_classes, 1, 1).to(device)
        label_target = torch.linspace(0, self.num_classes - 1, steps = self.num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)
        mask = (label_repeat == label_target)

        for batch_id in range(mask.shape[0]):
            for cls_id in range(mask.shape[1]):
                if cls_id != 0: 
                    # cls_id == 0 is the background
                    y, x = torch.where(mask[batch_id, cls_id] != 0)
                    # print("Mask:",y.numel())
                    if y.numel() >= _LABEL2MASK_THRESHOL:
                        bbx.append([batch_id, torch.min(x).item(), torch.min(y).item(), 
                                    torch.max(x).item(), torch.max(y).item(), cls_id])
        bbx = torch.tensor(bbx).to(device)
        return bbx
        
        
class TranslationBranch(nn.Module):
    """
    3D Translation Estimation Module for PoseCNN. 
    """    
    def __init__(self, num_classes = 10,input_dim=512, hidden_layer_dim = 128):
        super(TranslationBranch, self).__init__()
        
        ######################################################################
        # TODO: Initialize layers of translation branch for PoseCNN.         #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        ######################################################################
        # Replace "pass" statement with your code

        self.num_classes = num_classes



        self.conv1 = nn.Conv2d(input_dim, hidden_layer_dim, 1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.relu1 = nn.ReLU()


        self.conv2 = nn.Conv2d(input_dim, hidden_layer_dim, 1)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.relu2 = nn.ReLU()


        self.conv3 = nn.Conv2d(hidden_layer_dim,  3*self.num_classes, 1)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            translation: Map of object centroid predictions.
                translation size: (N,3*num_classes,H,W)
        """
        translation = None
        ######################################################################
        # TODO: Implement forward pass of translation branch.                #
        ######################################################################
        # Replace "pass" statement with your code
        
        x1 = self.relu1(self.conv1(feature1))
        x2 = self.relu2(self.conv2(feature2))

        x2_up = nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        temp = x2_up + x1

        # temp = nn.functional.interpolate(x2, scale_factor=2) + x1

        
        x3 = self.conv3(temp)   # x3 =(N,11,H,W)

        translation = nn.functional.interpolate(x3, size=(480,640), mode='bilinear')  # temp =(N,64,480,640)
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        return translation

class RotationBranch(nn.Module):
    """
    3D Rotation Regression Module for PoseCNN. 
    """    
    def __init__(self, feature_dim = 512, roi_shape = 7, hidden_dim = 4096, num_classes = 10):
        super(RotationBranch, self).__init__()

        ######################################################################
        # TODO: Initialize layers of rotation branch for PoseCNN.            #
        # It is recommended that you initialize each convolution kernel with #
        # the kaiming_normal initializer and each bias vector to zeros.      #
        ######################################################################
        # Replace "pass" statement with your code
        
        self.roi_shape = roi_shape
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.roi1 = RoIPool(self.roi_shape, 1/8)
        self.roi2 = RoIPool(self.roi_shape, 1/16)

        
        # self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_features=self.feature_dim * roi_shape * roi_shape, out_features=hidden_dim)
        nn.init.kaiming_normal_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)


        self.relu_lin = nn.ReLU()
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features= 4 * self.num_classes)
        nn.init.kaiming_normal_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, feature1, feature2, bbx):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
            bbx: Bounding boxes of regions of interst (N, 5) with (batch_ids, x1, y1, x2, y2)
        Returns:
            quaternion: Regressed components of a quaternion for each class at each ROI.
                quaternion size: (N,4*num_classes)
        """
        quaternion = None

        ######################################################################
        # TODO: Implement forward pass of rotation branch.                   #
        ######################################################################
        # Replace "pass" statement with your code
        out1 = self.roi1(feature1, bbx.to(dtype = torch.float32))
        out2 = self.roi2(feature2, bbx.to(dtype = torch.float32))
        out = out1 + out2
        quaternion = self.lin2(self.relu_lin(self.lin1(out.flatten(1))))

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return quaternion

class PoseCNN(nn.Module):
    """
    PoseCNN
    """
    def __init__(self, backbone_name, pretrained_backbone, input_dim, models_pcd, cam_intrinsic):
        super(PoseCNN, self).__init__()

        self.iou_threshold = 0.7
        self.models_pcd = models_pcd
        self.cam_intrinsic = cam_intrinsic

        ######################################################################
        # TODO: Initialize layers and components of PoseCNN.                 #
        #                                                                    #
        # Create an instance of FeatureExtraction, SegmentationBranch,       #
        # TranslationBranch, and RotationBranch for use in PoseCNN           #
        ######################################################################
        # Replace "pass" statement with your code
        # vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        self.input_dim = input_dim


        # if isinstance(pretrained_backbone, models.vgg16):
        #     self.input_dim = 512
        # elif isinstance(pretrained_backbone, models.resnet18):
        #     self.input_dim = 128
        # elif isinstance(pretrained_backbone, models.resnet50):
        #     self.input_dim = 2048

        self.feature_extractor = FeatureExtraction(
            pretrained_model=pretrained_backbone,
            backbone_name=backbone_name
        )
        self.segmentation_branch = SegmentationBranch(input_dim=self.input_dim)
        self.RotationBranch = RotationBranch(feature_dim=self.input_dim)
        self.TranslationBranch = TranslationBranch(input_dim=self.input_dim)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################


    def forward(self, input_dict):
        """
        input_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs'
        }
        """


        if self.training:
            loss_dict = {
                "loss_segmentation": 0,
                "loss_centermap": 0,
                "loss_R": 0
            }

            gt_bbx = self.getGTbbx(input_dict)

            ######################################################################
            # TODO: Implement PoseCNN's forward pass for training.               #
            #                                                                    #
            # Model should extract features, segment the objects, identify roi   #
            # object bounding boxes, and predict rotation and translations for   #
            # each roi box.                                                      #
            #                                                                    #
            # The training loss for semantic segmentation should be stored in    #
            # loss_dict["loss_segmentation"] and calculated using the            #
            # loss_cross_entropy(.) function.                                    #
            #                                                                    #
            # The training loss for translation should be stored in              #
            # loss_dict["loss_centermap"] using the L1loss function.             #
            #                                                                    #
            # The training loss for rotation should be stored in                 #
            # loss_dict["loss_R"] using the given loss_Rotation function.        #
            ######################################################################
            # Important: the rotation loss should be calculated only for regions
            # of interest that match with a ground truth object instance.
            # Note that the helper function, IOUselection, may be used for 
            # identifying the predicted regions of interest with acceptable IOU 
            # with the ground truth bounding boxes.
            # If no ROIs result from the selection, don't compute the loss_R
            
            # Replace "pass" statement with your code
            
            # print(input_dict.size())
            feat1, feat2 = self.feature_extractor(input_dict)
            probab, segmk , d_bbx = self.segmentation_branch(feat1, feat2)
            loss_dict["loss_segmentation"] = loss_cross_entropy(probab,input_dict['label'])
        
            trans = self.TranslationBranch(feat1, feat2)
            mae_loss = nn.L1Loss()
            loss_dict["loss_centermap"] = mae_loss(trans,input_dict['centermaps'])

            gt_bbx = gt_bbx.to(torch.float32)
            filter_bbx_R = IOUselection(d_bbx, gt_bbx, self.iou_threshold)
            if filter_bbx_R.shape[0] != 0:
                quater =  self.RotationBranch(feat1, feat2, gt_bbx[:,0:5])
                gt_R = self.gtRotation(filter_bbx_R, input_dict)    # gt_R
                pred_R, label = self.estimateRotation(quater , filter_bbx_R)  # pred_R , label
                label = label.long()
                loss_dict["loss_R"] = loss_Rotation(pred_R, gt_R, label, self.models_pcd)

            else:
                loss_dict["loss_R"] = 0


            ######################################################################
            #                            END OF YOUR CODE                        #
            ######################################################################
            
            return loss_dict
        else:
            output_dict = None
            segmentation = None

            with torch.no_grad():
                ######################################################################
                # TODO: Implement PoseCNN's forward pass for inference.              #
                ######################################################################
                # Replace "pass" statement with your code
                
                feat1, feat2 = self.feature_extractor(input_dict)
                # print("feat1:",feat1.shape)
                # print("feat2:",feat2.shape)
                _, segmentation, bb_xs = self.segmentation_branch(feat1, feat2)
                # print("segmentation:",segmentation.shape)
                # print("BB_XS:",bb_xs.shape)
                if bb_xs.ndim == 2 and bb_xs.shape[0] > 0:
                    trans_i = self.TranslationBranch(feat1, feat2)


                    bb_xs = bb_xs.to(torch.float32)

                    # print("BB_XS:",bb_xs.shape)
                    quater =  self.RotationBranch(feat1, feat2, bb_xs[:,0:5])
                    pred__R, _ = self.estimateRotation(quater , bb_xs)


                    pred_centers, pred_depths = HoughVoting(segmentation, trans_i, num_classes=10)

                    bb_xs = bb_xs.to(device=bb_xs.device, dtype=torch.long)
                    # bb_xs = bb_xs.to(torch.cuda.LongTensor())


                    output_dict = self.generate_pose(pred__R , pred_centers, pred_depths, bb_xs)
                else:
                    print("No valid bounding boxes detected for rotation branch.")
                    output_dict = {}
                


                ######################################################################
                #                            END OF YOUR CODE                        #
                ######################################################################

            return output_dict, segmentation
    
    def estimateTrans(self, translation_map, filter_bbx, pred_label):
        """
        translation_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        label: a tensor [batch_size, num_classes, height, width]
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            trans_map = translation_map[batch_id, (cls-1) * 3 : cls * 3, :]
            label = (pred_label[batch_id] == cls).detach()
            pred_T = trans_map[:, label].mean(dim=1)
            pred_Ts[idx] = pred_T
        return pred_Ts

    def gtTrans(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Ts[idx] = input_dict['RTs'][batch_id][cls - 1][:3, [3]].T
        return gt_Ts 

    def getGTbbx(self, input_dict):
        """
            bbx is N*6 (batch_ids, x1, y1, x2, y2, cls)
        """
        gt_bbx = []
        objs_id = input_dict['objs_id']
        device = objs_id.device
        ## [x_min, y_min, width, height]
        bbxes = input_dict['bbx']
        for batch_id in range(bbxes.shape[0]):
            for idx, obj_id in enumerate(objs_id[batch_id]):
                if obj_id.item() != 0:
                    # the obj appears in this image
                    bbx = bbxes[batch_id][idx]
                    gt_bbx.append([batch_id, bbx[0].item(), bbx[1].item(),
                                  bbx[0].item() + bbx[2].item(), bbx[1].item() + bbx[3].item(), obj_id.item()])
        return torch.tensor(gt_bbx).to(device=device, dtype=torch.int16)
        
    def estimateRotation(self, quaternion_map, filter_bbx):
        """
        quaternion_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Rs = torch.zeros(N_filter_bbx, 3, 3)
        label = []
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            quaternion = quaternion_map[idx, (cls-1) * 4 : cls * 4]
            quaternion = nn.functional.normalize(quaternion, dim=0)
            pred_Rs[idx] = quaternion_to_matrix(quaternion)
            label.append(cls)
        label = torch.tensor(label)
        return pred_Rs, label

    def gtRotation(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Rs = torch.zeros(N_filter_bbx, 3, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Rs[idx] = input_dict['RTs'][batch_id][cls - 1][:3, :3]
        return gt_Rs 

    def generate_pose(self, pred_Rs, pred_centers, pred_depths, bbxs):
        """
        pred_Rs: a tensor [pred_bbx_size, 3, 3]
        pred_centers: [batch_size, num_classes, 2]
        pred_depths: a tensor [batch_size, num_classes]
        bbx: a tensor [pred_bbx_size, 6]
        """        
        output_dict = {}
        for idx, bbx in enumerate(bbxs):
            bs, _, _, _, _, obj_id = bbx
            R = pred_Rs[idx].numpy()
            center = pred_centers[bs, obj_id - 1].numpy()
            depth = pred_depths[bs, obj_id - 1].numpy()
            if (center**2).sum().item() != 0:
                T = np.linalg.inv(self.cam_intrinsic) @ np.array([center[0], center[1], 1]) * depth
                T = T[:, np.newaxis]
                if bs.item() not in output_dict:
                    output_dict[bs.item()] = {}
                output_dict[bs.item()][obj_id.item()] = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]])))
        return output_dict


def eval(model, dataloader, device, alpha = 0.35):
    import cv2
    model.eval()

    sample_idx = random.randint(0,len(dataloader.dataset)-1)
    ## image version vis
    rgb = torch.tensor(dataloader.dataset[sample_idx]['rgb'][None, :]).to(device)
    inputdict = {'rgb': rgb}
    pose_dict, label = model(inputdict)
    poselist = []
    rgb =  (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return dataloader.dataset.visualizer.vis_oneview(
        ipt_im = rgb, 
        obj_pose_dict = pose_dict[0],
        alpha = alpha
        )

