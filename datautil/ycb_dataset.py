import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import cv2
from tqdm import tqdm

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train':
            self.path = './datautil/train_data_list.txt'
        elif mode == 'test':
            self.path = './datautil/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root
        # self.add_noise = add_noise
        self.add_noise = False
        self.noise_trans = noise_trans

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while (1):
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('./datautil/classes.txt')
        class_id = 1
        self.cld = {}
        while (1):
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()

            class_id += 1

        all_pcds = []
        max_points = max([pts.shape[0] for pts in self.cld.values()])
        for cid in sorted(self.cld.keys()):
            pts = self.cld[cid]
            if pts.shape[0] < max_points:
                pad = np.zeros((max_points - pts.shape[0], 3), dtype=np.float32)
                pts = np.vstack((pts, pad))

            all_pcds.append(pts)
        self.models_pcd = np.stack(all_pcds)  # shape: [num_classes, max_points, 3]

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.refine = refine
        self.front_num = 2

        self.cam_intrinsic = np.array([
            [self.cam_fx_2, 0.0, self.cam_cx_2],
            [0.0, self.cam_fy_2, self.cam_cy_2],
            [0.0, 0.0, 1.0]
        ])


        print(len(self.list))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.jpg'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        # print("max", label.max(), "min", label.min())
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))


        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise:
            for k in tqdm(range(5)):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break
        if self.add_noise:
            img = self.trancolor(img)
        
        img_orig = np.array(img)

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        my_mask = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
        mask_orig = my_mask.astype(int)
        mask_orig[mask_orig > 0] = 255
        my_mask = my_mask[rmin:rmax, cmin:cmax]
        my_mask = my_mask.astype(int)
        my_mask = np.reshape(my_mask, (my_mask.shape[0], my_mask.shape[1], -1))
        my_mask[my_mask > 0] = 255
        my_mask = cv2.cvtColor(my_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        img_masked = np.transpose(img_masked, (1, 2, 0))
        img_masked = cv2.bitwise_and(my_mask, img_masked)
        img_masked = np.transpose(img_masked, (2, 0, 1))
        my_mask = np.transpose(my_mask, (2, 0, 1))

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])


        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        dense_points = False
        if len(choose) >= 2000 and len(choose) <= 6000:
            dense_points = True

        if dense_points == True:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt + 1000] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        elif dense_points == False and len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax]

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        if self.refine:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)

        input_dict = {
            'rgb': self.norm(torch.from_numpy(img_masked.astype(np.float32))),             # [3, H, W] tensor, normalized
            'depth': torch.from_numpy(depth.astype(np.float32)),           # [H, W] or [1, H, W] tensor (optional in current PoseCNN)
            'label': torch.from_numpy(label.astype(np.int64)),           # [H, W] tensor for segmentation labels
            'objs_id': torch.LongTensor(obj),         # [num_obj] tensor, class ids in the current frame
            'bbx': self.extract_bbx_from_label(label, obj),
            'RTs': torch.from_numpy(meta['poses'].transpose(2, 0, 1)),             # [num_class, 3, 4] or [num_obj, 3, 4], GT poses
            'centermaps': self.generate_centermap(obj, meta, label)           # [3*num_classes, H, W] (as in your PoseCNN)
        }

        input_dict['rgb_full'] = torch.from_numpy(img_orig.transpose(2, 0, 1).astype(np.float32)) / 255.0

        return input_dict
        # return torch.from_numpy(cloud.astype(np.float32)), \
        #        torch.LongTensor(choose.astype(np.int32)), \
        #        self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
        #        torch.from_numpy(target.astype(np.float32)), \
        #        torch.from_numpy(model_points.astype(np.float32)), \
        #        torch.LongTensor([int(obj[idx]) - 1]), \
        #        torch.from_numpy(my_mask.astype(np.float32)), \
        #        dense_points

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    def generate_centermap(self, obj_ids, meta, label_img):
        """
        Generate a 3D centroid map with size [3*num_classes, H, W].
        Each channel group (3) contains x,y,z for a class.
        """
        centermap = np.zeros((3 * 21, 480, 640), dtype=np.float32)  # 21: max class id (YCB has 21 classes)

        for idx, cls_id in enumerate(obj_ids):
            if cls_id == 0:
                continue
            RT = meta['poses'][:, :, idx]  # shape [3, 4]
            x, y, z = RT[:, 3]  # center position
            mask = (label_img == cls_id)
            for i, val in enumerate([x, y, z]):
                centermap[3 * (cls_id - 1) + i][mask] = val
        return torch.from_numpy(centermap)

    def extract_bbx_from_label(self, label, objs_id):
        """
        Extract bounding boxes from label image for each object id.
        Output: tensor of shape [num_obj, 4] with [x, y, w, h]
        """
        bbx_list = []
        for cls_id in objs_id:
            mask = (label == cls_id)
            if mask.sum() == 0:
                bbx_list.append([0, 0, 0, 0])
                continue
            ys, xs = np.where(mask)
            x_min, y_min = xs.min(), ys.min()
            x_max, y_max = xs.max(), ys.max()
            bbx_list.append([x_min, y_min, x_max - x_min, y_max - y_min])
        return torch.tensor(bbx_list, dtype=torch.float32)


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def cal_pixel(label):
    pixel_num = 0
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] > 0:
                pixel_num += 1
    return pixel_num
