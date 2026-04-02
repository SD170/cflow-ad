import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T


__all__ = ('MVTecDataset', 'StcDataset')


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

# BTAD (e.g. Kaggle BTech_Dataset_transformed): 01/, 02/, 03/ with train/test/ground_truth
BTAD_CLASS_NAMES = ['01', '02', '03']

def _is_image_file(name):
    ext = os.path.splitext(name)[1].lower()
    return ext in ('.png', '.bmp', '.jpg', '.jpeg')


def _resolve_mask_path(gt_type_dir, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for name in (stem + '_mask.png', stem + '_mask.bmp', stem + '_mask.jpg'):
        cand = os.path.join(gt_type_dir, name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(gt_type_dir, stem + '_mask.png')

STC_CLASS_NAMES = ['01', '02', '03', '04', '05', '06', 
                '07', '08', '09', '10', '11', '12'] #, '13' - no ground-truth]


class StcDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in STC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, STC_CLASS_NAMES)
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        #
        if is_train:
            self.dataset_path = os.path.join(c.data_path, 'training')
            self.dataset_vid = os.path.join(self.dataset_path, 'videos')
            self.dataset_dir = os.path.join(self.dataset_path, 'frames')
            self.dataset_files = sorted([f for f in os.listdir(self.dataset_vid) if f.startswith(self.class_name)])
            if not os.path.isdir(self.dataset_dir):
                os.mkdir(self.dataset_dir)
            done_file = os.path.join(self.dataset_path, 'frames_{}.pt'.format(self.class_name))
            print(done_file)
            H, W = 480, 856
            if os.path.isfile(done_file):
                assert torch.load(done_file) == len(self.dataset_files), 'train frames are not processed!'
            else:
                count = 0
                for dataset_file in self.dataset_files:
                    print(dataset_file)
                    data = read_video(os.path.join(self.dataset_vid, dataset_file)) # read video file entirely -> mem issue!!!
                    vid = data[0] # weird read_video that returns byte tensor in format [T,H,W,C]
                    fps = data[2]['video_fps']
                    print('video mu/std: {}/{} {}'.format(torch.mean(vid/255.0, (0,1,2)), torch.std(vid/255.0, (0,1,2)), vid.shape))
                    assert [H, W] == [vid.size(1), vid.size(2)], 'same H/W'
                    dataset_file_dir = os.path.join(self.dataset_dir, os.path.splitext(dataset_file)[0])
                    os.mkdir(dataset_file_dir)
                    count = count + 1
                    for i, frame in enumerate(vid):
                        filename = '{0:08d}.jpg'.format(i)
                        write_jpeg(frame.permute((2, 0, 1)), os.path.join(dataset_file_dir, filename), 80)
                torch.save(torch.tensor(count), done_file)
            #
            self.x, self.y, self.mask = self.load_dataset_folder()
        else:
            self.dataset_path = os.path.join(c.data_path, 'testing')
            self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.ToPILImage(),
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.normalize(self.transform_x(x))
        if y == 0: #self.is_train:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = self.transform_mask(mask)
        #
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = list(), list(), list()
        img_dir = os.path.join(self.dataset_path, 'frames')
        img_types = sorted([f for f in os.listdir(img_dir) if f.startswith(self.class_name)])
        gt_frame_dir = os.path.join(self.dataset_path, 'test_frame_mask')
        gt_pixel_dir = os.path.join(self.dataset_path, 'test_pixel_mask')
        for i, img_type in enumerate(img_types):
            print('Folder:', img_type)
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg')])
            x.extend(img_fpath_list)
            # labels for every test image
            if phase == 'test':
                gt_pixel = np.load('{}.npy'.format(os.path.join(gt_pixel_dir, img_type)))
                gt_frame = np.load('{}.npy'.format(os.path.join(gt_frame_dir, img_type)))
                if i == 0:
                    m = gt_pixel
                    y = gt_frame
                else:
                    m = np.concatenate((m, gt_pixel), axis=0)
                    y = np.concatenate((y, gt_frame), axis=0)
                #
                mask = [e for e in m] # np.expand_dims(e, axis=0)
                assert len(x) == len(y), 'number of x and y should be same'
                assert len(x) == len(mask), 'number of x and mask should be same'
            else:
                mask.extend([None] * len(img_fpath_list))
                y.extend([0] * len(img_fpath_list))
        #
        return list(x), list(y), list(mask)


class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        ds = getattr(c, 'dataset', 'mvtec')
        if ds == 'btad':
            assert c.class_name in BTAD_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, BTAD_CLASS_NAMES)
        else:
            assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crp_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if _is_image_file(f)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                gt_fpath_list = [_resolve_mask_path(gt_type_dir, f) for f in img_fpath_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
