# https://github.com/fab-jul/hdf5_dataloader
import argparse
import glob
import h5py
import numpy as np
import os
import pickle
import gzip

import torch
from torch.utils.data import Dataset

default_opener = lambda p_: h5py.File(p_, 'r')


class HDF5Dataset(Dataset):

    @staticmethod
    def _get_num_in_shard(shard_p, opener=default_opener):
        print(f'\rh5: Opening {shard_p}... ', end='')
        try:
            with opener(shard_p) as f:
                num_per_shard = len(f['len'].keys())
        except:
            print(f"h5: Could not open {shard_p}!")
            num_per_shard = -1
        return num_per_shard

    @staticmethod
    def check_shard_lengths(file_paths, opener=default_opener):
        """
        Filter away the last shard, which is assumed to be smaller. this double checks that all other shards have the
        same number of entries.
        :param file_paths: list of .hdf5 files
        :param opener:
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        shard_lengths = []
        print("Checking shard_lengths in", file_paths)
        for i, p in enumerate(file_paths):
            shard_lengths.append(HDF5Dataset._get_num_in_shard(p, opener))
        return shard_lengths

    def __init__(self, data_path,   # hdf5 file, or directory of hdf5s
                 shuffle_shards=False,
                 opener=default_opener,
                 seed=29):
        self.data_path = data_path
        self.shuffle_shards = shuffle_shards
        self.opener = opener
        self.seed = seed

        # If `data_path` is an hdf5 file
        if os.path.splitext(self.data_path)[-1] == '.hdf5' or os.path.splitext(self.data_path)[-1] == '.h5':
            self.data_dir = os.path.dirname(self.data_path)
            self.shard_paths = [self.data_path]
        # Else, if `data_path` is a directory of hdf5s
        else:
            self.data_dir = self.data_path
            self.shard_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.hdf5')) + glob.glob(os.path.join(self.data_dir, '*.h5')))

        assert len(self.shard_paths) > 0, "h5: Directory does not have any .hdf5 files! Dir: " + self.data_dir

        self.shard_lengths = HDF5Dataset.check_shard_lengths(self.shard_paths, self.opener)
        self.num_per_shard = self.shard_lengths[0]
        self.total_num = sum(self.shard_lengths)

        assert len(self.shard_paths) > 0, "h5: Could not find .hdf5 files! Dir: " + self.data_dir + " ; len(self.shard_paths) = " + str(len(self.shard_paths))

        self.num_of_shards = len(self.shard_paths)

        print("h5: paths", len(self.shard_paths), "; shard_lengths", self.shard_lengths, "; total", self.total_num)

        # Shuffle shards
        if self.shuffle_shards:
            np.random.seed(seed)
            np.random.shuffle(self.shard_paths)

    def __len__(self):
        return self.total_num

    def get_indices(self, idx): #根据随机索引获取 hdf5的 实际位置
        shard_idx = np.digitize(idx, np.cumsum(self.shard_lengths))
        idx_in_shard = str(idx - sum(self.shard_lengths[:shard_idx]))
        return shard_idx, idx_in_shard

    def __getitem__(self, index):
        idx = index % self.total_num
        shard_idx, idx_in_shard = self.get_indices(idx)
        # Read from shard
        with self.opener(self.shard_paths[shard_idx]) as f:
            data = f[idx_in_shard][()]
        return data


class HDF5Maker():

    def __init__(self, out_path, num_per_shard=100000, max_shards=None, name=None, name_fmt='shard_{:04d}.hdf5', force=False, video=False):

        # `out_path` could be an hdf5 file, or a directory of hdf5s
        # If `out_path` is an hdf5 file, then `name` will be its basename
        # If `out_path` is a directory, then `name` will be used if provided else name_fmt will be used

        self.out_path = out_path
        self.num_per_shard = num_per_shard
        self.max_shards= max_shards
        self.name = name
        self.name_fmt = name_fmt
        self.force = force
        self.video = video

        # If `out_path` is an hdf5 file
        if os.path.splitext(self.out_path)[-1] == '.hdf5' or os.path.splitext(self.out_path)[-1] == '.h5':
            # If it exists, check if it should be deleted
            if os.path.isfile(self.out_path):
                if not self.force:
                    raise ValueError('{} already exists.'.format(self.out_path))
                print('Removing {}...'.format(self.out_path))
                os.remove(self.out_path)
            # Make the directory if it does not exist
            self.out_dir = os.path.dirname(self.out_path)
            os.makedirs(self.out_dir, exist_ok=True)
            # Extract its name
            self.name = os.path.basename(self.out_path)
        # Else, if `out_path` is a directory
        else:
            self.out_dir = self.out_path
            # If `out_dir` exists
            if os.path.isdir(self.out_dir):
                # Check if it should be deleted
                if not self.force:
                    raise ValueError('{} already exists.'.format(self.out_dir))
                print('Removing *.hdf5 files from {}...'.format(self.out_dir))
                files = glob.glob(os.path.join(self.out_dir, "*.hdf5"))
                files += glob.glob(os.path.join(self.out_dir, "*.h5"))
                for file in files:
                    os.remove(file)
            # Else, make the directory
            else:
                os.makedirs(self.out_dir)

        self.writer = None
        self.shard_paths = []
        self.shard_number = 0

        # To save num_of_objs in each item
        shard_idx = 0
        idx_in_shard = 0

        self.create_new_shard()
        self.add_video_info()

    def create_new_shard(self):

        if self.writer:
            self.writer.close()

        self.shard_number += 1

        if self.max_shards is not None and self.shard_number == self.max_shards + 1:
            print('Created {} shards, ENDING.'.format(self.max_shards))
            return

        self.shard_p = os.path.join(self.out_dir, self.name_fmt.format(self.shard_number) if self.name is None else self.name)
        assert not os.path.exists(self.shard_p), 'Record already exists! {}'.format(self.shard_p)
        self.shard_paths.append(self.shard_p)

        print('Creating shard # {}: {}...'.format(self.shard_number, self.shard_p))
        self.writer = h5py.File(self.shard_p, 'w')
        if self.video:
            self.create_video_groups()

        self.count = 0

    def add_video_info(self):
        pass

    def create_video_groups(self):
        self.writer.create_group('len')
        self.writer.create_group('videos')

    def add_video_data(self, data, dtype=None):
        self.writer['len'].create_dataset(str(self.count), data=len(data))
        self.writer.create_group(str(self.count))
        for i, frame in enumerate(data):
            self.writer[str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")

    def add_data(self, data, dtype=None, return_curr_count=False):

        if self.video:
            self.add_video_data(data, dtype)
        else:
            # self.writer.create_dataset(str(self.count), data=data, compression="gzip", compression_opts=9)
            self.writer.create_dataset(str(self.count), data=data, dtype=dtype, compression="lzf")

        curr_count = self.count
        self.count += 1

        if self.count == self.num_per_shard:
            self.create_new_shard()

        if return_curr_count:
            return curr_count

    def close(self):
        self.writer.close()
        assert len(self.shard_paths)


if __name__ == "__main__":

    # Make
    # h5_maker = HDF5Maker('./EXPERIMENTS/h5', num_per_shard=10, force=True)
    #
    # a = [torch.zeros(12, 255, 52, 52)] * 12
    # for data in a:
    #     h5_maker.add_data(data)
    #
    # h5_maker.close()
    #
    # # Read
    # h5_ds = HDF5Dataset('./EXPERIMENTS/h5')
    # data = h5_ds[0]
    # print(data)
    # assert torch.all(data == a[0])

    # with h5py.File('D:/DeepLearning_Projects/mcvd-pytorch-master/datasets/signLanguage/shard_0001.hdf5') as f:
    #     print("Keys: %s" % f.keys())
    #     print(f['len'])
    # video_ds = HDF5Dataset('D:/DeepLearning_Projects/mcvd-pytorch-master/datasets/signLanguages/shard_0001.hdf5')
    # # print(video_ds)
    # with video_ds.opener(video_ds.shard_paths[0]) as f:
    #     # video_ds.get_indices(200)
    #     print(f['len']['1'][()])
    #     # num_train = f['']['59'][()]
    #     # print(num_train)
    #     print(len(video_ds))

    # with gzip.open('D:\phoenix14t.pami0.train.annotations_only.gzip', 'rb') as file:
    #     f = file.read()  # 读取为列表，进行解码，并去除分隔符
    #     for line in f:
    #         print(line)  # 输出内容

    with h5py.File('/sda/home/immc_guest/dy/new_project/datasets/phoenix2014T_clip_handpose_motion/train/shard_0001.hdf5') as f:
        print("Keys: %s" % f.keys())
        print(len(f['motion']['5'].keys()))
        # print(f['text']['0']['1'])
        print(len(f['video']['5'].keys()))
        # print(f['folder'].keys())
        # print(f['671']['161'].shape)
        # print(f['video']['1'].keys())
        # print(len(f['len'].keys()))
        # print(f['pose_video']['1'].keys())
        # print(len(f['video'].keys()))
        # print(len(f['pose_video'].keys()))
        # print(f['video']['1']['0'].shape)
        # print(f['len']['671'][()])

    # from segment_anything import build_sam, SamAutomaticMaskGenerator
    # from PIL import Image
    # import matplotlib.pyplot as plt
    # import cv2
    # def show_anns(anns):
    #     if len(anns) == 0:
    #         return
    #     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #     ax = plt.gca()
    #     ax.set_autoscale_on(False)
    #
    #     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    #     img[:, :, 3] = 0
    #     for ann in sorted_anns:
    #         m = ann['segmentation']
    #         color_mask = np.concatenate([np.random.random(3), [0.35]])
    #         img[m] = color_mask
    #     ax.imshow(img)
    # # 加载图像
    # image = Image.open('/sda/home/immc_guest/pyItem/knowledge_distillation/FR-UNet/robust_vessel_segmentation/test-yg/16_9_json.png')
    # image = image.resize((2048,2048 )).convert('RGB')
    # device = "cuda:9"
    # # 将图像转换为NumPy数组
    # np_array = np.array(image)
    #
    # mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="/sda/home/immc_guest/dy/new_project/segment-anything-main/segment_anything/sam_vit_h_4b8939.pth"))
    #
    # masks = mask_generator.generate( np_array)
    #
    #
    # plt.figure(figsize=(40,40))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()
    # def min_max_normalization(data):
    #     min_val = torch.min(data)
    #     max_val = torch.max(data)
    #     normalized_data = (data - min_val) / (max_val - min_val)
    #     return normalized_data
    # # a = torch.randn((5,128,128))
    # # b = torch.randn((5,128,128))
    # #
    # # res = min_max_normalization((a- b))
    # # thres = 0.1
    # # mask = torch.where(res>thres,torch.tensor(1),torch.tensor(0))
    #
    # def create_mask(x, y):
    #     diff = min_max_normalization( x - y)
    #     mask = torch.where(diff > 1, torch.tensor(1), torch.tensor(0))
    #     return mask
    #
    # # 示例数据
    # x = torch.randn(1, 5, 128, 128)
    # y = torch.randn(1, 5, 128, 128)
    #
    # # 创建掩码
    # mask = create_mask(x, y)
    #
    # print("掩码形状:", mask.shape)

    # import numpy as np
    #
    # # 读取npy文件
    # data = np.load('/sda/data/guest/data/handpose/train/01July_2011_Friday_tagesschau-2758.npy')
    #
    # # 查看shape
    # print(data.shape)
    # from einops import rearrange, reduce, repeat
    #
    # # decomposition is the inverse process - represent an axis as a combination of new axes
    # # several decompositions possible, so b1=2 is to decompose 6 to b1=2 and b2=3
    # ims = torch.rand((2,15,128,128))
    #
    # ims = rearrange(ims, 'b (f c) h w -> b f c h w ', c=3)
    # print(ims.shape)

    # import pickle
    #
    # with open('/sda/home/immc_guest/dy/new_project/logs/clip_encoder/px_T_128_cond0/logs/meters.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print(data)



