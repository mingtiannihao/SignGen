

import numpy as np
import os
import pickle
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .h5 import HDF5Dataset


class SignLanguage(Dataset):

    def __init__(self, data_path, frames_per_sample=5, train=True, random_time=True, random_horizontal_flip=True,color_jitter=0,
                 total_videos=-1, skip_videos=0,image_size=64):

        self.data_path = data_path                    # '/path/to/Datasets/SignLanguage_h5' (with .hdf5 file in it), or to the hdf5 file itself
        self.train = train
        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        self.random_time = random_time
        self.color_jitter = color_jitter
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.jitter = transforms.ColorJitter(hue=color_jitter) #改变亮度
        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        # Train
        # Read h5 files as dataset
        # self.videos_ds = HDF5Dataset(self.data_path)

        print(f"Dataset length: {self.__len__()}")

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        print(shard_idx,'   ',idx_in_shard,'       ')
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.videos_ds)

    def max_index(self):
        return len(self.videos_ds)

    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        if int(idx_in_shard) >= self.__len__(): idx_in_shard = str(int(idx_in_shard))
        prefinals = []
        prefinals_depth = []
        prefinals_pose = []
        prefinals_motion = []
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:

            label = f['text'][str(idx_in_shard)][()]
            video_len = f['len'][str(idx_in_shard)][()]
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)

            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                img = f['video'][str(idx_in_shard)][str(i)][()]
                arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
                prefinals.append(arr)
                pose = f['pose_video'][str(idx_in_shard)][str(i)][()]
                prefinals_pose.append(torch.from_numpy(pose))
                depth = f['depth'][str(idx_in_shard)][str(i)][()]
                prefinals_depth.append(torch.from_numpy(depth))
                motion_img = f['motion'][str(idx_in_shard)][str(i)][()]
                prefinals_motion.append(torch.from_numpy(motion_img))


        data = torch.stack(prefinals)
        depth_data = torch.stack(prefinals_depth)
        # pose = torch.permute(deepth_data, (0,3,1,2))
        motion_data = torch.stack(prefinals_motion)
        pose_data = torch.stack(prefinals_pose)
        # motion_data = torch.permute(motion_data, (0, 3, 1, 2))
        data = self.jitter(data)
        # print(deepth_data.shape, 'dataloader pose')
        # print(motion_data.shape,'dataloader motion')
        return data,label,depth_data,motion_data,pose_data

if __name__ == "__main__":

    data_path ='/sda/home/immc_guest/dy/new_project/datasets/phoenix2014-release_T_en_clip/test/shard_0001.hdf5'
    dataset = SignLanguage(os.path.join(data_path), frames_per_sample=2, random_time=True,
                           random_horizontal_flip=True,
                           color_jitter= 0)

    dataset.__getitem__(0)