import argparse
import cv2
import glob
import imageio
import numpy as np
import os
import sys

from functools import partial
from multiprocessing import Pool

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse

from dwpose import DWposeDetector
from openpose import OpenposeDetector
from openpose.utils import  resize_image, HWC3
import os
import sys
import pickle
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from PIL import Image
from tensorflow.python.platform import gfile
from tqdm import tqdm
import csv
from pytorch_transformers import  BertModel, BertConfig,BertTokenizer
import numpy as np
from PIL import Image
import torch
import  clip
from PIL import Image
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F

from h5 import HDF5Maker

def textToBert_dim96(text):
    tokenizer = BertTokenizer.from_pretrained(r'/datasets/bertbasecased/vocab.txt')
    # s = 'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'
    tokens = tokenizer.tokenize(text)
    # print(tokens)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # print(tokens)

    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    # print(ids.shape)

    model = BertModel.from_pretrained(r'datasets/bertbasecased/')
    # pooled 可以暂时理解为最后一层每一个句子第一个单词[cls]的结果
    all_layers_all_words, pooled = model(ids)
    # print( pooled)
    # print(pooled.shape)
    net = nn.Sequential( nn.Linear(768, 768),nn.ReLU(),nn.Linear(768, 96),nn.Tanh())
    output = (net(pooled)-0.25)*0.1
    return output

def textToBert_dim96_csd(text):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese')
    # s = 'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # print(tokens)

    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

    model = BertModel.from_pretrained('bert-base-cased')
    # pooled 可以暂时理解为最后一层每一个句子第一个单词[cls]的结果
    all_layers_all_words, pooled = model(ids)
    net = nn.Sequential( nn.Linear(768, 768),nn.ReLU(),nn.Linear(768, 96),nn.Tanh())
    output = (net(pooled)-0.25)*0.1
    return output
def clip_text_encoder(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-B/32")

    # image = preprocess(Image.open("/sda/home/immc_guest/dy/new_project/CLIP-main/CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(text).to(device=device)

    with torch.no_grad():
     text_features = model.encode_text(text)

    return text_features

class SignLanguage_HDF5Maker(HDF5Maker):


    def create_video_groups(self):
        self.writer.create_group('video')
        self.writer.create_group('len')
        self.writer.create_group('text')
        self.writer.create_group('pose_video')
        self.writer.create_group('motion')
        self.writer.create_group('depth')

    def add_video_data_all(self, data,deepth_data,motion, text=None,origin_text=None,depth=None):
        self.writer['len'].create_dataset(str(self.count), data=len(data))
        self.writer['text'].create_dataset(str(self.count), data= text,)
        self.writer['pose_video'].create_group(str(self.count))
        self.writer['video'].create_group(str(self.count))
        self.writer['motion'].create_group(str(self.count))
        self.writer['depth'].create_group(str(self.count))
        for i, frame in enumerate(data):
            self.writer['video'][str(self.count)].create_dataset(str(i), data=frame, compression="lzf")
        for j, deepth_frame in enumerate(deepth_data):
            self.writer['pose_video'][str(self.count)].create_dataset(str(j), data=deepth_frame, compression="lzf")
        for k, motion_frame in enumerate(motion):
            self.writer['motion'][str(self.count)].create_dataset(str(k), data=motion_frame, compression="lzf")
        for l, depth_frame in enumerate(depth):
            self.writer['motion'][str(self.count)].create_dataset(str(l), data=depth_frame, compression="lzf")

        self.count += 1

        if self.count == self.num_per_shard:
            self.create_new_shard()


def center_crop(image):
    h, w, c = image.shape
    new_h, new_w = h if h < w else w, w if w < h else h
    r_min, r_max = h // 2 - new_h // 2, h // 2 + new_h // 2
    c_min, c_max = w // 2 - new_w // 2, w // 2 + new_w // 2
    return image[r_min:r_max, c_min:c_max, :]


def read_video(video_file, image_size):
    frames = []
    filenames = gfile.Glob(os.path.join('%s' % (video_file+'/1/'), '*'))
    print(len(filenames))
    h, w = image_size, image_size
    new_h = image_size
    new_w = int(new_h / h * w)
    for path in filenames:
        pil_im = Image.open(path)

        # pil_im = Image.fromarray(img)
        pil_im_rsz = pil_im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        frames.append(np.array(pil_im_rsz))

    np_frames = np.stack(frames)
    return np_frames

def read_video_handpose(video_file, image_size):
    # device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
    split = args.split
    parent_dir = '2014/'+split
    filename = video_file.split('/')[-1]
    path = os.path.join(parent_dir, (filename + '.npy') )
    if not os.path.exists(os.path.dirname(path)):
        print('fail:      ',path)
        raise
    data = np.load(path)
    # f h w c
    return data

def read_video_motion(video_file):
    split = args.split
    parent_dir = ''
    filename = video_file.split('/')[-1]
    path = os.path.join(parent_dir, (filename + '_raft.npy'))
    if not os.path.exists(os.path.dirname(path)):
        print('fail:      ', path)
        raise
    data = np.load(path)
    print(data.shape)
    # # f c h w
    return data

def read_video_depth(video_file):
    split = args.split
    parent_dir = ''
    filename = video_file.split('/')[-1]
    path = os.path.join(parent_dir, filename)
    if not os.path.exists(os.path.dirname(path)):
        print('fail:      ', path)
        raise
    data = np.load(path)
    print(data.shape)
    # # f c h w
    return data

def read_pictures(video_file, image_size=128):
    frames = []
    # reader = imageio.get_reader(video_file)
    filenames = gfile.Glob(os.path.join('%s' % (video_file), '*'))

    fi = gfile.Glob(os.path.join('path' , '*'))
    h, w = image_size, image_size
    new_h = image_size
    new_w = int(new_h / h * w)
    for path in filenames:
        # img_cc = center_crop(img)
        pil_im = Image.open(path)
        # pil_im = Image.fromarray(img)
        pil_im_rsz = pil_im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        frames.append(np.array(pil_im_rsz))
        # frames.append(np.array(img))
    return np.stack(frames)

def process_pictures(video_file, image_size):
    frames = []
    try:
        frames = read_pictures(video_file, image_size)
    except StopIteration:
        pass
        # break
    except (KeyboardInterrupt, SystemExit):
        print("Ctrl+C!!")
        return "break"
    except:
        e = sys.exc_info()[0]
        print("ERROR:", e)
    return frames

def process_video(video_file, image_size):
    frames = []
    handpose_video = []
    motion = []
    depth = []
    try:
        frames = read_video(video_file, image_size)
        handpose_video = read_video_handpose(video_file,image_size)
        motion = read_video_motion(video_file)
        depth = read_video_depth(video_file)
    except StopIteration:
        pass
        # break
    except (KeyboardInterrupt, SystemExit):
        print("Ctrl+C!!")
        return "break"
    except:
        e = sys.exc_info()[0]
        print("ERROR:", e)
    return frames,handpose_video,motion,depth

#phoenix-2014/2014T
def make_text_dic(dir):
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    records = []
    dictionary = {}
    ori_dictionary = {}
    j=1
    with open(dir) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            records.append(row)
        for i in tqdm(range(len(records))):
            c = ''.join(records[i])
            b = c.split('|')
            key = b[0]
            # print(key)
            ori_value = b[1]
            value = b[3].strip("__ON__").replace("__PU", '').replace("__EMOTION__ ", '').strip("__OFF__").replace(
                '__LEFTHAND__', '').replace("__", '')
            # print(value)
            clip = clip_text_encoder(value).to(device=device).detach()
            clip_info = clip.cpu().numpy()
            ori_dictionary[key] = ori_value
            dictionary[key] = clip_info
    return dictionary,ori_dictionary

#CS-daily
def make_csd_text_dic(dir):
    device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
    dictionary = {}
    ori_dictionary = {}
    with open(dir, 'rb') as f:
            data = pickle.load(f)
            lens = len(data['info'])
            print(lens)

            for i in tqdm(range(lens)):
                key = data['info'][i]['name']
                value = ''.join(data['info'][i]['label_char'])
                bert = textToBert_dim96_csd(value).to(device=device).detach()
                bert_info = bert.cpu().numpy()
                ori_dictionary[key] = value
                dictionary[key] = bert_info


    return dictionary,ori_dictionary


def make_h5_from_sl(sl_dir, text_path,split='train', out_dir='./h5_ds', image_size=128, vids_per_shard=100000, force_h5=False):


    #text dictionary
    #phoenix
    dic,ori_dictionary = make_text_dic(text_path)
    print('dic done')
    #csl-daily
    # dic, ori_dictionary = make_csd_text_dic(text_path)
    # print(dic.get('01December_2011_Thursday_tagesschau_default-12'))
    # H5 maker
    h5_maker = SignLanguage_HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)
    # split = "train"
    # filenames = gfile.Glob(os.path.join('%s/%s' % (data_dir, split), '*'))
    filenames = gfile.Glob(os.path.join('%s%s' % (sl_dir, split), '*'))

    for  i in tqdm(range(len(filenames))):
        try:
            video,handpose_video,motion,depth = process_video(filenames[i], image_size)
            text_key = filenames[i].split('/')[-1]
            text_value = dic[text_key]
            ori_value = ori_dictionary[text_key]

            h5_maker.add_video_data_all(video,handpose_video,motion,text_value,ori_value,depth)

        except StopIteration:
            break

        except (KeyboardInterrupt, SystemExit):
            print("Ctrl+C!!")
            break

        except:
            e = sys.exc_info()[0]
            print("ERROR:", e)

    h5_maker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--sl_dir', type=str, help="Directory with videos")
    parser.add_argument('--vids_per_shard', type=int, default=100000)
    parser.add_argument('--force_h5', type=eval, default=False)
    parser.add_argument('--split', type=str, default=False)
    parser.add_argument('--image_size', type=eval,  default=False)
    parser.add_argument('--text_path', type=str, default=False)

    args = parser.parse_args()


    make_h5_from_sl(out_dir=os.path.join(args.out_dir),
                    sl_dir=args.sl_dir,
                    split=args.split,
                    image_size=args.image_size, vids_per_shard=args.vids_per_shard,
                    force_h5=args.force_h5,
                    text_path = args.text_path)
    print('done')


