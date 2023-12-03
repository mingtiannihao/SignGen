# coding=utf-8

import os
import cv2

# mp4存放的路径，路径下只有mp4
videos_src_path = r'D:/DeepLearning_Projects/mcvd-pytorch-master/datasets/videos/1/'
# 保存的路径，会在路径下创建mp4文件名的文件夹保存图片
videos_save_path = r'D:/DeepLearning_Projects/mcvd-pytorch-master/datasets/videos/1'

videos = os.listdir(videos_src_path)
# videos = filter(lambda x: x.endswith('MP4'), videos)

for each_video in videos:
    print('Video Name :', each_video)
    # get the name of each video, and make the directory to save frames
    each_video_name, _ = each_video.split('.')
    os.mkdir(videos_save_path + '/' + each_video_name)

    each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

    # get the full path of each video, which will open the video tp extract frames
    each_video_full_path = os.path.join(videos_src_path, each_video)

    cap = cv2.VideoCapture(each_video_full_path)
    # 第几帧
    frame_count = 1
    # 隔着多少帧取一张
    frame_rate = 1
    success = True
    # 计数
    num = 0
    while (success):
        success, frame = cap.read()
        if success == True:

            if frame_count % frame_rate == 0:
                cv2.imwrite(each_video_save_full_path + each_video_name + "%06d.jpg" % num, frame)
                num += 1

        frame_count = frame_count + 1
    print('Final frame:', num)

