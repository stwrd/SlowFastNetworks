import os
import random
import glob
import cv2
from config import params
import shutil

train_percent = 0.9
dataset_path = r'E:\workspace\data\tmp'
dst_dataset_path = r'E:\workspace\data\fighting_data'

train_path = os.path.join(dst_dataset_path, 'train')
test_path = os.path.join(dst_dataset_path, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

sub_classlist = os.listdir(dataset_path)
for sub_class in sub_classlist:
    sub_class_path = os.path.join(dataset_path,sub_class)
    find_str = os.path.join(sub_class_path,'*.mp4')
    total_videolist = glob.glob(find_str)

    #filter invalid video
    filter_videolist = []
    for video_path in total_videolist:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            all_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if all_cnt >= params['clip_len']+2:
                filter_videolist.append(video_path)

    num=len(filter_videolist)
    list=range(num)
    tr=int(num*train_percent)
    te=int(tr*train_percent)
    trainlist= random.sample(list,tr)

    sub_train_path = os.path.join(train_path,sub_class)
    os.makedirs(sub_train_path,exist_ok=True)
    sub_test_path = os.path.join(test_path,sub_class)
    os.makedirs(sub_test_path,exist_ok=True)
    for i in list:
        if i in trainlist:
            shutil.copy(filter_videolist[i],sub_train_path)
        else:
            shutil.copy(filter_videolist[i],sub_test_path)
