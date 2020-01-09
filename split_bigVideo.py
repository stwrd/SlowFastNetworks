import cv2
import os
import sys
import numpy as np


if __name__ == '__main__':
    tar_path = '/media/hzh/work/workspace/data/fighting_data/dj/Arson'
    filename_list = os.listdir(tar_path)
    img_stack = []
    standard_width = 0
    standard_height = 0
    ratio = 1

    unit_len = 40#单位长度
#1126,1130,1143,1146,
    for filename in filename_list:
        print('process ',filename)
        if not filename.endswith('.mp4'):
            continue
        full_path = os.path.join(tar_path,filename)
        basename = os.path.splitext(filename)[0]
        video_cap = cv2.VideoCapture(full_path)
        frame_list = []
        count = 0#帧标记
        idx = 1#视频序号

        fps = video_cap.get(cv2.CAP_PROP_FPS)
        fps = np.ceil(fps)
        step_num = np.floor(fps // 6.0)
        vw = None
        if step_num == 0:
            step_num = 1
        while True:
            is_read, img = video_cap.read()
            if is_read:
                #1%5=1
                #(1-1)%5=1-1
                #(1%1)=0
                #(1-1)%1=0
                if count % step_num == 0:
                    ratio = 400.0 / max(img.shape[1], img.shape[0])
                    standard_height = int(float(img.shape[0]) * ratio)
                    standard_width = int(float(img.shape[1]) * ratio)
                    img = cv2.resize(img, (standard_width, standard_height))
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    if (count//step_num)%unit_len == 0:
                        if vw is not None:
                            print("current frame idx:%d,relative frame idx:%d,save video total frames:%d" % (
                            count, count // step_num, small_video_frame_num))
                            vw.release()
                            split_name = os.path.join(tar_path,'split','%s_%03d.mp4'%(basename,idx))
                            print('create video ',split_name)
                            vw = cv2.VideoWriter(split_name,fourcc,6.0, (standard_width,standard_height))
                            idx += 1
                            small_video_frame_num = 0
                        else:
                            os.makedirs(os.path.join(tar_path, 'split'), exist_ok=True)
                            split_name = os.path.join(tar_path, 'split', '%s_%03d.mp4' % (basename, idx))
                            print('create video ', split_name)
                            vw = cv2.VideoWriter(split_name, fourcc, 6.0, (standard_width, standard_height))
                            idx += 1
                            small_video_frame_num = 0
                    vw.write(img)
                    small_video_frame_num+=1
                count += 1
            else:
                if vw.isOpened():
                    print("End frame!!!----------->current frame idx:%d,relative frame idx:%d,save video total frames:%d" % (
                        count, count // step_num, small_video_frame_num))
                    vw.release()
                break


