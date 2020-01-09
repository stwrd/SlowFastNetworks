import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
from lib import slowfastnet

from tensorboardX import SummaryWriter
import cv2

CROP_SIZE = 224

def main():
    # prepare images path
    tar_path = r'/media/hzh/ssd_disk/打架标注数据/fight_data20191114done/test'
    out_path = r'/media/hzh/work/workspace/data/fighting_data/dj/out'
    os.makedirs(out_path,exist_ok=True)
    videos_name = os.listdir(tar_path)
    videos_name = [os.path.join(tar_path, name) for name in videos_name]

    if len(videos_name) == 0:
        raise Exception("no image found in {}".format(params["test_video_path"]))

    # Start inference
    batch_size = 1

    model = slowfastnet.resnet50(class_num=params['num_classes'])
    model.train(False)

    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load pretrain model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.cuda(params['gpu'][0])
    with torch.no_grad():
        def fix_images(img_list, frame_num):
            img_list = img_list[frame_num:]
            return img_list

        for i in range(0, len(videos_name)):
            capture = cv2.VideoCapture(videos_name[i])
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print('total frame:',frame_count)
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_list = []
            src_frame_list = []
            new_video_idx = 1
            while True:
                ret,frame = capture.read()
                if ret:
                    src_frame_list.append(frame)
                    # will resize frames if not already final size
                    # if (frame_height != resize_height) or (frame_width != resize_width):
                    #     frame = cv2.resize(frame, (resize_width, resize_height))
                    frame = cv2.resize(frame, (CROP_SIZE, CROP_SIZE))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32)
                    # frame = frame[top:top+CROP_SIZE,left:left+CROP_SIZE]
                    frame = (frame/128.0).astype(np.float32) - np.array([[[1.0, 1.0, 1.0]]],np.float32)
                    frame_list.append(frame)
                    if len(frame_list) == params['clip_len']:
                        tensor_frame = np.array(frame_list).transpose((3, 0, 1, 2))
                        inputs = torch.from_numpy(tensor_frame).cuda()
                        outputs = model(inputs.unsqueeze(0))
                        # outputs = model(torch.ones_like(inputs).unsqueeze(0).cuda())
                        outputs = torch.nn.functional.softmax(outputs).cpu().numpy()
                        if np.argmax(outputs,1) == 1:
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            basename = os.path.splitext(os.path.split(videos_name[i])[1])[0]
                            save_video_name = os.path.join(out_path,basename + '_%03d' %new_video_idx + '.mp4')
                            print('find fighting save to ',save_video_name)
                            vw = cv2.VideoWriter(save_video_name, fourcc, 6.0, (frame_width, frame_height))
                            for im in src_frame_list:
                                vw.write(im)
                            vw.release()
                            new_video_idx+=1
                        frame_list = fix_images(frame_list,12)
                        src_frame_list = fix_images(src_frame_list,12)

                else:
                    break

            capture.release()

if __name__ == '__main__':
    main()
