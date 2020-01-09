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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = 1
    output = torch.nn.functional.softmax(output)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(torch.from_numpy(np.array([target])).cuda().view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    # prepare images path
    cls_name = os.listdir(params["test_video_path"])
    videos_name = []
    labels = np.empty([0],np.int64)
    for i,cls in enumerate(cls_name):
        sub_path = os.path.join(params["test_video_path"],cls)
        sub_names = os.listdir(sub_path)
        videos_name += [os.path.join(sub_path, name) for name in sub_names]
        labels = np.concatenate((labels,np.ones([len(sub_names)],np.int64)*i),0)

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

        top1 = AverageMeter()
        for i in range(0, len(videos_name)):
            capture = cv2.VideoCapture(videos_name[i])
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            print('total frame:',frame_count)
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            left=0
            top=0
            if frame_height < frame_width:
                resize_height = CROP_SIZE
                resize_width = int(float(resize_height) / frame_height * frame_width)
                left = (resize_width - CROP_SIZE)//2
            else:
                resize_width = CROP_SIZE
                resize_height = int(float(resize_width) / frame_width * frame_height)
                top = (resize_height-CROP_SIZE)//2

            frame_list = []
            trace_torch_model = True
            while True:
                ret,frame = capture.read()
                if ret:
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

                        if trace_torch_model:
                            trace_model = torch.jit.trace(model,inputs.unsqueeze(0))
                            pred_out = trace_model(inputs.unsqueeze(0))
                            pred_out1 = trace_model(torch.ones_like(inputs).unsqueeze(0).cuda())
                            trace_torch_model = False
                            os.makedirs('./weights',exist_ok=True)
                            trace_model.save('./weights/fighting_detect_model.pt')
                            break

                        prec1, = accuracy(outputs.data, labels[i])
                        if prec1.item()< 1:
                            print('current presicion ',prec1.item(),'avg presicon ',top1.avg,'video name ',videos_name[i])
                        top1.update(prec1.item(), outputs.size(0))
                        frame_list = fix_images(frame_list,1)

                else:
                    break

            capture.release()
            break
        print_string = 'Top-1 accuracy: {top1_acc:.2f}%'.format(top1_acc=top1.avg)
        print(print_string)

if __name__ == '__main__':
    main()
