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
import cv2

CROP_SIZE = 224

def main():

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
        dummy_inputs = torch.zeros([1,3,36,CROP_SIZE,CROP_SIZE]).cuda()
        trace_model = torch.jit.trace(model,dummy_inputs)
        pred_out = trace_model(dummy_inputs)
        pred_out1 = trace_model(torch.zeros_like(dummy_inputs).cuda())
        outs = model((torch.ones_like(dummy_inputs)*0.5).cuda())
        trace_torch_model = False
        os.makedirs('./weights',exist_ok=True)
        trace_model.save('./weights/fighting_detect_model.pt')

if __name__ == '__main__':
    main()
