import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import augmentations
import imgaug.augmenters as iaa
import imgaug as ia


class VideoDataset(Dataset):

    def __init__(self, directory, mode='train', clip_len=8, frame_sample_rate=1):
        folder = Path(directory)/mode  # get the directory of the specified split
        self.clip_len = clip_len
        self.short_side = [128, 160]
        self.crop_size = 112
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode

        self.fnames, labels = [], []
        if mode == 'train':
            with open(os.path.join(directory,mode+'.txt'), 'r') as file:
                video_files = file.readlines()
                video_files = [video_file.replace('\n', '') for video_file in video_files]
                for video_file in video_files:
                    f, l = video_file.split(' ')
                    self.fnames.append(os.path.join(directory,f))
                    labels.append(int(l))
                self.label_array = np.array(labels) -1
        else:
            with open(os.path.join(directory,'classInd.txt')) as file:
                pairs = file.readlines()
                pairs = [pair.replace('\n', '') for pair in pairs]
                self.label2index = {p.split(' ')[1]: int(p.split(' ')[0]) for p in pairs}
            with open(os.path.join(directory,mode+'.txt'), 'r') as file:
                video_files = file.readlines()
                video_files = [video_file.replace('\n', '') for video_file in video_files]
                for video_file in video_files:
                    self.fnames.append(os.path.join(directory,video_file))
                    name = video_file.split('/',1)[0]
                    labels.append(self.label2index[name])
                self.label_array = np.array(labels) -1
        # for label in sorted(os.listdir(folder)):
        #     for fname in os.listdir(os.path.join(folder, label)):
        #         self.fnames.append(os.path.join(folder, label, fname))
        #         labels.append(label)
        # # prepare a mapping between the label names (strings) and indices (ints)
        # self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))}
        # # convert the list of label names into an array of label indices
        # self.label_array = np.array([self.label2index[label] for label in labels], dtype=np.int64)
        #
        # label_file = str(len(os.listdir(folder)))+'class_labels.txt'
        # with open(label_file, 'w') as f:
        #     for id, label in enumerate(sorted(self.label2index)):
        #         f.writelines(str(id + 1) + ' ' + label + '\n')

        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                # iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                # sometimes(iaa.CropAndPad(
                #     percent=(-0.05, 0.1),
                #     pad_mode=ia.ALL,
                #     pad_cval=(0, 255)
                # )),
                # sometimes(iaa.Affine(
                #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                #     # scale images to 80-120% of their size, individually per axis
                #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                #     # translate by -20 to +20 percent (per axis)
                #     rotate=(-45, 45),  # rotate by -45 to +45 degrees
                #     shear=(-16, 16),  # shear by -16 to +16 degrees
                #     order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                #     cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                #     mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                # )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               # # convert images into their superpixel representation
                               # iaa.OneOf([
                               #     iaa.GaussianBlur((0, 1.5)),  # blur images with a sigma between 0 and 3.0
                               #     iaa.AverageBlur(k=(2, 5)),
                               #     # blur image using local means with kernel sizes between 2 and 7
                               #     iaa.MedianBlur(k=(3, 7)),
                               #     # blur image using local medians with kernel sizes between 2 and 7
                               # ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               # iaa.OneOf([
                               #     iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               # ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   iaa.FrequencyNoiseAlpha(
                                       exponent=(-4, 0),
                                       first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                       second=iaa.ContrastNormalization((0.5, 2.0))
                                   )
                               ]),
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # # sometimes move parts of the image around
                               # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.new_loadvideo(self.fnames[index],self.crop_size)

        while buffer.shape[0]<self.clip_len:
            index = np.random.randint(self.__len__())
            buffer = self.new_loadvideo(self.fnames[index],self.crop_size)

        if self.mode == 'train' or self.mode == 'training':
            buffer = self.randomflip(buffer)
        # buffer = self.crop(buffer, self.clip_len, self.crop_size)

            # cv2.imshow('before',cv2.cvtColor(buffer[0], cv2.COLOR_BGR2RGB).astype('uint8'))
            buffer = buffer.reshape([1,buffer.shape[1]*buffer.shape[0],buffer.shape[2],buffer.shape[3]])
            # cv2.imshow('before', cv2.cvtColor(buffer[0], cv2.COLOR_BGR2RGB).astype('uint8'))
            # cv2.waitKey(0)
            buffer = self.seq.augment_images(buffer)
            buffer = buffer.reshape([self.clip_len,self.crop_size,self.crop_size,-1])
            #augmentation
            # image_augment = augmentations.PhotometricDistort()
            # buffer = image_augment(buffer)
            # buffer[buffer>255] = 255
            # buffer[buffer<0] = 0
            # if self.label_array[index] == 1:
            #     for image in buffer:
            #         cv2.imshow('after', cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8'))
            #         cv2.waitKey(0)
        buffer = buffer.astype(np.float32)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return buffer, self.label_array[index]

    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def loadvideo(self, fname):
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count-1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count>300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count>end_idx:
                break
            if count%self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size

                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()
        return buffer
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    #optimize data read method
    def new_loadvideo(self, fname, crop_size):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Randomly select start indices in order to crop the video
        top = np.random.randint(frame_height * 0.1)
        bottom = np.random.randint(frame_height*0.9, frame_height)
        left = np.random.randint(frame_width * 0.1)
        right = np.random.randint(frame_width*0.9 , frame_width)

        # if frame_height < frame_width:
        #     resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
        #     resize_width = int(float(resize_height) / frame_height * frame_width)
        # else:
        #     resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
        #     resize_height = int(float(resize_width) / frame_width * frame_height)

        # randomly select time index for temporal jittering
        if frame_count < self.clip_len:
            time_index = 0
        else:
            time_index = np.random.randint(frame_count - self.clip_len + 1)


        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        buffer = np.empty((self.clip_len, crop_size, crop_size, 3), np.dtype('uint8'))

        count = 0
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        capture.set(cv2.CAP_PROP_POS_FRAMES,time_index)
        while (sample_count < self.clip_len):
            ret, frame = capture.read()
            if ret:
                # will resize frames if not already final size
                # if (frame_height != resize_height) or (frame_width != resize_width):
                #     frame = cv2.resize(frame, (resize_width, resize_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame[top:bottom,left:right, :]
                frame = cv2.resize(frame, (crop_size, crop_size))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            else:
                break
            count += 1
        capture.release()
        return buffer

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':

    datapath = r'H:\BaiduNetdiskDownload\UCF-101'
    train_dataloader = \
        DataLoader( VideoDataset(datapath, mode='test',clip_len=36), batch_size=8, shuffle=True, num_workers=0)
    for step, (buffer, label) in enumerate(train_dataloader):
        print("label: ", label)
