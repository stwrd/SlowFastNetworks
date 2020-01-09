import csv
import os
import shutil

outpath = r'/media/hzh/ssd_disk/BaiduNetdiskDownload/Charades'
inputpath = r'/media/hzh/ssd_disk/BaiduNetdiskDownload/Charades/Charades_v1_480'
os.makedirs(os.path.join(outpath,'throw'),exist_ok=True)
os.makedirs(os.path.join(outpath,'other'),exist_ok=True)

with open(r'/media/hzh/ssd_disk/BaiduNetdiskDownload/Charades/Charades/Charades_v1_train.csv') as f:
    throw_action = ['c019']
    reader = csv.DictReader(f)
    for row in reader:
        videoname = row['id']
        print(videoname)
        if 'EGO' not in videoname:
            verify = row['verified']
            if verify == 'Yes':
                actions = row['actions'].split(';')
                input_video = os.path.join(inputpath, '{}.mp4'.format(videoname))
                idx = 1
                for action in actions:
                    if action == '':
                        continue
                    vid, s, e = action.split(' ')
                    duaring = max(7,float(e) - float(s))
                    if vid in throw_action:
                        output_video = os.path.join(outpath, 'throw', '{}_{:0>2}.mp4'.format(videoname, idx))
                        cmd = 'ffmpeg -y -i {}  -ss {} -t {} -r 6 {}'.format(input_video,s,duaring,output_video)
                        os.system(cmd)
                    idx += 1
                # if idx > 1:
                #     output_video = os.path.join(outpath, 'other', '{}.mp4'.format(videoname))
                #     cmd = 'ffmpeg -y -i {} -r 6 {}'.format(input_video,output_video)
                #     os.system(cmd)
