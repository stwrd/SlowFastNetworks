import cv2
import os
import sys
import numpy as np
import shutil
if __name__ == '__main__':
    tar_path = sys.argv[1]
    for sub_folder in os.listdir(tar_path):
        full_sub_floder = os.path.join(tar_path, sub_folder)
        if os.path.isdir(full_sub_floder):
            filename_list = os.listdir(full_sub_floder)
            img_stack = []
            standard_width = 0
            standard_height = 0
            ratio = 1

        #1126,1130,1143,1146,
            for filename in filename_list:
                print('process ',filename)
                if not filename.endswith('.avi'):
                    continue
                full_path = os.path.join(full_sub_floder,filename)
                basename = os.path.splitext(filename)[0]
                video_cap = cv2.VideoCapture(full_path)
                frame_list = []
                count = 0#帧标记
                idx = 1#视频序号

                fps = video_cap.get(cv2.CAP_PROP_FPS)
                fps = np.ceil(fps)
                step_num = np.floor(fps // 6.0)
                if step_num == 0:
                    step_num = 1
                vw = None
                while True:
                    is_read, img = video_cap.read()
                    if is_read:
                        count += 1

                        #1%5=1
                        #(1-1)%5=1-1
                        #(1%1)=0
                        #(1-1)%1=0
                        if (count-1) % step_num == 0:
                            ratio = 400.0 / max(img.shape[1], img.shape[0])
                            standard_height = int(float(img.shape[0]) * ratio)
                            standard_width = int(float(img.shape[1]) * ratio)
                            img = cv2.resize(img, (standard_width, standard_height))
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            if vw is None:
                                vw = cv2.VideoWriter(os.path.join(full_sub_floder,basename+'_tmp.avi'), fourcc, 6.0, (standard_width, standard_height))
                            vw.write(img)
                    else:
                        vw.release()
                        tmp_path = os.path.join(full_sub_floder,basename+'_tmp.avi')
                        if os.path.exists(full_path) and os.path.exists(tmp_path):
                            print('copy ',tmp_path,' to ',full_path)
                            shutil.move(tmp_path,full_path)
                        break


