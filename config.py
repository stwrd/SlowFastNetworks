params = dict()

params['num_classes'] = 2

params['dataset'] = '/media/hzh/work/workspace/data/fighting_data'
params['test_video_path'] = '/media/hzh/work/workspace/data/fighting_data/test'

params['epoch_num'] = 400
params['batch_size'] = 8
params['step'] = 200
params['num_workers'] = 1
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 1
params['pretrained']='UCF101/2020-01-04-18-17-04/clip_len_36frame_sample_rate_1_checkpoint_40.pth.tar'
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'UCF101'
params['clip_len'] = 36
params['frame_sample_rate'] = 1
