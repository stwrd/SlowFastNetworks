params = dict()

params['num_classes'] = 2

params['dataset'] = 'E:/workspace/data/fighting_data'
params['test_video_path'] = 'E:/workspace/data/fighting_data/test'

params['epoch_num'] = 400
params['batch_size'] = 8
params['step'] = 200
params['num_workers'] = 4
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 1
params['pretrained']='E:/workspace/SlowFastNetworks/UCF101/clip_len_36frame_sample_rate_1_checkpoint_90.pth.tar'
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'UCF101'
params['clip_len'] = 36
params['frame_sample_rate'] = 1
