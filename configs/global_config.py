## Device
cuda_visible_devices = '0'
device = 'cuda:0'

## Logs
training_step = 1
image_rec_result_log_snapshot = 100
pivotal_training_steps = 0
model_snapshot_interval = 400


#
# input_data_path = '/home/ssd1/Database/CelebA-HQ/test/'
input_data_path = 'ood_images/'
generator_path = '/home/ssd2/caopu/workspace/PTI/ood_results/'
encoder_name = 'e4e'
codes_paths = 'codes'
run_name = f'{encoder_name}-face'