import sys
sys.path.append('..')
from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config
import wandb
from argparse import ArgumentParser
from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset


def run_PTI(opts, run_name='', use_wandb=False, use_multi_id_training=False):
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    global_config.run_name = global_config.encoder_name = opts.encoder

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    result_path = f'results/{global_config.run_name}'
    os.makedirs(result_path, exist_ok=True)

    dataset = ImagesDataset(global_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]), opts.split)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, use_wandb)
    else:
        coach = SingleIDCoach(dataloader, use_wandb)

    coach.edit(opts.edit, opts.factor)

    return global_config.run_name


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--edit', type=str)
    parser.add_argument('--factor', type=float)
    opts = parser.parse_args()
    run_PTI(opts, run_name='', use_wandb=False, use_multi_id_training=False)
