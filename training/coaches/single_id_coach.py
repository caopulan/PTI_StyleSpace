import sys
sys.path.append('/home/ssd2/caopu/workspace/PTI')
import os
import torch
import numpy as np
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from utils.data_utils import tensor2im
from utils.models_utils import load_tuned_G_v2

class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        save_path = f'results/{global_config.run_name}'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f'{save_path}/model', exist_ok=True)
        os.makedirs(f'{save_path}/images', exist_ok=True)

        use_ball_holder = True

        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]
            mask = torch.Tensor(np.load(f'{image_name}.npy')).cuda()

            self.restart_training()
            feature = torch.zeros([1, 64, 512, 512]).cuda()
            feature.requires_grad = True
            optimizer_feature = torch.optim.Adam([feature], lr=0.05)

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = torch.load(f'{global_config.codes_paths}/{global_config.encoder_name}/{image_name}.pt')[None]
            self.w_pivots[image_name] = w_pivot

            w_pivot = w_pivot.to(global_config.device)

            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            for i in tqdm(range(hyperparameters.max_pti_steps)):

                img = self.forward(w_pivot)
                loss, l2_loss_val, loss_lpips = self.calc_loss(img, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot, mask)

                self.optimizer.zero_grad()
                # optimizer_feature.zero_grad()
                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()
                # optimizer_feature.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                log_images_counter += 1
                if i % 20 == 0:
                    img = tensor2im(img[0])
                    img.save(f'{save_path}/images/PTI-{image_name}-step-{i}.jpg')

            for i in tqdm(range(500)):

                img = self.forward(w_pivot, mask, feature)

                loss, l2_loss_val, loss_lpips = self.calc_loss(img, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot, mask=None)
                optimizer_feature.zero_grad()
                loss.backward()
                optimizer_feature.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                log_images_counter += 1

                if i % 10 == 0:
                    img = tensor2im(img[0])
                    img.save(f'{save_path}/images/{image_name}-step-{i}.jpg')
            self.image_counter += 1
            img = tensor2im(img[0])
            img.save(f'{save_path}/images/{image_name}.jpg')
            torch.save(self.G, f'{save_path}/model/{image_name}.pt')

    def edit(self, edit_vector, factor):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        save_path = f'results/{global_config.run_name}'
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f'{save_path}/model', exist_ok=True)
        os.makedirs(f'{save_path}/{os.path.basename(edit_vector)[:-3]}-{factor}', exist_ok=True)

        use_ball_holder = True

        edit = torch.load(edit_vector, map_location='cpu')[None].cuda()
        for fname, image in tqdm(self.data_loader):
            image_name = fname[0]

            G = load_tuned_G_v2(os.path.join('/media/sariel/disk2T', save_path, 'model', f'{image_name}.pt'))
            w_pivot = torch.load(f'{global_config.codes_paths}/{global_config.encoder_name}/{image_name}.pt')[None]
            w_pivot = w_pivot.to(global_config.device)

            w_edit = w_pivot + edit * factor
            generated_images = G.synthesis(w_edit, noise_mode='const', force_fp32=True)

            img = tensor2im(generated_images[0])
            img.save(f'{save_path}/{os.path.basename(edit_vector)[:-3]}-{factor}/{image_name}.jpg')
