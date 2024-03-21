import argparse
import os
import random
import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from MusicControlNet import MusicControlNet
from MusicDataset import MusicDataset

def apply_snr_weight(loss, timesteps, noise_scheduler, min_snr_gamma, v_parameterization):
    if v_parameterization:
        snr = noise_scheduler.get_v_posterior_log_variance(timesteps)
    else:
        snr = noise_scheduler.get_posterior_log_variance(timesteps)
    
    snr_exp = torch.exp(-min_snr_gamma * snr)
    loss_weights = snr_exp / snr_exp.mean()
    loss = loss * loss_weights
    return loss

def train(args):
    accelerator = Accelerator()
    
    controlnet = MusicControlNet(num_genres=args.num_genres, num_moods=args.num_moods, num_controls=args.num_controls)
    unet = UNet2DConditionModel.from_pretrained(args.unet_model_name_or_path)
    
    train_dataset = MusicDataset(args.train_data_dir, args.conditioning_data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.learning_rate)
    
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    controlnet, unet, optimizer, train_dataloader = accelerator.prepare(controlnet, unet, optimizer, train_dataloader)
    
    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            with accelerator.accumulate(controlnet):
                latents = batch["latents"]
                genre_ids = batch["genre_ids"]
                mood_ids = batch["mood_ids"]
                controls = batch["controls"]
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                down_block_res_samples, mid_block_res_sample = controlnet(noisy_latents, timesteps, genre_ids, mood_ids, controls)
                
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                if args.v_parameterization:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])
                loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                loss = loss.mean()
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
    
    accelerator.wait_for_everyone()
    accelerator.save_state(os.path.join(args.output_dir, "final"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--conditioning_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--unet_model_name_or_path", type=str, required=True)
    parser.add_argument("--num_genres", type=int, required=True)
    parser.add_argument("--num_moods", type=int, required=True)
    parser.add_argument("--num_controls", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--min_snr_gamma", type=float, default=5.0)
    parser.add_argument("--v_parameterization", action="store_true")
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()