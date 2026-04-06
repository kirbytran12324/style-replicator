from collections import OrderedDict
from typing import TYPE_CHECKING
import torch
from jobs.process import BaseSDTrainProcess
from toolkit.train_tools import apply_snr_weight

if TYPE_CHECKING:
    from jobs import TrainJob


class TrainFineTuneProcess(BaseSDTrainProcess):
    def __init__(self, process_id: int, job: 'TrainJob', config: OrderedDict):
        super().__init__(process_id, job, config)

    def hook_train_loop(self, batch_list):
        loss_dict = OrderedDict()
        total_loss = 0.0
        processed_batches = 0

        for batch in batch_list:
            # Skip None batches (can occur during gradient accumulation with multi-dataset loading)
            if batch is None:
                continue
            processed_batches += 1
                
            noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)

            # NOTE: Do NOT set requires_grad_(True) on noisy_latents or target.
            # Only LoRA weight parameters should have gradients. Setting requires_grad
            # on input tensors breaks the gradient flow through the LoRA layers.

            text_embeddings = self.sd.encode_prompt(conditioned_prompts)

            model_pred = self.sd.predict_noise(
                noisy_latents,
                text_embeddings=text_embeddings,
                timestep=timesteps,
            )

            target = noise

            if self.train_config.noise_scheduler == 'flowmatch':
                clean_latents = batch.latents.to(self.device_torch, dtype=model_pred.dtype)
                target = noise - clean_latents
            elif self.model_config.is_v_pred:
                if hasattr(self.sd.noise_scheduler, "get_velocity"):
                    clean_latents = batch.latents.to(self.device_torch, dtype=model_pred.dtype)
                    target = self.sd.noise_scheduler.get_velocity(clean_latents, noise, timesteps)

            # target is a reference tensor — it must NOT require gradients
            # Calculate MSE loss with per-sample reduction to apply SNR weighting
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.detach().float(), reduction="none")
            
            # Apply SNR-weighted loss if configured (prevents high-noise timesteps from dominating)
            if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0:
                if hasattr(self.sd.noise_scheduler, "alphas_cumprod"):
                    loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

            # Flowmatch schedulers do not expose alphas_cumprod; use scheduler-native timestep weights instead.
            if (
                self.train_config.noise_scheduler == 'flowmatch'
                and self.train_config.timestep_type == 'weighted'
                and hasattr(self.sd.noise_scheduler, 'get_weights_for_timesteps')
            ):
                timestep_weights = self.sd.noise_scheduler.get_weights_for_timesteps(
                    timesteps,
                    timestep_type=self.train_config.timestep_type,
                ).to(loss.device, dtype=loss.dtype)
                while timestep_weights.ndim < loss.ndim:
                    timestep_weights = timestep_weights.unsqueeze(-1)
                loss = loss * timestep_weights
            
            # Average across spatial dimensions (keep batch dimension for potential future batch weighting)
            loss = loss.mean()

            self.accelerator.backward(loss)
            total_loss += loss.item()

        # self.params is a list of optimizer param group dicts: [{"params": [...], "lr": ...}, ...]
        # Flatten to a single list of tensors that actually have gradients.
        parameters = []
        for group in self.params:
            if isinstance(group, dict):
                for p in group.get('params', []):
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        parameters.append(p)
            elif isinstance(group, torch.Tensor) and group.grad is not None:
                parameters.append(group)

        if self.train_config.max_grad_norm is not None and self.train_config.max_grad_norm > 0 and len(parameters) > 0:
            self.accelerator.clip_grad_norm_(parameters, self.train_config.max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        if processed_batches == 0:
            loss_dict['loss'] = 0.0
        else:
            loss_dict['loss'] = total_loss / processed_batches
        return loss_dict