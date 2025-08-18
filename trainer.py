"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time


class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda", 
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()

    def train_iteration(
        self,
        loss_fn,
        dataloader,
    ):

        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        total_batches = len(dataloader)
        print(f"\n**** Starting training iteration with {total_batches} batches")
        
        for i, trajs in enumerate(dataloader):
            loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)

            # Print progress for first 10 batches to see if anything is happening
            if i < 10:
                print(f"**** Trainer: Batch {i+1}/{total_batches}: loss={loss:.4f}, nll={nll:.4f}, entropy={entropy:.4f}")
            
            # Print progress every 1% of batches for debugging
            if (i + 1) % max(1, total_batches // 100) == 0:
                print(f"**** Trainer: Progress: {i+1}/{total_batches} batches ({(i+1)/total_batches*100:.1f}%)")
                print(f"**** Trainer: Current loss: {loss:.4f}, nll: {nll:.4f}, entropy: {entropy:.4f}")
            
            # # zahra EWA #### 
            # if i >=100: 
            #     break

        print(f"Completed training iteration in {time.time() - train_start:.2f} seconds")
        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )
