import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import copy


class RewardModel(nn.Module):
    """reward model for RLHF training"""

    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model

        self.value_head = nn.Sequential(
            nn.Linear(base_model.d_model, base_model.d_model),
            nn.Tanh(),
            nn.Linear(base_model.d_model, 1)
        )

    def forward(self, input_ids, mask=None):
        x = self.base_model.token_embedding(input_ids)

        for layer in self.base_model.layers:
            x = layer(x, mask)

        x = self.base_model.norm(x)

        last_hidden = x[:, -1, :]

        reward = self.value_head(last_hidden)

        return reward


class ValueModel(nn.Module):
    """value model for PPO"""

    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model

        self.value_head = nn.Sequential(
            nn.Linear(base_model.d_model, base_model.d_model),
            nn.ReLU(),
            nn.Linear(base_model.d_model, 1)
        )

    def forward(self, input_ids, mask=None):
        x = self.base_model.token_embedding(input_ids)

        for layer in self.base_model.layers:
            x = layer(x, mask)

        x = self.base_model.norm(x)

        values = self.value_head(x)

        return values.squeeze(-1)


class PreferenceDataset(Dataset):
    """dataset for preference learning"""

    def __init__(self, prompts, chosen_responses, rejected_responses, tokenizer):
        self.prompts = prompts
        self.chosen = chosen_responses
        self.rejected = rejected_responses
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen = self.chosen[idx]
        rejected = self.rejected[idx]

        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen)
        rejected_ids = self.tokenizer.encode(rejected)

        return {
            'prompt_ids': torch.tensor(prompt_ids),
            'chosen_ids': torch.tensor(chosen_ids),
            'rejected_ids': torch.tensor(rejected_ids)
        }


class RLHFTrainer:
    """reinforcement learning from human feedback trainer"""

    def __init__(self, model, reward_model, value_model, tokenizer, device='cuda'):
        self.model = model
        self.reward_model = reward_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(device)
        self.reward_model.to(device)
        self.value_model.to(device)

        self.reference_model = copy.deepcopy(model)
        self.reference_model.eval()

        self.kl_coef = 0.1
        self.clip_range = 0.2
        self.gamma = 0.99
        self.lam = 0.95

    def train_reward_model(self, dataloader, epochs=3, lr=1e-5):
        """trains reward model on preference data"""

        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=lr)

        self.reward_model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch in dataloader:
                prompt_ids = batch['prompt_ids'].to(self.device)
                chosen_ids = batch['chosen_ids'].to(self.device)
                rejected_ids = batch['rejected_ids'].to(self.device)

                chosen_input = torch.cat([prompt_ids, chosen_ids], dim=1)
                rejected_input = torch.cat([prompt_ids, rejected_ids], dim=1)

                chosen_reward = self.reward_model(chosen_input)
                rejected_reward = self.reward_model(rejected_input)

                loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                predictions = (chosen_reward > rejected_reward).float()
                correct += predictions.sum().item()
                total += len(predictions)

            accuracy = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.4f}")

    def compute_advantages(self, rewards, values, masks):
        """computes GAE advantages"""

        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]

            advantages[t] = delta + self.gamma * self.lam * masks[t] * last_advantage

            last_advantage = advantages[t]

        returns = advantages + values

        return advantages, returns

    def ppo_step(self, prompts, max_length=512):
        """performs one PPO training step"""

        self.model.train()
        self.value_model.train()

        experiences = []

        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([prompt_ids], device=self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    prompt_tensor,
                    max_new_tokens=max_length,
                    do_sample=True
                )

                response = generated[:, len(prompt_ids):]

                reward = self.reward_model(generated).item()

                with torch.no_grad():
                    ref_logits = self.reference_model(generated)

                logits = self.model(generated)

                log_probs = F.log_softmax(logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                selected_log_probs = log_probs.gather(2, generated.unsqueeze(-1)).squeeze(-1)
                selected_ref_log_probs = ref_log_probs.gather(2, generated.unsqueeze(-1)).squeeze(-1)

                kl_div = (selected_log_probs - selected_ref_log_probs).sum(-1)

                values = self.value_model(generated)

            experiences.append({
                'input_ids': generated,
                'reward': reward,
                'kl_div': kl_div,
                'values': values,
                'log_probs': selected_log_probs
            })

        return experiences

    def train_ppo(self, prompts, epochs=10, batch_size=4):
        """trains model with PPO"""

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)
        value_optimizer = torch.optim.AdamW(self.value_model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            experiences = self.ppo_step(prompts[:batch_size])

            for exp in experiences:
                input_ids = exp['input_ids']
                old_log_probs = exp['log_probs'].detach()
                rewards = torch.tensor([exp['reward'] - self.kl_coef * exp['kl_div'].item()], device=self.device)
                old_values = exp['values'].detach()

                values = self.value_model(input_ids)

                advantages, returns = self.compute_advantages(
                    rewards,
                    old_values,
                    torch.ones_like(rewards)
                )

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                logits = self.model(input_ids)
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns)

                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
                value_optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")


class ConstitutionalAI:
    """implements constitutional AI principles"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.principles = [
            "Be helpful, harmless, and honest",
            "Refuse harmful requests politely",
            "Provide accurate information",
            "Acknowledge uncertainty when appropriate",
            "Respect privacy and confidentiality",
            "Avoid bias and discrimination",
            "Be transparent about limitations"
        ]

    def critique_response(self, prompt, response):
        """critiques response against principles"""

        critique_prompt = f"""
Evaluate the following response against these principles:
{chr(10).join(f"{i+1}. {p}" for i, p in enumerate(self.principles))}

Original prompt: {prompt}
Response: {response}

Critique:"""

        critique_ids = self.tokenizer.encode(critique_prompt)
        critique_tensor = torch.tensor([critique_ids], device=self.model.device if hasattr(self.model, 'device') else 'cpu')

        with torch.no_grad():
            critique = self.model.generate(critique_tensor, max_new_tokens=200)

        critique_text = self.tokenizer.decode(critique[0].tolist())

        return critique_text

    def revise_response(self, prompt, response, critique):
        """revises response based on critique"""

        revision_prompt = f"""
Given this critique, revise the response to better align with the principles:

Original prompt: {prompt}
Original response: {response}
Critique: {critique}

Revised response:"""

        revision_ids = self.tokenizer.encode(revision_prompt)
        revision_tensor = torch.tensor([revision_ids], device=self.model.device if hasattr(self.model, 'device') else 'cpu')

        with torch.no_grad():
            revised = self.model.generate(revision_tensor, max_new_tokens=256)

        revised_text = self.tokenizer.decode(revised[0].tolist())

        return revised_text


class ModelQuantizer:
    """quantizes models to INT8/INT4 for efficient inference"""

    @staticmethod
    def quantize_int8(model):
        """quantizes model to INT8"""

        quantized_model = copy.deepcopy(model)

        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data

                scale = weight.abs().max() / 127.0
                quantized_weight = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)

                module.weight.data = quantized_weight.to(torch.float32) * scale

        return quantized_model

    @staticmethod
    def quantize_int4(model):
        """quantizes model to INT4"""

        quantized_model = copy.deepcopy(model)

        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data

                scale = weight.abs().max() / 7.0
                quantized_weight = torch.round(weight / scale).clamp(-8, 7)

                module.weight.data = quantized_weight * scale

        return quantized_model
