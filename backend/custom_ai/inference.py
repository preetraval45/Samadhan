import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
import time


class InferenceEngine:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.to(self.device)
        self.model.eval()


    def generate_text(self, prompt, max_length=200, temperature=0.8,
                     top_k=40, top_p=0.9, repetition_penalty=1.2):

        input_ids = self.tokenizer.encode(prompt)

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        generated_ids = self.model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        generated_text = self.tokenizer.decode(generated_ids[0].cpu().tolist())

        return generated_text


    def chat(self, message, conversation_history=None, max_history=5):

        if conversation_history is None:
            conversation_history = []

        conversation_history.append({'role': 'user', 'content': message})

        if len(conversation_history) > max_history * 2:
            conversation_history = conversation_history[-(max_history * 2):]

        prompt = self._format_conversation(conversation_history)

        response = self.generate_text(prompt, max_length=150, temperature=0.7)

        conversation_history.append({'role': 'assistant', 'content': response})

        return response, conversation_history


    def _format_conversation(self, history):
        formatted = ""
        for turn in history:
            role = turn['role']
            content = turn['content']
            formatted += f"{role}: {content}\n"
        formatted += "assistant: "
        return formatted


    def batch_generate(self, prompts, max_length=200, temperature=0.8):

        results = []

        for prompt in prompts:
            result = self.generate_text(prompt, max_length, temperature)
            results.append(result)

        return results


    def score_text(self, text):

        input_ids = self.tokenizer.encode(text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)

            log_probs = F.log_softmax(logits, dim=-1)

            target_ids = input_ids[1:]
            selected_log_probs = []

            for i, target_id in enumerate(target_ids):
                if target_id < log_probs.size(-1):
                    selected_log_probs.append(log_probs[0, i, target_id].item())

            if selected_log_probs:
                avg_log_prob = sum(selected_log_probs) / len(selected_log_probs)
                perplexity = np.exp(-avg_log_prob)
            else:
                perplexity = float('inf')

        return {
            'perplexity': perplexity,
            'avg_log_prob': avg_log_prob if selected_log_probs else None
        }


    def complete(self, prefix, num_completions=3, max_length=100):

        completions = []

        for _ in range(num_completions):
            completion = self.generate_text(
                prefix,
                max_length=max_length,
                temperature=0.9,
                top_k=50,
                top_p=0.95
            )
            completions.append(completion)

        return completions


    def extract_embeddings(self, text):

        input_ids = self.tokenizer.encode(text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            embeddings = self.model.embedding(input_tensor)

            mean_embedding = embeddings.mean(dim=1)

        return mean_embedding.cpu().numpy()



class ConversationManager:
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.conversations = {}


    def create_conversation(self, conversation_id):
        self.conversations[conversation_id] = []


    def add_message(self, conversation_id, role, content):
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        self.conversations[conversation_id].append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })


    def get_response(self, conversation_id, message):
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)

        self.add_message(conversation_id, 'user', message)

        history = self.conversations[conversation_id]

        response, updated_history = self.inference_engine.chat(message, history)

        self.conversations[conversation_id] = updated_history

        return response


    def get_history(self, conversation_id):
        return self.conversations.get(conversation_id, [])


    def clear_conversation(self, conversation_id):
        if conversation_id in self.conversations:
            self.conversations[conversation_id] = []



class ResponseQualityFilter:
    def __init__(self, min_length=10, max_repetition=0.3):
        self.min_length = min_length
        self.max_repetition = max_repetition


    def is_valid_response(self, text):
        if len(text) < self.min_length:
            return False

        words = text.split()
        if len(words) < 3:
            return False

        unique_words = len(set(words))
        total_words = len(words)

        if total_words > 0:
            repetition_ratio = 1 - (unique_words / total_words)
            if repetition_ratio > self.max_repetition:
                return False

        return True


    def filter_responses(self, responses):
        return [r for r in responses if self.is_valid_response(r)]
