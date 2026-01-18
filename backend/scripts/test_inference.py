import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from custom_ai import CustomTokenizer, CustomTransformer, InferenceEngine
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Test model inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='Hello, how are you?', help='Input prompt')
    parser.add_argument('--max_length', type=int, default=100, help='Max generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    print('Loading tokenizer...')
    tokenizer = CustomTokenizer()
    tokenizer.load(args.tokenizer)
    print(f'Tokenizer loaded with vocab size: {len(tokenizer.word_to_id)}')

    print('Loading model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('model_config', {})

    model = CustomTransformer(
        vocab_size=config.get('vocab_size', len(tokenizer.word_to_id)),
        d_model=config.get('d_model', 512)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully')

    inference_engine = InferenceEngine(model, tokenizer, device=device)

    if args.interactive:
        print('\nInteractive mode - type "quit" to exit')
        conversation_history = []

        while True:
            user_input = input('\nYou: ').strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            response, conversation_history = inference_engine.chat(
                user_input,
                conversation_history=conversation_history
            )

            print(f'Assistant: {response}')

    else:
        print(f'\nPrompt: {args.prompt}')
        print('Generating...')

        output = inference_engine.generate_text(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )

        print(f'\nGenerated text:\n{output}')


if __name__ == '__main__':
    main()
