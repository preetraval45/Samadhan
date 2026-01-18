import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from custom_ai import (
    CustomTokenizer,
    CustomTransformer,
    CustomTrainer,
    DataPreprocessor
)
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Train custom AI model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--vocab_size', type=int, default=30000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed forward dimension')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')

    args = parser.parse_args()

    print('Loading dataset...')
    preprocessor = DataPreprocessor()

    if args.dataset.endswith('.json'):
        texts = preprocessor.load_json_dataset(args.dataset)
    elif args.dataset.endswith('.txt'):
        with open(args.dataset, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = preprocessor.load_text_files([args.dataset])

    print(f'Loaded {len(texts)} texts')

    print('Training tokenizer...')
    tokenizer = CustomTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts)
    print(f'Tokenizer trained with vocab size: {len(tokenizer.word_to_id)}')

    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.pkl')
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f'Tokenizer saved to {tokenizer_path}')

    print('Splitting dataset...')
    train_texts, val_texts, test_texts = preprocessor.split_dataset(texts)
    print(f'Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}')

    print('Initializing model...')
    model = CustomTransformer(
        vocab_size=len(tokenizer.word_to_id),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {param_count:,}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    trainer = CustomTrainer(
        model,
        tokenizer,
        device=device,
        learning_rate=args.learning_rate
    )

    print('Starting training...')
    trainer.train(
        train_texts,
        val_texts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.output_dir
    )

    print('Training complete!')

    config = {
        'vocab_size': len(tokenizer.word_to_id),
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'training_samples': len(train_texts),
        'epochs_trained': args.epochs
    }

    config_path = os.path.join(args.output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'Model config saved to {config_path}')


if __name__ == '__main__':
    main()
