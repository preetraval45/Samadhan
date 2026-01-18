import json
import os
import sys


def create_sample_dataset():
    """creates a sample training dataset for getting started"""

    sample_conversations = [
        "Hello! How can I help you today?",
        "I'm doing great, thanks for asking!",
        "The weather is beautiful today, perfect for a walk outside.",
        "Python is a versatile programming language used for web development, data science, and more.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are inspired by the structure of the human brain.",
        "Deep learning has revolutionized computer vision and natural language processing.",
        "Transformers are the architecture behind modern language models.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Training large models requires significant computational resources.",
        "Fine-tuning allows adapting pre-trained models to specific tasks.",
        "What programming language would you like to learn?",
        "I can help you with coding, writing, analysis, and much more.",
        "Let me explain that concept in simpler terms.",
        "That's an interesting question! Let me think about it.",
        "Mathematics is the foundation of computer science and AI.",
        "Algorithms are step-by-step procedures for solving problems.",
        "Data structures organize information efficiently in memory.",
        "Recursion is when a function calls itself to solve a problem.",
        "The internet connects billions of devices worldwide.",
        "Cybersecurity protects systems and data from threats.",
        "Cloud computing provides on-demand access to computing resources.",
        "APIs allow different software systems to communicate.",
        "Version control helps manage changes to code over time.",
        "Testing ensures software works correctly before deployment.",
        "What would you like to create today?",
        "I'm here to assist with your questions and tasks.",
        "Learning to code opens up many career opportunities.",
        "Practice is essential for mastering any skill.",
        "Don't be afraid to make mistakes - that's how we learn!",
        "Breaking down complex problems makes them easier to solve.",
        "Documentation helps others understand your code.",
        "Clean code is easier to maintain and debug.",
        "Collaboration makes projects better through diverse perspectives.",
        "The scientific method guides systematic problem solving.",
        "Critical thinking helps evaluate information and arguments.",
        "Creativity drives innovation and new solutions.",
        "Persistence is key to overcoming challenges.",
        "Continuous learning keeps skills relevant and sharp.",
        "Communication skills are crucial in any field.",
    ]

    dataset = []
    for text in sample_conversations:
        dataset.append({
            "text": text,
            "source": "sample"
        })

    return dataset


def save_dataset(dataset, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f'Dataset saved to {output_path}')
    print(f'Total samples: {len(dataset)}')


def main():
    dataset = create_sample_dataset()

    output_dir = 'training_data'
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'sample_dataset.json')
    save_dataset(dataset, output_path)

    print('\nTo train the model, run:')
    print(f'python scripts/train_custom_model.py --dataset {output_path} --epochs 20')


if __name__ == '__main__':
    main()
