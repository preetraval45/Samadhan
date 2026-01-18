import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ai import UnifiedAI
import argparse


def demo_text_generation(ai):
    print("\n" + "="*60)
    print("TEXT GENERATION DEMO")
    print("="*60)

    prompts = [
        "What is artificial intelligence?",
        "Tell me a short story about a robot",
        "Explain quantum computing in simple terms"
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = ai.generate_text(prompt, max_length=100, temperature=0.8)
        print(f"Response: {response}")


def demo_image_generation(ai):
    print("\n" + "="*60)
    print("IMAGE GENERATION DEMO")
    print("="*60)

    prompts = [
        "beautiful sunset over ocean",
        "mountain landscape with lake",
        "futuristic city at night"
    ]

    os.makedirs("demo_outputs", exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating image {i+1}: {prompt}")
        image = ai.generate_image(prompt, height=256, width=256)

        output_path = f"demo_outputs/image_{i+1}.png"
        ai.save_image(image, output_path)
        print(f"Saved to: {output_path}")


def demo_video_generation(ai):
    print("\n" + "="*60)
    print("VIDEO GENERATION DEMO")
    print("="*60)

    prompt = "waves crashing on beach"
    print(f"\nGenerating video: {prompt}")

    os.makedirs("demo_outputs", exist_ok=True)

    video = ai.generate_video(prompt, num_frames=8, fps=8)

    output_path = "demo_outputs/video_demo.mp4"
    ai.save_video(video, output_path)
    print(f"Saved to: {output_path}")

    gif_path = "demo_outputs/video_demo.gif"
    ai.save_gif(video, gif_path)
    print(f"GIF saved to: {gif_path}")


def demo_audio_generation(ai):
    print("\n" + "="*60)
    print("AUDIO GENERATION DEMO")
    print("="*60)

    os.makedirs("demo_outputs", exist_ok=True)

    print("\nGenerating speech...")
    speech = ai.generate_speech("Hello, this is a test of speech synthesis")
    ai.save_audio(speech, "demo_outputs/speech.wav")
    print("Speech saved to: demo_outputs/speech.wav")

    print("\nGenerating music...")
    music = ai.generate_music("calming piano melody", duration=10.0)
    ai.save_audio(music, "demo_outputs/music.wav")
    print("Music saved to: demo_outputs/music.wav")


def demo_chat(ai):
    print("\n" + "="*60)
    print("CHAT DEMO")
    print("="*60)

    conversation_history = []

    messages = [
        "Hello! How are you?",
        "What can you help me with?",
        "Tell me something interesting"
    ]

    for message in messages:
        print(f"\nUser: {message}")
        response, conversation_history = ai.chat(message, conversation_history)
        print(f"AI: {response}")


def demo_auto_detection(ai):
    print("\n" + "="*60)
    print("AUTOMATIC TASK DETECTION DEMO")
    print("="*60)

    tasks = [
        ("generate an image of a cat", "image"),
        ("create a video of dancing", "video"),
        ("make some music with drums", "music"),
        ("what is machine learning?", "text")
    ]

    for prompt, expected_type in tasks:
        detected_type = ai._detect_task_type(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Detected: {detected_type} (expected: {expected_type})")
        print(f"Match: {'✓' if detected_type == expected_type else '✗'}")


def main():
    parser = argparse.ArgumentParser(description='Demo multimodal AI system')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'text', 'image', 'video', 'audio', 'chat', 'auto'],
                       help='Which demo to run')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    print("="*60)
    print("CUSTOM MULTIMODAL AI SYSTEM DEMO")
    print("="*60)
    print(f"\nInitializing on device: {args.device}")
    print("This may take a moment...")

    ai = UnifiedAI(vocab_size=10000, device=args.device)

    print(f"\nDevice: {ai.device}")
    print(f"Capabilities: {len(ai.capabilities)}")

    if args.demo == 'all' or args.demo == 'text':
        demo_text_generation(ai)

    if args.demo == 'all' or args.demo == 'chat':
        demo_chat(ai)

    if args.demo == 'all' or args.demo == 'auto':
        demo_auto_detection(ai)

    if args.demo == 'all' or args.demo == 'image':
        demo_image_generation(ai)

    if args.demo == 'all' or args.demo == 'video':
        demo_video_generation(ai)

    if args.demo == 'all' or args.demo == 'audio':
        demo_audio_generation(ai)

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nCheck the 'demo_outputs' directory for generated files")
    print("\nCapabilities:")
    for capability in ai.capabilities:
        print(f"  - {capability}")

    print("\nAll systems operational!")


if __name__ == '__main__':
    main()
