#!/bin/bash

# Samadhan Training Script - Optimized for 16GB GPU
# RTX 5070 Ti / RTX 4080 / Similar consumer GPUs

set -e

echo "=========================================="
echo "Samadhan Training - 16GB GPU Edition"
echo "=========================================="
echo ""
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check available VRAM
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM" -lt 14000 ]; then
    echo "⚠️  Warning: Less than 14GB VRAM detected"
    echo "Training may fail. Consider using smaller batch sizes."
    echo ""
fi

echo "What would you like to train?"
echo ""
echo "Recommended for 16GB GPU:"
echo "1) Small LLM (7B) - ✅ FITS (12-14GB)"
echo "2) Image Generator (Base) - ✅ FITS (8-10GB)"
echo "3) Voice Cloning - ✅ FITS (4-6GB)"
echo "4) Deepfake (Base) - ✅ FITS (6-8GB)"
echo ""
echo "Not recommended for 16GB:"
echo "5) Medium LLM (13B) - ⚠️  TIGHT (16GB+)"
echo "6) Advanced Image (ControlNet/SR) - ⚠️  TIGHT (14-16GB)"
echo ""
echo "7) Exit"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "Training Small LLM (7B Parameters)"
        echo "=========================================="
        echo ""
        echo "Memory optimizations enabled:"
        echo "  - Mixed precision (FP16): ✅"
        echo "  - Gradient checkpointing: ✅"
        echo "  - Batch size: 1 (with accumulation)"
        echo "  - Max sequence length: 2048"
        echo ""

        read -p "Path to training data (JSON files): " DATA_PATH
        if [ ! -e "$DATA_PATH" ]; then
            echo "❌ Data path not found: $DATA_PATH"
            exit 1
        fi

        echo ""
        echo "Estimated requirements:"
        echo "  - VRAM: 12-14GB"
        echo "  - Storage: ~50GB"
        echo "  - Time: 2-4 weeks"
        echo ""
        read -p "Continue? (y/n): " confirm

        if [ "$confirm" = "y" ]; then
            echo ""
            echo "Starting training..."
            python backend/scripts/train_large_llm.py \
                --model_size small \
                --data_path "$DATA_PATH" \
                --batch_size 1 \
                --gradient_accumulation_steps 16 \
                --max_length 2048 \
                --epochs 3 \
                --learning_rate 2e-5 \
                --output_dir checkpoints/llm_small_16gb \
                --mixed_precision fp16 \
                --gradient_checkpointing
        fi
        ;;

    2)
        echo ""
        echo "=========================================="
        echo "Training Base Image Generator"
        echo "=========================================="
        echo ""

        read -p "Path to image dataset directory: " DATA_DIR
        if [ ! -d "$DATA_DIR" ]; then
            echo "❌ Directory not found: $DATA_DIR"
            exit 1
        fi

        echo ""
        echo "Estimated requirements:"
        echo "  - VRAM: 8-10GB"
        echo "  - Storage: ~200GB"
        echo "  - Time: 1-2 weeks"
        echo ""
        read -p "Continue? (y/n): " confirm

        if [ "$confirm" = "y" ]; then
            echo ""
            echo "Starting training..."
            python backend/scripts/train_advanced_image_model.py \
                --mode base \
                --data_dir "$DATA_DIR" \
                --batch_size 2 \
                --epochs 100 \
                --learning_rate 1e-4 \
                --output_dir checkpoints/image_base_16gb
        fi
        ;;

    3)
        echo ""
        echo "=========================================="
        echo "Training Voice Cloning Model"
        echo "=========================================="
        echo ""
        echo "Memory optimizations:"
        echo "  - Batch size: 8"
        echo "  - Audio length: 3 seconds"
        echo "  - Mixed precision: FP16"
        echo ""
        echo "Estimated requirements:"
        echo "  - VRAM: 4-6GB"
        echo "  - Storage: ~100GB"
        echo "  - Time: 3-5 days"
        echo ""
        echo "⚠️  Voice cloning training script coming soon"
        ;;

    4)
        echo ""
        echo "=========================================="
        echo "Training Deepfake Base Model"
        echo "=========================================="
        echo ""
        echo "Estimated requirements:"
        echo "  - VRAM: 6-8GB"
        echo "  - Storage: ~150GB"
        echo "  - Time: 3-5 days"
        echo ""
        echo "⚠️  Deepfake training script coming soon"
        ;;

    5)
        echo ""
        echo "=========================================="
        echo "Medium LLM (13B) - Advanced Users Only"
        echo "=========================================="
        echo ""
        echo "⚠️  WARNING: This will use ALL 16GB VRAM"
        echo ""
        echo "Required optimizations:"
        echo "  - INT8 quantization: ✅"
        echo "  - Aggressive gradient checkpointing: ✅"
        echo "  - Batch size: 1 (no accumulation)"
        echo "  - Max sequence length: 1024"
        echo ""
        echo "Expect:"
        echo "  - Very slow training"
        echo "  - Possible OOM errors"
        echo "  - 4-6 weeks training time"
        echo ""
        read -p "Continue anyway? (y/n): " confirm

        if [ "$confirm" = "y" ]; then
            read -p "Path to training data: " DATA_PATH

            echo ""
            echo "Starting training with extreme optimizations..."
            python backend/scripts/train_large_llm.py \
                --model_size medium \
                --data_path "$DATA_PATH" \
                --batch_size 1 \
                --gradient_accumulation_steps 8 \
                --max_length 1024 \
                --epochs 3 \
                --learning_rate 1e-5 \
                --output_dir checkpoints/llm_medium_16gb \
                --mixed_precision fp16 \
                --gradient_checkpointing \
                --quantize_training
        fi
        ;;

    6)
        echo ""
        echo "=========================================="
        echo "Advanced Image Models"
        echo "=========================================="
        echo ""
        echo "⚠️  These models require 14-16GB VRAM"
        echo ""
        echo "Select:"
        echo "1) ControlNet - 14GB"
        echo "2) Super-Resolution 8x - 16GB"
        echo ""
        read -p "Choice [1-2]: " adv_choice

        read -p "Path to dataset: " DATA_DIR

        case $adv_choice in
            1)
                echo "Training ControlNet..."
                python backend/scripts/train_advanced_image_model.py \
                    --mode controlnet \
                    --data_dir "$DATA_DIR" \
                    --batch_size 1 \
                    --epochs 50
                ;;
            2)
                echo "Training Super-Resolution 8x..."
                python backend/scripts/train_advanced_image_model.py \
                    --mode super_resolution \
                    --data_dir "$DATA_DIR" \
                    --batch_size 1 \
                    --epochs 100
                ;;
        esac
        ;;

    7)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to: checkpoints/"
echo ""
echo "Monitor VRAM usage:"
echo "  nvidia-smi -l 1"
echo ""
echo "View TensorBoard:"
echo "  tensorboard --logdir=logs/"
echo ""
