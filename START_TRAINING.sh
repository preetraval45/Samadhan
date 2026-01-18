#!/bin/bash

# Samadhan Phase 1 Training Starter
# This script starts training for all Phase 1 models

set -e

echo "=========================================="
echo "Samadhan Phase 1 Training System"
echo "=========================================="
echo ""

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  CUDA not detected, training will be slow on CPU"
fi

echo ""
echo "Select training mode:"
echo "1) Train Large Language Model (LLM)"
echo "2) Train Advanced Image Models"
echo "3) Train Advanced Video Models"
echo "4) Train Deepfake Models"
echo "5) Train Voice Cloning"
echo "6) Train ALL Models (Full Pipeline)"
echo "7) Start Distributed Training (Multi-GPU)"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "Training Large Language Model"
        echo "=========================================="
        echo ""
        echo "Select model size:"
        echo "1) Small (7B parameters) - 2-3 weeks on 8xA100"
        echo "2) Medium (13B parameters) - 3-4 weeks on 8xA100"
        echo "3) Large (70B parameters) - 6-8 weeks on 64xA100"
        echo "4) Grok (314B parameters) - 10-12 weeks on 128xA100"
        read -p "Enter choice [1-4]: " model_choice

        case $model_choice in
            1) MODEL_SIZE="small" ;;
            2) MODEL_SIZE="medium" ;;
            3) MODEL_SIZE="large" ;;
            4) MODEL_SIZE="grok" ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        read -p "Path to training data (JSON files): " DATA_PATH
        read -p "Number of epochs [default: 3]: " EPOCHS
        EPOCHS=${EPOCHS:-3}
        read -p "Batch size per GPU [default: 4]: " BATCH_SIZE
        BATCH_SIZE=${BATCH_SIZE:-4}
        read -p "Use RLHF after base training? (y/n) [default: y]: " USE_RLHF
        USE_RLHF=${USE_RLHF:-y}

        RLHF_FLAG=""
        if [ "$USE_RLHF" = "y" ]; then
            RLHF_FLAG="--use_rlhf"
        fi

        echo ""
        echo "Starting LLM training..."
        python backend/scripts/train_large_llm.py \
            --model_size $MODEL_SIZE \
            --data_path $DATA_PATH \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --output_dir checkpoints/llm_${MODEL_SIZE} \
            $RLHF_FLAG
        ;;

    2)
        echo ""
        echo "=========================================="
        echo "Training Advanced Image Models"
        echo "=========================================="
        echo ""
        echo "Select training mode:"
        echo "1) Base Image Generator"
        echo "2) ControlNet"
        echo "3) Super-Resolution (8x & 16x)"
        echo "4) All Image Models"
        read -p "Enter choice [1-4]: " img_choice

        case $img_choice in
            1) MODE="base" ;;
            2) MODE="controlnet" ;;
            3) MODE="super_resolution" ;;
            4) MODE="all" ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac

        read -p "Path to image dataset directory: " DATA_DIR
        read -p "Number of epochs [default: 100]: " EPOCHS
        EPOCHS=${EPOCHS:-100}

        echo ""
        echo "Starting image model training..."
        python backend/scripts/train_advanced_image_model.py \
            --mode $MODE \
            --data_dir $DATA_DIR \
            --epochs $EPOCHS \
            --output_dir checkpoints/image_models
        ;;

    3)
        echo ""
        echo "=========================================="
        echo "Training Advanced Video Models"
        echo "=========================================="
        echo ""
        echo "Video model training coming soon..."
        echo "Models are code-ready, training pipeline in development"
        ;;

    4)
        echo ""
        echo "=========================================="
        echo "Training Deepfake Models"
        echo "=========================================="
        echo ""
        echo "Deepfake training requires ethical approval"
        echo "Models are code-ready for authorized use"
        ;;

    5)
        echo ""
        echo "=========================================="
        echo "Training Voice Cloning"
        echo "=========================================="
        echo ""
        echo "Voice cloning training coming soon..."
        echo "Models are code-ready, training pipeline in development"
        ;;

    6)
        echo ""
        echo "=========================================="
        echo "Training ALL Models (Full Pipeline)"
        echo "=========================================="
        echo ""
        echo "This will train all Phase 1 models sequentially"
        echo "Estimated time: 6-12 months on 64xA100 cluster"
        echo ""
        read -p "Are you sure? This requires significant compute resources (y/n): " confirm

        if [ "$confirm" = "y" ]; then
            echo "Starting full training pipeline..."
            # Train LLM first
            python backend/scripts/train_large_llm.py \
                --model_size small \
                --data_path training_data/*.json \
                --epochs 3 \
                --distributed

            # Train image models
            python backend/scripts/train_advanced_image_model.py \
                --mode all \
                --data_dir training_data/images \
                --epochs 100

            echo "✅ Full training pipeline complete!"
        fi
        ;;

    7)
        echo ""
        echo "=========================================="
        echo "Distributed Training (Multi-GPU)"
        echo "=========================================="
        echo ""

        read -p "Number of GPUs: " NUM_GPUS
        read -p "Model to train (llm/image/video): " MODEL_TYPE

        if [ "$MODEL_TYPE" = "llm" ]; then
            read -p "Model size (small/medium/large/grok): " MODEL_SIZE
            read -p "Data path: " DATA_PATH

            echo ""
            echo "Starting distributed LLM training on $NUM_GPUS GPUs..."

            torchrun \
                --nproc_per_node=$NUM_GPUS \
                --master_port=29500 \
                backend/scripts/train_large_llm.py \
                --model_size $MODEL_SIZE \
                --data_path $DATA_PATH \
                --distributed \
                --epochs 3 \
                --batch_size 4
        fi
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
echo "View training logs: logs/"
echo ""
echo "Next steps:"
echo "1. Evaluate model performance"
echo "2. Load trained model into API"
echo "3. Test inference endpoints"
echo ""
