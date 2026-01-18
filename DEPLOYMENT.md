# Samadhan Deployment Guide

Complete deployment guide for the Samadhan Decision Intelligence Platform with Phase 1 Advanced AI capabilities.

## Table of Contents
- [Quick Start](#quick-start)
- [Deployment Modes](#deployment-modes)
- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [Training Infrastructure](#training-infrastructure)
- [Service Ports](#service-ports)

## Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (for containerized deployment)
- NVIDIA GPU + CUDA 12.1+ (recommended for production)
- kubectl (for Kubernetes deployment)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Samadhan
chmod +x START_API.sh START_TRAINING.sh
```

### 2. Choose Deployment Mode
```bash
./START_API.sh
```

## Deployment Modes

### Mode 1: Local Development
Direct Python execution without Docker.

```bash
./START_API.sh
# Select option 1

# Or manually:
cd backend
pip install -r requirements.txt
pip install -r requirements-custom-ai.txt
pip install -r requirements-multimodal.txt
python main.py
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/api/docs

### Mode 2: Docker Compose - API & Services
Full production stack with all services.

```bash
./START_API.sh
# Select option 2

# Or manually:
docker-compose up -d
```

**Services Started:**
- Backend API (FastAPI)
- Frontend (Next.js)
- PostgreSQL Database
- Redis Cache
- Qdrant Vector DB
- MLflow Model Registry
- Neo4j Knowledge Graph
- Nginx Reverse Proxy

**Access:**
- Backend API: http://localhost:401
- Frontend: http://localhost:402
- API Docs: http://localhost:401/api/docs
- Nginx: http://localhost:400
- MLflow: http://localhost:407
- Neo4j Browser: http://localhost:408

### Mode 3: Docker Compose - Training Mode
Training infrastructure with GPU support.

```bash
./START_API.sh
# Select option 3

# Or manually:
docker-compose --profile training up -d
```

**Training Services:**
- Training Master Node (8 GPU support)
- TensorBoard (visualization)
- Weights & Biases (experiment tracking)
- MLflow Training (model registry)

**Access:**
- TensorBoard: http://localhost:412
- Weights & Biases: http://localhost:413
- MLflow Training: http://localhost:414

**Enter Training Environment:**
```bash
docker exec -it samadhan-training-master bash
cd /training
./START_TRAINING.sh
```

### Mode 4: Kubernetes Production
Scalable production deployment with auto-scaling.

```bash
./START_API.sh
# Select option 4

# Or manually:
kubectl apply -f k8s/deployment.yaml
```

**Features:**
- Auto-scaling (2-10 replicas)
- GPU node selection
- Persistent volumes (checkpoints, training data, outputs)
- Health checks & liveness probes
- LoadBalancer service

**Manage Deployment:**
```bash
# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/samadhan-backend

# Scale manually
kubectl scale deployment/samadhan-backend --replicas=5
```

### Mode 5: API Only
Minimal API server without models or databases.

```bash
./START_API.sh
# Select option 5

# Or manually:
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Mode 6: Full System with Pre-trained Models
Complete system with trained model checkpoints.

```bash
./START_API.sh
# Select option 6
```

**Requirements:**
- Trained model checkpoints in `./checkpoints/`
- Use `./START_TRAINING.sh` to train models first

## Docker Compose

### Single Unified Configuration
All services are in one `docker-compose.yml` file with profiles for different use cases.

### Service Profiles

**Default Profile** (no profile needed):
- nginx
- backend
- frontend
- postgres
- redis
- qdrant
- mlflow
- neo4j

**Training Profile** (`--profile training`):
- training-master
- tensorboard
- wandb-local
- mlflow-training

### Commands

**Start API Services:**
```bash
docker-compose up -d
```

**Start Training Services:**
```bash
docker-compose --profile training up -d
```

**Start Everything:**
```bash
docker-compose --profile training up -d
```

**Stop Services:**
```bash
# Stop API services
docker-compose down

# Stop training services
docker-compose --profile training down

# Stop everything and remove volumes
docker-compose --profile training down -v
```

**View Logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f training-master
```

**Rebuild Images:**
```bash
# Rebuild all
docker-compose build

# Rebuild specific service
docker-compose build backend
docker-compose build training-master
```

## Training Infrastructure

### Start Training

**Using Starter Script:**
```bash
./START_TRAINING.sh
```

**Options:**
1. Train Large Language Model (LLM)
   - Small (7B params) - 2-3 weeks on 8xA100
   - Medium (13B params) - 3-4 weeks on 8xA100
   - Large (70B params) - 6-8 weeks on 64xA100
   - Grok (314B params) - 10-12 weeks on 128xA100

2. Train Advanced Image Models
   - Base Image Generator
   - ControlNet
   - Super-Resolution (8x & 16x)
   - All Image Models

3. Train Advanced Video Models
4. Train Deepfake Models
5. Train Voice Cloning
6. Train ALL Models (Full Pipeline)
7. Start Distributed Training (Multi-GPU)

### Manual Training Commands

**LLM Training:**
```bash
python backend/scripts/train_large_llm.py \
  --model_size grok \
  --data_path training_data/*.json \
  --epochs 3 \
  --batch_size 4 \
  --distributed \
  --use_rlhf
```

**Image Model Training:**
```bash
python backend/scripts/train_advanced_image_model.py \
  --mode all \
  --data_dir training_data/images \
  --epochs 100 \
  --output_dir checkpoints/image_models
```

**Distributed Training (Multi-GPU):**
```bash
torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  backend/scripts/train_large_llm.py \
  --model_size large \
  --data_path training_data/*.json \
  --distributed
```

### Training in Docker

```bash
# Start training infrastructure
docker-compose --profile training up -d

# Enter training container
docker exec -it samadhan-training-master bash

# Inside container
cd /training
./START_TRAINING.sh
```

### Monitor Training

**TensorBoard:**
```bash
# Local
tensorboard --logdir=./logs

# Docker
open http://localhost:412
```

**Weights & Biases:**
```bash
open http://localhost:413
```

**MLflow:**
```bash
# Production
open http://localhost:407

# Training
open http://localhost:414
```

## Service Ports

### API Services
| Service | Port | URL |
|---------|------|-----|
| Nginx | 400 | http://localhost:400 |
| Backend API | 401 | http://localhost:401 |
| Frontend | 402 | http://localhost:402 |
| PostgreSQL | 403 | localhost:403 |
| Redis | 404 | localhost:404 |
| Qdrant | 405 | http://localhost:405 |
| Qdrant gRPC | 406 | localhost:406 |
| MLflow | 407 | http://localhost:407 |
| Neo4j Browser | 408 | http://localhost:408 |
| Neo4j Bolt | 409 | bolt://localhost:409 |

### Training Services
| Service | Port | URL |
|---------|------|-----|
| TensorBoard (Master) | 410 | http://localhost:410 |
| Distributed Training | 411 | localhost:411 |
| TensorBoard | 412 | http://localhost:412 |
| Weights & Biases | 413 | http://localhost:413 |
| MLflow Training | 414 | http://localhost:414 |

## API Endpoints

### Phase 1 Advanced Endpoints

**Initialize Models:**
```bash
curl -X POST http://localhost:401/api/v1/phase1/init_phase1_models \
  -H "Content-Type: application/json" \
  -d '{"model_sizes": ["small"]}'
```

**LLM Generation:**
```bash
curl -X POST http://localhost:401/api/v1/phase1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "model_size": "small",
    "max_tokens": 512,
    "use_constitutional_ai": true
  }'
```

**Advanced Image Generation:**
```bash
curl -X POST http://localhost:401/api/v1/phase1/image/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "mode": "generate",
    "height": 512,
    "width": 512
  }'
```

**Advanced Video Generation:**
```bash
curl -X POST http://localhost:401/api/v1/phase1/video/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A drone flying over a city",
    "num_frames": 16,
    "resolution": "4k",
    "camera_motion": "pan_right"
  }'
```

**Voice Cloning:**
```bash
curl -X POST http://localhost:401/api/v1/phase1/voice/clone \
  -F "reference_audio=@reference.wav" \
  -F "text=Hello, this is a cloned voice" \
  -F "emotion=neutral"
```

**Get Capabilities:**
```bash
curl http://localhost:401/api/v1/phase1/capabilities
```

## Troubleshooting

### Docker Issues

**Permission Denied:**
```bash
sudo chown -R $USER:$USER .
chmod +x START_API.sh START_TRAINING.sh
```

**GPU Not Detected:**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Update docker-compose.yml if needed
```

**Port Already in Use:**
```bash
# Check what's using the port
lsof -i :401

# Kill the process or change port in docker-compose.yml
```

### Training Issues

**Out of Memory:**
- Reduce batch size
- Use gradient accumulation
- Enable model quantization
- Use smaller model size

**CUDA Out of Memory:**
```bash
# Reduce CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1  # Use only 2 GPUs instead of 8
```

**Slow Training:**
- Enable mixed precision (FP16)
- Use distributed training
- Increase batch size if GPU memory allows

### API Issues

**Models Not Loading:**
```bash
# Check if checkpoints exist
ls -la checkpoints/

# Verify GPU availability
nvidia-smi

# Check backend logs
docker-compose logs -f backend
```

**Health Check Failing:**
```bash
# Check service status
docker-compose ps

# Restart services
docker-compose restart backend

# Check logs
docker-compose logs backend
```

## Production Recommendations

1. **GPU Requirements:**
   - Minimum: 1x NVIDIA A100 (40GB)
   - Recommended: 8x NVIDIA A100 (80GB)
   - For Grok model: 128x NVIDIA A100

2. **Storage:**
   - Checkpoints: 500GB - 5TB
   - Training Data: 1TB - 10TB
   - Outputs: 500GB

3. **Memory:**
   - API Server: 32GB RAM minimum
   - Training: 256GB - 512GB RAM

4. **Network:**
   - InfiniBand for distributed training
   - 10Gb Ethernet minimum

5. **Security:**
   - Use environment variables for secrets
   - Enable SSL/TLS for production
   - Implement authentication & authorization
   - Regular security updates

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify GPU: `nvidia-smi`
3. Check service health: `curl http://localhost:401/api/v1/health`
4. Review documentation: `/api/docs`
