# Samadhan Quick Start Guide

Get up and running with Samadhan in under 5 minutes.

## Prerequisites
- Docker & Docker Compose installed
- NVIDIA GPU with CUDA 12.1+ (optional but recommended)
- 16GB RAM minimum

## 1. Clone Repository
```bash
git clone <repository-url>
cd Samadhan
```

## 2. Start Services

### Option A: Interactive Menu (Recommended)
```bash
chmod +x START_API.sh
./START_API.sh
```
Select option 2 (Docker Compose - API & Services)

### Option B: Direct Command
```bash
docker-compose up -d
```

## 3. Verify Services
```bash
# Check all services are running
docker-compose ps

# Test API health
curl http://localhost:401/api/v1/health
```

## 4. Access the Platform

### API Documentation
Open in browser: http://localhost:401/api/docs

### Frontend Application
Open in browser: http://localhost:402

### MLflow (Model Registry)
Open in browser: http://localhost:407

### Neo4j (Knowledge Graph)
Open in browser: http://localhost:408
- Username: neo4j
- Password: samadhan123

## 5. Initialize Phase 1 Models

### Small Model (Recommended for testing)
```bash
curl -X POST http://localhost:401/api/v1/phase1/init_phase1_models \
  -H "Content-Type: application/json" \
  -d '{"model_sizes": ["small"]}'
```

### All Models (Requires significant GPU memory)
```bash
curl -X POST http://localhost:401/api/v1/phase1/init_phase1_models \
  -H "Content-Type: application/json" \
  -d '{"model_sizes": ["small", "medium", "large", "grok"]}'
```

## 6. Test the API

### Text Generation
```bash
curl -X POST http://localhost:401/api/v1/phase1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain artificial intelligence",
    "model_size": "small",
    "max_tokens": 256
  }'
```

### Image Generation
```bash
curl -X POST http://localhost:401/api/v1/phase1/image/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "mode": "generate",
    "height": 512,
    "width": 512
  }'
```

### Video Generation
```bash
curl -X POST http://localhost:401/api/v1/phase1/video/advanced \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ocean waves at sunset",
    "num_frames": 16,
    "resolution": "hd"
  }'
```

### Check Capabilities
```bash
curl http://localhost:401/api/v1/phase1/capabilities
```

## 7. View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

## 8. Stop Services
```bash
docker-compose down
```

## Common Issues

### Port Already in Use
Edit `docker-compose.yml` and change the port mappings.

### GPU Not Detected
Ensure NVIDIA Docker runtime is installed:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory
Use smaller model sizes or increase GPU/RAM allocation.

## Next Steps

1. **Train Custom Models**: Use `./START_TRAINING.sh`
2. **Read Full Documentation**: See [DEPLOYMENT.md](DEPLOYMENT.md)
3. **Explore API**: Visit http://localhost:401/api/docs
4. **View Roadmap**: See [ADVANCED_FEATURES_ROADMAP.md](ADVANCED_FEATURES_ROADMAP.md)

## Training Your Own Models

```bash
# Start training infrastructure
docker-compose --profile training up -d

# Enter training container
docker exec -it samadhan-training-master bash

# Start training
cd /training
./START_TRAINING.sh
```

## Service URLs

### API Services (Default)
| Service | URL |
|---------|-----|
| Backend API | http://localhost:401 |
| API Docs | http://localhost:401/api/docs |
| Frontend | http://localhost:402 |
| Nginx | http://localhost:400 |
| MLflow | http://localhost:407 |
| Neo4j | http://localhost:408 |

### Training Services (--profile training)
| Service | URL |
|---------|-----|
| TensorBoard | http://localhost:412 |
| Weights & Biases | http://localhost:413 |
| MLflow Training | http://localhost:414 |

## Support

- API Documentation: http://localhost:401/api/docs
- Health Check: http://localhost:401/api/v1/health
- Logs: `docker-compose logs -f`

---

**Ready to build decision intelligence! 🚀**
