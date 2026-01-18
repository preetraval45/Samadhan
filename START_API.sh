#!/bin/bash

# Samadhan API Starter
# Starts the complete Samadhan platform with all Phase 1 models

set -e

echo "=========================================="
echo "Samadhan Decision Intelligence Platform"
echo "Phase 1 Advanced AI System"
echo "=========================================="
echo ""

# Check dependencies
echo "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker not found. Docker deployment unavailable."
    DOCKER_AVAILABLE=false
else
    echo "✅ Docker available"
    DOCKER_AVAILABLE=true
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_AVAILABLE=true
else
    echo "⚠️  CUDA not detected. GPU acceleration unavailable."
    GPU_AVAILABLE=false
fi

echo ""
echo "Select deployment mode:"
echo "1) Local Development (Direct Python)"
echo "2) Docker Compose - API & Services"
echo "3) Docker Compose - Training Mode"
echo "4) Kubernetes Production"
echo "5) API Only (No Models)"
echo "6) Full System with Pre-trained Models"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "Starting Local Development Server"
        echo "=========================================="
        echo ""

        # Install dependencies
        echo "Installing dependencies..."
        cd backend
        pip install -r requirements.txt
        pip install -r requirements-custom-ai.txt
        pip install -r requirements-multimodal.txt

        # Create necessary directories
        mkdir -p checkpoints training_data generated_outputs uploaded_faces

        echo ""
        echo "Starting FastAPI server..."
        echo "API will be available at: http://localhost:8000"
        echo "API docs: http://localhost:8000/api/docs"
        echo ""
        echo "Press Ctrl+C to stop"
        echo ""

        python main.py
        ;;

    2)
        if [ "$DOCKER_AVAILABLE" = false ]; then
            echo "❌ Docker not available. Please install Docker first."
            exit 1
        fi

        echo ""
        echo "=========================================="
        echo "Starting Docker Compose - API & Services"
        echo "=========================================="
        echo ""

        # Build and start services
        echo "Building Docker images..."
        docker-compose build

        echo ""
        echo "Starting services..."
        docker-compose up -d

        echo ""
        echo "✅ All services started!"
        echo ""
        echo "Services:"
        echo "  - Backend API: http://localhost:401"
        echo "  - Frontend: http://localhost:402"
        echo "  - API Docs: http://localhost:401/api/docs"
        echo "  - Nginx: http://localhost:400"
        echo "  - PostgreSQL: localhost:403"
        echo "  - Redis: localhost:404"
        echo "  - Qdrant: localhost:405"
        echo "  - MLflow: http://localhost:407"
        echo "  - Neo4j: http://localhost:408"
        echo ""
        echo "View logs: docker-compose logs -f"
        echo "Stop: docker-compose down"
        echo ""

        # Wait for services to be ready
        echo "Waiting for services to be ready..."
        sleep 10

        # Check service health
        echo ""
        echo "Checking service health..."
        curl -s http://localhost:401/api/v1/health || echo "⚠️  Backend not ready yet"

        echo ""
        echo "To initialize Phase 1 models:"
        echo "curl -X POST http://localhost:401/api/v1/phase1/init_phase1_models"
        ;;

    3)
        if [ "$DOCKER_AVAILABLE" = false ]; then
            echo "❌ Docker not available. Please install Docker first."
            exit 1
        fi

        echo ""
        echo "=========================================="
        echo "Starting Docker Compose - Training Mode"
        echo "=========================================="
        echo ""

        # Build and start training services
        echo "Building Docker images..."
        docker-compose build training-master

        echo ""
        echo "Starting training infrastructure..."
        docker-compose --profile training up -d

        echo ""
        echo "✅ Training infrastructure started!"
        echo ""
        echo "Training Services:"
        echo "  - Training Master: samadhan-training-master"
        echo "  - TensorBoard: http://localhost:412"
        echo "  - Weights & Biases: http://localhost:413"
        echo "  - MLflow Training: http://localhost:414"
        echo ""
        echo "To enter training environment:"
        echo "  docker exec -it samadhan-training-master bash"
        echo "  cd /training"
        echo "  ./START_TRAINING.sh"
        echo ""
        echo "View logs:"
        echo "  docker-compose logs -f training-master"
        echo ""
        echo "Stop training:"
        echo "  docker-compose --profile training down"
        ;;

    4)
        echo ""
        echo "=========================================="
        echo "Kubernetes Production Deployment"
        echo "=========================================="
        echo ""

        if ! command -v kubectl &> /dev/null; then
            echo "❌ kubectl not found. Please install kubectl first."
            exit 1
        fi

        # Check cluster connection
        if ! kubectl cluster-info &> /dev/null; then
            echo "❌ Cannot connect to Kubernetes cluster"
            exit 1
        fi

        echo "Connected to Kubernetes cluster:"
        kubectl cluster-info

        echo ""
        read -p "Deploy to production? (y/n): " confirm

        if [ "$confirm" = "y" ]; then
            echo ""
            echo "Deploying to Kubernetes..."

            # Apply deployments
            kubectl apply -f k8s/deployment.yaml

            echo ""
            echo "✅ Deployment complete!"
            echo ""
            echo "Check status:"
            echo "  kubectl get pods"
            echo "  kubectl get services"
            echo ""
            echo "View logs:"
            echo "  kubectl logs -f deployment/samadhan-backend"
            echo ""
            echo "Scale deployment:"
            echo "  kubectl scale deployment/samadhan-backend --replicas=5"
        fi
        ;;

    5)
        echo ""
        echo "=========================================="
        echo "Starting API Only (No Models)"
        echo "=========================================="
        echo ""

        cd backend
        echo "Starting minimal API server..."
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        ;;

    6)
        echo ""
        echo "=========================================="
        echo "Full System with Pre-trained Models"
        echo "=========================================="
        echo ""

        # Check for checkpoints
        if [ ! -d "checkpoints" ] || [ -z "$(ls -A checkpoints)" ]; then
            echo "⚠️  No checkpoints found!"
            echo ""
            echo "You need to either:"
            echo "1. Train models using: ./START_TRAINING.sh"
            echo "2. Download pre-trained checkpoints"
            echo "3. Place checkpoint files in ./checkpoints/"
            echo ""
            read -p "Continue anyway? (y/n): " confirm
            if [ "$confirm" != "y" ]; then
                exit 0
            fi
        fi

        echo "Starting full system..."

        if [ "$DOCKER_AVAILABLE" = true ]; then
            docker-compose up -d
            echo ""
            echo "✅ Full system started with Docker"
            echo "Backend: http://localhost:401"
            echo ""
            echo "Initializing models..."
            sleep 10
            curl -X POST http://localhost:401/api/v1/phase1/init_phase1_models \
                -H "Content-Type: application/json" \
                -d '{"model_sizes": ["small"]}'
        else
            cd backend
            python main.py
        fi
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "🚀 Samadhan is running!"
echo "=========================================="
