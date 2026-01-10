# Samādhān - Quick Start Guide

## 🚀 Start in 3 Steps

### 1. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your API keys (optional for testing)
```

### 2. Start All Services
```bash
docker-compose up -d
```

### 3. Access the Platform
- **Main App (via Nginx)**: http://localhost:400
- **API Docs**: http://localhost:400/api/docs

## 📍 All Service Ports (400-409)

| Service | Port | URL |
|---------|------|-----|
| **Nginx (Main Gateway)** | **400** | **http://localhost:400** |
| Backend API | 401 | http://localhost:401 |
| Frontend | 402 | http://localhost:402 |
| PostgreSQL | 403 | localhost:403 |
| Redis | 404 | localhost:404 |
| Qdrant | 405 | http://localhost:405/dashboard |
| Qdrant gRPC | 406 | localhost:406 |
| MLflow | 407 | http://localhost:407 |
| Neo4j Browser | 408 | http://localhost:408 |
| Neo4j Bolt | 409 | bolt://localhost:409 |

**💡 Pro Tip**: Use port 400 (Nginx) to access everything in one place!

## ✨ What's Implemented

### ✅ Core Features
- Multi-provider LLM (OpenAI, Anthropic)
- RAG with vector search
- Knowledge graph (Neo4j)
- Contextual memory system
- Explainable AI with confidence scores

### ✅ Domain Modules
- Healthcare: Clinical support, drug interactions
- Legal: Contract analysis, compliance
- Finance: Risk assessment, fraud detection

### ✅ Multi-Modal
- Computer Vision: OCR, image analysis
- Audio: Transcription, meeting intelligence

### ✅ UI Pages
- Chat interface
- Analytics dashboard
- Document management
- Model management

## 🛑 Stop Services
```bash
docker-compose down
```

## 📚 Documentation
- Full setup: [SETUP.md](SETUP.md)
- Project structure: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- Main README: [README.md](README.md)
