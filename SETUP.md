# Samādhān Setup Guide

## Quick Start with Docker

### Prerequisites
- Docker & Docker Compose installed
- API keys for OpenAI/Anthropic (optional but recommended)

### Step 1: Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### Step 2: Start All Services

```bash
# Build and start all containers
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 3: Access the Application

- **Frontend**: http://localhost:4002
- **Backend API**: http://localhost:4001
- **API Documentation**: http://localhost:4001/api/docs
- **MLflow**: http://localhost:4007
- **Qdrant Dashboard**: http://localhost:4005/dashboard

### Services & Ports

| Service | Port | Description |
|---------|------|-------------|
| Backend API | 4001 | FastAPI backend |
| Frontend | 4002 | Next.js frontend |
| PostgreSQL | 4003 | Database |
| Redis | 4004 | Cache |
| Qdrant | 4005 | Vector database |
| Qdrant gRPC | 4006 | Vector database gRPC |
| MLflow | 4007 | ML experiment tracking |

## Manual Setup (Without Docker)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp ../.env.example .env
# Edit .env with your configuration

# Run the server
python main.py
```

Backend will be available at http://localhost:8000

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Run development server
npm run dev
```

Frontend will be available at http://localhost:3000

### Database Setup

```bash
# Install PostgreSQL
# Create database
createdb samadhan

# Install Redis
# Install Qdrant (https://qdrant.tech/documentation/quick-start/)
```

## Testing the Setup

### 1. Health Check

```bash
curl http://localhost:4001/api/v1/health
```

### 2. Test Chat Endpoint

```bash
curl -X POST http://localhost:4001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, Samādhān!",
    "use_rag": false
  }'
```

### 3. Upload Document

```bash
curl -X POST http://localhost:4001/api/v1/documents/upload \
  -F "file=@/path/to/document.pdf"
```

## Stopping Services

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Troubleshooting

### Port Conflicts
If ports are already in use, edit `docker-compose.yml` and change the port mappings.

### API Key Issues
- Ensure API keys are properly set in `.env`
- Check that the keys have the correct permissions

### Database Connection Issues
- Verify PostgreSQL is running: `docker-compose ps`
- Check logs: `docker-compose logs postgres`

### Frontend Can't Connect to Backend
- Verify backend is running: `curl http://localhost:4001/health`
- Check CORS settings in backend configuration

## Development Tips

### Hot Reload
Both frontend and backend support hot reload:
- Backend: Changes to Python files automatically reload the server
- Frontend: Changes to React components automatically refresh the browser

### Debugging
- Backend logs: `docker-compose logs -f backend`
- Frontend logs: `docker-compose logs -f frontend`
- Database logs: `docker-compose logs -f postgres`

### Adding New Dependencies
```bash
# Backend
docker-compose exec backend pip install package-name
docker-compose exec backend pip freeze > requirements.txt

# Frontend
docker-compose exec frontend npm install package-name
```

## Production Deployment

For production deployment, see [DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Need Help?

- Documentation: `/docs` directory
- Issues: https://github.com/yourusername/samadhan/issues
- API Docs: http://localhost:4001/api/docs
