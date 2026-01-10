# Samādhān - Access Guide

## 🌐 All Service URLs

### Primary Access (Through Nginx - Port 400)
All services are accessible through the unified Nginx gateway:

| Service | URL | Description |
|---------|-----|-------------|
| **Main Application** | http://localhost:400 | Frontend UI (Next.js) |
| **API Documentation** | http://localhost:400/api/docs | Swagger UI for API |
| **Qdrant Dashboard** | http://localhost:400/qdrant/dashboard | Vector database UI |
| **MLflow UI** | http://localhost:400/mlflow/ | ML experiment tracking |
| **Neo4j Browser** | http://localhost:400/neo4j/ | Knowledge graph browser |

### Direct Service Access (Individual Ports)
You can also access services directly on their individual ports:

| Service | Port | Direct URL |
|---------|------|------------|
| Backend API | 401 | http://localhost:401/api/docs |
| Frontend | 402 | http://localhost:402 |
| PostgreSQL | 403 | localhost:403 (DB connection) |
| Redis | 404 | localhost:404 (Cache connection) |
| Qdrant | 405 | http://localhost:405/dashboard |
| Qdrant gRPC | 406 | localhost:406 (Internal) |
| MLflow | 407 | http://localhost:407 |
| Neo4j Browser | 408 | http://localhost:408 |
| Neo4j Bolt | 409 | bolt://localhost:409 (DB connection) |

---

## ✅ Service Status Check

### Quick Health Checks

**1. Check all containers are running:**
```bash
docker-compose ps
```

**2. Test Nginx (unified gateway):**
```bash
curl http://localhost:400/health
```

**3. Test Backend API:**
```bash
curl http://localhost:400/api/v1/health
```

**4. Test Frontend:**
- Open browser: http://localhost:400
- Should see Samādhān chat interface

**5. Test Qdrant:**
```bash
curl http://localhost:400/qdrant/
```

**6. Test MLflow:**
```bash
curl http://localhost:400/mlflow/
```

**7. Test Neo4j:**
```bash
curl http://localhost:400/neo4j/
```

---

## 🎯 Quick Start Testing

### 1. Open the Main Application
```
http://localhost:400
```
You should see:
- Welcome screen with Samādhān branding
- Chat interface
- Sidebar with navigation

### 2. Test Chat API
Using curl:
```bash
curl -X POST "http://localhost:400/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, what can you help me with?",
    "conversation_id": "test-123"
  }'
```

### 3. Upload a Document
```bash
curl -X POST "http://localhost:400/api/v1/documents/upload" \
  -F "file=@document.pdf"
```

### 4. List Available Models
```bash
curl http://localhost:400/api/v1/models
```

---

## 🔧 Environment Configuration

### Required Environment Variables

Create a `.env` file in the root directory:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database
DATABASE_URL=postgresql://samadhan:samadhan123@postgres:5432/samadhan

# Redis
REDIS_URL=redis://redis:6379/0

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=samadhan123

# Application
ENVIRONMENT=development
API_VERSION=v1.0.0
DEBUG=true
```

---

## 🚀 Usage Examples

### Example 1: Healthcare Query
```bash
curl -X POST "http://localhost:400/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the side effects of aspirin?",
    "domain": "healthcare",
    "conversation_id": "health-001"
  }'
```

### Example 2: Legal Analysis
```bash
curl -X POST "http://localhost:400/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze this NDA for potential risks",
    "domain": "legal",
    "conversation_id": "legal-001"
  }'
```

### Example 3: Financial Assessment
```bash
curl -X POST "http://localhost:400/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the risk profile of this portfolio?",
    "domain": "finance",
    "conversation_id": "finance-001"
  }'
```

---

## 📊 Monitoring & Debugging

### View Logs

**All services:**
```bash
docker-compose logs -f
```

**Specific service:**
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f nginx
```

**Last 50 lines:**
```bash
docker-compose logs --tail=50 backend
```

### Restart a Service

```bash
docker-compose restart backend
docker-compose restart frontend
docker-compose restart nginx
```

### Rebuild and Restart

```bash
docker-compose down
docker-compose up -d --build
```

---

## 🎨 Frontend Pages

Navigate to these pages in the UI:

| Page | URL | Description |
|------|-----|-------------|
| Home | http://localhost:400/ | Chat interface |
| Analytics | http://localhost:400/analytics | Metrics dashboard |
| Documents | http://localhost:400/documents | Document management |
| Models | http://localhost:400/models | AI model comparison |
| Knowledge | http://localhost:400/knowledge | Knowledge graph |
| History | http://localhost:400/history | Conversation history |
| Settings | http://localhost:400/settings | User preferences |

---

## 🔐 Database Connections

### PostgreSQL
```bash
psql -h localhost -p 403 -U samadhan -d samadhan
# Password: samadhan123
```

### Redis
```bash
redis-cli -h localhost -p 404
```

### Neo4j Cypher Shell
```bash
docker exec -it samadhan-neo4j cypher-shell -u neo4j -p samadhan123
```

---

## ⚡ Performance Tips

1. **First request is slow**: The first API call initializes LLM connections. Subsequent requests are faster.

2. **Use caching**: Redis caching is enabled for common queries.

3. **Adjust model selection**: Use faster models (GPT-3.5) for simple queries, GPT-4 for complex ones.

4. **Vector search tuning**: Adjust `k` parameter in RAG retrieval based on context needed.

---

## 🐛 Troubleshooting

### Issue: "Connection refused" error
**Solution**: Check if all containers are running
```bash
docker-compose ps
```

### Issue: Frontend shows 404
**Solution**: Check nginx logs and restart
```bash
docker-compose logs nginx
docker-compose restart nginx
```

### Issue: API returns empty responses
**Solution**: Check if API keys are configured in `.env`

### Issue: Slow responses
**Solution**:
1. Check backend logs for errors
2. Verify Qdrant and Redis are running
3. Use faster models for testing

---

## 📞 Support

- Documentation: See `README.md`, `SETUP.md`, `PROJECT_STRUCTURE.md`
- Implementation Status: See `IMPLEMENTATION_STATUS.md`
- Improvements Guide: See `IMPROVEMENTS_AND_AI_TRAINING.md`

---

**🎉 Platform is now fully operational!**

Access the main application at: **http://localhost:400**
