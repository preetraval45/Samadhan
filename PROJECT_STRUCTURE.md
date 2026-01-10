# Samādhān Project Structure

## 📁 Complete Directory Structure

```
Samadhan/
├── 🎨 Branding
│   ├── logo-full.svg                    # Full logo with tagline
│   └── logo-tagline.svg                 # Horizontal tagline logo
│
├── 🔧 Configuration
│   ├── .env.example                     # Environment variables template
│   ├── .gitignore                       # Git ignore rules
│   ├── docker-compose.yml               # Docker orchestration (ports 4001-4007)
│   ├── README.md                        # Project overview
│   ├── SETUP.md                         # Setup instructions
│   └── PROJECT_STRUCTURE.md             # This file
│
├── 🔙 Backend (FastAPI)
│   ├── main.py                          # Application entry point
│   ├── requirements.txt                 # Python dependencies
│   ├── Dockerfile                       # Backend container
│   │
│   ├── core/                            # Core functionality
│   │   ├── config.py                    # Configuration management
│   │   └── logging.py                   # Structured logging
│   │
│   ├── api/routes/                      # API endpoints
│   │   ├── health.py                    # Health checks
│   │   ├── chat.py                      # Chat/conversation API
│   │   ├── models.py                    # Model management
│   │   └── documents.py                 # Document upload/management
│   │
│   ├── rag/                             # RAG Architecture
│   │   ├── retriever.py                 # Context retrieval
│   │   └── generator.py                 # Response generation
│   │
│   ├── llm/                             # LLM Engine
│   │   └── engine.py                    # Multi-provider LLM client
│   │
│   ├── vector_store/                    # Vector Databases
│   │   └── qdrant_store.py              # Qdrant integration
│   │
│   ├── explainability/                  # AI Explainability Layer
│   │   └── interpreter.py               # SHAP-like attribution, confidence scoring
│   │
│   ├── domains/                         # Domain-Specific Modules
│   │   ├── healthcare.py                # Medical AI (clinical support, drug interactions)
│   │   ├── legal.py                     # Legal AI (contract analysis, compliance)
│   │   └── finance.py                   # Financial AI (risk assessment, fraud detection)
│   │
│   └── multimodal/                      # Multi-Modal Processing
│       ├── vision.py                    # Computer vision, OCR, medical imaging
│       └── audio.py                     # Speech-to-text, meeting intelligence
│
├── 🎨 Frontend (Next.js 14 + Tailwind CSS)
│   ├── package.json                     # Node dependencies
│   ├── tsconfig.json                    # TypeScript config
│   ├── tailwind.config.ts               # Tailwind with brand colors
│   ├── postcss.config.js                # PostCSS config
│   ├── next.config.js                   # Next.js config
│   ├── Dockerfile                       # Frontend container
│   │
│   └── src/
│       ├── app/                         # App Router
│       │   ├── layout.tsx               # Root layout with sidebar
│       │   ├── page.tsx                 # Home/Chat page
│       │   ├── globals.css              # Global styles
│       │   ├── analytics/               # Analytics Dashboard
│       │   │   └── page.tsx             # Metrics, charts, insights
│       │   └── documents/               # Document Management
│       │       └── page.tsx             # Upload, list, manage docs
│       │
│       └── components/
│           ├── providers.tsx            # React Query provider
│           ├── layout/                  # Layout components
│           │   ├── Sidebar.tsx          # Side navigation
│           │   └── Header.tsx           # Top header with search
│           └── chat/                    # Chat components
│               ├── ChatInterface.tsx    # Main chat UI
│               ├── ChatMessage.tsx      # Message display
│               └── WelcomeScreen.tsx    # Landing page
│
└── 🗄️ Data & Infrastructure
    ├── data/                            # Data storage (gitignored)
    ├── docs/                            # Documentation
    ├── scripts/                         # Utility scripts
    └── tests/                           # Test files
```

## 🚀 Key Features Implemented

### ✅ Core Platform
- [x] FastAPI backend with async support
- [x] Next.js 14 frontend with App Router
- [x] Tailwind CSS with custom brand theme
- [x] Docker Compose orchestration (Ports 4001-4007)
- [x] PostgreSQL, Redis, Qdrant, MLflow integration

### ✅ AI Capabilities
- [x] RAG (Retrieval-Augmented Generation) architecture
- [x] Multi-provider LLM support (OpenAI, Anthropic)
- [x] Vector database integration (Qdrant)
- [x] Semantic search and embeddings

### ✅ Explainability & Trust
- [x] Confidence scoring system
- [x] Source attribution and citations
- [x] Decision explanation generation
- [x] Audit trail logging
- [x] Bias detection framework

### ✅ Domain Expertise
- [x] **Healthcare Module**
  - Clinical decision support
  - Drug interaction analysis
  - Medical image analysis (research only)
  - Evidence-based recommendations

- [x] **Legal Module**
  - Contract analysis and risk assessment
  - Compliance checking (GDPR, CCPA, etc.)
  - Obligation extraction
  - Case law research framework

- [x] **Finance Module**
  - Investment risk assessment
  - Fraud detection
  - Portfolio analysis
  - Regulatory compliance (KYC, AML)
  - Value at Risk (VaR) calculations

### ✅ Multi-Modal Processing
- [x] **Computer Vision**
  - Image analysis and description
  - OCR (Optical Character Recognition)
  - Medical imaging analysis
  - Visual Q&A
  - Object detection

- [x] **Audio Processing**
  - Speech-to-text transcription
  - Speaker diarization
  - Meeting intelligence
  - Emotion detection
  - Language identification
  - Auto-generated summaries and chapters

### ✅ User Interface
- [x] Modern dark theme with brand colors
- [x] Responsive sidebar navigation
- [x] Real-time chat interface
- [x] Analytics dashboard with metrics
- [x] Document management page
- [x] Welcome screen with features

## 🎯 Technology Stack

### Backend
- **Framework**: FastAPI 0.110+
- **AI/ML**: LangChain, Transformers, Sentence-Transformers
- **LLMs**: OpenAI, Anthropic Claude
- **Vector DB**: Qdrant
- **Database**: PostgreSQL, Redis
- **MLOps**: MLflow

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **State**: React Query, Zustand
- **UI**: Lucide Icons, Framer Motion

### Infrastructure
- **Containers**: Docker, Docker Compose
- **Orchestration**: Kubernetes-ready
- **Ports**: 4001-4007 series

## 🔌 API Endpoints

### Health & Status
- `GET /` - Root endpoint
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/ready` - Readiness probe
- `GET /api/v1/health/live` - Liveness probe

### Chat & Conversation
- `POST /api/v1/chat` - Send message with RAG
- `GET /api/v1/chat/history/{id}` - Get conversation
- `DELETE /api/v1/chat/history/{id}` - Delete conversation
- `POST /api/v1/chat/stream` - Streaming responses

### Models
- `GET /api/v1/models` - List available models
- `GET /api/v1/models/{id}` - Model details

### Documents
- `POST /api/v1/documents/upload` - Upload document
- `GET /api/v1/documents` - List documents
- `DELETE /api/v1/documents/{id}` - Delete document

## 🌐 Access Points

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| Frontend | 4002 | http://localhost:4002 | Web interface |
| Backend API | 4001 | http://localhost:4001 | REST API |
| API Docs | 4001 | http://localhost:4001/api/docs | Swagger UI |
| PostgreSQL | 4003 | localhost:4003 | Database |
| Redis | 4004 | localhost:4004 | Cache |
| Qdrant | 4005 | http://localhost:4005/dashboard | Vector DB |
| Qdrant gRPC | 4006 | localhost:4006 | Vector DB gRPC |
| MLflow | 4007 | http://localhost:4007 | ML tracking |

## 🎨 Brand Colors

```css
/* Primary Gradient */
#00d4ff → #0099ff → #0066ff (Cyan to Blue)

/* Accent Gradient */
#ff6b00 → #ff9500 (Orange)

/* Background */
#0a0e27 (Dark primary)
#1a1f3a (Dark secondary)
#252b45 (Dark tertiary)
```

## 🔐 Security & Compliance

- Environment-based configuration
- API key management
- Rate limiting ready
- CORS configuration
- Health monitoring
- Audit logging framework

## 📝 Next Steps

### Immediate
1. Add API keys to `.env` file
2. Run `docker-compose up -d`
3. Access frontend at http://localhost:4002

### Future Enhancements
- [ ] Knowledge graph integration (Neo4j)
- [ ] Contextual memory system
- [ ] Advanced analytics with D3.js/Plotly
- [ ] Multi-language support
- [ ] Real-time collaboration features
- [ ] Mobile app
- [ ] Enterprise SSO integration

## 📚 Documentation

- **Setup**: See [SETUP.md](SETUP.md)
- **API Docs**: http://localhost:4001/api/docs
- **README**: See [README.md](README.md)

---

Built with ❤️ for better decision-making through AI
