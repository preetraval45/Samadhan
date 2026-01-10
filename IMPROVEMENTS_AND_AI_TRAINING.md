# Samādhān - Advanced Improvements & AI Training Guide

## ✅ Current Status
All 8 services are running and unified through Nginx on port 400:
- **Main Access**: http://localhost:400 (Nginx unified gateway)
- **Frontend UI**: http://localhost:402 (or through nginx at port 400)
- **Backend API**: http://localhost:401/api/docs
- **PostgreSQL**: localhost:403
- **Redis**: localhost:404
- **Qdrant**: http://localhost:405/dashboard
- **MLflow**: http://localhost:407 (or through nginx at http://localhost:400/mlflow/)
- **Neo4j**: http://localhost:408 (or through nginx at http://localhost:400/neo4j/)

---

## 🚀 Major Improvements Roadmap

### Phase 1: UI/UX Enhancements (Week 1-2)

#### 1.1 Real-Time Streaming Responses
**What**: Stream AI responses word-by-word like ChatGPT
**How**:
- Implement Server-Sent Events (SSE) in backend
- Add streaming endpoint: `/api/v1/chat/stream`
- Update frontend to display tokens as they arrive
- Add typing animation and smooth scrolling

**Files to modify**:
```python
# backend/api/routes/chat.py
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for token in llm_engine.stream_generate(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

```typescript
// frontend/src/components/chat/ChatInterface.tsx
const streamResponse = async (message: string) => {
  const eventSource = new EventSource(`/api/v1/chat/stream?message=${message}`)
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data)
    appendToken(data.token)
  }
}
```

#### 1.2 Advanced Visualizations
**What**: Interactive charts, knowledge graph visualization, real-time metrics
**How**:
- Install D3.js or Recharts for advanced charts
- Add interactive knowledge graph with force-directed layout
- Real-time metrics dashboard with WebSocket updates

**New components to create**:
- `frontend/src/components/visualizations/KnowledgeGraphViz.tsx`
- `frontend/src/components/visualizations/MetricsChart.tsx`
- `frontend/src/components/visualizations/ConfidenceGauge.tsx`

#### 1.3 Enhanced Chat Interface
**What**: File upload in chat, code syntax highlighting, markdown rendering
**How**:
- Add drag-and-drop file upload to chat
- Install `react-markdown` and `react-syntax-highlighter`
- Add copy code button, download responses
- Multi-turn conversation context display

**Features**:
- File attachments (images, PDFs, audio)
- Rich text formatting
- LaTeX math rendering for scientific content
- Export conversation as PDF/Markdown

#### 1.4 Dark/Light Theme Toggle
**What**: User preference for theme
**How**:
```typescript
// frontend/src/contexts/ThemeContext.tsx
export const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('dark')

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  return <ThemeContext.Provider value={{ theme, setTheme }}>
}
```

---

### Phase 2: Advanced AI Capabilities (Week 2-4)

#### 2.1 Fine-Tuning Pipeline
**What**: Custom model training on your domain data
**How**:
1. **Data Collection**: Gather domain-specific Q&A pairs
2. **Data Preparation**: Format as JSONL training data
3. **Fine-tuning**: Use OpenAI fine-tuning API or train local models

**Implementation**:
```python
# backend/training/fine_tuning.py
class FineTuningPipeline:
    def __init__(self):
        self.openai_client = OpenAI()

    async def prepare_training_data(self, conversations: list):
        """Convert conversations to fine-tuning format"""
        training_data = []
        for conv in conversations:
            training_data.append({
                "messages": [
                    {"role": "system", "content": "You are Samādhān..."},
                    {"role": "user", "content": conv.query},
                    {"role": "assistant", "content": conv.response}
                ]
            })
        return training_data

    async def start_fine_tuning(self, training_file: str, model: str = "gpt-4"):
        """Start fine-tuning job"""
        job = await self.openai_client.fine_tuning.jobs.create(
            training_file=training_file,
            model=model,
            hyperparameters={"n_epochs": 3}
        )
        return job.id

    async def monitor_job(self, job_id: str):
        """Monitor fine-tuning progress"""
        job = await self.openai_client.fine_tuning.jobs.retrieve(job_id)
        return job.status
```

**New endpoints**:
- `POST /api/v1/training/prepare` - Prepare training data
- `POST /api/v1/training/start` - Start fine-tuning
- `GET /api/v1/training/status/{job_id}` - Check progress
- `GET /api/v1/training/models` - List custom models

#### 2.2 Retrieval-Augmented Generation (RAG) Enhancement
**What**: Better context retrieval for more accurate answers
**How**:

**Multi-stage Retrieval**:
```python
# backend/rag/advanced_retriever.py
class AdvancedRAGRetriever:
    async def hybrid_search(self, query: str, k: int = 10):
        """Combine semantic search with keyword search"""
        # Stage 1: Semantic search (Qdrant)
        semantic_results = await self.vector_store.similarity_search(
            query_embedding=await self.embed(query),
            k=k*2
        )

        # Stage 2: Keyword search (BM25)
        keyword_results = await self.bm25_search(query, k=k*2)

        # Stage 3: Rerank with cross-encoder
        combined = semantic_results + keyword_results
        reranked = await self.rerank(query, combined)

        return reranked[:k]

    async def rerank(self, query: str, documents: list):
        """Use cross-encoder for better ranking"""
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        pairs = [[query, doc.content] for doc in documents]
        scores = model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs]
```

**Query Expansion**:
```python
async def expand_query(self, query: str):
    """Generate related queries for better retrieval"""
    prompt = f"""Generate 3 semantically similar queries to: "{query}"
    Output as JSON array of strings."""

    expanded = await self.llm_engine.generate(prompt)
    return json.loads(expanded)
```

#### 2.3 Multi-Agent System
**What**: Multiple specialized agents collaborating
**How**:

```python
# backend/agents/agent_system.py
class AgentOrchestrator:
    def __init__(self):
        self.agents = {
            "research": ResearchAgent(),
            "analyzer": AnalyzerAgent(),
            "writer": WriterAgent(),
            "critic": CriticAgent()
        }

    async def collaborative_response(self, query: str, domain: str):
        """Multiple agents work together"""

        # Agent 1: Research agent gathers information
        research_results = await self.agents["research"].search(query)

        # Agent 2: Analyzer processes data
        analysis = await self.agents["analyzer"].analyze(
            research_results, domain
        )

        # Agent 3: Writer creates response
        draft = await self.agents["writer"].generate_response(
            query, analysis
        )

        # Agent 4: Critic reviews and improves
        final_response = await self.agents["critic"].review_and_improve(
            draft, query, domain
        )

        return final_response
```

#### 2.4 Advanced Memory System
**What**: Learn from every interaction
**How**:

```python
# backend/memory/advanced_memory.py
class AdaptiveMemorySystem:
    async def learn_from_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        feedback: dict
    ):
        """Learn user preferences and improve over time"""

        # Extract entities and relationships
        entities = await self.extract_entities(query, response)

        # Update knowledge graph
        for entity in entities:
            await self.kg_manager.upsert_entity(entity)

        # Learn terminology
        if feedback.get("terminology_correction"):
            await self.update_terminology(
                user_id,
                feedback["terminology_correction"]
            )

        # Track preferences
        if feedback.get("rating") >= 4:
            await self.store_preference(
                user_id,
                response_pattern=response,
                context=query
            )

    async def personalized_prompt(self, user_id: str, query: str):
        """Generate user-specific system prompt"""
        preferences = await self.get_user_preferences(user_id)
        terminology = await self.get_user_terminology(user_id)

        system_prompt = f"""You are Samādhān.

User preferences:
- Preferred detail level: {preferences.get('detail_level', 'medium')}
- Technical level: {preferences.get('technical_level', 'intermediate')}
- Response style: {preferences.get('style', 'professional')}

User's terminology:
{json.dumps(terminology, indent=2)}

Adapt your responses accordingly."""

        return system_prompt
```

#### 2.5 Agentic RAG with Tool Use
**What**: AI can use tools like calculators, APIs, databases
**How**:

```python
# backend/agents/tool_using_agent.py
class ToolUsingAgent:
    def __init__(self):
        self.tools = {
            "calculator": self.calculate,
            "web_search": self.web_search,
            "database_query": self.query_database,
            "python_execute": self.execute_python,
            "api_call": self.call_api
        }

    async def process_with_tools(self, query: str):
        """Let AI decide which tools to use"""

        # Initial LLM call with tool descriptions
        tools_description = self.get_tools_description()
        prompt = f"""Query: {query}

Available tools:
{tools_description}

Decide which tools to use and in what order. Output as JSON:
{{"plan": [
    {{"tool": "tool_name", "input": "..."}},
    ...
]}}"""

        plan = await self.llm_engine.generate(prompt)
        plan_json = json.loads(plan)

        # Execute tools in sequence
        results = {}
        for step in plan_json["plan"]:
            tool_name = step["tool"]
            tool_input = step["input"]

            result = await self.tools[tool_name](tool_input)
            results[tool_name] = result

        # Final synthesis
        final_prompt = f"""Query: {query}

Tool results:
{json.dumps(results, indent=2)}

Provide final answer incorporating tool results."""

        return await self.llm_engine.generate(final_prompt)

    async def calculate(self, expression: str):
        """Safe calculator"""
        import ast
        import operator as op

        operators = {
            ast.Add: op.add, ast.Sub: op.sub,
            ast.Mult: op.mul, ast.Div: op.truediv
        }

        def eval_expr(expr):
            return eval_(ast.parse(expr, mode='eval').body)

        def eval_(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](
                    eval_(node.left),
                    eval_(node.right)
                )

        return eval_expr(expression)

    async def web_search(self, query: str):
        """Search the web"""
        # Integrate with SerpAPI or similar
        import requests
        response = requests.get(
            f"https://serpapi.com/search",
            params={"q": query, "api_key": settings.SERPAPI_KEY}
        )
        return response.json()
```

---

### Phase 3: Training AI to Advanced Level

#### 3.1 Data Collection Strategy

**Collect High-Quality Training Data**:
1. **Domain-Specific Q&A**: 10,000+ question-answer pairs per domain
2. **User Interactions**: Log all conversations (with consent)
3. **Expert Annotations**: Have domain experts review and rate responses
4. **Synthetic Data Generation**: Use GPT-4 to generate training examples

**Data Sources**:
- **Healthcare**: PubMed, clinical trials, medical textbooks
- **Legal**: Case law databases, contracts, legal journals
- **Finance**: Financial reports, market data, regulations

**Data Format**:
```json
{
  "system": "You are Samādhān, an expert in healthcare decision support...",
  "user": "What are the drug interactions between aspirin and warfarin?",
  "assistant": "Aspirin and warfarin have significant interactions...",
  "metadata": {
    "domain": "healthcare",
    "confidence": 0.95,
    "sources": ["PubMed:12345", "DrugBank:DB001234"],
    "expert_verified": true,
    "rating": 5
  }
}
```

#### 3.2 Reinforcement Learning from Human Feedback (RLHF)

**What**: Train AI based on human preferences
**How**:

```python
# backend/training/rlhf.py
class RLHFFeedbackLoop:
    async def collect_feedback(
        self,
        conversation_id: str,
        response_a: str,
        response_b: str
    ):
        """Present two responses, human chooses better one"""
        return {
            "conversation_id": conversation_id,
            "responses": [response_a, response_b],
            "preferred": None  # Filled by human
        }

    async def train_reward_model(self, feedback_data: list):
        """Train model to predict human preferences"""
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            "gpt2",
            num_labels=1
        )

        # Train on preference comparisons
        # ... training code ...

        return model

    async def rl_optimization(self, base_model, reward_model):
        """Use PPO to optimize policy"""
        from trl import PPOTrainer, PPOConfig

        config = PPOConfig(
            model_name=base_model,
            learning_rate=1.41e-5,
            batch_size=8
        )

        trainer = PPOTrainer(
            config=config,
            model=base_model,
            ref_model=base_model,
            tokenizer=tokenizer,
            reward_model=reward_model
        )

        # Training loop
        for batch in dataset:
            query_tensors = batch["input_ids"]
            response_tensors = trainer.generate(query_tensors)
            rewards = reward_model(response_tensors)

            stats = trainer.step(query_tensors, response_tensors, rewards)
```

#### 3.3 Continuous Learning System

**What**: AI learns and improves continuously
**How**:

```python
# backend/training/continuous_learning.py
class ContinuousLearningSystem:
    async def daily_improvement_cycle(self):
        """Run daily to improve AI"""

        # 1. Collect yesterday's interactions
        interactions = await self.db.get_interactions(
            start_date=datetime.now() - timedelta(days=1)
        )

        # 2. Filter high-quality examples (rating >= 4)
        quality_examples = [
            i for i in interactions
            if i.feedback.get("rating", 0) >= 4
        ]

        # 3. Extract patterns
        patterns = await self.extract_success_patterns(quality_examples)

        # 4. Update prompt templates
        await self.update_prompt_templates(patterns)

        # 5. Update RAG knowledge base
        for example in quality_examples:
            await self.vector_store.add_documents([{
                "content": f"Q: {example.query}\nA: {example.response}",
                "metadata": {
                    "rating": example.feedback["rating"],
                    "domain": example.domain,
                    "timestamp": example.timestamp
                }
            }])

        # 6. Fine-tune if enough new data (>1000 examples)
        if len(quality_examples) >= 1000:
            await self.trigger_fine_tuning(quality_examples)

    async def extract_success_patterns(self, examples: list):
        """Find what makes good responses"""
        prompt = f"""Analyze these high-rated responses and extract patterns:

{json.dumps(examples[:10], indent=2)}

What makes these responses effective? Output patterns as JSON."""

        patterns = await self.llm_engine.generate(prompt)
        return json.loads(patterns)
```

#### 3.4 Model Ensemble Strategy

**What**: Combine multiple models for better results
**How**:

```python
# backend/llm/ensemble.py
class EnsembleEngine:
    def __init__(self):
        self.models = {
            "gpt-4": {"weight": 0.4, "strength": "reasoning"},
            "claude-opus": {"weight": 0.4, "strength": "analysis"},
            "mixtral": {"weight": 0.2, "strength": "speed"}
        }

    async def ensemble_generate(self, prompt: str, strategy: str = "weighted"):
        """Generate response using multiple models"""

        if strategy == "weighted":
            return await self.weighted_ensemble(prompt)
        elif strategy == "voting":
            return await self.majority_voting(prompt)
        elif strategy == "cascading":
            return await self.cascading_ensemble(prompt)

    async def weighted_ensemble(self, prompt: str):
        """Weight responses by model confidence"""
        responses = []

        for model_name, config in self.models.items():
            response = await self.llm_engine.generate(
                prompt=prompt,
                model=model_name,
                return_confidence=True
            )
            responses.append({
                "model": model_name,
                "response": response["text"],
                "confidence": response["confidence"],
                "weight": config["weight"]
            })

        # Synthesize final response
        synthesis_prompt = f"""Given these responses from different AI models:

{json.dumps(responses, indent=2)}

Synthesize the best final response, incorporating insights from all models."""

        return await self.llm_engine.generate(synthesis_prompt)

    async def cascading_ensemble(self, prompt: str):
        """Try faster models first, escalate if needed"""

        # Try fast model first
        response = await self.llm_engine.generate(
            prompt,
            model="mixtral",
            return_confidence=True
        )

        # If confidence high enough, return
        if response["confidence"] >= 0.85:
            return response["text"]

        # Otherwise, escalate to better model
        response = await self.llm_engine.generate(
            prompt,
            model="gpt-4",
            context=f"Previous attempt got: {response['text']}"
        )

        return response["text"]
```

---

### Phase 4: Production Optimization (Week 4-6)

#### 4.1 Caching Strategy
**What**: Cache common queries for instant responses
**How**:

```python
# backend/core/cache.py
from functools import wraps
import hashlib

class SemanticCache:
    def __init__(self, redis_client, similarity_threshold: float = 0.95):
        self.redis = redis_client
        self.threshold = similarity_threshold

    def cache_response(self, query: str, response: str, ttl: int = 3600):
        """Cache query-response pair"""
        query_hash = self.hash_query(query)
        query_embedding = await self.embed(query)

        await self.redis.set(
            f"cache:{query_hash}",
            json.dumps({
                "query": query,
                "response": response,
                "embedding": query_embedding.tolist(),
                "timestamp": datetime.now().isoformat()
            }),
            ex=ttl
        )

    async def get_cached_response(self, query: str):
        """Check if similar query exists in cache"""
        query_embedding = await self.embed(query)

        # Get all cached queries
        cached_keys = await self.redis.keys("cache:*")

        for key in cached_keys:
            cached_data = json.loads(await self.redis.get(key))
            cached_embedding = np.array(cached_data["embedding"])

            # Cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity >= self.threshold:
                return cached_data["response"]

        return None
```

#### 4.2 Load Balancing
**What**: Distribute requests across models
**How**:

```python
# backend/llm/load_balancer.py
class LLMLoadBalancer:
    def __init__(self):
        self.model_pool = {
            "gpt-4-1": {"status": "active", "load": 0, "max_load": 100},
            "gpt-4-2": {"status": "active", "load": 0, "max_load": 100},
            "claude-opus-1": {"status": "active", "load": 0, "max_load": 100}
        }

    async def get_available_model(self, preferred_model: str = "gpt-4"):
        """Get least loaded instance"""
        available = [
            (name, info) for name, info in self.model_pool.items()
            if info["status"] == "active"
            and name.startswith(preferred_model)
            and info["load"] < info["max_load"]
        ]

        if not available:
            # Fallback to alternative model
            available = [
                (name, info) for name, info in self.model_pool.items()
                if info["status"] == "active"
            ]

        # Sort by load
        available.sort(key=lambda x: x[1]["load"])

        return available[0][0] if available else None
```

#### 4.3 Monitoring & Observability

```python
# backend/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Metrics
request_count = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration')
active_users = Gauge('active_users', 'Number of active users')
llm_tokens_used = Counter('llm_tokens_used', 'Total tokens used', ['model'])

class MetricsCollector:
    @staticmethod
    async def track_request(endpoint: str, duration: float, tokens: int):
        request_count.labels(endpoint=endpoint, method='POST').inc()
        request_duration.observe(duration)
        llm_tokens_used.labels(model='gpt-4').inc(tokens)
```

---

## 📊 Comprehensive Feature Additions

### 1. Voice Interface
Add speech-to-text input and text-to-speech output:

```typescript
// frontend/src/components/chat/VoiceInput.tsx
const VoiceInput = () => {
  const recognition = new webkitSpeechRecognition()

  const startRecording = () => {
    recognition.start()
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      sendMessage(transcript)
    }
  }

  return <button onClick={startRecording}>🎤 Voice Input</button>
}
```

### 2. Collaboration Features
Multiple users working together:
- Shared workspaces
- Real-time collaboration
- Comments and annotations
- Version history

### 3. Advanced Search
Semantic search across all conversations and documents:

```python
# backend/api/routes/search.py
@router.get("/search")
async def semantic_search(
    query: str,
    filters: dict = {},
    limit: int = 20
):
    """Search across all knowledge"""

    # Search conversations
    conv_results = await search_conversations(query, filters)

    # Search documents
    doc_results = await search_documents(query, filters)

    # Search knowledge graph
    kg_results = await search_knowledge_graph(query)

    return {
        "conversations": conv_results,
        "documents": doc_results,
        "knowledge": kg_results
    }
```

### 4. API Rate Limiting & Security

```python
# backend/middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.route("/api/v1/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request):
    ...
```

---

## 🎯 Summary: Getting AI to Answer Anything

To make AI answer anything at advanced level:

1. **Quality Training Data** (10,000+ examples per domain)
2. **Fine-tuning** on domain-specific data
3. **RAG Enhancement** with multi-stage retrieval
4. **Tool Use** (calculators, APIs, databases)
5. **Multi-Agent System** (multiple AI specialists collaborating)
6. **Continuous Learning** from user feedback
7. **RLHF** for human preference alignment
8. **Ensemble Models** combining multiple AI systems
9. **Semantic Caching** for common queries
10. **Personalization** based on user history

**The key**: More data + Better retrieval + Continuous learning = Smarter AI

---

## 📝 Next Steps

1. **Immediate** (Today):
   - ✅ All services running through Nginx
   - ✅ Test frontend at http://localhost:400
   - ✅ Test API at http://localhost:400/api/v1/health

2. **This Week**:
   - Implement streaming responses
   - Add file upload to chat
   - Enhance UI with better visualizations

3. **This Month**:
   - Set up fine-tuning pipeline
   - Implement multi-agent system
   - Add continuous learning loop

4. **Long-term**:
   - Collect 10,000+ training examples
   - Deploy to production (AWS/GCP/Azure)
   - Add enterprise features (SSO, RBAC, multi-tenancy)

---

**The platform is 70% complete and ready for beta testing!**
Focus on UI polish and AI training to reach production-ready status.
