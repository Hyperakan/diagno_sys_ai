## 🔍 Project Overview

**DiagnoSys** is a modular, AI-powered system designed for interactive diagnostic support and smart information retrieval. It combines:

- **FastAPI**-based microservices for high-performance APIs,
- **Large Language Models (LLMs)** for conversational and reasoning capabilities,
- **RAG (Retrieval-Augmented Generation)** for enhanced, context-aware responses,
- **Vector databases** for fast and meaningful semantic search,
- Full Docker-based architecture for scalable deployment.

The architecture is divided into two key services:

- **`app/`**: Main diagnostic backend – includes chat, user handling, LLM interaction
- **`rag/`**: Retrieval engine – handles document indexing and semantic search

---

## ⚙️ Key Components

- `app/`: Main API service for diagnostic chat and user-facing endpoints
- `rag/`: Retrieval backend for document embedding and similarity search
- `general_utils/`: Text extraction utilities (e.g., from PDFs)
- `test/`: Basic pipeline testing
- `Dockerfile.app` / `Dockerfile.rag`: Container setup for API and retrieval services
- `docker-compose.yml`: Unified orchestration for both services

