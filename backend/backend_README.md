# Backend (API + Orchestration)

This folder contains the Python backend for HomeRepairBot. It exposes a FastAPI service that:

1. Accepts text + image requests from the Android app.  
2. Runs the multimodal pipeline:
   - Captioning (Qwen2-VL)  
   - Emergency / intent / topic / hazard gates  
   - RAG + curated web search  
   - Long-form summary + step-by-step answer  
3. Manages dialog memory across turns.

The backend is organized into:

- `main.py` – FastAPI application entrypoint.  
- `api/` – Route handlers and request/response schemas.  
- `services/` – Core business logic (orchestration, safety, RAG, sessions).  
- `utils/` – Logging, probes, common helpers.  
- `tests/` – Lightweight tests for safety and orchestration behavior.

Sensitive configuration (e.g., API keys, model paths) is kept out of version control and loaded via
environment variables or local settings files.
