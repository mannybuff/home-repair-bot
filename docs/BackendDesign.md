# Backend Design – HomeRepairBot

The backend is implemented in **FastAPI** and structured around modular services.

## 1. Folder Structure

```
backend/
├─ main.py
├─ api/
├─ services/
│  ├─ orchestrator.py
│  ├─ qwen_use.py
│  ├─ safety.py
│  ├─ sessions.py
│  └─ rag_search.py
└─ utils/
```

## 2. main.py
Defines:
- FastAPI application instance  
- Routers  
- CORS configuration  
- Health checks  

## 3. API Layer
Routes include:
- `/synthesize`  
- `/summarize`  
- `/intent`  
- `/hazard`  
- `/caption`  

Each route:
- Validates input  
- Calls into the correct service chain  
- Returns structured DTOs  

## 4. Orchestrator
Coordinates the reasoning pipeline:

1. Caption image → text  
2. Build bundle  
3. Emergency check  
4. Intent/topic check  
5. RAG + web search  
6. Hazard evaluation  
7. Long-form synthesis  
8. Final step-by-step plan  

## 5. Safety System
`safety.py` implements:
- Emergency classification  
- Intent filtering  
- Topic classification  
- Hazard detection  

Each is isolated, testable, and logged.

## 6. RAG System
`rag_search.py`:
- Generates queries  
- Runs local index search  
- Runs whitelisted web search  
- Normalizes results  
- Produces structured snippet batches  

## 7. Session Management
`sessions.py` handles:
- Dialog ID creation  
- Writing memory JSON  
- Loading memory on new turns  

## 8. Utilities
`utils/` contains:
- Logging helpers  
- Curl-style probe wrappers  
- Common shared functions  
