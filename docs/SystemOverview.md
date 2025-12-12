# System Overview – HomeRepairBot

This document provides a high‑level walkthrough of the complete HomeRepairBot system architecture.

## 1. Goals
- Provide reliable, structured home‑repair guidance.
- Accept *image + text* input.
- Maintain *safety-first* reasoning (emergency + hazard gates).
- Operate on consumer-grade hardware.
- Allow multi-turn conversations.

## 2. End-to-End Data Flow

```
Android App
   ↓
Cloudflare → Nginx → FastAPI Backend
   ↓
Captioning (Qwen2-VL)
   ↓
Bundle Assembly (caption + user text)
   ↓
Emergency Gate (hard stop if triggered)
   ↓
Intent/Topic Gate
   ↓
Query Generation
   ↓
RAG + Whitelisted Web Search
   ↓
Hazard Gate
   ↓
Long-form Repair Synthesis
   ↓
Step-by-Step Plan
   ↓
Memory Store
   ↓
Android App UI
```

## 3. Core Concepts

### 3.1 Bundles
A bundle is a structured JSON combining:
- Raw user text  
- Caption text  
- Derived keywords  
- Session metadata  

### 3.2 Safety Pipeline
1. **Emergency Gate** – stops everything immediately upon detection.  
2. **Intent Gate** – refuses non-home-repair requests.  
3. **Topic Gate** – classifies into plumbing/electrical/etc.  
4. **Hazard Gate** – flags high-risk tasks.  

### 3.3 RAG
- Uses a curated set of PDFs  
- Uses whitelisted web domains  
- Produces merged long-form context blocks  

### 3.4 Memory System
Stores:
- Topic  
- Key steps already given  
- Hazard flags  

Used to determine whether a new turn:
- Requires the full RAG pipeline  
- Can be answered with lightweight reasoning  
