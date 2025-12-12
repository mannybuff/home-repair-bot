
# HomeRepairBot

HomeRepairBot is a multimodal home repair assistant that runs on **consumer‑grade hardware** and uses a
small vision–language model plus Retrieval‑Augmented Generation (RAG), safety gates, and an Android
frontend.

Users take a photo of a problem (for example, a faucet leak or cracked drywall), add a short description,
and receive a **step‑by‑step repair plan** that blends model reasoning with curated home‑repair resources.

This repository contains:

- A **Python backend** (FastAPI) hosting Qwen2‑VL, RAG, safety logic, and multi‑turn memory.
- An **Android frontend** for image capture, text input, and displaying structured repair plans.
- A modular design intended for controlled, local, privacy‑respecting deployment.

---

## 1. Motivation

Many organizations want AI assistance, but relying solely on large, closed‑source models can be limiting:

- Cost and rate‑limits can become unpredictable.
- Sensitive user images and home‑interior data may not be suitable for external APIs.
- Model output behavior is harder to tune for narrow, safety‑critical domains.

Home repair is an especially challenging domain requiring:

- **Visual understanding** of real household environments  
- **Safety‑critical decision making** (electrical, gas, structural hazards)  
- **Ordered, practical instructions** rather than a conversational summary  

HomeRepairBot investigates whether a carefully engineered system, built around a **small 2B‑parameter
vision‑language model**, can reliably provide structured guidance in a complex, real‑world domain when
combined with RAG, domain filtering, and explicit safety gates.

---

## 2. High‑Level Architecture

**Flow overview:**

1. User sends text and/or image from Android app  
2. Traffic routes via Cloudflare → Nginx → FastAPI backend  
3. Backend executes the multimodal pipeline:  
   - Caption (Qwen2‑VL)  
   - Merge caption + user text into a structured bundle  
   - **Emergency Gate** (fires, gas leaks, electrical shorts)  
   - **Intent + Topic Gate** (is it home repair? which category?)  
   - Query generation for RAG and whitelisted web search  
   - **Hazard Gate** (high‑voltage, gas line, structural load‑bearing, etc.)  
   - Long‑form repair synthesis  
   - Final structured step‑by‑step plan  
   - Memory JSON written for multi‑turn follow‑ups  

4. Android app renders overview, hazard notices, and step‑by‑step instructions.

---

## 3. Repository Layout

```
home-repair-bot/
├─ README.md
├─ Report.md
├─ requirements.txt
├─ .gitignore
├─ artifacts/
├─ backend/
├─ frontend/
└─ docs/
```

### Backend (Python)

Includes:

- `main.py`
- `api/` – routing layer
- `services/` – model calls, safety gates, RAG, orchestrator
- `utils/` – logging, probe utilities, shared helpers
- `tests/` – lightweight repeatable test cases

Sensitive files (API keys, auth, local settings) are intentionally excluded.

### Frontend (Android)

Includes:

- `MainActivity.kt`  
- `SessionStore.kt`  
- `Dtos.kt`, `Repository.kt`, `SynthesisDtos.kt`  
- XML layouts (chat UI, cards, controls)

Sensitive configuration files like API base URLs and interceptors are intentionally excluded.

---

## 4. Key Features

- **Multimodal Input** — Image + text combined prior to reasoning  
- **Safety Stack** — Four layers: Emergency, Intent, Topic, Hazard  
- **Domain RAG** — Curated PDF corpus + whitelisted search  
- **Step‑By‑Step Reasoning** — Structured plan generation  
- **Multi‑Turn Memory** — Per‑dialog context persistence  
- **Local‑First Architecture** — Designed for workstation deployment

---

## 5. Getting Started (Backend)

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Requires:

- Python 3.10+
- Local or near‑local model paths for Qwen2‑VL
- RAG index folders and whitelisted search configuration

---

## 6. Getting Started (Android)

- Open `frontend/` in Android Studio  
- Modify local API base URL  
- Build & run on device or emulator  

---

## 7. Status, Limitations & Future Work

Documented thoroughly in `Report.md`. Key themes:

- Captioner over‑interpretation  
- Prompt‑chain error propagation  
- Latency improvements  
- Safety‑gate consolidation  
- Upgrading to newer Qwen or hybrid VLM/Lang‑only architectures  

---

# 8. License — **GNU Affero General Public License v3.0 (AGPL‑3.0)**

HomeRepairBot is released under the **GNU AGPL‑3.0**, a strong copyleft license designed to guarantee that:

- **Any modifications, extensions, or derivative works must also be released under AGPL‑3.0**,  
- **Even when the software is used to provide a network‑accessible service** (SaaS, hosted API, cloud inference),  
- Users interacting with modified versions — including over the network — retain the right to inspect the
  source code.

This prevents third parties from:

- Taking the backend logic private  
- Hosting a closed, proprietary version of the service  
- Withholding improvements made to the system  

AGPL‑3.0 is the preferred license for server‑side AI systems where **openness, transparency, and reciprocal
software freedom** are core goals.

Full legal text:  
https://www.gnu.org/licenses/agpl-3.0.en.html

---

For additional architecture notes, see `docs/SystemOverview.md`,
`docs/BackendDesign.md`, and `docs/FrontendDesign.md`.
