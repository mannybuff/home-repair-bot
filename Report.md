# HomeRepairBot – Project Report

This report provides a narrative overview of the HomeRepairBot project:

- Why it was built  
- How the system is structured  
- What experiments were run  
- Current limitations and next steps  

It complements the top‑level `README.md` by going into more detail while staying shorter and more
readable than a full academic paper.

---

## 1. Motivation and Background

Many organizations want to leverage AI but may not be able to rely solely on large, closed models:

- Cost can grow with heavy usage.  
- Data privacy and legal concerns arise when sending sensitive data off‑prem.  
- The model’s behavior may not be tunable enough for narrow, safety‑critical domains.  

Home repair is an excellent example of a domain where:

- The user’s environment is highly variable.  
- Visual inspection is essential (photos of fittings, wiring, walls).  
- Safety matters (e.g., gas leaks, electrical faults, structural issues).  
- Users benefit from **step‑by‑step instructions**, not just quick answers.  

HomeRepairBot explores whether a **small vision‑language model** (Qwen2‑VL‑2B), combined with carefully
designed prompts, RAG, and safety gates, can deliver useful home‑repair assistance on consumer hardware.

---

## 2. System Overview

### 2.1 User Workflow

1. A user takes a photo of a home‑repair problem (e.g., under‑sink plumbing, cracked drywall).  
2. They add a short text description such as “How do I fix this?” or “Why is this leaking?”.  
3. The Android app sends this data to the backend API.  
4. The backend:
   - Inspects the image and text.  
   - Runs safety checks.  
   - Retrieves relevant reference material.  
   - Synthesizes a structured repair plan.  
5. The app displays the result in a chat‑style interface, with room for follow‑up questions.

### 2.2 Backend Pipeline

The backend pipeline can be summarized as:

1. **Ingress**  
   - Request arrives via Nginx / Cloudflare to a FastAPI endpoint.

2. **Captioning**  
   - If an image is included, Qwen2‑VL generates a caption and visual description.

3. **Bundle Construction**  
   - The caption (if any) and user text are combined into a single **bundle** with labeled fields:
     - Raw user text  
     - Caption text  
     - Parsed intent / topic placeholders  
     - Session IDs  

4. **Emergency Gate**  
   - A dedicated model call evaluates whether the situation is an emergency (e.g., gas smell, fire,
     major electrical short).  
   - If emergency is detected, the system responds with a “call 911 / professional immediately”
     message and stops.

5. **Intent & Topic Gate**  
   - Another model call classifies:
     - Is this about home repair / DIY at all?  
     - What high‑level topic cluster does it belong to? (plumbing, electrical, structural, etc.)  
   - Non–home‑repair queries receive a gentle refusal and a suggestion to start a new session.

6. **Query Generation and Retrieval**  
   - The bundle is used to generate search queries.  
   - The system runs:
     - RAG over a curated set of home‑repair PDFs.  
     - Whitelisted web searches over trusted sites.  
   - Around a dozen snippets are stored per channel (RAG, web).

7. **Hazard Gate**  
   - Qwen is prompted to detect potential hazards based on the snippets:
     - High‑voltage work  
     - Gas lines  
     - Load‑bearing structures  
   - Flags are recorded indicating whether extra caution or professional help is required.

8. **Long‑Form Summary**  
   - Relevant snippets are consolidated into a long‑form project description:
     - Tools and materials  
     - Preparation steps  
     - Step‑by‑step procedure  
   - This acts as a structured context block.

9. **Answer Generation**  
   - A final call produces the user‑facing answer:
     - Short overview of the issue  
     - Numbered repair steps  
     - Hazard warnings and when to call a professional  
   - For multi‑turn conversations, the answer is influenced by prior memory and current topic.

10. **Memory Management**  
    - The system writes out a **memory JSON** file with:
      - Topic and intent  
      - Key steps already taken  
      - Hazard and emergency status  
    - On subsequent turns, the memory is consulted to decide:
      - Whether to re‑run the full RAG pipeline  
      - Whether to treat the message as a new topic  

---

## 3. Experimental Setup

### 3.1 Domain Knowledge

To handle realistic home‑repair tasks, the system uses:

- A curated set of PDFs (e.g., home repair guides, manuals).  
- A whitelist of websites known for:
  - Accurate, practical instructions  
  - Safety‑conscious advice  

These sources feed the RAG and web‑search components, providing detailed instructions the model can then
summarize and rephrase.

### 3.2 Safety Scenarios

A series of safety‑oriented test cases were created, such as:

- Gas line upgrade vs active gas leak.  
- Electrical panel issues (e.g., repeatedly tripping breaker).  
- Kitchen fire damage.  
- Non‑emergency, fantasy input to test refusal.  

For each scenario, the system logs:

- Emergency gate decision  
- Hazard flags  
- Final user‑facing response  

Overall, the **Emergency Gate** behaved consistently and is one of the more reliable parts of the
system.

### 3.3 Latency Tests

The system was run on a local workstation (RTX 4080 + modern CPU). Latency was measured under different
levels of concurrency and for both:

- Text‑only requests  
- Image + text requests  

Findings (at a high level):

- Single‑user latency for a full first turn (including RAG and multiple model calls) is on the order of
  tens of seconds.  
- Concurrency increases latency significantly; under higher load, some timeouts occurred.  
- Image + text requests incur additional overhead from captioning and RAG.  

Potential improvements include:

- Model fine‑tuning and quantization.  
- Batching or restructuring calls to reduce round‑trips.  
- Scaling out across more GPUs or machines.  

---

## 4. Results and Observations

### 4.1 Strengths

- **Emergency detection** – Good at distinguishing genuine emergencies from normal DIY.  
- **Simple repairs** – For 2–3 step tasks (e.g., basic patching, simple leaks), the system often
  produces:
  - Reasonable, ordered instructions.  
  - Clear caveats and suggestions to check local codes.  
- **Domain focus** – Compared to a general chatbot, the system stays anchored in home repair and
  surfaces relevant how‑to information.

### 4.2 Common Issues

- **Over‑diagnosis from images**  
  - Example: a relatively minor crack in a wall sometimes triggers “foundation repair” language.  
- **Prompt sensitivity**  
  - Multiple chained model calls mean that small prompt changes can ripple through.  
- **Overview vs first‑step ordering**  
  - Sometimes the “overview” and “first step” blur together or appear in the wrong sections.  
- **Hazard flags fine‑tuning**  
  - Hazard detection works but is somewhat noisy; it would benefit from more targeted iteration.  

### 4.3 Overall Assessment

From a research / capstone perspective:

- The project successfully demonstrates that a **small VLM + RAG + safety stack** can provide useful,
  domain‑specific guidance for home repair tasks.  
- It shows how to integrate:
  - Vision  
  - Text  
  - Retrieval  
  - Safety gating  
  - A mobile frontend  
  into a cohesive system.

From a “can this beat large corporate models?” perspective:

- The current setup does not consistently outperform large, hosted models on quality or latency.  
- However, it provides significantly more **control** over:
  - Data flow  
  - Safety behavior  
  - System evolution  

---

## 5. Limitations

Key limitations identified:

1. **Model size and robustness**  
   - A 2B‑parameter VLM is powerful but still limited in nuanced, multi‑call pipelines.  
   - Errors can accumulate across captioning, intent, query generation, and answer stages.  

2. **Latency**  
   - Multiple serial model calls, plus RAG and web search, add up.  
   - Concurrency magnifies latency and can lead to timeouts.  

3. **Next‑step image retrieval**  
   - A v2 design for image retrieval (showing the next step visually) was not completed.  
   - v1 used a simpler cosine similarity on image citations; v2 aimed for a richer pipeline.  

4. **Engineering complexity**  
   - Coordinating several model calls, safety checks, and RAG makes debugging non‑trivial.  
   - The system has many moving parts; a more modular, test‑driven approach would help further.  

---

## 6. Future Work

Several promising directions:

1. **Model upgrades**  
   - Move to newer Qwen models with better instruction following and robustness.  
   - Consider splitting vision and language across specialized models.  

2. **Safety stack simplification**  
   - Combine emergency, hazard, intent, and topic checks into fewer, more robust prompts and calls.  
   - Add more systematic testing for edge cases.  

3. **Next‑step image retrieval**  
   - Implement a stable v2 with:
     - Embedding‑based retrieval over curated “how‑to” images.  
     - Clear alignment between text steps and corresponding images.  

4. **Latency and scalability**  
   - Adopt quantization and more aggressive optimization for inference.  
   - Introduce batching, caching, and possibly a small queue system for heavy loads.  
   - Explore deployment on a multi‑GPU home lab or cloud GPUs.  

5. **User experience improvements**  
   - Make the Android app more conversational and less “single‑shot”.  
   - Add tutorials, tool lists, and checklists alongside the instructions.  
   - Log anonymized repair sessions for offline analysis (with user consent).  

---

## 7. Conclusion

HomeRepairBot is a **working prototype** of a home‑repair‑focused multimodal assistant. It demonstrates:

- Practical integration of a small VLM with RAG and safety gating.  
- End‑to‑end architecture from phone to local GPU and back.  
- A strong foundation for future work on specialized, privacy‑friendly AI assistants.  

While it does not fully match large hosted models on every axis, it shows that a **carefully engineered
system on modest hardware** can deliver meaningful help in a complex, safety‑critical domain.
