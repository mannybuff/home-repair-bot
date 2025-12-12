# Frontend Design – HomeRepairBot

The Android client is written in **Kotlin**, designed for simplicity and clarity.

## 1. Folder Structure

```
frontend/
├─ app/
└─ src_main/
   ├─ java/com_dit_ai_homerepairbot/
   │  ├─ MainActivity.kt
   │  ├─ util/SessionStore.kt
   │  └─ net/
   │     ├─ Dtos.kt
   │     ├─ Repository.kt
   │     └─ SynthesisDtos.kt
   └─ res/
      ├─ layout/
      └─ values/
```

## 2. MainActivity.kt
Manages:
- Image capture or file selection  
- Text input collection  
- UI event routing  
- Launching API requests  
- Rendering results into message bubbles and repair cards  

## 3. DTO Layer
`net/Dtos.kt` and `SynthesisDtos.kt` define:
- Synthesis request schema  
- Synthesis response schema  
- Hazard payload  
- Step-by-step entries  

These maintain a 1:1 mapping with backend FastAPI response types.

## 4. Networking Layer
`Repository.kt` includes:
- Multipart construction (image + text)  
- Backend POST requests  
- JSON parsing  
- Error-handling hooks  
- Session-aware dialog ID handling  

## 5. Session System
`SessionStore.kt`:
- Persists the active dialog ID  
- Supports reset logic  
- Allows topic-shift detection on next turn  

## 6. UI Components
The UI uses:
- Chat-style message list  
- Card layout for repair steps  
- Hazard warning banners  
- Loading indicators  

UI is kept intentionally lightweight so development can focus on backend reasoning quality and safety behavior.
