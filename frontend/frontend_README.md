# Frontend (Android App)

This folder contains the Android client for HomeRepairBot. It is a Kotlin app that:

- Captures or selects a photo of a home-repair problem.  
- Lets the user enter a short text description.  
- Sends the data to the backend API.  
- Displays the structured repair plan and hazard information.

The key pieces are:

- `app/` – Android app module (Manifest, Gradle config, etc.).  
- `src_main/java/com_dit_ai_homerepairbot/`:
  - `MainActivity.kt` – main UI and event handling.  
  - `util/SessionStore.kt` – simple session tracking.  
  - `net/Dtos.kt` – data transfer objects matching backend responses.  
  - `net/Repository.kt` – networking and API calls.  
  - `net/SynthesisDtos.kt` – specific DTOs for synthesis responses.  
- `src_main/res/` – layouts and styles.

Sensitive or environment-specific files, such as API base URL configs and interceptors with keys, should
be provided via local, untracked files (for example `ApiConfigLocal.kt`) and are intentionally excluded
from this repository.
