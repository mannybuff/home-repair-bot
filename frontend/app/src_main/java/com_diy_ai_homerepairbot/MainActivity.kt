// File: app/src/main/java/com/diy_ai/homerepairbot/MainActivity.kt
package com.diy_ai.homerepairbot

import android.Manifest
import kotlin.math.max
import android.util.Log
import android.content.ClipData
import android.content.ClipboardManager
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import androidx.core.net.toUri
import androidx.lifecycle.lifecycleScope
import com.bumptech.glide.Glide
import com.bumptech.glide.load.model.GlideUrl
import com.bumptech.glide.load.model.LazyHeaders
import com.diy_ai.homerepairbot.databinding.ActivityMainBinding
import com.diy_ai.homerepairbot.net.ApiConfig
import com.diy_ai.homerepairbot.net.OrchestratorResponse
import com.diy_ai.homerepairbot.net.Repository
import com.diy_ai.homerepairbot.net.SafetySummary
import com.diy_ai.homerepairbot.net.prettyText
import com.diy_ai.homerepairbot.net.bullets
import com.diy_ai.homerepairbot.util.SessionStore
import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.HttpException
import java.io.ByteArrayOutputStream
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.util.UUID
import com.diy_ai.homerepairbot.net.DialogStateResponse
import com.diy_ai.homerepairbot.net.SessionMemoryResponse
import android.annotation.SuppressLint


class MainActivity : AppCompatActivity() {

    // ---- View binding & session store ----
    private lateinit var binding: ActivityMainBinding
    private val sessionStore by lazy { SessionStore(this) }
    private val repository by lazy { Repository() }

    // ---- Simple UI / state helpers ----
    private var currentDialogId: String = ""
    private var currentImageUri: Uri? = null
    private var currentImageBitmap: Bitmap? = null

    // Upload constraints
    private val maxUploadBytes: Int = 45 * 1024 * 1024 // 45 MB
    private val maxDimension: Int = 2560               // max side in px

    // Pretty-printing for debug JSON
    private val gsonPretty by lazy { GsonBuilder().setPrettyPrinting().create() }

    // ---- Activity result launchers ----

    // Gallery picker
    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            currentImageUri = it
            currentImageBitmap = null
            binding.imageViewPhoto.visibility = View.VISIBLE
            binding.imageViewPhoto.setImageURI(it)
        }
    }

    // Camera preview (bitmap in-memory)
    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        bitmap?.let {
            currentImageBitmap = it
            currentImageUri = null
            binding.imageViewPhoto.visibility = View.VISIBLE
            binding.imageViewPhoto.setImageBitmap(it)
        }
    }

    // Runtime permission prompt for camera
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        Toast.makeText(
            this,
            if (granted) getString(R.string.permission_granted) else getString(R.string.permission_denied),
            Toast.LENGTH_SHORT
        ).show()
        if (granted) {
            // Immediately launch camera if user just granted permission
            cameraLauncher.launch(null)
        }
    }

    // ---- Lifecycle ----
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // TEMP: prove the conversation container is visible
        addAssistantText(
            title = null,
            body = "[debug] Conversation container is alive."
        )

        // Debug / log fields should be scrollable + selectable
        binding.textViewResponse.movementMethod = ScrollingMovementMethod()

        binding.textViewResponse.isVerticalScrollBarEnabled = true
        binding.textViewResponse.setTextIsSelectable(true)

        binding.textLastPrompt.setTextIsSelectable(true)

        // Input scrolling
        binding.inputDescription.isVerticalScrollBarEnabled = true
        binding.inputDescription.setHorizontallyScrolling(false)
        binding.inputDescription.maxLines = 6

        initCardLongPressCopy()
        initDialogSpinner()
        initButtons()

        // Initial spinner contents from local store + best-effort fetch from server
        refreshDialogSpinner(preserveSelection = false)
        lifecycleScope.launch(Dispatchers.IO) {
            refreshSpinnerFromServer()
        }
    }

    // ---- UI helpers ----

    private fun convoContainer(): LinearLayout = binding.conversationContainer

    private fun scrollToBottom() {
        binding.scrollResponse.post {
            binding.scrollResponse.fullScroll(View.FOCUS_DOWN)
        }
    }

    private fun addUserImageFromPicker() {
        val drawable = binding.imageViewPhoto.drawable ?: return
        val iv = ImageView(this).apply {
            adjustViewBounds = true
            setImageDrawable(drawable)
            contentDescription = getString(R.string.preview_image)
            setPadding(12, 12, 12, 8)
        }
        convoContainer().addView(iv)
        scrollToBottom()
    }

    @SuppressLint("SetTextI18n")
    private fun addUserText(text: String) {
        if (text.isBlank()) return
        val tv = TextView(this).apply {
            setTextColor(0xFFBEBEEA.toInt())
            textSize = 15f
            setPadding(12, 8, 12, 12)
            this.text = text
            setTextIsSelectable(true)
        }
        convoContainer().addView(tv)

        // Debug: confirm user bubble was added
        binding.textViewResponse.text = "User bubble added."
        Toast.makeText(this, "addUserText() called", Toast.LENGTH_SHORT).show()

        scrollToBottom()
    }


    @SuppressLint("SetTextI18n")
    private fun addAssistantText(title: String?, body: String) {
        // Debug: mark that we're about to add an assistant bubble
        binding.textViewResponse.text = "Rendering assistant plan bubble..."

        if (!title.isNullOrBlank()) {
            val titleView = TextView(this).apply {
                setTextColor(0xFFECECF5.toInt())
                textSize = 16f
                setPadding(12, 8, 12, 4)
                text = title
                setTypeface(typeface, android.graphics.Typeface.BOLD)
                setTextIsSelectable(true)
            }
            convoContainer().addView(titleView)
        }

        val bodyView = TextView(this).apply {
            setTextColor(0xFFE0E0F5.toInt())
            textSize = 15f
            setPadding(12, 4, 12, 12)
            text = body
            setTextIsSelectable(true)
        }
        convoContainer().addView(bodyView)
        scrollToBottom()

        // Clear debug now that bubble exists
        binding.textViewResponse.text = ""
    }



    /**
     * Shows a "best match" preview image for the current plan.
     * Uses the dedicated ImageView in the output card.
     */
    private fun showBestPreview(absUrl: String) {
        val view = binding.bestPreview
        view.visibility = View.VISIBLE
        loadImageWithApiKeyInto(view, absUrl)
        view.setOnClickListener {
            startActivity(android.content.Intent(android.content.Intent.ACTION_VIEW, absUrl.toUri()))
        }
        scrollToBottom()
    }

    private fun setSendLoading(isLoading: Boolean) {
        binding.buttonSend.isEnabled = !isLoading
    }

    private fun absoluteFrom(relOrAbs: String): String {
        if (relOrAbs.startsWith("http://") || relOrAbs.startsWith("https://")) {
            return relOrAbs
        }
        val base = ApiConfig.BASE_URL.trimEnd('/')
        val rel = if (relOrAbs.startsWith("/")) relOrAbs else "/$relOrAbs"
        return base + rel
    }

    private fun extractTaskFrom(text: String): String =
        when {
            text.isBlank() -> "home repair issue"
            text.length <= 40 -> text
            else -> text.take(40) + "…"
        }

    private fun newDialogId(): String {
        val hex = UUID.randomUUID().toString().replace("-", "").take(8)
        return "dlg-$hex"
    }

    private fun getCurrentSpinnerLabel(): String {
        return binding.spinnerDialogs.selectedItem?.toString()
            ?: SessionStore.NEW_SESSION_LABEL
    }

    private fun clearUserInputs() {
        binding.inputDescription.setText("")

        currentImageUri = null
        currentImageBitmap = null

        binding.imageViewPhoto.setImageDrawable(null)
        // keep visibility; selectors will make it visible when needed

        binding.bestPreview.setImageDrawable(null)
        binding.bestPreview.visibility = View.GONE
    }

    private fun clearConversationUiForDialogSwitch() {
        binding.textViewResponse.text = ""
        binding.textLastPrompt.text = ""
        convoContainer().removeAllViews()
        clearUserInputs()
    }

    // Start a brand-new dialog from any gate (intent or emergency).
    private fun startNewDialogFromGate() {
        val newId = newDialogId()
        currentDialogId = newId
        sessionStore.add(newId)

        // Force spinner to select the new dialog and clear the conversation UI.
        refreshDialogSpinner(
            preserveSelection = false,
            forceSelectId = newId
        )
        clearConversationUiForDialogSwitch()
    }

    // ---- Dialog spinner / session handling ----

    private fun initDialogSpinner() {
        binding.spinnerDialogs.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    val chosen = parent.getItemAtPosition(position) as String
                    onDialogChosen(chosen)
                }

                override fun onNothingSelected(parent: AdapterView<*>) {
                    // no-op
                }
            }
    }

    private fun refreshDialogSpinner(
        preserveSelection: Boolean = true,
        forceSelectId: String? = null
    ) {
        val recent = sessionStore.list()
        val items = mutableListOf(SessionStore.NEW_SESSION_LABEL)
        items.addAll(recent)

        val previous = forceSelectId
            ?: if (preserveSelection) binding.spinnerDialogs.selectedItem?.toString() else null

        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            items
        )
        binding.spinnerDialogs.adapter = adapter

        val target = previous ?: items.firstOrNull()
        val idx = items.indexOfFirst { it == target }.coerceAtLeast(0)
        if (idx in items.indices) {
            binding.spinnerDialogs.setSelection(idx)
        }
    }

    private suspend fun refreshSpinnerFromServer() {
        try {
            val idx = repository.getSessionIndex()
            val dialogs = idx.getAsJsonObject("dialogs")
                ?.keySet()
                ?.toList()
                ?.sorted()
                ?: emptyList()

            dialogs.forEach { sessionStore.add(it) }

            withContext(Dispatchers.Main) {
                refreshDialogSpinner(preserveSelection = true)
            }
        } catch (_: Exception) {
            // non-fatal; local list still works
        }
    }

    private fun ensureDialogIdForSend(): String {
        val label = getCurrentSpinnerLabel()
        if (label == SessionStore.NEW_SESSION_LABEL || !label.startsWith("dlg-")) {
            val id = newDialogId()
            currentDialogId = id
            sessionStore.add(id)
            refreshDialogSpinner(preserveSelection = false, forceSelectId = id)
            return id
        }
        currentDialogId = label
        return currentDialogId
    }

    private fun onDialogChosen(chosen: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                if (chosen == SessionStore.NEW_SESSION_LABEL) {
                    // New dialog selected: generate id locally and let backend create its state
                    val newId = sessionStore.createDialog()
                    currentDialogId = newId

                    withContext(Dispatchers.Main) {
                        binding.textViewResponse.text = getString(
                            R.string.dialog_created_template,
                            newId.take(8)
                        )
                        clearConversationUiForDialogSwitch()
                        refreshDialogSpinner(
                            preserveSelection = false,
                            forceSelectId = newId
                        )
                    }
                    return@launch
                }

                // Existing dialog: set currentDialogId and reload last state + memory
                currentDialogId = chosen
                val dialogId = currentDialogId

                // Fetch dialog state (required) and memory (best-effort)
                val state: DialogStateResponse = repository.getDialogState(dialogId)
                val memory: SessionMemoryResponse? = runCatching {
                    repository.getSessionMemory(dialogId)
                }.getOrNull()

                val (lastQuery, assistantBody) = buildReplayFromMemory(state, memory)

                withContext(Dispatchers.Main) {
                    clearConversationUiForDialogSwitch()

                    // If we have a last user query, show it as the last user turn
                    if (!lastQuery.isNullOrBlank()) {
                        addUserText(lastQuery)
                    }

                    // Show the assistant body as a single assistant bubble
                    addAssistantText(
                        title = null,
                        body = assistantBody
                    )
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error loading dialog state", e)
                withContext(Dispatchers.Main) {
                    binding.textViewResponse.text = getString(
                        R.string.dialog_state_load_error_template,
                        e.message ?: "unknown"
                    )
                }
            }
        }
    }

    /**
     * Build a simple replay view from the dialog state + session memory.
     *
     * Returns:
     *  - first: last user query (or null)
     *  - second: assistant body text to show in the plan bubble
     */
    /**
     * Build a simple replay view from the dialog state + session memory.
     *
     * Returns:
     *  - first: last user query (or null)
     *  - second: assistant body text to show in the plan bubble
     */
    private fun buildReplayFromMemory(
        state: DialogStateResponse,
        memory: SessionMemoryResponse?
    ): Pair<String?, String> {

        val lastUserQuery = state.lastQuery?.takeIf { it.isNotBlank() }

        // Summary text comes from SessionMemoryResponse.memory.summary
        val summaryText = memory
            ?.memory
            ?.summary
            ?.takeIf { it.isNotBlank() }

        // Number of events already in this dialog comes from DialogStateResponse.events
        val eventsCount = state.events.size
        val eventsLine = if (eventsCount > 0) {
            "\n\n(This dialog already has $eventsCount steps saved. " +
                    "You can continue by asking a follow-up question.)"
        } else {
            ""
        }

        val body = when {
            summaryText != null -> {
                // Use the backend’s written summary as the main body
                summaryText.trim() + eventsLine
            }

            !lastUserQuery.isNullOrBlank() -> {
                // Fallback: no summary yet, just acknowledge the dialog
                "Dialog state loaded for:\n“${lastUserQuery.trim()}”.$eventsLine"
            }

            else -> {
                // Very defensive fallback
                "Dialog state loaded.$eventsLine"
            }
        }

        return lastUserQuery to body
    }

    // ---- Buttons / interactions ----
    private fun initButtons() {
        // Camera
        binding.buttonCamera.setOnClickListener {
            launchCameraWithPermission()
        }

        // Gallery
        binding.buttonGallery.setOnClickListener {
            galleryLauncher.launch("image/*")
        }

        // Clear everything
        binding.buttonClear.setOnClickListener {
            clearConversationUiForDialogSwitch()
        }

        // Check API (short tap → ping /healthz)
        binding.buttonCheckApi.setOnClickListener {
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val resp: JsonObject = repository.healthz()
                    val ok = resp.get("ok")?.asBoolean == true
                    val llmLoaded = resp.getAsJsonObject("models")
                        ?.get("text_llm_loaded")
                        ?.asBoolean ?: false

                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@MainActivity,
                            if (ok) "API OK • LLM=$llmLoaded" else "API not OK",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            this@MainActivity,
                            "API error: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            }
        }

        // Check API (long press → show dialog state + memory snapshot)
        binding.buttonCheckApi.setOnLongClickListener {
            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    val selected = binding.spinnerDialogs.selectedItem?.toString()
                    val id = when {
                        selected != null && selected.startsWith("dlg-") -> selected
                        currentDialogId.isNotBlank() -> currentDialogId
                        else -> {
                            val fresh = newDialogId()
                            currentDialogId = fresh
                            sessionStore.add(fresh)
                            withContext(Dispatchers.Main) {
                                refreshDialogSpinner(
                                    preserveSelection = false,
                                    forceSelectId = fresh
                                )
                            }
                            fresh
                        }
                    }

                    // Core dialog state
                    val state = repository.getDialogState(id)
                    val eventsCount = state.events.size
                    val source = if (eventsCount > 0) {
                        "latest.json (fallback or present)"
                    } else {
                        "chat_info.json (no events found)"
                    }

                    // Best-effort session memory (may fail independently)
                    val memory = runCatching {
                        repository.getSessionMemory(id)
                    }.getOrNull()
                    val summary = memory?.memory?.summary

                    val debugText = buildString {
                        appendLine(getString(R.string.dialog_state_for, id))
                        appendLine("last_query: ${state.lastQuery ?: ""}")
                        appendLine("events: $eventsCount, source: $source")
                        appendLine()

                        appendLine("memory.summary:")
                        if (!summary.isNullOrBlank()) {
                            // Trim very long summaries so the debug card stays readable
                            val trimmed = if (summary.length > 800) {
                                summary.take(800) + " …[trimmed]"
                            } else {
                                summary
                            }
                            appendLine(trimmed)
                        } else {
                            appendLine("<none>")
                        }

                        if (eventsCount > 0) {
                            appendLine()
                            appendLine("--- first event snapshot ---")
                            appendLine(gsonPretty.toJson(state.events.first()))
                        }
                    }

                    withContext(Dispatchers.Main) {
                        binding.textViewResponse.text = debugText
                        binding.textViewResponse.scrollTo(0, 0)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        binding.textViewResponse.text = getString(
                            R.string.fusion_error,
                            e.message ?: "unknown"
                        )
                    }
                }
            }
            true
        }


        // IME action "Send"
        binding.inputDescription.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND) {
                binding.buttonSend.performClick()
                true
            } else {
                false
            }
        }

        // Main send → Orchestrator dialog (text + optional image)
        binding.buttonSend.setOnClickListener {
            val userText = binding.inputDescription.text?.toString().orEmpty()
            if (userText.isBlank()) {
                Toast.makeText(
                    this,
                    getString(R.string.describe_issue_hint),
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }

            val dialogId = ensureDialogIdForSend()
            binding.textLastPrompt.text = userText

            // Snapshot and echo user message into conversation
            val imagePart = buildJpegPartFromSelection()
            if (imagePart != null) {
                addUserImageFromPicker()
            }
            addUserText(userText)

            // Clear inputs so they don't bleed into next send
            clearUserInputs()

            lifecycleScope.launch(Dispatchers.IO) {
                withContext(Dispatchers.Main) {
                    setSendLoading(true)
                    binding.textViewResponse.text = getString(
                        R.string.sending_request_for_dialog,
                        dialogId
                    )
                }

                try {
                    val resp = if (imagePart != null) {
                        repository.dialogWithImage(
                            imagePart = imagePart,
                            text = userText,
                            dialogId = dialogId
                        )
                    } else {
                        repository.dialogTextOnly(
                            text = userText,
                            dialogId = dialogId
                        )
                    }

                    val safety = resp.synthesis?.safetySummary
                    val blocked = (resp.blocked == true) || (safety?.blocked == true)
                    val requiresAck = (resp.requiresAck == true) || (safety?.requiresAck == true)
                    val ackStage = safety?.ackStage ?: resp.ackStage ?: 0

                    // Total number of acknowledgement stages – for now we assume
                    // at least 1 if acks are required.
                    val stageCount = if (requiresAck) max(ackStage + 1, 1) else 0
                    val safetyMessage = buildSafetyMessage(resp, safety)

                    // Interpret backend gates as intent / emergency / hazard
                    withContext(Dispatchers.Main) {
                        setSendLoading(false)

                        // Emergency gate = hard block, no ack, session should effectively be "done".
                        val isEmergencyGate = blocked && !requiresAck

                        // Hazard gate = staged acknowledgement, then normal instructions.
                        val isHazardGate = !blocked && requiresAck

                        // Intent gate = not home-repair, soft refusal, no ack, no synthesis payload.
                        val isIntentGate =
                            !blocked &&
                                    !requiresAck && !resp.ok && resp.synthesis == null

                        when {
                            isEmergencyGate -> {
                                // Hard stop: show emergency dialog and encourage starting a new task.
                                showEmergencyGateDialog(safetyMessage) {
                                    startNewDialogFromGate()
                                }
                            }

                            isHazardGate -> {
                                // Soft stop: require a single acknowledgement, then show instructions.
                                showAckDialog(
                                    ackStage,
                                    stageCount,
                                    safetyMessage
                                ) {
                                    // After user acknowledges, render the normal synthesis result.
                                    lifecycleScope.launch(Dispatchers.IO) {
                                        renderSynthesisResult(userText, dialogId, resp)
                                    }
                                }
                            }

                            isIntentGate -> {
                                // Not a home repair request: soft refusal + "New Task" affordance.
                                showIntentGateDialog(safetyMessage) {
                                    startNewDialogFromGate()
                                }
                            }

                            else -> {
                                // No gate triggered: render synthesis as usual.
                                lifecycleScope.launch(Dispatchers.IO) {
                                    renderSynthesisResult(userText, dialogId, resp)
                                }
                            }
                        }
                    }

                } catch (e: Exception) {
                    e.printStackTrace()
                    val msg = when (e) {
                        is HttpException -> {
                            val body = e.response()?.errorBody()?.string()?.take(600)
                            "HTTP ${e.code()} • $body"
                        }
                        is SocketTimeoutException -> "Timeout while waiting for server"
                        is ConnectException -> "Cannot connect to ${ApiConfig.BASE_URL}"
                        else -> "${e::class.java.simpleName}: ${e.message ?: "unknown"}"
                    }

                    withContext(Dispatchers.Main) {
                        binding.textViewResponse.text =
                            getString(R.string.fusion_error, msg)
                        setSendLoading(false)
                    }
                }
            }
        }
    }

    private fun initCardLongPressCopy() {
        // Single handler we can attach to multiple views (card and inner layout)
        val handler = View.OnLongClickListener {
            val container = convoContainer()
            val buf = StringBuilder()

            // Include last prompt + debug text
            buf.appendLine(binding.textLastPrompt.text?.toString().orEmpty())
            buf.appendLine()
            buf.appendLine(binding.textViewResponse.text?.toString().orEmpty())
            buf.appendLine()

            for (i in 0 until container.childCount) {
                val v = container.getChildAt(i)
                if (v is TextView) {
                    val txt = v.text?.toString().orEmpty()
                    if (txt.isNotBlank()) {
                        buf.appendLine(txt)
                        buf.appendLine()
                    }
                }
            }

            val textToCopy = buf.toString().trim()
            if (textToCopy.isNotEmpty()) {
                val cm = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("DIY-AI dialog", textToCopy))
                Toast.makeText(this, "Dialog copied to clipboard", Toast.LENGTH_SHORT).show()
            }
            true
        }

        // Long-press anywhere on the card OR its inner content
        binding.cardOutput.setOnLongClickListener(handler)
        binding.scrollContent.setOnLongClickListener(handler)
    }


    // ---- Safety & gating helpers ----

    private fun showAckDialog(stage: Int, total: Int, message: String, onAck: () -> Unit) {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.safety_ack_title, stage + 1, total))
            .setMessage(message.ifBlank { getString(R.string.safety_ack_default) })
            .setPositiveButton(R.string.safety_ack_continue) { _, _ -> onAck() }
            .setNegativeButton(R.string.safety_ack_cancel, null)
            .show()
    }

    private fun showEmergencyGateDialog(
        message: String,
        onNewTask: () -> Unit
    ) {
        val body = message.ifBlank {
            getString(R.string.emergency_gate_message_default)
        }

        AlertDialog.Builder(this)
            .setTitle(R.string.emergency_gate_title)
            .setMessage(body)
            .setPositiveButton(R.string.emergency_gate_new_task) { _, _ ->
                onNewTask()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun showIntentGateDialog(
        message: String,
        onNewTask: () -> Unit
    ) {
        val body = message.ifBlank {
            getString(R.string.intent_gate_message_default)
        }

        AlertDialog.Builder(this)
            .setTitle(R.string.intent_gate_title)
            .setMessage(body)
            .setPositiveButton(R.string.intent_gate_new_task) { _, _ ->
                onNewTask()
            }
            .setNegativeButton(R.string.intent_gate_stay_here, null)
            .show()
    }

    private fun buildSafetyMessage(
        resp: OrchestratorResponse,
        safety: SafetySummary?
    ): String {
        if (safety == null && resp.answer.isNullOrBlank()) {
            return getString(R.string.safety_ack_default)
        }

        return buildString {
            val gate = safety?.gate.orEmpty().trim()
            if (gate.isNotEmpty()) {
                appendLine(gate)
                appendLine()
            }

            val warnings = safety?.warnings.orEmpty()
            for (w in warnings) {
                val reason = w.reason?.takeIf { it.isNotBlank() }
                val cat = w.category?.takeIf { it.isNotBlank() }
                val level = w.level?.takeIf { it.isNotBlank() }

                val line = when {
                    reason != null && cat != null && level != null ->
                        "$cat ($level): $reason"
                    reason != null && cat != null ->
                        "$cat: $reason"
                    reason != null ->
                        reason
                    cat != null && level != null ->
                        "$cat ($level)"
                    cat != null ->
                        cat
                    else -> null
                }

                if (!line.isNullOrBlank()) {
                    append("• ")
                    appendLine(line.trim())
                }
            }

            if (isBlank()) {
                append(resp.answer.orEmpty().ifBlank {
                    getString(R.string.safety_ack_default)
                })
            }
        }.trim()
    }

    /**
     * Optional: render a brief safety bubble inside the conversation
     * once we've passed the gate.
     *
     * NOTE: currently not used in the main happy path, since we keep the
     * primary output in textViewResponse. Left here for future chat-style UI.
     */
    private fun renderSafetyBubble(resp: OrchestratorResponse, safety: SafetySummary?) {
        val blocked = (resp.blocked == true) || (safety?.blocked == true)
        val requiresAck = (resp.requiresAck == true) || (safety?.requiresAck == true)
        if (!blocked && !requiresAck) return

        val ackStage = safety?.ackStage ?: resp.ackStage ?: 0
        val stageIndex = ackStage + 1
        val totalStages = stageIndex.coerceAtLeast(1)

        val title = getString(R.string.safety_ack_title, stageIndex, totalStages)
        val body = buildSafetyMessage(resp, safety)
        if (body.isNotBlank()) {
            addAssistantText(title = title, body = body)
        }
    }

    // ---- Synthesis rendering ----

    @SuppressLint("SetTextI18n")
    private suspend fun renderSynthesisResult(
        userText: String,
        fallbackDialogId: String,
        response: OrchestratorResponse
    ) {
        try {
            val synthesis = response.synthesis
            val finalDialogId = response.dialogId ?: fallbackDialogId

            // Remember the dialog ID locally, but DO NOT refresh the spinner here.
            // Spinner changes would fire onDialogChosen(), which clears the bubbles.
            if (finalDialogId.startsWith("dlg-")) {
                currentDialogId = finalDialogId
                sessionStore.add(finalDialogId)
            }

            // If there is no structured synthesis, fall back to the raw answer string
            if (synthesis == null) {
                val fallbackBody = response.answer.orEmpty().ifBlank {
                    "I couldn’t build a full repair plan from this request, " +
                            "but here is the assistant’s direct answer."
                }

                withContext(Dispatchers.Main) {
                    addAssistantText(
                        title = extractTaskFrom(userText),
                        body = fallbackBody
                    )
                    binding.textViewResponse.text = ""
                    setSendLoading(false)
                }
                return
            }

            // Build main plan text from synthesis fields
            val scopeText = synthesis.scopeOverview
                ?.prettyText()
                ?.takeIf { it.isNotBlank() }

            val toolsText = synthesis.toolsRequired
                ?.bullets()
                ?.takeIf { it.isNotBlank() }

            val firstStepText = synthesis.firstStep
                ?.prettyText()
                ?.takeIf { it.isNotBlank() }

            val planBody = buildString {
                // Overview
                scopeText?.let {
                    appendLine("Overview:")
                    appendLine(it.trim())
                    appendLine()
                }

                // Tools
                toolsText?.let {
                    appendLine("Tools you’ll need:")
                    appendLine(it.trim())
                    appendLine()
                }

                // First step
                firstStepText?.let {
                    appendLine("First step:")
                    appendLine(it.trim())
                    appendLine()
                }
            }.trim().ifBlank {
                // If the structured fields are empty, fall back to the answer text
                response.answer.orEmpty().ifBlank {
                    "I generated a repair plan, but couldn’t format it into sections."
                }
            }

            // Optional: preview image from citations
            val previewUrl = synthesis.imageCitations
                ?.firstOrNull()
                ?.takeIf { it.isNotBlank() }
                ?.let { absoluteFrom(it) }

            withContext(Dispatchers.Main) {
                addAssistantText(
                    title = extractTaskFrom(userText),
                    body = planBody
                )

                if (!previewUrl.isNullOrBlank()) {
                    showBestPreview(previewUrl)
                }

                binding.textViewResponse.text = ""
                setSendLoading(false)
            }
        } catch (e: Exception) {
            // Surface any crash in this renderer into the debug field
            withContext(Dispatchers.Main) {
                binding.textViewResponse.text =
                    "renderSynthesisResult error: ${e::class.java.simpleName}: ${e.message}"
                setSendLoading(false)
            }
        }
    }

    // ---- Image helpers ----

    private fun buildJpegPartFromSelection(): MultipartBody.Part? {
        currentImageUri?.let { uri ->
            return buildJpegPartFromUri(uri, "photo.jpg")
        }
        currentImageBitmap?.let { bmp ->
            return buildJpegPartFromBitmap(bmp, "camera.jpg")
        }
        return null
    }

    private fun buildJpegPartFromUri(uri: Uri, filename: String): MultipartBody.Part? {
        return try {
            contentResolver.openInputStream(uri)?.use { input ->
                val data = input.readBytes()

                // First pass: just decode bounds
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeByteArray(data, 0, data.size, bounds)

                val sample = computeInSampleSize(bounds.outWidth, bounds.outHeight, maxDimension)

                val opts = BitmapFactory.Options().apply { inSampleSize = sample }
                val bmp = BitmapFactory.decodeByteArray(data, 0, data.size, opts) ?: return null

                val payload = compressToCap(bmp) ?: return null
                val req = payload.toRequestBody("image/jpeg".toMediaType())
                MultipartBody.Part.createFormData("image", filename, req)
            }
        } catch (_: Throwable) {
            null
        }
    }

    private fun buildJpegPartFromBitmap(src: Bitmap, filename: String): MultipartBody.Part? {
        return try {
            val scaled = downscaleIfNeeded(src, maxDimension)
            val payload = compressToCap(scaled) ?: return null
            val req = payload.toRequestBody("image/jpeg".toMediaType())
            MultipartBody.Part.createFormData("image", filename, req)
        } catch (_: Throwable) {
            null
        }
    }

    private fun computeInSampleSize(w: Int, h: Int, maxDim: Int): Int {
        var inSampleSize = 1
        var width = w
        var height = h
        while (width > maxDim || height > maxDim) {
            width /= 2
            height /= 2
            inSampleSize *= 2
        }
        return inSampleSize.coerceAtLeast(1)
    }

    private fun downscaleIfNeeded(src: Bitmap, maxDim: Int): Bitmap {
        val w = src.width
        val h = src.height
        if (w <= maxDim && h <= maxDim) return src
        val scale = if (w >= h) maxDim / w.toFloat() else maxDim / h.toFloat()
        val nw = (w * scale).toInt().coerceAtLeast(1)
        val nh = (h * scale).toInt().coerceAtLeast(1)
        return src.scale(nw, nh)
    }

    private fun compressToCap(bitmap: Bitmap): ByteArray? {
        val qualities = listOf(90, 80, 70, 60, 50, 40, 30)
        for (q in qualities) {
            val baos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, q, baos)
            val arr = baos.toByteArray()
            if (arr.size <= maxUploadBytes) return arr
        }

        // One more try: half-scale, medium quality
        val half = bitmap.scale(bitmap.width / 2, bitmap.height / 2)
        val baos = ByteArrayOutputStream()
        half.compress(Bitmap.CompressFormat.JPEG, 70, baos)
        val downsized = baos.toByteArray()
        return if (downsized.size <= maxUploadBytes) downsized else null
    }

    private fun loadImageWithApiKeyInto(view: ImageView, absUrl: String) {
        val glideUrl = GlideUrl(
            absUrl,
            LazyHeaders.Builder()
                .addHeader("X-API-Key", ApiConfig.API_KEY)
                .build()
        )
        Glide.with(this).load(glideUrl).into(view)
        view.visibility = View.VISIBLE
    }

    // ---- Permissions ----

    private fun launchCameraWithPermission() {
        val perm = Manifest.permission.CAMERA
        if (ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED) {
            cameraLauncher.launch(null)
        } else {
            requestPermissionLauncher.launch(perm)
        }
    }
}
