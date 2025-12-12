package com.diy_ai.homerepairbot.net

import com.google.gson.JsonObject
import com.google.gson.annotations.SerializedName


// ---------- Safety summary ----------

data class SafetyWarning(
    @SerializedName("category") val category: String? = null,
    @SerializedName("level")    val level: String? = null,
    @SerializedName("reason")   val reason: String? = null,
    @SerializedName("signals")  val signals: List<String>? = null
)

data class SafetySummary(
    @SerializedName("blocked")       val blocked: Boolean? = null,
    @SerializedName("requires_ack")  val requiresAck: Boolean? = null,
    @SerializedName("ack_stage")     val ackStage: Int? = null,
    @SerializedName("gate")          val gate: String? = null,
    @SerializedName("warnings")      val warnings: List<SafetyWarning>? = null
)

// ---------- Orchestrator envelope (/api/v1/rag/orchestrator/dialog) ----------

data class OrchestratorResponse(
    @SerializedName("ok")             val ok: Boolean = true,
    @SerializedName("schema_version") val schemaVersion: String? = null,
    @SerializedName("dialog_id")      val dialogId: String? = null,

    @SerializedName("blocked")        val blocked: Boolean? = null,
    @SerializedName("requires_ack")   val requiresAck: Boolean? = null,
    @SerializedName("ack_stage")      val ackStage: Int? = null,

    @SerializedName("session_mode")   val sessionMode: String? = null,
    @SerializedName("orchestration")  val orchestration: JsonObject? = null,

    @SerializedName("synthesis")      val synthesis: Synthesis? = null,
    @SerializedName("answer")         val answer: String? = null
)

// ---------- Synthesis payload (server -> app) ----------

data class Synthesis(
    @SerializedName("scope_overview") val scopeOverview: Any? = null,
    @SerializedName("tools_required") val toolsRequired: Any? = null,
    @SerializedName("first_step")     val firstStep: Any? = null,
    @SerializedName("image_citations") val imageCitations: List<String>? = null,
    @SerializedName("longform_refs")   val longformRefs: List<String>? = null,


    // New: safety summary included in synthesis object
    @SerializedName("safety_summary")  val safetySummary: SafetySummary? = null
)

// ---------- Fusion envelope (/api/v1/rag/fusion/search) ----------

data class IntentDecision(
    @SerializedName("decision")          val decision: String? = null,
    @SerializedName("message")           val message: String? = null,
    @SerializedName("require_ack_stage") val requireAckStage: Int? = null,
    @SerializedName("work_types")        val workTypes: List<String>? = null
)

// ---------- Dialog state (/api/v1/dialog/state) ----------

data class DialogStateResponse(
    @SerializedName("dialog_id")       val dialogId: String,
    @SerializedName("last_query")      val lastQuery: String?,
    @SerializedName("blocked")         val blocked: Boolean? = null,
    @SerializedName("requires_ack")    val requiresAck: Boolean? = null,
    @SerializedName("ack_stage")       val ackStage: Int? = null,
    @SerializedName("safety_summary")  val safetySummary: SafetySummary? = null,
    @SerializedName("events")          val events: List<JsonObject> = emptyList()
)
// ---------- Next-step (/api/v1/dialog/next-step) ----------

data class NextStepPayload(
    @SerializedName("dialog_id")      val dialogId: String?,
    @SerializedName("topic")          val topic: String?,
    @SerializedName("topic_intent")   val topicIntent: String?,
    @SerializedName("summary")        val summary: String?,
    @SerializedName("hazards_seen")   val hazardsSeen: List<String>?,
    @SerializedName("next_step_text") val nextStepText: String?,
    @SerializedName("raw")            val raw: String?,
    @SerializedName("_source")        val source: String?,
    @SerializedName("ack_stage")      val ackStage: Int?
)

data class NextStepResponse(
    @SerializedName("ok")             val ok: Boolean,
    @SerializedName("schema_version") val schemaVersion: String?,
    @SerializedName("dialog_id")      val dialogId: String,
    @SerializedName("next_step")      val nextStep: NextStepPayload?
)

// ---------- Session memory (/api/v1/rag/session/memory) ----------

data class MemorySource(
    @SerializedName("recent_events_used") val recentEventsUsed: Int?
)

data class SessionMemory(
    @SerializedName("dialog_id")      val dialogId: String,
    @SerializedName("updated_at")     val updatedAt: String?,
    @SerializedName("last_ack_stage") val lastAckStage: Int?,
    @SerializedName("hazards_seen")   val hazardsSeen: List<String>?,
    @SerializedName("key_points")     val keyPoints: List<String>?,
    @SerializedName("summary")        val summary: String?,
    @SerializedName("source")         val source: MemorySource?
)

data class SessionMemoryResponse(
    @SerializedName("ok")             val ok: Boolean,
    @SerializedName("schema_version") val schemaVersion: String?,
    @SerializedName("memory")         val memory: SessionMemory?
)
