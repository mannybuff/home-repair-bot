// File: app/src/main/java/com/diy_ai/homerepairbot/net/Repository.kt
package com.diy_ai.homerepairbot.net

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MultipartBody
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import okhttp3.RequestBody

class Repository(
    private val api: ManbproApi = ApiConfig.retrofit().create(ManbproApi::class.java)
) {

    // ---------- Health ----------

    suspend fun healthz(): JsonObject = api.healthz()

    // ---------- Session index & state ----------

    suspend fun getSessionIndex(): JsonObject = api.getSessionIndex()

    suspend fun getDialogState(dialogId: String): DialogStateResponse =
        api.getDialogState(dialogId)

    suspend fun getDialogStateJson(dialogId: String): JsonObject {
        val body = api.resumeSession(dialogId)
        val text = body.string()
        return runCatching { JsonParser.parseString(text).asJsonObject }
            .getOrElse { JsonObject() }
    }

    // ---------- Orchestrator dialog (v2 unified entry) ----------

    suspend fun dialogTextOnly(
        text: String,
        dialogId: String
    ): OrchestratorResponse {
        val mediaType = "text/plain".toMediaType()
        val textPart = text.toRequestBody(mediaType)
        val dialogIdPart = dialogId.toRequestBody(mediaType)

        return api.orchestratorDialog(
            image = null,
            text = textPart,
            dialogId = dialogIdPart
        )
    }

    suspend fun dialogWithImage(
        imagePart: MultipartBody.Part,
        text: String,
        dialogId: String
    ): OrchestratorResponse {
        val mediaType = "text/plain".toMediaType()
        val textPart = text.toRequestBody(mediaType)
        val dialogIdPart = dialogId.toRequestBody(mediaType)

        return api.orchestratorDialog(
            image = imagePart,
            text = textPart,
            dialogId = dialogIdPart
        )
    }

    // ---------- Next-step and memory (kept for future use) ----------

    suspend fun dialogNextStep(
        dialogId: String,
        text: String,
        caption: String?,
        ackStage: Int
    ): NextStepResponse {
        val mediaType = "text/plain".toMediaType()

        val dialogIdPart = dialogId.toRequestBody(mediaType)
        val textPart = text.toRequestBody(mediaType)
        val captionPart = caption?.toRequestBody(mediaType)
        val ackStagePart = ackStage.toString().toRequestBody(mediaType)

        return api.dialogNextStep(
            dialogId = dialogIdPart,
            text = textPart,
            caption = captionPart,
            ackStage = ackStagePart
        )
    }

    suspend fun getSessionMemory(dialogId: String): SessionMemoryResponse {
        return api.sessionMemory(dialogId)
    }
}
