package com.diy_ai.homerepairbot.util

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import java.util.UUID

class SessionStore(private val context: Context) {

    companion object {
        const val NEW_SESSION_LABEL = "New Session"

        private const val PREF_NAME = "session_store"
        private const val KEY_LIST  = "dialog_ids_csv"
        private const val MAX_ITEMS = 30
    }

    private fun prefs(): SharedPreferences =
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    private fun load(): List<String> {
        val csv = prefs().getString(KEY_LIST, "") ?: ""
        return csv.split(',')
            .map { it.trim() }
            .filter { it.startsWith("dlg-") }
            .distinct()
    }

    private fun save(list: List<String>) {
        val csv = list.joinToString(",")
        prefs().edit { putString(KEY_LIST, csv) }
    }

    fun list(): List<String> = load()

    fun add(id: String) {
        if (!id.startsWith("dlg-")) return
        val set = LinkedHashSet(load())
        set.remove(id)
        val list = mutableListOf(id)
        list.addAll(set)
        save(list.take(MAX_ITEMS))
    }

    fun clear() {
        save(emptyList())
    }

    /**
     * Hybrid model helper:
     *  - generate a new dialog id (dlg-xxxxxxxx)
     *  - store it locally
     *  - return it to the caller so backend can start using it.
     */
    fun createDialog(): String {
        val hex = UUID.randomUUID().toString().replace("-", "").take(8)
        val id = "dlg-$hex"
        add(id)
        return id
    }
}
