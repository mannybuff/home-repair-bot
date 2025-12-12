package com.diy_ai.homerepairbot.net

// Keep this file for UI helpers only. No data classes here to avoid redeclarations.

fun Any.prettyText(): String = when (this) {
    is String -> this
    is List<*> -> this.joinToString("\n") { it?.toString() ?: "" }
    is Map<*, *> -> this.entries.joinToString("\n") { "${it.key}: ${it.value}" }
    else -> this.toString()
}

fun Any.bullets(): String = when (this) {
    is List<*> -> this.joinToString("\n") { "• ${it?.toString() ?: ""}" }
    is String -> "• $this"
    else -> "• $this"
}
