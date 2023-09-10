package dev.flower.android

import android.os.Build
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.util.UUID

enum class Event {
    START_CLIENT_ENTER, START_CLIENT_LEAVE
}

fun createEvent(event: Event) {
   mapOf(
        "context" to mapOf(
            "source" to UUID.randomUUID(),
            "cluster" to UUID.randomUUID().toString(),
            "date" to LocalDateTime.now(ZoneOffset.UTC).toString(),
            "flower" to mapOf(
                "package_name" to "flwr",
                "package_version" to "1.5.0",
            ),
            "hw" to mapOf(
                "cpu_count" to Runtime.getRuntime().availableProcessors(),
            ),
            "platform" to mapOf(
                "system" to Build.,
                "release" to System.getProperty("os.version"),
                "platform" to System.getProperty("os.name").toLowerCase(),
                "python_implementation" to System.getProperty("python.version_info.implementation"),
                "python_version" to System.getProperty("python.version"),
                "machine" to System.getProperty("os.arch"),
                "architecture" to System.getProperty("os.arch.data"),
                "version" to System.getProperty("os.version"),
            ),
        ),
    )
}
