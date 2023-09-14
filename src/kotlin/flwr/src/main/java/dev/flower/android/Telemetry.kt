package dev.flower.android

import android.os.Build
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.util.UUID

@Serializable
data class Payload(@SerialName("event_type")
                   val eventType: Event,
                   @SerialName("event_details")
                   val eventDetails: Map<String, String>,
                   val context: Context)

enum class Event {
    START_CLIENT_ENTER, START_CLIENT_LEAVE
}

@Serializable
data class Context(val source: String,
                   val cluster: String,
                   val date: String,
                   val flower: Flower,
                   val hw: HW,
                   val platform: Platform)

@Serializable
data class Flower(@SerialName("package_name")
                  val packageName: String,
                  @SerialName("package_version")
                  val packageVersion: String)

@Serializable
data class HW(@SerialName("cpu_count")
              val cpuCount: String)

@Serializable
data class Platform(val system: String,
                    val release: String,
                    val platform: String,
                    @SerialName("python_implementation")
                    val pythonImplementation: String,
                    @SerialName("python_version")
                    val pythonVersion: String,
                    @SerialName("android_sdk_version")
                    val androidSdkVersion: String,
                    val machine: String,
                    val architecture: String,
                    val version: String)

fun getSourceId(homeDir: File): UUID {
    val flwrFile = File(homeDir, ".flwr")
    if (flwrFile.exists()) {
        return UUID.fromString(flwrFile.readText().trim())
    } else {
        val uuid = UUID.randomUUID()
        flwrFile.createNewFile()
        flwrFile.writeText(uuid.toString())
        return uuid
    }
}

fun createEventContext(homeDir: File): Context {
    val date = LocalDateTime.now(ZoneOffset.UTC).toString()
    val version = "0.0.2"
    val hw = HW(Runtime.getRuntime().availableProcessors().toString())
    val platform = Platform(
        "Android",
        Build.VERSION.RELEASE.toString(),
        Build.FINGERPRINT.toString(),
        "",
        "",
        Build.VERSION.SDK_INT.toString(),
        Build.SUPPORTED_ABIS[0].toString(),
        "",
        ""
    )
    val flower = Flower("flwr", version)

    return Context(getSourceId(homeDir).toString(), UUID.randomUUID().toString(), date, flower, hw, platform)
}


fun createEvent(event: Event, eventDetails: Map<String, String>, context: android.content.Context) {
    val homeDir = context.getExternalFilesDir(null)
    val JSON = "application/json; charset=utf-8".toMediaType()
    val payload = homeDir?.let { createEventContext(it) }?.let { Payload(event, eventDetails, it) }
    val json = Json.encodeToString(payload)
    val url = "https://telemetry.flower.dev/api/v1/event"
    val client = OkHttpClient()

    val body = json.toRequestBody(JSON);
    val request = Request.Builder()
        .url(url)
        .post(body)
        .build();

    client.newCall(request).execute().use { response ->
        println(response.body?.string())
    }
}
