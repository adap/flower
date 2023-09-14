package dev.flower.android

import android.os.Build
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Serializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import java.text.SimpleDateFormat
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.util.Date
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

@Serializer(forClass = Date::class)
object DateSerializer : KSerializer<Date> {

    private val df = SimpleDateFormat("yyyy-MM-dd HH:mm:ss")

    override val descriptor = PrimitiveSerialDescriptor("String", PrimitiveKind.STRING)
    override fun serialize(encoder: Encoder, value: Date) = encoder.encodeString(df.format(value))
    override fun deserialize(decoder: Decoder): Date = df.parse(decoder.decodeString())
}
@Serializable
data class Context(val source: String,
                   val cluster: String,
                   @Serializable(DateSerializer::class)
                   val date: Date,
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
                "system" to Build.VERSION.CODENAME,
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
