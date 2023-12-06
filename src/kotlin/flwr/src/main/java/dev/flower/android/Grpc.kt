package dev.flower.android

import android.content.Context
import android.util.Log
import androidx.datastore.core.CorruptionException
import androidx.datastore.core.DataStore
import androidx.datastore.core.Serializer
import androidx.datastore.dataStore
import androidx.datastore.preferences.protobuf.InvalidProtocolBufferException
import flwr.proto.FleetGrpc
import flwr.proto.FleetOuterClass.CreateNodeRequest
import flwr.proto.FleetOuterClass.CreateNodeResponse
import flwr.proto.FleetOuterClass.DeleteNodeRequest
import flwr.proto.FleetOuterClass.DeleteNodeResponse
import flwr.proto.FleetOuterClass.PullTaskInsRequest
import flwr.proto.FleetOuterClass.PullTaskInsResponse
import flwr.proto.FleetOuterClass.PushTaskResRequest
import flwr.proto.FlowerServiceGrpc
import flwr.proto.NodeOuterClass.Node
import flwr.proto.TaskOuterClass.TaskIns
import flwr.proto.TaskOuterClass.TaskRes
import flwr.proto.Transport.ServerMessage
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import io.grpc.stub.StreamObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.CountDownLatch

object NodeSerializer : Serializer<Node> {
    override val defaultValue: Node = Node.getDefaultInstance()

    override suspend fun readFrom(input: InputStream): Node {
        try {
            return Node.parseFrom(input)
        } catch (exception: InvalidProtocolBufferException) {
            throw CorruptionException("Cannot read proto.", exception)
        }
    }

    override suspend fun writeTo(
        t: Node,
        output: OutputStream
    ) = t.writeTo(output)
}

val Context.nodeDataStore: DataStore<Node> by dataStore(
    fileName = "fleet.pb",
    serializer = NodeSerializer
)

internal class FlowerGrpc
@Throws constructor(
    channel: ManagedChannel,
    private val client: Client,
) {
    private val finishLatch = CountDownLatch(1)

    private val asyncStub = FlowerServiceGrpc.newStub(channel)!!

    private val requestObserver = asyncStub.join(object : StreamObserver<ServerMessage> {
        override fun onNext(msg: ServerMessage) {
            Log.i("Flower Grpc", "Receive message: $msg.")
            try {
                sendResponse(msg)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        override fun onError(t: Throwable) {
            t.printStackTrace()
            finishLatch.countDown()
        }

        override fun onCompleted() {
            finishLatch.countDown()
        }
    })!!

    fun sendResponse(msg: ServerMessage) {
        val response = handleLegacyMessage(client, msg)
        Log.i("Flower Grpc", "Send message: $response.")
        requestObserver.onNext(response.first)
    }
}

/**
 * Start a Flower client node which connects to a Flower server.
 *
 * @param serverAddress The IPv4 or IPv6 address of the server. If the Flower server runs on the
 * same machine on port 8080, then server_address would be “[::]:8080”.
 * @param useTLS Whether to use TLS to connect to the Flower server.
 * @param client The Flower client implementation.
 */
suspend fun startClient(
    serverAddress: String,
    useTls: Boolean,
    client: Client,
) {
    FlowerGrpc(createChannel(serverAddress, useTls), client)
}

internal suspend fun createChannel(address: String, useTLS: Boolean = false): ManagedChannel {
    val channelBuilder =
        ManagedChannelBuilder.forTarget(address).maxInboundMessageSize(HUNDRED_MEBIBYTE)
    if (!useTLS) {
        channelBuilder.usePlaintext()
    }
    return withContext(Dispatchers.IO) {
        channelBuilder.build()
    }
}

suspend fun startClient(host: String, port: Int, useTls: Boolean, client: Client) {
    FlowerGrpc(createChannel(host, port, useTls), client)
}

internal suspend fun createChannel(host: String, port: Int, useTls: Boolean): ManagedChannel {
    val channelBuilder =
        ManagedChannelBuilder.forAddress(host, port).maxInboundMessageSize(HUNDRED_MEBIBYTE)
    if (!useTls) {
        channelBuilder.usePlaintext()
    }
    return withContext(Dispatchers.IO) {
        channelBuilder.build()
    }
}


const val HUNDRED_MEBIBYTE = 100 * 1024 * 1024

internal class FlwrRere
@Throws constructor(
    channel: ManagedChannel,
    private val client: Client,
    private val context: Context
) {

    private val KEYNODE = "node"
    private val KEYTASKINS = "currentTaskIns"

    private val finishLatch = CountDownLatch(1)

    private val asyncStub = FleetGrpc.newStub(channel)

    private val state = mutableMapOf<String, TaskIns?>()
    private val nodeStore = mutableMapOf<String, Node?>()

    private fun createNode() {
        val createNodeRequest = CreateNodeRequest.newBuilder().build()

        try {
            asyncStub.createNode(createNodeRequest, object : StreamObserver<CreateNodeResponse> {
                override fun onNext(value: CreateNodeResponse?) {
                    value?.let { response ->
                        runBlocking {
                            context.nodeDataStore.updateData { node ->
                                node.toBuilder()
                                    .setNodeId(response.node.nodeId)
                                    .setAnonymous(response.node.anonymous)
                                    .build()
                            }
                        }

                    }
                }

                override fun onError(t: Throwable?) {
                    t?.printStackTrace()
                    finishLatch.countDown()
                }

                override fun onCompleted() {
                    finishLatch.countDown()
                }
            })
        }catch(e: Exception) {
            e.printStackTrace()
            Log.i("Flower Grpc", "Create node not implemented.")
        }
    }

    private suspend fun deleteNode() {
        context.nodeDataStore.data.collect { data ->
            val deleteNodeRequest = DeleteNodeRequest.newBuilder().setNode(data).build()
            asyncStub.deleteNode(deleteNodeRequest, object : StreamObserver<DeleteNodeResponse> {
                override fun onNext(value: DeleteNodeResponse?) {
                    runBlocking {
                        context.nodeDataStore.updateData { node ->
                            node.toBuilder()
                                .setNodeId(0)
                                .setAnonymous(false)
                                .build()
                        }
                    }
                }

                override fun onError(t: Throwable?) {
                    t?.printStackTrace()
                    finishLatch.countDown()
                }

                override fun onCompleted() {
                    finishLatch.countDown()
                }
            })
        }
    }

    private suspend fun request(requestChannel: Channel<PullTaskInsRequest>, node: Node?) {
        val request = if (node != null) {
            PullTaskInsRequest.newBuilder().setNode(node).build()
        } else {
            PullTaskInsRequest.newBuilder().build()
        }

        Log.i("Flower Grpc", "Sending request $request")
        requestChannel.send(request)
    }

    private suspend fun receive(requestChannel: Channel<PullTaskInsRequest>, node: Node?, withTimeout: Boolean = false) = flow {
        coroutineScope {
            var numberOfTries = 0
            val responses = Channel<TaskIns?>(1)
            for (request in requestChannel)
                asyncStub.pullTaskIns(request, object : StreamObserver<PullTaskInsResponse> {
                    override fun onNext(value: PullTaskInsResponse?) {
                        val taskIns = value?.let { getTaskIns(it) }
                        Log.i("Flower Grpc", "Receive $taskIns")
                        if (taskIns != null && validateTaskIns(taskIns, true)) {
                            state[KEYTASKINS] = taskIns
                            responses.trySend(taskIns).isSuccess
                        }

                    }

                    override fun onError(t: Throwable?) {
                        t?.printStackTrace()
                    }

                    override fun onCompleted() {
                    }
                })

            for (response in responses) {
                if (response == null) {
                    if (numberOfTries >= 10) {
                        cancel("Timeout")
                    }
                    delay(3000)
                    numberOfTries++
                    request(requestChannel, node)
                } else {
                    numberOfTries = 0
                    emit(response)
                }
            }
        }
    }

    suspend fun startGrpcRere() {
        createNode()

        val node: Node? = nodeStore[KEYNODE]
        if (node == null) {
            println("Node not available")
        }

        val requestChannel = Channel<PullTaskInsRequest>(1)
        request(requestChannel, node)
        receive(requestChannel, node)
            .collect {
                val (taskRes, sleepDuration, keepGoing) = handle(client, it)
                send(taskRes)
                delay(sleepDuration.toLong())
                if (keepGoing) {
                    request(requestChannel, node)
                } else {
                    deleteNode()
                    requestChannel.close()
                }
            }
    }

    private fun send(taskRes: TaskRes) {
        nodeStore[KEYNODE]?.let { node ->
            state[KEYTASKINS]?.let { taskIns ->
                if (validateTaskRes(taskRes)) {
                    taskRes
                        .toBuilder()
                        .setTaskId("")
                        .setGroupId(taskIns.groupId)
                        .setWorkloadId(taskIns.workloadId)
                        .task.toBuilder()
                        .setProducer(node)
                        .setConsumer(taskIns.task.producer)
                        .addAncestry(taskIns.taskId)
                    val request = PushTaskResRequest.newBuilder().addTaskResList(taskRes).build()
                    asyncStub.pushTaskRes(request, null)
                }
                state[KEYTASKINS] = null
            }
        }
    }
}

suspend fun startFlowerRere(
    serverAddress: String,
    useTLS: Boolean,
    client: Client,
    context: Context
) {
    FlwrRere(createChannel(serverAddress, useTLS), client, context).startGrpcRere()
}
