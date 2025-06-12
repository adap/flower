package dev.flower.android

import flwr.proto.FleetGrpc
import flwr.proto.FleetOuterClass.CreateNodeRequest
import flwr.proto.FleetOuterClass.CreateNodeResponse
import flwr.proto.FleetOuterClass.DeleteNodeRequest
import flwr.proto.FleetOuterClass.DeleteNodeResponse
import flwr.proto.FleetOuterClass.PullTaskInsRequest
import flwr.proto.FleetOuterClass.PullTaskInsResponse
import flwr.proto.FleetOuterClass.PushTaskResRequest
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import io.grpc.stub.StreamObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.CountDownLatch
import flwr.proto.FlowerServiceGrpc
import flwr.proto.NodeOuterClass.Node
import flwr.proto.TaskOuterClass.TaskIns
import flwr.proto.TaskOuterClass.TaskRes
import flwr.proto.Transport.ServerMessage
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.flow

internal class FlowerGrpc
@Throws constructor(
    channel: ManagedChannel,
    private val client: Client,
) {
    private val finishLatch = CountDownLatch(1)

    private val asyncStub = FlowerServiceGrpc.newStub(channel)!!

    private val requestObserver = asyncStub.join(object : StreamObserver<ServerMessage> {
        override fun onNext(msg: ServerMessage) {
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

const val HUNDRED_MEBIBYTE = 100 * 1024 * 1024

internal class FlwrRere
@Throws constructor(
    channel: ManagedChannel,
    private val client: Client,
) {

    private val KEYNODE = "node"
    private val KEYTASKINS = "currentTaskIns"

    private val finishLatch = CountDownLatch(1)

    private val asyncStub = FleetGrpc.newStub(channel)

    private val state = mutableMapOf<String, TaskIns?>()
    private val nodeStore = mutableMapOf<String, Node?>()

    private fun createNode() {
        val createNodeRequest = CreateNodeRequest.newBuilder().build()

        asyncStub.createNode(createNodeRequest, object : StreamObserver<CreateNodeResponse> {
            override fun onNext(value: CreateNodeResponse?) {
                nodeStore[KEYNODE] = value?.node
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

    private fun deleteNode() {
        nodeStore[KEYNODE]?.let { node ->
            val deleteNodeRequest = DeleteNodeRequest.newBuilder().setNode(node).build()
            asyncStub.deleteNode(deleteNodeRequest, object : StreamObserver<DeleteNodeResponse> {
                override fun onNext(value: DeleteNodeResponse?) {
                    nodeStore[KEYNODE] = null
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

    private suspend fun request(requestChannel: Channel<PullTaskInsRequest>, node: Node) {
        val request = PullTaskInsRequest.newBuilder().setNode(node).build()
        requestChannel.send(request)
    }

    private suspend fun receive(requestChannel: Channel<PullTaskInsRequest>, node: Node) = flow {
        coroutineScope {
            val responses = Channel<TaskIns?>(1)
            for (request in requestChannel)
                asyncStub.pullTaskIns(request, object : StreamObserver<PullTaskInsResponse> {
                    override fun onNext(value: PullTaskInsResponse?) {
                        val taskIns = value?.let { getTaskIns(it) }
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
                    delay(3000)
                    request(requestChannel, node)
                } else {
                    emit(response)
                }
            }
        }
    }

    suspend fun startGrpcRere() {
        createNode()

        val node = nodeStore[KEYNODE]
        if (node == null) {
            println("Node not available")
            return
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

suspend fun createFlowerRere(
    serverAddress: String,
    useTLS: Boolean,
    client: Client,
) {
    FlwrRere(createChannel(serverAddress, useTLS), client)
}
