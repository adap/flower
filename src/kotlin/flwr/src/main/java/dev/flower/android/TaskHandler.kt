package dev.flower.android

import flwr.proto.FleetOuterClass.PullTaskInsResponse
import flwr.proto.TaskOuterClass.Task
import flwr.proto.TaskOuterClass.TaskIns
import flwr.proto.TaskOuterClass.TaskRes
import flwr.proto.Transport.ClientMessage
import flwr.proto.Transport.ServerMessage

internal fun validateTaskIns(taskIns: TaskIns, discardReconnectIns: Boolean): Boolean {
    if (!taskIns.hasTask() || (!taskIns.task.hasLegacyServerMessage() && !taskIns.task.hasSa()) ||
        (discardReconnectIns && taskIns.task.legacyServerMessage.msgCase == ServerMessage.MsgCase.RECONNECT_INS)
    ) {
        return false
    }
    return true
}

internal fun validateTaskRes(taskRes: TaskRes): Boolean {
    return taskRes.hasTask()
}

internal fun getTaskIns(pullTaskInsResponse: PullTaskInsResponse): TaskIns? {
    if (pullTaskInsResponse.taskInsListCount == 0) {
        return null
    }
    return pullTaskInsResponse.getTaskInsList(0)
}

internal fun getServerMessageFromTaskIns(taskIns: TaskIns, excludeReconnectIns: Boolean): ServerMessage? {
    if (!validateTaskIns(taskIns, excludeReconnectIns) || !taskIns.task.hasLegacyServerMessage()) {
        return null
    }
    return taskIns.task.legacyServerMessage
}

internal fun wrapClientMessageInTaskRes(clientMessage: ClientMessage): TaskRes {
    return TaskRes.newBuilder()
        .setTaskId("")
        .setGroupId("")
        .setWorkloadId(0)
        .setTask(Task.newBuilder().addAllAncestry(emptyList()).setLegacyClientMessage(clientMessage))
        .build()
}
