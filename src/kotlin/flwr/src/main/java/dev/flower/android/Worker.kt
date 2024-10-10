package dev.flower.android

import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.PeriodicWorkRequest
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import java.util.UUID
import java.util.concurrent.TimeUnit

internal class BackgroundWorker(private val context: Context,
                 workerParams: WorkerParameters,
                 private val client: Client,
                 private val grpcRere: Boolean) : CoroutineWorker(context, workerParams) {

    override suspend fun doWork(): Result {
        inputData.getString("serverAddress")?.let { startClient(it, false, client, grpcRere) }
        val workManager: WorkManager = WorkManager.getInstance(context)
        workManager.cancelWorkById(id)
        return Result.success()
    }
}

/**
* Start a PeriodicWorker that runs Flower client in the background.
*
* @param interval The interval for the PeriodicWorker to resume its work periodically in minutes.
* @param serverAddress The IPv4 or IPv6 address of the server. If the Flower server runs on the
* same machine on port 8080, then server_address would be “[::]:8080”.
*/
fun startPeriodicWorker(interval: Long, serverAddress: String) {
    val constraints: Constraints = Constraints.Builder().build()

    val workRequest: PeriodicWorkRequest = PeriodicWorkRequestBuilder<BackgroundWorker>(
        interval, TimeUnit.MINUTES
    )
        .setInitialDelay(0, TimeUnit.MILLISECONDS)
        .setInputData(
            Data.Builder()
                .putString("serverAddress", serverAddress)
                .build()
        )
        .setConstraints(constraints)
        .build()

    val uniqueWorkName = "Worker${UUID.randomUUID()}"

    WorkManager.getInstance(context)
        .enqueueUniquePeriodicWork(uniqueWorkName, ExistingPeriodicWorkPolicy.KEEP, workRequest)
}
