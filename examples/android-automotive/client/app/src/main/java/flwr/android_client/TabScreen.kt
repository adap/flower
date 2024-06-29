package flwr.android_client

import android.os.Handler
import android.text.TextUtils
import android.util.Log
import androidx.car.app.CarContext
import androidx.car.app.CarToast
import androidx.car.app.Screen
import androidx.car.app.model.Action
import androidx.car.app.model.CarColor
import androidx.car.app.model.CarIcon
import androidx.car.app.model.ItemList
import androidx.car.app.model.ListTemplate
import androidx.car.app.model.MessageTemplate
import androidx.car.app.model.Pane
import androidx.car.app.model.PaneTemplate
import androidx.car.app.model.Row
import androidx.car.app.model.Tab
import androidx.car.app.model.TabContents
import androidx.car.app.model.TabTemplate
import androidx.car.app.model.TabTemplate.TabCallback
import androidx.car.app.model.Template
import androidx.core.graphics.drawable.IconCompat
import dev.flower.flower_tflite.FlowerClient
import dev.flower.flower_tflite.FlowerServiceRunnable
import dev.flower.flower_tflite.SampleSpec
import dev.flower.flower_tflite.createFlowerService
import dev.flower.flower_tflite.helpers.classifierAccuracy
import dev.flower.flower_tflite.helpers.loadMappedAssetFile
import dev.flower.flower_tflite.helpers.categoricalCrossEntropyLoss
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

import java.util.*

// TabScreen: shows the status of the engine, settings, and logs
class TabScreen(carContext: CarContext) : Screen(carContext) {
    object EngineStatusService {
        var isEngineOK: Boolean = true // Engine status, true if OK, false if not OK
            private set

        val engineProblem: String // Error message to show when the engine is not OK
            get() = if (isEngineOK) "" else "Engine overheating"

        fun simulateEngineStatusChange() {
            isEngineOK = !isEngineOK // Simulate engine status change
        }
    }

    private val mTabs: MutableMap<String, Tab> = HashMap() // Tabs map
    private val mTabContentsMap: MutableMap<String, TabContents> = HashMap() // Tab contents map
    private var mTabTemplateBuilder: TabTemplate.Builder? = null // Tab template builder
    private var mActiveContentId: String? = null // Active content ID

    private var flEnabled = false // Federated Learning enabled
    private var notificationsEnabled = false // Notifications enabled

    private val logs = StringBuilder() // Logs buffer

    private val handler = Handler()
    private var engineCheckRunnable: Runnable? = null

    private val scope = MainScope()
    lateinit var flowerClient: FlowerClient<FeatureArray, FloatArray>
    lateinit var flowerServiceRunnable: FlowerServiceRunnable<FeatureArray, FloatArray>
    lateinit var db: Db

    init {
        initializeEngineCheckRunnable()
        createFlowerClient()
        scope.launch {
            loadDataInBackground()
        }
    }

    override fun onGetTemplate(): Template {
        mTabTemplateBuilder = TabTemplate.Builder(object : TabCallback {
            override fun onTabSelected(tabContentId: String) {
                mActiveContentId = tabContentId
                invalidate()
            }
        }).setHeaderAction(Action.APP_ICON)

        mTabContentsMap.clear()
        mTabs.clear()

        for (i in ICON_RES_IDS.indices) {
            val contentId = i.toString()
            var contentTemplate = when (i) {
                0 -> createShortMessageTemplate()
                1 -> createPaneTemplate("Settings Content")
                2 -> createLogsTemplate()
                else -> createLogsTemplate()
            }
            val tabContents = TabContents.Builder(contentTemplate).build()
            mTabContentsMap[contentId] = tabContents

            val tabBuilder = Tab.Builder()
                .setTitle(TITLES[i])
                .setIcon(
                    CarIcon.Builder(
                        IconCompat.createWithResource(
                            carContext,
                            ICON_RES_IDS[i]
                        )
                    ).build()
                )
                .setContentId(contentId)
            if (TextUtils.isEmpty(mActiveContentId) && i == 0) {
                mActiveContentId = contentId
                mTabTemplateBuilder!!.setTabContents(tabContents)
            } else if (TextUtils.equals(mActiveContentId, contentId)) {
                mTabTemplateBuilder!!.setTabContents(tabContents)
            }

            val tab = tabBuilder.build()
            mTabs[tab.contentId] = tab
            mTabTemplateBuilder!!.addTab(tab)
        }
        return mTabTemplateBuilder!!
            .setActiveTabContentId(mActiveContentId!!)
            .build()
    }

    private fun createLogsTemplate(): ListTemplate {
        val listBuilder = ItemList.Builder()
        val logEntries = logs.toString().split("\n".toRegex()).dropLastWhile { it.isEmpty() }
            .toTypedArray()
        for (entry in logEntries) {
            if (!entry.isEmpty()) {
                listBuilder.addItem(Row.Builder().setTitle(entry).build())
            }
        }
        return ListTemplate.Builder()
            .setSingleList(listBuilder.build())
            .build()
    }

    private fun buildRowForTemplate(title: CharSequence, clickable: Boolean): Row {
        val rowBuilder = Row.Builder()
            .setTitle(title)
        if (clickable) {
            rowBuilder.setOnClickListener {
                CarToast.makeText(
                    carContext, title,
                    CarToast.LENGTH_SHORT
                ).show()
            }
        }
        return rowBuilder.build()
    }

    private fun createShortMessageTemplate(): MessageTemplate {
        val isEngineOK = EngineStatusService.isEngineOK
        val message =
            if (isEngineOK) "Engine is running smoothly" else "Problem: " + EngineStatusService.engineProblem

        val iconResId = if (isEngineOK) R.drawable.ic_engine else R.drawable.ic_engine
        val icon = CarIcon.Builder(IconCompat.createWithResource(carContext, iconResId)).build()

        return MessageTemplate.Builder(message)
            .setIcon(icon)
            .build()
    }

    private fun createPaneTemplate(paneTitle: String): PaneTemplate {
        val paneBuilder = Pane.Builder()
            .setImage(
                CarIcon.Builder(
                    IconCompat.createWithResource(
                        carContext,
                        R.drawable.ic_logo
                    )
                ).build()
            )

        paneBuilder.addRow(
            Row.Builder()
                .setTitle("Informed Consent")
                .addText("This app uses Federated Learning to improve engine fault predictions, making your car safer and more reliable. By giving your consent, you allow your car to contribute to this effort without compromising your privacy. Only the insights learned from your car's data, not the raw data itself, are shared with others.")
                .build()
        )

        // Pulsante per attivare il Federated Learning
        paneBuilder.addAction(
            Action.Builder()
                .setTitle("Consent Federated Learning")
                .setOnClickListener {
                    flEnabled = true
                    logAction("[SYSTEM] Show Notification switched to " + (if (true) "ON" else "OFF"))
                    invalidate()
                }
                .build()
        )
        // Pulsante per attivare la Notifica quando si sta facendo training della rete
        paneBuilder.addAction(
            createSwitchAction("Show Notification: ", notificationsEnabled, object : SwitchClickListener {
                override fun onSwitchClick(isOn: Boolean) {
                    notificationsEnabled = !notificationsEnabled
                    logAction("[SYSTEM] Show Notification switched to " + (if (!isOn) "ON" else "OFF"))
                    invalidate()
                }
            })
        )

        return PaneTemplate.Builder(paneBuilder.build())
            .build()
    }

    private fun initializeEngineCheckRunnable() {
        engineCheckRunnable = object : Runnable {
            override fun run() {
                val isEngineOK = EngineStatusService.isEngineOK
                val statusMessage =
                    if (isEngineOK) "Engine is running smoothly" else "Problem: " + EngineStatusService.engineProblem
                val logTag = if (isEngineOK) "" else "[ERROR]"
                logAction("$logTag $statusMessage")
                EngineStatusService.simulateEngineStatusChange() // Simulazione del cambiamento di stato del motore
                invalidate()
                handler.postDelayed(this, ENGINE_CHECK_INTERVAL.toLong())
            }
        }
        handler.post(engineCheckRunnable as Runnable) // Avvio del controllo periodico dello stato del motore
    }

    private fun logAction(action: String) {
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
        logs.append(action).append(" at [").append(timestamp).append("]\n")
    }

    private fun createSwitchAction(
        title: String,
        isOn: Boolean,
        listener: SwitchClickListener
    ): Action {
        return Action.Builder()
            .setTitle(title + " " + (if (isOn) "ON" else "OFF"))
            .setOnClickListener {
                val newIsOn = !isOn
                CarToast.makeText(
                    carContext,
                    title + " " + (if (newIsOn) "ON" else "OFF"),
                    CarToast.LENGTH_SHORT
                ).show()
                listener.onSwitchClick(newIsOn)
            }
            .build()
    }

    private fun createFabBackAction(): Action {
        val action = Action.Builder()
            .setIcon(CarIcon.BACK)
            .setBackgroundColor(CarColor.BLUE)
            .setOnClickListener { screenManager.pop() }
            .build()
        return action
    }

    private interface SwitchClickListener {
        fun onSwitchClick(isOn: Boolean)
    }

    companion object {
        private const val LIST_SIZE = 10

        private const val ENGINE_CHECK_INTERVAL =
            5000 // Intervallo di controllo motore in millisecondi

        // Titoli delle Tab
        private val TITLES = arrayOf(
            "Engine",
            "Settings",
            "Logs"
        )

        // Icone per le tabelle
        private val ICON_RES_IDS = intArrayOf(
            R.drawable.ic_engine,
            R.drawable.ic_settings,
            R.drawable.ic_logs
        )
    }

    private fun createFlowerClient() {
        val buffer = loadMappedAssetFile(carContext, "model/enginefaultdb.tflite")
        val layersSizes = intArrayOf(504,36,144,16)
        val sampleSpec = SampleSpec<FeatureArray, FloatArray>(
            { it.toTypedArray() },
            { it.toTypedArray() },
            {Array(it) { FloatArray(CLASSES.size) } },
            ::categoricalCrossEntropyLoss,
            ::classifierAccuracy,
        )
        flowerClient = FlowerClient(buffer, layersSizes, sampleSpec)
    }

    suspend fun runWithStacktrace(call: suspend () -> Unit) {
        try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
        }
    }

    suspend fun <T> runWithStacktraceOr(or: T, call: suspend () -> T): T {
        return try {
            call()
        } catch (err: Error) {
            Log.e(TAG, Log.getStackTraceString(err))
            or
        }
    }

    suspend fun loadDataInBackground() {
        val result = runWithStacktraceOr("Failed to load training dataset.") {
            loadData(carContext, flowerClient, DEVICE_ID)
            "Training dataset is loaded in memory. Ready to train!"
        }
        CarToast.makeText(carContext, result, CarToast.LENGTH_SHORT).show()

        scope.launch {
            runGrpcInBackground(IP, PORT)
        }
        //trainButton.isEnabled = true
    }

    suspend fun runGrpcInBackground(host: String, port: Int) {
        val address = "dns:///$host:$port"
        //TODO: add error handling if connection fails
        runWithStacktraceOr("Failed to connect to the FL server \n") {
             createFlowerService(address, false, flowerClient) {
                 println("Result of Flower Service's creation: $it")
                 CarToast.makeText(carContext, it, CarToast.LENGTH_SHORT).show()
            }
            "Connection to the FL server successful \n"
        }
    }
}

private const val TAG = "TabScreen"

typealias FeatureArray = FloatArray

private val DEVICE_ID = 1
private val IP = "10.0.2.2"
private const val PORT = 8080
