package dev.bootnet.shieldtube

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import com.google.android.gms.common.moduleinstall.InstallStatusListener
import com.google.android.gms.common.moduleinstall.ModuleInstall
import com.google.android.gms.common.moduleinstall.ModuleInstallRequest
import com.google.android.gms.common.moduleinstall.ModuleInstallStatusUpdate
import com.google.android.gms.common.moduleinstall.ModuleInstallStatusUpdate.InstallState
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.Executors

/**
 * Seguimiento de cabeza para el Sonido 3D. Dos fuentes:
 *
 * - Sensor de los audífonos (Sensor.TYPE_HEAD_TRACKER, p. ej. Sony WH-1000XM5).
 *   Android lo reserva al sistema; solo aparece si se levanta la restricción con
 *   `adb shell cmd sensorservice unrestrict-ht` (hasta reiniciar). Funciona
 *   también con la pantalla apagada.
 * - Cámara frontal + detección facial de ML Kit, en el propio móvil (las
 *   imágenes no se guardan ni se envían). Solo mientras se mira la pantalla.
 *
 * Entrega el yaw (grados, positivo = izquierda) y un texto de estado para la
 * pantalla, ambos en el hilo principal (donde ocurre también todo el cálculo).
 */
class HeadTracker(
    private val context: Context,
    private val onYaw: (Double) -> Unit,
    private val onStatus: (String) -> Unit,
) {
    private val main = Handler(Looper.getMainLooper())
    private val sensors = context.getSystemService(SensorManager::class.java)
    private val smoother = HeadMath.Smoother()
    private val recenter = HeadMath.Recenter()
    private var lastSensorNanos = 0L
    private var sensor: Sensor? = null
    private var sensorWanted = false
    private var callbackRegistered = false
    private var camera: CameraSession? = null

    /** El sensor de los audífonos, si Android (13+) deja usarlo a las apps. */
    fun headSensor(): Sensor? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        sensors.getDynamicSensorList(Sensor.TYPE_HEAD_TRACKER).firstOrNull()
    } else {
        null
    }

    private val dynamicCallback = object : SensorManager.DynamicSensorCallback() {
        override fun onDynamicSensorConnected(connected: Sensor) {
            if (sensorWanted && connected == headSensor()) startSensor()
        }

        override fun onDynamicSensorDisconnected(disconnected: Sensor) {
            if (disconnected != sensor) return
            sensors.unregisterListener(sensorListener)
            sensor = null
            center()
        }
    }

    private val sensorListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && event.firstEventAfterDiscontinuity) {
                recenter.reset()
            }
            val dt = if (lastSensorNanos == 0L) 0.0 else (event.timestamp - lastSensorNanos) / 1e9
            lastSensorNanos = event.timestamp
            val yaw = HeadMath.yawFromEulerVector(
                event.values[0].toDouble(), event.values[1].toDouble(), event.values[2].toDouble(),
            )
            val smooth = emit(recenter.update(yaw, dt))
            status(R.string.head_status_sensor, smooth.toInt())
        }

        override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit
    }

    /** Sensor de los audífonos: sigue funcionando en segundo plano. */
    fun startSensor(): Boolean {
        sensorWanted = true
        if (!callbackRegistered) {
            sensors.registerDynamicSensorCallback(dynamicCallback, main)
            callbackRegistered = true
        }
        if (sensor != null) return true
        val found = headSensor() ?: return false
        sensor = found
        recenter.reset()
        lastSensorNanos = 0L
        sensors.registerListener(sensorListener, found, SensorManager.SENSOR_DELAY_GAME, main)
        Log.i(TAG, "Seguimiento de cabeza con el sensor de ${found.name}")
        return true
    }

    fun stopSensor() {
        sensorWanted = false
        if (callbackRegistered) {
            sensors.unregisterDynamicSensorCallback(dynamicCallback)
            callbackRegistered = false
        }
        if (sensor == null) return
        sensors.unregisterListener(sensorListener)
        sensor = null
        center()
    }

    /** Cámara frontal: solo con la app en pantalla (Android no deja usarla en segundo plano). */
    fun startCamera() {
        if (camera == null) camera = CameraSession().also { it.start() }
    }

    fun stopCamera() {
        val session = camera ?: return
        session.stop()
        camera = null
        center()
    }

    fun stop() {
        stopSensor()
        stopCamera()
    }

    private fun center() {
        smoother.reset()
        main.post { onYaw(0.0) }
    }

    private fun emit(yaw: Double): Double {
        val smooth = smoother.update(yaw.coerceIn(-90.0, 90.0))
        main.post { onYaw(smooth) }
        return smooth
    }

    private fun status(resId: Int, vararg args: Any) {
        val text = context.getString(resId, *args)
        main.post { onStatus(text) }
    }

    /** Cámara frontal a baja resolución (~15 fps) + ML Kit en modo rápido. */
    private inner class CameraSession : LifecycleOwner {
        private val registry = LifecycleRegistry(this)
        private val executor = Executors.newSingleThreadExecutor()
        private val detector = FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setMinFaceSize(0.2f)
                .build(),
        )
        private var lastFrame = 0L
        private var missed = 0

        override val lifecycle: Lifecycle get() = registry

        /**
         * El modelo de detección facial lo pone Google Play Services. Con el APK
         * instalado a mano (no desde Play Store) puede no estar descargado: se pide
         * explícitamente y se espera a que termine.
         */
        fun start() {
            registry.currentState = Lifecycle.State.CREATED
            status(R.string.head_status_starting)
            val modules = ModuleInstall.getClient(context)
            modules.areModulesAvailable(detector)
                .addOnSuccessListener { response ->
                    if (response.areModulesAvailable()) openCamera() else downloadModel()
                }
                .addOnFailureListener { error ->
                    Log.w(TAG, "Google Play Services no responde", error)
                    status(R.string.head_status_no_play_services)
                }
        }

        private fun downloadModel() {
            status(R.string.head_status_downloading, 0)
            val modules = ModuleInstall.getClient(context)
            val listener = object : InstallStatusListener {
                override fun onInstallStatusUpdated(update: ModuleInstallStatusUpdate) {
                    val progress = update.progressInfo
                    if (progress != null && progress.totalBytesToDownload > 0) {
                        status(
                            R.string.head_status_downloading,
                            (progress.bytesDownloaded * 100 / progress.totalBytesToDownload).toInt(),
                        )
                    }
                    when (update.installState) {
                        InstallState.STATE_COMPLETED -> {
                            modules.unregisterListener(this)
                            if (registry.currentState != Lifecycle.State.DESTROYED) openCamera()
                        }
                        InstallState.STATE_FAILED, InstallState.STATE_CANCELED -> {
                            modules.unregisterListener(this)
                            status(R.string.head_status_download_failed)
                        }
                        else -> Unit
                    }
                }
            }
            val request = ModuleInstallRequest.newBuilder().addApi(detector).setListener(listener, main::post).build()
            modules.installModules(request)
                .addOnSuccessListener { response ->
                    if (response.areModulesAlreadyInstalled()) {
                        modules.unregisterListener(listener)
                        openCamera()
                    }
                }
                .addOnFailureListener { error ->
                    Log.w(TAG, "No se pudo descargar el modelo facial", error)
                    status(R.string.head_status_download_failed)
                }
        }

        private fun openCamera() {
            if (registry.currentState == Lifecycle.State.DESTROYED) return
            status(R.string.head_status_looking)
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener({
                if (registry.currentState == Lifecycle.State.DESTROYED) return@addListener
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(Size(480, 360), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                            )
                            .build(),
                    )
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(executor, ::analyze)
                runCatching {
                    future.get().bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, analysis)
                    registry.currentState = Lifecycle.State.RESUMED
                }.onFailure {
                    Log.w(TAG, "No se pudo abrir la cámara frontal", it)
                    status(R.string.head_status_camera_failed)
                }
            }, main::post)
        }

        @SuppressLint("UnsafeOptInUsageError")
        private fun analyze(proxy: androidx.camera.core.ImageProxy) {
            val now = System.nanoTime()
            val image = proxy.image
            if (image == null || now - lastFrame < 66_000_000L) {
                proxy.close()
                return
            }
            lastFrame = now
            detector.process(InputImage.fromMediaImage(image, proxy.imageInfo.rotationDegrees))
                .addOnSuccessListener { faces ->
                    val face = faces.maxByOrNull { it.boundingBox.width() }
                    if (face != null) {
                        missed = 0
                        val yaw = emit(HeadMath.yawFromCameraFace(face.headEulerAngleY))
                        status(R.string.head_status_camera, yaw.toInt())
                    } else if (++missed == 8) {
                        emit(0.0) // sin cara ~0,5 s: el sonido vuelve al frente
                        status(R.string.head_status_looking)
                    }
                }
                .addOnFailureListener { error ->
                    Log.w(TAG, "Fallo de la detección facial", error)
                    status(R.string.head_status_detector_failed)
                }
                .addOnCompleteListener { proxy.close() }
        }

        fun stop() {
            registry.currentState = Lifecycle.State.DESTROYED
            ProcessCameraProvider.getInstance(context).addListener({
                runCatching { ProcessCameraProvider.getInstance(context).get().unbindAll() }
                detector.close()
                executor.shutdown()
            }, main::post)
        }
    }

    private companion object {
        const val TAG = "ShieldTube"
    }
}
