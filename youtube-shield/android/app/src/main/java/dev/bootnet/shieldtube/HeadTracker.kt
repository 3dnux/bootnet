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
 * Entrega el yaw (grados, positivo = izquierda) en el hilo principal; todo el
 * cálculo ocurre también en el hilo principal.
 */
class HeadTracker(private val context: Context, private val onYaw: (Double) -> Unit) {
    enum class Source { SENSOR, CAMERA }

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
            emit(recenter.update(yaw, dt))
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

    private fun emit(yaw: Double) {
        val smooth = smoother.update(yaw.coerceIn(-90.0, 90.0))
        main.post { onYaw(smooth) }
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

        fun start() {
            registry.currentState = Lifecycle.State.CREATED
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
                }.onFailure { Log.w(TAG, "No se pudo abrir la cámara frontal", it) }
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
                        emit(HeadMath.yawFromCameraFace(face.headEulerAngleY))
                    } else if (++missed == 8) {
                        emit(0.0) // sin cara ~0,5 s: el sonido vuelve al frente
                    }
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
