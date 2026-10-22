package dev.bootnet.shieldtube

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Cálculos del seguimiento de cabeza (sin dependencias de Android, para poder
 * testearlos). Convención de toda la app: yaw en grados, positivo = cabeza
 * girada hacia la izquierda del usuario.
 */
object HeadMath {
    private const val DEG = 180.0 / Math.PI

    /** Lleva un ángulo a [-180, 180). */
    fun wrap(degrees: Double): Double {
        var d = (degrees + 180.0) % 360.0
        if (d < 0) d += 360.0
        return d - 180.0
    }

    /**
     * Yaw a partir del vector de Euler de Sensor.TYPE_HEAD_TRACKER (ejes de la
     * cabeza: X oreja derecha, Y nariz, Z coronilla). El vector es la rotación
     * del marco de referencia a la cabeza; se gira el eje Y (la nariz) y se mide
     * su ángulo en el plano horizontal.
     */
    fun yawFromEulerVector(rx: Double, ry: Double, rz: Double): Double {
        val angle = sqrt(rx * rx + ry * ry + rz * rz)
        if (angle < 1e-9) return 0.0
        val kx = rx / angle
        val ky = ry / angle
        val kz = rz / angle
        val c = cos(angle)
        val s = sin(angle)
        // Rodrigues aplicado a (0, 1, 0): v·c + (k×v)·s + k(k·v)(1-c)
        val fx = -kz * s + kx * ky * (1 - c)
        val fy = c + ky * ky * (1 - c)
        return atan2(-fx, fy) * DEG
    }

    /**
     * Yaw a partir de la cara detectada por la cámara frontal (ML Kit:
     * headEulerAngleY positivo = cara girada hacia la derecha de la imagen; en la
     * imagen sin espejo de la cámara frontal, la derecha de la imagen es la
     * izquierda del usuario).
     */
    fun yawFromCameraFace(headEulerAngleY: Float): Double = headEulerAngleY.toDouble()

    /**
     * Para el sensor de los audífonos, cuyo marco de referencia es arbitrario:
     * el "frente" se recentra despacio hacia donde mira el usuario (como el
     * audio espacial de Apple), así el sonido sigue delante si cambias de postura.
     */
    class Recenter(private val timeConstantSeconds: Double = 8.0) {
        private var center: Double? = null

        fun reset() {
            center = null
        }

        /** Devuelve el yaw relativo al frente actual. */
        fun update(yaw: Double, dtSeconds: Double): Double {
            val current = center ?: yaw.also { center = it }
            val relative = wrap(yaw - current)
            center = wrap(current + relative * (dtSeconds / timeConstantSeconds).coerceIn(0.0, 1.0))
            return relative
        }
    }

    /** Suavizado exponencial + zona muerta, para que el sonido no tiemble. */
    class Smoother(private val alpha: Double = 0.5, private val deadZone: Double = 1.5) {
        var value = 0.0
            private set

        fun reset() {
            value = 0.0
        }

        fun update(target: Double): Double {
            val next = value + alpha * wrap(target - value)
            if (abs(next - value) >= deadZone || abs(target) < deadZone) value = next
            return value
        }
    }
}
