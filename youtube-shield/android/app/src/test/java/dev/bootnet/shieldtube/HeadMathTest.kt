package dev.bootnet.shieldtube

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.PI

class HeadMathTest {
    private fun rad(deg: Double) = deg * PI / 180

    @Test
    fun normalizaAngulos() {
        assertEquals(-170.0, HeadMath.wrap(190.0), 1e-9)
        assertEquals(170.0, HeadMath.wrap(-190.0), 1e-9)
        assertEquals(0.0, HeadMath.wrap(360.0), 1e-9)
    }

    @Test
    fun yawDelSensorDeCabeza() {
        // Giro puro alrededor de la coronilla (Z): positivo = izquierda.
        assertEquals(30.0, HeadMath.yawFromEulerVector(0.0, 0.0, rad(30.0)), 1e-6)
        assertEquals(-45.0, HeadMath.yawFromEulerVector(0.0, 0.0, rad(-45.0)), 1e-6)
        // Mirar arriba/abajo (eje X, orejas) no cambia el yaw.
        assertEquals(0.0, HeadMath.yawFromEulerVector(rad(25.0), 0.0, 0.0), 1e-6)
        // Inclinar la cabeza hacia el hombro (eje Y, nariz) tampoco.
        assertEquals(0.0, HeadMath.yawFromEulerVector(0.0, rad(20.0), 0.0), 1e-6)
    }

    @Test
    fun recentraDespacio() {
        val recenter = HeadMath.Recenter(timeConstantSeconds = 8.0)
        assertEquals(0.0, recenter.update(100.0, 0.05), 1e-9) // la primera lectura es el frente
        assertEquals(30.0, recenter.update(130.0, 0.05), 1e-9) // giro de 30° a la izquierda
        var relative = 0.0
        repeat(20 * 30) { relative = recenter.update(130.0, 0.05) } // 30 s quieto en esa postura
        assertEquals(0.0, relative, 1.0)
    }

    @Test
    fun suavizaSinTemblar() {
        val smoother = HeadMath.Smoother(alpha = 0.5, deadZone = 1.5)
        assertEquals(20.0, smoother.update(40.0), 1e-9)
        assertEquals(30.0, smoother.update(40.0), 1e-9)
        val before = smoother.value
        smoother.update(before + 1.0) // ruido pequeño: se ignora
        assertEquals(before, smoother.value, 1e-9)
    }
}
