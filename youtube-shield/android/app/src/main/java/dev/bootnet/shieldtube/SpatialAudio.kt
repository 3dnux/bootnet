package dev.bootnet.shieldtube

import android.content.Context

/** Nivel de Sonido 3D (assets/spatial.js): 0 apagado, 1 suave, 2 media, 3 fuerte. */
object SpatialAudio {
    private const val PREFS = "shieldtube"
    private const val KEY = "spatial_level"
    private val NAMES = intArrayOf(R.string.spatial_off, R.string.spatial_soft, R.string.spatial_medium, R.string.spatial_strong)

    /** La actividad lo usa para aplicar el nivel a la página. Solo en el hilo principal. */
    var listener: ((Int) -> Unit)? = null

    fun level(context: Context): Int =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getInt(KEY, 0).coerceIn(0, NAMES.size - 1)

    /** Pasa al siguiente nivel (apagado → suave → media → fuerte → apagado). */
    fun cycle(context: Context): Int {
        val next = (level(context) + 1) % NAMES.size
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putInt(KEY, next).apply()
        listener?.invoke(next)
        return next
    }

    fun label(context: Context, level: Int): String =
        context.getString(R.string.spatial_label, context.getString(NAMES[level]))
}
