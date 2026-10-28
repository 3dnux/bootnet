package dev.bootnet.shieldtube

import android.content.Context

/** Nivel de Sonido 3D (assets/spatial.js): 0 apagado, 1 suave, 2 media, 3 inmersivo. */
object SpatialAudio {
    private const val PREFS = "shieldtube"
    private const val KEY = "spatial_level"
    private const val KEY_HEAD = "head_tracking"
    private val NAMES = intArrayOf(R.string.spatial_off, R.string.spatial_soft, R.string.spatial_medium, R.string.spatial_strong)

    /** La actividad lo usa para aplicar el nivel a la página. Solo en el hilo principal. */
    var listener: ((Int) -> Unit)? = null

    fun level(context: Context): Int =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getInt(KEY, 0).coerceIn(0, NAMES.size - 1)

    /** Pasa al siguiente nivel (apagado → suave → media → inmersivo → apagado). */
    fun cycle(context: Context): Int = setLevel(context, (level(context) + 1) % NAMES.size)

    fun setLevel(context: Context, level: Int): Int {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putInt(KEY, level).apply()
        listener?.invoke(level)
        return level
    }

    /** Seguimiento de cabeza (HeadTracker) activado por el usuario. */
    fun headTracking(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_HEAD, false)

    fun setHeadTracking(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY_HEAD, enabled).apply()
    }

    fun label(context: Context, level: Int): String =
        context.getString(R.string.spatial_label, context.getString(NAMES[level]))
}
