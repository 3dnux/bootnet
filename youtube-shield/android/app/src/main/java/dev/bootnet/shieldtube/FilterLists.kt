package dev.bootnet.shieldtube

import android.util.Log
import org.json.JSONArray
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Descarga (con caché de 24 h) las listas que Brave activa por defecto en Android. */
object FilterLists {
    private const val TAG = "ShieldTube"
    private const val CATALOG_URL =
        "https://raw.githubusercontent.com/brave/adblock-resources/master/filter_lists/list_catalog.json"
    private const val PLATFORM = "ANDROID"
    val MAX_AGE_MS = TimeUnit.HOURS.toMillis(24)

    private class Source(val url: String, val permission: Int, val hosts: Boolean)

    // Listas de Brave usadas si no se puede leer el catálogo.
    private val FALLBACK = listOf(
        "https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/filters.txt",
        "https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/quick-fixes.txt",
        "https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/unbreak.txt",
        "https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/privacy.txt",
        "https://easylist.to/easylist/easylist.txt",
        "https://easylist.to/easylist/easyprivacy.txt",
        "https://raw.githubusercontent.com/brave/adblock-lists/master/brave-lists/brave-specific.txt",
        "https://raw.githubusercontent.com/brave/adblock-lists/master/brave-lists/brave-unbreak.txt",
        "https://raw.githubusercontent.com/brave/adblock-lists/master/brave-lists/brave-android-specific.txt",
    ).map { Source(it, permission = 1, hosts = false) }

    fun load(cacheDir: File): List<AdblockEngine.FilterList> {
        cacheDir.mkdirs()
        val sources = cachedFetch(CATALOG_URL, cacheDir)
            ?.let { runCatching { parseCatalog(it) }.getOrNull() }
            ?.takeIf { it.isNotEmpty() }
            ?: FALLBACK
        val pool = Executors.newFixedThreadPool(6)
        try {
            val futures = sources.map { source -> pool.submit<String?> { cachedFetch(source.url, cacheDir) } }
            return sources.zip(futures).mapNotNull { (source, future) ->
                future.get()?.let { AdblockEngine.FilterList(it, source.permission, source.hosts) }
            }
        } finally {
            pool.shutdown()
        }
    }

    private fun parseCatalog(json: String): List<Source> {
        val catalog = JSONArray(json)
        val sources = mutableListOf<Source>()
        for (i in 0 until catalog.length()) {
            val list = catalog.getJSONObject(i)
            if (!list.optBoolean("default_enabled")) continue
            val platforms = list.optJSONArray("platforms")
            if (platforms != null && (0 until platforms.length()).none { platforms.getString(it) == PLATFORM }) continue
            val permission = list.optInt("permission_mask", 0)
            val listSources = list.getJSONArray("sources")
            for (j in 0 until listSources.length()) {
                val source = listSources.getJSONObject(j)
                sources += Source(source.getString("url"), permission, source.optString("format") == "Hosts")
            }
        }
        return sources
    }

    /** Copia en caché si es reciente; si no, descarga. Si la descarga falla, usa la caché vieja. */
    private fun cachedFetch(url: String, cacheDir: File): String? {
        val file = File(cacheDir, url.replace(Regex("^https?://"), "").replace(Regex("[^a-zA-Z0-9.-]+"), "_"))
        if (file.exists() && System.currentTimeMillis() - file.lastModified() < MAX_AGE_MS) return file.readText()
        return try {
            val connection = URL(url).openConnection() as HttpURLConnection
            connection.connectTimeout = 15_000
            connection.readTimeout = 30_000
            try {
                if (connection.responseCode != 200) error("HTTP ${connection.responseCode}")
                connection.inputStream.bufferedReader().use { it.readText() }.also { file.writeText(it) }
            } finally {
                connection.disconnect()
            }
        } catch (e: Exception) {
            Log.w(TAG, "No se pudo descargar $url: ${e.message}")
            if (file.exists()) file.readText() else null
        }
    }
}
