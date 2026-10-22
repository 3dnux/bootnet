package dev.bootnet.shieldtube

import java.net.URI
import java.util.Base64

/** Utilidades puras de red (sin dependencias de Android, para poder testearlas). */
object Net {
    const val HOME_URL = "https://m.youtube.com/"

    /** Dominios que se navegan dentro de la app; el resto se abre fuera. */
    private val IN_APP_HOSTS = listOf("youtube.com", "youtu.be", "youtube-nocookie.com", "google.com")

    /** Orígenes de YouTube donde se inyectan scriptlets y CSS cosmético. */
    val YOUTUBE_ORIGINS = listOf(
        "https://m.youtube.com",
        "https://www.youtube.com",
        "https://youtube.com",
        "https://music.youtube.com",
    )

    private val EXTENSION_TYPES = mapOf(
        "js" to "script", "mjs" to "script",
        "css" to "stylesheet",
        "png" to "image", "jpg" to "image", "jpeg" to "image", "gif" to "image", "webp" to "image",
        "svg" to "image", "ico" to "image", "avif" to "image",
        "woff" to "font", "woff2" to "font", "ttf" to "font", "otf" to "font",
        "mp4" to "media", "webm" to "media", "m3u8" to "media", "mpd" to "media", "mp3" to "media",
        "m4a" to "media",
    )

    fun isAllowedInApp(url: String): Boolean {
        val uri = runCatching { URI(url) }.getOrNull() ?: return false
        val host = uri.host ?: return false
        return uri.scheme == "https" && IN_APP_HOSTS.any { host == it || host.endsWith(".$it") }
    }

    /**
     * WebView no dice qué tipo de recurso es cada petición (Brave sí lo sabe porque
     * va dentro de Chromium), así que se deduce del marco, la cabecera Accept y la
     * extensión. Lo que no encaja suele ser fetch/XHR en una web como YouTube.
     */
    fun inferRequestType(url: String, isMainFrame: Boolean, accept: String?): String {
        if (isMainFrame) return "document"
        val acceptLower = accept?.lowercase().orEmpty()
        when {
            acceptLower.startsWith("text/html") -> return "subdocument"
            acceptLower.startsWith("text/css") -> return "stylesheet"
            acceptLower.startsWith("image/") -> return "image"
        }
        val path = runCatching { URI(url).rawPath }.getOrNull().orEmpty()
        val extension = path.substringAfterLast('/').substringAfterLast('.', "").lowercase()
        return EXTENSION_TYPES[extension] ?: "xmlhttprequest"
    }

    class DataUrl(val mimeType: String, val bytes: ByteArray)

    /** Decodifica los `data:` que devuelve el motor como recurso de sustitución. */
    fun parseDataUrl(url: String): DataUrl? {
        if (!url.startsWith("data:")) return null
        val comma = url.indexOf(',')
        if (comma < 0) return null
        val meta = url.substring(5, comma)
        val payload = url.substring(comma + 1)
        val mimeType = meta.substringBefore(';').ifEmpty { "text/plain" }
        val bytes = if (meta.endsWith(";base64")) {
            runCatching { Base64.getDecoder().decode(payload) }.getOrNull() ?: return null
        } else {
            java.net.URLDecoder.decode(payload, "UTF-8").toByteArray()
        }
        return DataUrl(mimeType, bytes)
    }
}
