package dev.bootnet.shieldtube

import java.net.URI
import java.net.URLDecoder
import java.net.URLEncoder

/**
 * Reconoce enlaces de YouTube en un texto (portapapeles, "Compartir") y los
 * convierte a la URL equivalente de m.youtube.com, que es la que abre la app.
 * Sin dependencias de Android, para poder testearlo.
 */
object YoutubeLinks {
    /** Enlace reconocido. `videoId` es null en listas y canales. */
    data class Link(val url: String, val videoId: String?)

    private val CANDIDATE = Regex(
        """(?:https?://)?(?:[\w-]+\.)*(?:youtube\.com|youtu\.be|youtube-nocookie\.com)(?:/[^\s"'<>]*)?""",
        RegexOption.IGNORE_CASE,
    )
    private val VIDEO_ID = Regex("^[A-Za-z0-9_-]{11}$")

    /** Parámetros que se conservan; el resto (si, feature, pp…) es seguimiento. */
    private val KEEP = listOf("v", "t", "list", "index")

    fun find(text: String?): Link? {
        if (text.isNullOrBlank()) return null
        return CANDIDATE.findAll(text).firstNotNullOfOrNull { parse(it.value.trimEnd('.', ',', ')', ';', '!', '?')) }
    }

    fun parse(raw: String): Link? {
        val withScheme = if (raw.startsWith("http", ignoreCase = true)) raw else "https://$raw"
        val uri = runCatching { URI(withScheme) }.getOrNull() ?: return null
        val host = uri.host?.lowercase() ?: return null
        val segments = uri.rawPath.orEmpty().split('/').filter { it.isNotEmpty() }
        val query = parseQuery(uri.rawQuery)

        if (host == "youtu.be" || host.endsWith(".youtu.be")) {
            val id = segments.firstOrNull()?.takeIf { VIDEO_ID.matches(it) } ?: return null
            return watch(id, query)
        }
        if (!(host == "youtube.com" || host.endsWith(".youtube.com") || host.endsWith("youtube-nocookie.com"))) {
            return null
        }
        val first = segments.firstOrNull()
        return when {
            first == "watch" -> query["v"]?.takeIf { VIDEO_ID.matches(it) }?.let { watch(it, query) }
            first in listOf("shorts", "live", "embed", "v") ->
                segments.getOrNull(1)?.takeIf { VIDEO_ID.matches(it) }?.let { id ->
                    if (first == "shorts") Link("${Net.HOME_URL}shorts/$id", id) else watch(id, query)
                }
            first == "playlist" -> query["list"]?.let { Link("${Net.HOME_URL}playlist?list=${encode(it)}", null) }
            // Canales (@nombre, channel/…, c/…, user/…).
            first != null && (first.startsWith("@") || first in listOf("channel", "c", "user")) ->
                Link(Net.HOME_URL + segments.joinToString("/"), null)
            else -> null
        }
    }

    private fun watch(id: String, query: Map<String, String>): Link {
        val params = mutableListOf("v=$id")
        for (key in KEEP.drop(1)) query[key]?.let { params += "$key=${encode(it)}" }
        return Link("${Net.HOME_URL}watch?${params.joinToString("&")}", id)
    }

    private fun parseQuery(raw: String?): Map<String, String> =
        raw.orEmpty().split('&').mapNotNull { pair ->
            val key = pair.substringBefore('=')
            if (key.isEmpty()) null else key to URLDecoder.decode(pair.substringAfter('=', ""), "UTF-8")
        }.toMap()

    private fun encode(value: String) = URLEncoder.encode(value, "UTF-8")

    /** ID del vídeo de una URL de la app (para no ofrecer el que ya se está viendo). */
    fun videoIdOf(url: String?): String? = url?.let { parse(it)?.videoId }

    /** Consulta oEmbed de YouTube para mostrar el título del vídeo en el aviso. */
    fun oEmbedUrl(link: Link): String =
        "https://www.youtube.com/oembed?format=json&url=" + encode("https://www.youtube.com/watch?v=${link.videoId}")
}
