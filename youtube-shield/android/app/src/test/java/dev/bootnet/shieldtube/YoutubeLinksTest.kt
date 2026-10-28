package dev.bootnet.shieldtube

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class YoutubeLinksTest {
    private fun url(text: String?) = YoutubeLinks.find(text)?.url

    @Test
    fun reconoceVideos() {
        assertEquals("https://m.youtube.com/watch?v=dQw4w9WgXcQ", url("https://youtu.be/dQw4w9WgXcQ?si=AbCdEf123"))
        assertEquals("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=42", url("https://youtu.be/dQw4w9WgXcQ?t=42"))
        assertEquals(
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=1m5s&list=PL123",
            url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share&t=1m5s&list=PL123&pp=x"),
        )
        assertEquals("https://m.youtube.com/watch?v=dQw4w9WgXcQ", url("https://music.youtube.com/watch?v=dQw4w9WgXcQ"))
        assertEquals("https://m.youtube.com/watch?v=dQw4w9WgXcQ", url("https://www.youtube.com/live/dQw4w9WgXcQ?si=x"))
        assertEquals("https://m.youtube.com/watch?v=dQw4w9WgXcQ", url("youtube.com/embed/dQw4w9WgXcQ"))
        assertEquals("https://m.youtube.com/shorts/abcDEF12345", url("https://youtube.com/shorts/abcDEF12345?si=q"))
    }

    @Test
    fun encuentraElEnlaceDentroDeUnTexto() {
        // Así copia el enlace el botón "Compartir" de algunas apps.
        assertEquals(
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            url("Mira este vídeo: https://youtu.be/dQw4w9WgXcQ?si=abc."),
        )
        assertEquals("https://m.youtube.com/watch?v=dQw4w9WgXcQ", url("youtu.be/dQw4w9WgXcQ"))
    }

    @Test
    fun listasYCanales() {
        val list = YoutubeLinks.find("https://www.youtube.com/playlist?list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG")!!
        assertEquals("https://m.youtube.com/playlist?list=PLx0sYbCqOb8TBPRdmBHs5Iftvv9TPboYG", list.url)
        assertNull(list.videoId)
        assertEquals("https://m.youtube.com/@MrBeast", url("https://youtube.com/@MrBeast?si=x"))
        assertEquals("https://m.youtube.com/channel/UC123", url("https://www.youtube.com/channel/UC123"))
    }

    @Test
    fun ignoraLoQueNoEsDeYoutube() {
        assertNull(url("https://evil.com/watch?v=dQw4w9WgXcQ"))
        assertNull(url("https://youtube.com.evil.net/watch?v=dQw4w9WgXcQ"))
        assertNull(url("https://www.youtube.com/watch?v=corto"))
        assertNull(url("https://www.youtube.com/"))
        assertNull(url("hola, ¿qué tal?"))
        assertNull(url(null))
    }

    @Test
    fun idDelVideoActual() {
        assertEquals("dQw4w9WgXcQ", YoutubeLinks.videoIdOf("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=3"))
        assertNull(YoutubeLinks.videoIdOf("https://m.youtube.com/"))
    }
}
