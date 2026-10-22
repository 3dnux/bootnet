package dev.bootnet.shieldtube

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NetTest {
    @Test
    fun soloNavegaDentroPorYoutubeYGoogle() {
        assertTrue(Net.isAllowedInApp("https://m.youtube.com/watch?v=x"))
        assertTrue(Net.isAllowedInApp("https://youtu.be/x"))
        assertTrue(Net.isAllowedInApp("https://accounts.google.com/ServiceLogin"))
        assertFalse(Net.isAllowedInApp("http://m.youtube.com/"))
        assertFalse(Net.isAllowedInApp("https://evil-youtube.com/"))
        assertFalse(Net.isAllowedInApp("https://youtube.com.evil.net/"))
        assertFalse(Net.isAllowedInApp("intent://watch#Intent;scheme=vnd.youtube;end"))
        assertFalse(Net.isAllowedInApp("no es una url"))
    }

    @Test
    fun deduceElTipoDePeticion() {
        assertEquals("document", Net.inferRequestType("https://m.youtube.com/", true, null))
        assertEquals("subdocument", Net.inferRequestType("https://x.com/frame", false, "text/html,application/xhtml+xml"))
        assertEquals("stylesheet", Net.inferRequestType("https://x.com/a", false, "text/css,*/*;q=0.1"))
        assertEquals("image", Net.inferRequestType("https://i.ytimg.com/vi/x/hq.jpg", false, "*/*"))
        assertEquals("script", Net.inferRequestType("https://x.com/s/base.js?v=1", false, "*/*"))
        assertEquals("font", Net.inferRequestType("https://x.com/f.woff2", false, null))
        assertEquals("xmlhttprequest", Net.inferRequestType("https://m.youtube.com/api/stats/ads?x=1", false, "*/*"))
    }

    @Test
    fun decodificaDataUrls() {
        val js = Net.parseDataUrl("data:application/javascript;base64,KGZ1bmN0aW9uKCl7fSkoKQ==")!!
        assertEquals("application/javascript", js.mimeType)
        assertArrayEquals("(function(){})()".toByteArray(), js.bytes)
        val text = Net.parseDataUrl("data:text/plain,hola%20mundo")!!
        assertEquals("hola mundo", String(text.bytes))
        assertNull(Net.parseDataUrl("https://example.com"))
    }
}
