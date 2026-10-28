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
    fun seHacePasarPorChromeOFirefox() {
        val webView = "Mozilla/5.0 (Linux; Android 14; SM-S911B Build/UP1A; wv) AppleWebKit/537.36 " +
            "(KHTML, like Gecko) Version/4.0 Chrome/141.0.7390.122 Mobile Safari/537.36"
        assertEquals(
            "Mozilla/5.0 (Linux; Android 14; SM-S911B Build/UP1A) AppleWebKit/537.36 " +
                "(KHTML, like Gecko) Chrome/141.0.7390.122 Mobile Safari/537.36",
            Net.chromeUserAgent(webView),
        )
        // 22-10-2026: 13 ciclos de 4 semanas después de Firefox 144.
        assertEquals(
            "Mozilla/5.0 (Android 14; Mobile; rv:157.0) Gecko/157.0 Firefox/157.0",
            Net.firefoxUserAgent("14", 1_792_627_200_000L),
        )
    }

    @Test
    fun detectaElRechazoDeGoogle() {
        assertTrue(Net.isSignInRejected("https://accounts.google.com/v3/signin/rejected?continue=x"))
        assertTrue(Net.isSignInRejected("https://accounts.google.com/signin/rejected"))
        assertFalse(Net.isSignInRejected("https://accounts.google.com/v3/signin/identifier"))
        assertFalse(Net.isSignInRejected("https://evil.com/signin/rejected"))
    }

    @Test
    fun soloGoogleUsaLaCamara() {
        assertTrue(Net.mayUseCamera("https://accounts.google.com/"))
        assertTrue(Net.mayUseCamera("https://accounts.google.com"))
        assertFalse(Net.mayUseCamera("https://m.youtube.com/"))
        assertFalse(Net.mayUseCamera("http://accounts.google.com/"))
        assertFalse(Net.mayUseCamera("https://accounts.google.com.evil.net/"))
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
