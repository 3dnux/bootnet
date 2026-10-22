package dev.bootnet.shieldtube

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.webkit.CookieManager
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import android.window.OnBackInvokedDispatcher
import androidx.webkit.ScriptHandler
import androidx.webkit.ServiceWorkerClientCompat
import androidx.webkit.ServiceWorkerControllerCompat
import androidx.webkit.UserAgentMetadata
import androidx.webkit.WebSettingsCompat
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private lateinit var root: FrameLayout
    private lateinit var webView: KeepAliveWebView
    private lateinit var loading: View
    private var fullscreenView: View? = null
    private var fullscreenCallback: WebChromeClient.CustomViewCallback? = null
    private val worker = Executors.newSingleThreadExecutor()
    private var engineReady = false
    private var pendingUrl: String? = null
    private var stopped = false
    private lateinit var spatialButton: TextView
    private var spatialLevelScript: ScriptHandler? = null

    /** URL de la página actual: origen de las peticiones que se consultan al motor. */
    @Volatile
    private var pageUrl = Net.HOME_URL

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        root = FrameLayout(this).apply { setBackgroundColor(BACKGROUND) }
        webView = KeepAliveWebView(this) { PlaybackService.running }.apply { setBackgroundColor(BACKGROUND) }
        loading = loadingView()
        root.addView(webView, matchParent())
        root.addView(loading, matchParent())
        spatialButton = spatialButton()
        root.addView(
            spatialButton,
            FrameLayout.LayoutParams(dp(44), dp(44), Gravity.BOTTOM or Gravity.END).apply {
                setMargins(0, 0, dp(12), dp(16))
            },
        )
        setContentView(root)
        applySystemBarInsets()
        setupWebView()
        registerBack()
        requestNotificationPermission()
        PlaybackService.onStopped = {
            // Fin de la reproducción en segundo plano: la WebView vuelve a su estado real.
            webView.syncWindowVisibility()
            if (stopped) webView.onPause()
        }
        SpatialAudio.listener = { level -> applySpatial(level) }

        pendingUrl = youtubeUrlFrom(intent) ?: Net.HOME_URL
        worker.execute {
            val engine = runCatching { Shields.load(applicationContext) }.getOrNull()
            runOnUiThread { onEngineReady(engine) }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        val url = youtubeUrlFrom(intent) ?: return
        if (engineReady) webView.loadUrl(url) else pendingUrl = url
    }

    // Con reproducción activa, la WebView sigue funcionando al salir o bloquear el móvil.
    override fun onPause() {
        super.onPause()
        // En pantalla completa el vídeo vive en otra vista de Chromium que sí se
        // daría por oculta; se sale de ella antes de que se oculte la ventana.
        if (PlaybackService.running && fullscreenView != null) {
            (webView.webChromeClient as WebChromeClient).onHideCustomView()
        }
    }

    override fun onStop() {
        super.onStop()
        stopped = true
        if (!PlaybackService.running) webView.onPause()
    }

    override fun onStart() {
        super.onStart()
        stopped = false
        webView.onResume()
    }

    override fun onDestroy() {
        PlaybackService.onStopped = null
        SpatialAudio.listener = null
        PlaybackService.stopPlayback(this)
        worker.shutdown()
        webView.destroy()
        super.onDestroy()
    }

    private fun youtubeUrlFrom(intent: Intent?): String? =
        intent?.dataString?.takeIf { intent.action == Intent.ACTION_VIEW && Net.isAllowedInApp(it) }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            mediaPlaybackRequiresUserGesture = false
        }
        applyBrowserIdentity()
        CookieManager.getInstance().setAcceptThirdPartyCookies(webView, true)
        // Que Android no rebaje la prioridad del renderizador cuando la app no se ve.
        webView.setRendererPriorityPolicy(WebView.RENDERER_PRIORITY_IMPORTANT, false)
        installBackgroundPlayback()
        webView.webViewClient = ShieldClient()
        webView.webChromeClient = FullscreenClient()
        if (WebViewFeature.isFeatureSupported(WebViewFeature.SERVICE_WORKER_BASIC_USAGE) &&
            WebViewFeature.isFeatureSupported(WebViewFeature.SERVICE_WORKER_SHOULD_INTERCEPT_REQUEST)
        ) {
            ServiceWorkerControllerCompat.getInstance().setServiceWorkerClient(object : ServiceWorkerClientCompat() {
                override fun shouldInterceptRequest(request: WebResourceRequest) = intercept(request)
            })
        }
    }

    /**
     * Google no deja iniciar sesión en WebViews y las reconoce por "; wv" en el user
     * agent y la marca "Android WebView" de las Client Hints (la cabecera
     * X-Requested-With ya no la envía WebView por defecto). Se quitan para parecer
     * Chrome. Si aun así Google rechaza el inicio de sesión, se pasa a modo Firefox
     * (ver onSignInRejected).
     */
    private fun applyBrowserIdentity() {
        val settings = webView.settings
        if (WebViewFeature.isFeatureSupported(WebViewFeature.USER_AGENT_METADATA)) {
            val metadata = WebSettingsCompat.getUserAgentMetadata(settings)
            val brands = metadata.brandVersionList.map { brand ->
                if (brand.brand.contains("WebView")) {
                    UserAgentMetadata.BrandVersion.Builder(brand).setBrand("Google Chrome").build()
                } else {
                    brand
                }
            }
            WebSettingsCompat.setUserAgentMetadata(
                settings, UserAgentMetadata.Builder(metadata).setBrandVersionList(brands).build(),
            )
        }
        settings.userAgentString = if (firefoxMode) {
            Net.firefoxUserAgent(Build.VERSION.RELEASE, System.currentTimeMillis())
        } else {
            Net.chromeUserAgent(WebSettings.getDefaultUserAgent(this))
        }
    }

    private val prefs by lazy { getSharedPreferences("shieldtube", MODE_PRIVATE) }

    private var firefoxMode: Boolean
        get() = prefs.getBoolean("firefox_mode", false)
        set(value) = prefs.edit().putBoolean("firefox_mode", value).apply()

    /** Google rechazó el inicio de sesión: se reintenta presentándose como Firefox. */
    private fun onSignInRejected() {
        if (firefoxMode) return
        firefoxMode = true
        applyBrowserIdentity()
        Toast.makeText(this, R.string.sign_in_retry, Toast.LENGTH_LONG).show()
        webView.loadUrl(Net.SIGN_IN_URL)
    }

    /** background.js: evita que YouTube se pause en segundo plano y controla el servicio. */
    private fun installBackgroundPlayback() {
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) return
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) return
        val origins = Net.YOUTUBE_ORIGINS.toSet()
        WebViewCompat.addWebMessageListener(webView, "shieldtubeMedia", origins) { _, message, _, isMainFrame, reply ->
            val data = message.data
            if (isMainFrame && data != null) PlaybackService.Player.onPageState(applicationContext, data, reply)
        }
        val script = assets.open("background.js").bufferedReader().use { it.readText() }
        WebViewCompat.addDocumentStartJavaScript(webView, script, origins)
        val spatial = assets.open("spatial.js").bufferedReader().use { it.readText() }
        WebViewCompat.addDocumentStartJavaScript(webView, spatial, origins)
        setSpatialLevelScript(SpatialAudio.level(this))
    }

    /** Nivel de Sonido 3D con el que arrancan las páginas nuevas. */
    private fun setSpatialLevelScript(level: Int) {
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) return
        spatialLevelScript?.remove()
        spatialLevelScript = WebViewCompat.addDocumentStartJavaScript(
            webView, "window.__shieldtubeSpatial = $level;", Net.YOUTUBE_ORIGINS.toSet(),
        )
    }

    /** Aplica el nivel de Sonido 3D a la página actual y a las siguientes. */
    private fun applySpatial(level: Int) {
        setSpatialLevelScript(level)
        webView.evaluateJavascript("window.shieldtubeSetSpatial && shieldtubeSetSpatial($level)", null)
        styleSpatialButton(level)
        Toast.makeText(this, SpatialAudio.label(this, level), Toast.LENGTH_SHORT).show()
    }

    /** Botón "3D" sobre la página de un vídeo: cambia la intensidad del Sonido 3D. */
    private fun spatialButton(): TextView = TextView(this).apply {
        text = getString(R.string.spatial_button)
        textSize = 13f
        typeface = Typeface.DEFAULT_BOLD
        gravity = Gravity.CENTER
        visibility = View.GONE
        contentDescription = SpatialAudio.label(this@MainActivity, SpatialAudio.level(this@MainActivity))
        setOnClickListener { SpatialAudio.cycle(this@MainActivity) }
    }.also { styleSpatialButton(SpatialAudio.level(this), it) }

    private fun styleSpatialButton(level: Int, button: TextView = spatialButton) {
        val active = level > 0
        val accent = getColor(R.color.accent)
        button.setTextColor(if (active) accent else Color.LTGRAY)
        button.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.argb(200, 15, 15, 15))
            setStroke(dp(2), if (active) accent else Color.GRAY)
        }
        button.contentDescription = SpatialAudio.label(this, level)
    }

    private fun updateSpatialButton(url: String) {
        val path = runCatching { Uri.parse(url).path }.getOrNull().orEmpty()
        spatialButton.visibility = if (path.startsWith("/watch")) View.VISIBLE else View.GONE
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        }
    }

    private fun onEngineReady(engine: AdblockEngine?) {
        engineReady = true
        if (engine != null) installCosmetics(engine)
        loading.visibility = View.GONE
        webView.loadUrl(pendingUrl ?: Net.HOME_URL)
        pendingUrl = null
    }

    /** Scriptlets y CSS al inicio de cada documento de YouTube, antes que sus scripts. */
    private fun installCosmetics(engine: AdblockEngine) {
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) return
        for (origin in Net.YOUTUBE_ORIGINS) {
            val script = Shields.documentStartScript(this, engine, origin)
            WebViewCompat.addDocumentStartJavaScript(webView, script, setOf(origin))
        }
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) return
        val exceptions = Net.YOUTUBE_ORIGINS.associateWith { Shields.exceptionsFor(engine, it) }
        WebViewCompat.addWebMessageListener(webView, "shieldtube", Net.YOUTUBE_ORIGINS.toSet()) { _, message, origin, _, reply ->
            val data = message.data ?: return@addWebMessageListener
            val originExceptions = exceptions[origin.toString().trimEnd('/')] ?: "[]"
            worker.execute {
                val css = runCatching {
                    val request = JSONObject(data)
                    val selectors = org.json.JSONArray(engine.hiddenClassIdSelectors(
                        request.getJSONArray("classes").toString(),
                        request.getJSONArray("ids").toString(),
                        originExceptions,
                    ))
                    (0 until selectors.length()).joinToString("\n") { "${selectors.getString(it)}{display:none!important}" }
                }.getOrDefault("")
                if (css.isNotEmpty()) runOnUiThread { reply.postMessage(css) }
            }
        }
    }

    /** Bloqueo de red: cada petición pasa por el motor de Brave. */
    private fun intercept(request: WebResourceRequest): WebResourceResponse? {
        val engine = Shields.engine ?: return null
        val url = request.url.toString()
        if (!url.startsWith("http")) return null
        val type = Net.inferRequestType(url, request.isForMainFrame, request.requestHeaders["Accept"])
        val source = if (request.isForMainFrame) url else pageUrl
        return when (val decision = engine.check(url, source, type, request.method)) {
            null -> null
            is AdblockEngine.Decision.Block -> emptyResponse()
            is AdblockEngine.Decision.Redirect -> Net.parseDataUrl(decision.dataUrl)
                ?.let { WebResourceResponse(it.mimeType, "utf-8", ByteArrayInputStream(it.bytes)) }
                ?: emptyResponse()
        }
    }

    private fun emptyResponse() = WebResourceResponse("text/plain", "utf-8", ByteArrayInputStream(ByteArray(0)))

    private inner class ShieldClient : WebViewClient() {
        override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest) = intercept(request)

        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
            val url = request.url.toString()
            if (Net.isAllowedInApp(url)) return false
            if (url.startsWith("http://") || url.startsWith("https://")) {
                try {
                    startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
                } catch (_: ActivityNotFoundException) {
                }
            }
            // intent://, vnd.youtube:... (abrir la app oficial) se ignoran.
            return true
        }

        override fun onPageStarted(view: WebView, url: String, favicon: android.graphics.Bitmap?) {
            pageUrl = url
            if (Net.isSignInRejected(url)) view.post { onSignInRejected() }
        }

        override fun doUpdateVisitedHistory(view: WebView, url: String, isReload: Boolean) {
            pageUrl = url
            updateSpatialButton(url)
        }
    }

    /** Pantalla completa de los vídeos. */
    private inner class FullscreenClient : WebChromeClient() {
        override fun onShowCustomView(view: View, callback: CustomViewCallback) {
            if (fullscreenView != null) {
                callback.onCustomViewHidden()
                return
            }
            fullscreenView = view
            fullscreenCallback = callback
            (window.decorView as ViewGroup).addView(view, matchParent())
            webView.visibility = View.INVISIBLE
            requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
            setSystemBarsHidden(true)
        }

        override fun onHideCustomView() {
            val view = fullscreenView ?: return
            (window.decorView as ViewGroup).removeView(view)
            fullscreenView = null
            fullscreenCallback?.onCustomViewHidden()
            fullscreenCallback = null
            webView.visibility = View.VISIBLE
            requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
            setSystemBarsHidden(false)
        }
    }

    @Suppress("DEPRECATION")
    private fun setSystemBarsHidden(hidden: Boolean) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val controller = window.insetsController ?: return
            if (hidden) {
                controller.hide(WindowInsets.Type.systemBars())
                controller.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            } else {
                controller.show(WindowInsets.Type.systemBars())
            }
        } else {
            window.decorView.systemUiVisibility = if (hidden) {
                View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            } else {
                0
            }
        }
    }

    /** Android 15+ dibuja bajo las barras del sistema: se deja hueco para ellas. */
    @Suppress("DEPRECATION")
    private fun applySystemBarInsets() {
        root.setOnApplyWindowInsetsListener { view, insets ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val bars = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            } else {
                view.setPadding(
                    insets.systemWindowInsetLeft, insets.systemWindowInsetTop,
                    insets.systemWindowInsetRight, insets.systemWindowInsetBottom,
                )
            }
            insets
        }
    }

    private fun handleBack() {
        when {
            fullscreenView != null -> (webView.webChromeClient as WebChromeClient).onHideCustomView()
            webView.canGoBack() -> webView.goBack()
            else -> finish()
        }
    }

    private fun registerBack() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            onBackInvokedDispatcher.registerOnBackInvokedCallback(OnBackInvokedDispatcher.PRIORITY_DEFAULT) { handleBack() }
        }
    }

    // Solo antes de Android 13; después se usa el OnBackInvokedCallback de registerBack().
    @Deprecated("Solo se usa antes de Android 13")
    @Suppress("DEPRECATION")
    @SuppressLint("GestureBackNavigation")
    override fun onBackPressed() {
        handleBack()
    }

    private fun loadingView(): View = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        gravity = Gravity.CENTER
        setBackgroundColor(BACKGROUND)
        addView(ProgressBar(this@MainActivity))
        addView(TextView(this@MainActivity).apply {
            text = getString(R.string.loading)
            setTextColor(Color.WHITE)
            textSize = 16f
            gravity = Gravity.CENTER
            setPadding(48, 32, 48, 8)
        })
        addView(TextView(this@MainActivity).apply {
            text = getString(R.string.loading_hint)
            setTextColor(Color.GRAY)
            textSize = 13f
            gravity = Gravity.CENTER
            setPadding(48, 0, 48, 0)
        })
    }

    private fun matchParent() =
        FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)

    private companion object {
        val BACKGROUND = Color.rgb(15, 15, 15)
    }
}
