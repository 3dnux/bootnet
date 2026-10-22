package dev.bootnet.shieldtube

import android.annotation.SuppressLint
import android.content.Context
import android.view.View
import android.webkit.WebView

/**
 * WebView que, mientras `keepAlive()` es cierto, no le dice a Chromium que su
 * ventana se ha ocultado. Chromium marca la página como oculta al bloquear el
 * móvil o salir de la app, y en Android pausa los vídeos de páginas ocultas y no
 * los deja volver a reproducirse; eso cortaba la reproducción en segundo plano.
 */
@SuppressLint("ViewConstructor")
class KeepAliveWebView(context: Context, private val keepAlive: () -> Boolean) : WebView(context) {

    override fun onWindowVisibilityChanged(visibility: Int) {
        super.onWindowVisibilityChanged(if (keepAlive()) View.VISIBLE else visibility)
    }

    /** Vuelve a informar de la visibilidad real (cuando termina la reproducción). */
    fun syncWindowVisibility() {
        super.onWindowVisibilityChanged(windowVisibility)
    }
}
