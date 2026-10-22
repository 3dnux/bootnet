package dev.bootnet.shieldtube

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.drawable.Icon
import android.media.MediaMetadata
import android.media.session.MediaSession
import android.media.session.PlaybackState
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.webkit.JavaScriptReplyProxy
import androidx.webkit.WebViewFeature

/**
 * Mantiene viva la reproducción con la pantalla bloqueada: servicio en primer
 * plano de tipo mediaPlayback, notificación con controles, MediaSession (pantalla
 * de bloqueo y auriculares) y wake locks mientras suena.
 */
class PlaybackService : Service() {
    private lateinit var session: MediaSession
    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null

    override fun onCreate() {
        super.onCreate()
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, getString(R.string.playback_channel), NotificationManager.IMPORTANCE_LOW),
        )
        session = MediaSession(this, "ShieldTube").apply {
            setCallback(object : MediaSession.Callback() {
                override fun onPlay() = Player.send("play")
                override fun onPause() = Player.send("pause")
                override fun onStop() = stopPlayback(this@PlaybackService)
                override fun onSeekTo(pos: Long) = Player.send("seek:${pos / 1000.0}")
            })
            setSessionActivity(openAppIntent())
            isActive = true
        }
        val power = getSystemService(PowerManager::class.java)
        wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ShieldTube:playback").apply {
            setReferenceCounted(false)
        }
        @Suppress("DEPRECATION")
        wifiLock = getSystemService(WifiManager::class.java)
            .createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "ShieldTube:playback").apply {
                setReferenceCounted(false)
            }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_PLAY -> Player.send("play")
            ACTION_PAUSE -> Player.send("pause")
            ACTION_SPATIAL -> SpatialAudio.cycle(this)
            ACTION_STOP -> {
                Player.send("pause")
                stopSelf()
                return START_NOT_STICKY
            }
        }
        update(Player.state)
        return START_NOT_STICKY
    }

    private fun update(state: Player.State) {
        session.setMetadata(
            MediaMetadata.Builder()
                .putString(MediaMetadata.METADATA_KEY_TITLE, state.title)
                .putString(MediaMetadata.METADATA_KEY_ARTIST, state.artist)
                .putLong(MediaMetadata.METADATA_KEY_DURATION, state.duration)
                .build(),
        )
        session.setPlaybackState(
            PlaybackState.Builder()
                .setActions(
                    PlaybackState.ACTION_PLAY or PlaybackState.ACTION_PAUSE or PlaybackState.ACTION_PLAY_PAUSE or
                        PlaybackState.ACTION_STOP or PlaybackState.ACTION_SEEK_TO,
                )
                .setState(
                    if (state.playing) PlaybackState.STATE_PLAYING else PlaybackState.STATE_PAUSED,
                    state.position,
                    if (state.playing) 1f else 0f,
                    SystemClock.elapsedRealtime(),
                )
                .build(),
        )
        val notification = notification(state)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        if (state.playing) {
            wakeLock?.acquire(MAX_WAKE_MS)
            wifiLock?.acquire()
        } else {
            wakeLock?.release()
            wifiLock?.release()
        }
    }

    private fun notification(state: Player.State): Notification {
        val toggle = if (state.playing) {
            Notification.Action.Builder(
                Icon.createWithResource(this, R.drawable.ic_pause), getString(R.string.pause), serviceIntent(ACTION_PAUSE),
            ).build()
        } else {
            Notification.Action.Builder(
                Icon.createWithResource(this, R.drawable.ic_play), getString(R.string.play), serviceIntent(ACTION_PLAY),
            ).build()
        }
        val spatial = Notification.Action.Builder(
            Icon.createWithResource(this, R.drawable.ic_spatial),
            SpatialAudio.label(this, SpatialAudio.level(this)),
            serviceIntent(ACTION_SPATIAL),
        ).build()
        val stop = Notification.Action.Builder(
            Icon.createWithResource(this, R.drawable.ic_close), getString(R.string.stop), serviceIntent(ACTION_STOP),
        ).build()
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(state.title)
            .setContentText(state.artist)
            .setContentIntent(openAppIntent())
            .setDeleteIntent(serviceIntent(ACTION_STOP))
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .setOngoing(state.playing)
            .addAction(toggle)
            .addAction(spatial)
            .addAction(stop)
            .setStyle(Notification.MediaStyle().setMediaSession(session.sessionToken).setShowActionsInCompactView(0, 1, 2))
            .build()
    }

    private fun serviceIntent(action: String): PendingIntent = PendingIntent.getService(
        this, action.hashCode(), Intent(this, PlaybackService::class.java).setAction(action),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )

    private fun openAppIntent(): PendingIntent = PendingIntent.getActivity(
        this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )

    override fun onDestroy() {
        wakeLock?.release()
        wifiLock?.release()
        session.release()
        running = false
        onStopped?.invoke()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    /** Puente entre la página (background.js) y el servicio. Solo se usa en el hilo principal. */
    object Player {
        data class State(
            val playing: Boolean = false,
            val title: String = "YouTube",
            val artist: String = "",
            val position: Long = 0,
            val duration: Long = 0,
        )

        var state = State()
            private set
        private var page: JavaScriptReplyProxy? = null

        /** Mensaje de background.js; arranca o actualiza el servicio. */
        fun onPageState(context: Context, json: String, reply: JavaScriptReplyProxy) {
            val data = runCatching { org.json.JSONObject(json) }.getOrNull() ?: return
            page = reply
            state = State(
                playing = data.optBoolean("playing"),
                title = data.optString("title").ifEmpty { "YouTube" },
                artist = data.optString("artist"),
                position = data.optLong("position"),
                duration = data.optLong("duration"),
            )
            if (!running && !state.playing) return
            val intent = Intent(context, PlaybackService::class.java)
            try {
                if (running) {
                    context.startService(intent)
                } else {
                    context.startForegroundService(intent)
                    running = true
                }
            } catch (e: IllegalStateException) {
                // Android no deja arrancar el servicio si la app ya está en segundo plano.
                Log.w("ShieldTube", "No se pudo iniciar la reproducción en segundo plano", e)
            }
        }

        fun send(command: String) {
            if (WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) page?.postMessage(command)
        }
    }

    companion object {
        private const val CHANNEL_ID = "playback"
        private const val NOTIFICATION_ID = 1
        private const val ACTION_PLAY = "dev.bootnet.shieldtube.PLAY"
        private const val ACTION_PAUSE = "dev.bootnet.shieldtube.PAUSE"
        private const val ACTION_STOP = "dev.bootnet.shieldtube.STOP"
        private const val ACTION_SPATIAL = "dev.bootnet.shieldtube.SPATIAL"
        private const val MAX_WAKE_MS = 4 * 60 * 60 * 1000L

        @Volatile
        var running = false

        /** Se llama (en el hilo principal) cuando el servicio termina. */
        var onStopped: (() -> Unit)? = null

        fun stopPlayback(context: Context) {
            Player.send("pause")
            context.stopService(Intent(context, PlaybackService::class.java))
        }
    }
}
