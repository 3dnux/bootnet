# ShieldTube para Android (APK)

App Android que abre YouTube (`m.youtube.com`) a pantalla completa con **el motor de bloqueo de Brave Shields** ([`adblock-rust`](https://github.com/brave/adblock-rust)) compilado como librería nativa. Usa las mismas listas y scriptlets que la versión de escritorio (ver [`../README.md`](../README.md)).

## Cómo funciona

| Capa de Brave Shields | En la app |
|---|---|
| Bloqueo de red | `WebViewClient.shouldInterceptRequest` (y el de service workers) → motor Rust vía JNI; devuelve respuesta vacía o el recurso de sustitución |
| Scriptlets (`##+js(...)`) | `WebViewCompat.addDocumentStartJavaScript`: se ejecutan antes que los scripts de YouTube |
| CSS cosmético y filtros genéricos | `assets/shields.js` + `addWebMessageListener` para pedir selectores al motor |
| Filtros procedurales | `selector:has-text(...)` |

- Listas: las que Brave activa por defecto **en Android** según su `list_catalog.json`. Se descargan la primera vez y se refrescan cada 24 h; el motor compilado se guarda en caché para arrancar rápido.
- Los enlaces de YouTube pueden abrirse con ShieldTube (aparece en el selector de apps).
- Pantalla completa con giro automático y botón atrás para volver.
- **Reproducción en segundo plano:** el vídeo sigue sonando al salir de la app o bloquear el móvil, con notificación y controles en la pantalla de bloqueo y en los auriculares (reproducir/pausar/avanzar).
- Los enlaces externos se abren en el navegador; los `intent://` a la app oficial se ignoran.

## Reproducción en segundo plano

- `assets/background.js` (inyectado al inicio de cada página de YouTube) le dice a YouTube que la página sigue visible: bloquea `visibilitychange`/`pagehide` y fija `document.hidden = false`, que es lo que usa para pausar.
- El mismo script avisa a la app de qué suena (título, posición, duración) y recibe las órdenes de los controles.
- `PlaybackService` es un servicio en primer plano de tipo `mediaPlayback`: notificación con controles, `MediaSession` para la pantalla de bloqueo y bluetooth, y wake lock + WiFi lock mientras suena.
- Chromium, dentro de la WebView, marca la página como oculta cuando la ventana deja de verse, y en Android pausa los vídeos de páginas ocultas. `KeepAliveWebView` no le informa de que la ventana se ha ocultado mientras la reproducción en segundo plano está activa, y la WebView tampoco se pausa. El renderizador mantiene prioridad alta.
- Si bloqueas el móvil en pantalla completa, la app sale de pantalla completa para que el audio siga.
- En Android 13+ la app pide permiso de notificaciones; sin él el audio sigue, pero la notificación no se ve. Si el móvil corta el audio tras un rato, quita a ShieldTube de la optimización de batería (*Ajustes → Apps → ShieldTube → Batería → Sin restricciones*).
- Para pararlo del todo: botón ✕ de la notificación o cerrar la app desde recientes.

## Instalar el APK

1. Copia `ShieldTube-x.y.z.apk` al móvil.
2. Ábrelo y permite *Instalar apps de origen desconocido* para la app con la que lo abres.
3. Play Protect puede avisar porque la app no viene de Google Play: elige *Instalar de todas formas*.

El APK está firmado con una clave de depuración. Si más adelante instalas uno compilado en otro equipo, Android pedirá desinstalar el anterior primero.

## Compilar

Requisitos: JDK 17+, Android SDK (plataforma 36 y NDK), Rust con los targets de Android, y Node.js + git para generar los scriptlets.

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
sdkmanager "platforms;android-36" "build-tools;36.0.0" "ndk;29.0.14206865"

cd youtube-shield
npm install                       # genera data/resources.json (scriptlets de uBO + Brave)

cd android
./gradlew assembleRelease         # compila Rust para cada ABI y empaqueta el APK
# → app/build/outputs/apk/release/app-release.apk
```

Tests:

```bash
./gradlew testReleaseUnitTest                         # Kotlin
cargo test --release --manifest-path rust/Cargo.toml  # puente JNI y motor
```

## Limitaciones

- WebView no indica el tipo de cada petición (Brave sí lo sabe porque va dentro de Chromium); se deduce de la cabecera `Accept` y la extensión, así que algún filtro que dependa del tipo puede no aplicarse.
- Hace falta un WebView reciente (Android System WebView / Chrome actualizados) para inyectar scriptlets al inicio del documento. Sin eso solo funciona el bloqueo de red.
- Iniciar sesión con Google: la app se presenta como Chrome (sin `; wv` ni la marca "Android WebView" en las Client Hints). Si Google aun así muestra su página de rechazo, la app cambia sola a modo Firefox y reinicia el inicio de sesión. Google puede cambiar su detección en cualquier momento, así que no está garantizado. Para volver al modo normal: *Ajustes → Apps → ShieldTube → Almacenamiento → Borrar datos*.
