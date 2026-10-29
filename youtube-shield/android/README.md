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
- Los enlaces de YouTube pueden abrirse con ShieldTube (aparece en el selector de apps) o enviarse con **Compartir → ShieldTube** desde la app de YouTube.
- **Enlaces copiados:** al abrir o volver a la app, si hay un enlace de YouTube en el portapapeles (vídeo, Short, directo, lista o canal; también `youtu.be` y `music.youtube.com`), pregunta "¿Ver este vídeo?" con el título y el canal (oEmbed de YouTube) y lo abre. No pregunta dos veces por el mismo enlace ni por el vídeo que ya se está viendo, y solo lee el portapapeles cuando cambia lo copiado. Los parámetros de rastreo (`si`, `feature`, `pp`) se quitan; el minuto (`t`) y la lista se conservan.
- Pantalla completa con giro automático y botón atrás para volver.
- **Reproducción en segundo plano:** el vídeo sigue sonando al salir de la app o bloquear el móvil, con notificación y controles en la pantalla de bloqueo y en los auriculares (reproducir/pausar/avanzar).
- Los enlaces externos se abren en el navegador; los `intent://` a la app oficial se ignoran.
- **Sonido 3D** para cualquier audífono: botón "3D" en la página del vídeo y en la notificación (apagado → suave → media → inmersivo).

## Sonido 3D

`assets/spatial.js` convierte el estéreo del vídeo en un envolvente virtual, como los sistemas de cine en casa para audífonos. Todo con Web Audio:

| Parte | Cómo | Para qué |
|---|---|---|
| Graves (< 120 Hz) | Divisor Linkwitz-Riley de 4.º orden, sin HRTF, realce de +2/+3/+4 dB | El bajo conserva cuerpo y golpe (el HRTF lo adelgaza) |
| Bocinas frontales | L y R a ±30°, ±40° o ±50° | Escenario estéreo delante de ti |
| Centro reforzado | (L+R)/2 a las dos frontales por igual | Voces y diálogos anclados delante |
| Traseras | (L−R)/2 con signo y retardo distintos (11/17 ms), paso bajo de 6,5 kHz y "sombra" de la cabeza hacia el otro oído (0,55 ms, paso bajo de 2,2 kHz) | Ambiente, público y efectos alrededor |
| Sala | Convolución con una respuesta generada de 0,45 s (reflexiones tempranas + cola de ruido que pierde agudos), distinta por canal | Sensación de estar en un lugar real |
| Brillo | Realce de agudos de +1/+1,5/+2 dB a 8 kHz | Claridad |

Las frontales son `PannerNode` en modo HRTF (filtrado binaural). Un limitador evita saturación, y una compensación medida con ruido rosa (centrado, lateral y estéreo ancho) deja cada nivel a ±1,3 dB del original. Los cambios de nivel van con rampa para no dar chasquidos.

Medido en Chromium: la correlación entre oídos de un sonido centrado baja de 1,0 (sin efecto) a 0,98 / 0,96 / 0,83 (suave / media / inmersivo). Cuanto más baja, más amplio y envolvente se percibe.

**Rendimiento y calidad.** Historial de la sala:
- **0.7.0:** convolución de 1,8 s más 5 filtros HRTF. Demasiado trabajo para el hilo de audio del móvil, y el sonido se cortaba.
- **0.7.1:** red de peines realimentados (tipo Freeverb), barata pero con una cola tonal que sonaba metálica. En la respuesta al impulso, su "lisura espectral" (de 0 = tonos puros a 1 = ruido liso) era 0,00.
- **Ahora:** convolución corta de 0,45 s. Su lisura es 0,54, más natural incluso que la de 1,8 s (0,41), y solo hay 2 HRTF.

Renderizando 30 s sin tiempo real cuesta un 6-7 % del tiempo real, igual que la 0.6.1, frente al 9-10 % de la 0.7.0. Si aun así Chromium informa de cortes (`playbackStats.underrunEvents`: 3 o más en 8 s), el 3D pasa solo a un modo ligero, sin sala ni traseras.

- El sonido sale "de delante" y te rodea, en vez de quedarse dentro de la cabeza. YouTube es estéreo, así que es una simulación, no Dolby Atmos.
- Se recuerda entre sesiones y funciona en segundo plano.
- Solo se aplica a audio de MediaSource (`blob:`), que es lo que usa YouTube; cualquier otro audio se deja intacto para que no salga en silencio.
- Si se combina con el ecualizador o el "360 Reality Audio" de la app de Sony, el efecto se suma. Mejor usar solo uno de los dos procesados espaciales.

### Seguimiento de cabeza

Mantén pulsado el botón "3D" para activarlo (el botón se rellena de naranja). Al girar la cabeza, las bocinas virtuales se quedan quietas delante de ti, como el audio espacial de los AirPods: se gira el oyente de Web Audio (`shieldtubeSetHeadYaw`). `HeadTracker` usa la mejor fuente disponible:

1. **Sensor de los audífonos** (`Sensor.TYPE_HEAD_TRACKER`, p. ej. Sony WH-1000XM5 con su firmware de seguimiento de cabeza). Android lo reserva al sistema; para que la app pueda leerlo hay que ejecutar, desde una computadora con depuración USB (o depuración inalámbrica), `adb shell cmd sensorservice unrestrict-ht`. Dura hasta reiniciar el móvil. Además, el móvil debe soportar el protocolo de head tracking de Android 13+, que algunos fabricantes no activan. Funciona también con la pantalla apagada, y el frente se recentra despacio si cambias de postura.
2. **Cámara frontal** con detección facial de ML Kit (a través de Google Play Services), procesada en el propio móvil: las imágenes no se guardan ni se envían. Solo funciona con la app en pantalla y en la página de un vídeo, y mientras miras la pantalla. El frente es la pantalla. Android muestra el indicador verde de cámara mientras está activa. Si no detecta tu cara, el sonido vuelve al frente. Con el APK instalado a mano, la app pide a Play Services que descargue el modelo facial (`ModuleInstallClient`) y muestra el progreso.

Junto al botón 3D aparece el estado del seguimiento: "Descargando modelo facial 40 %", "Buscando tu cara…", "Cara 12°", "Audífonos 8°", o el error si algo falla.

La matemática (vector de Euler → yaw, recentrado, suavizado) está en `HeadMath.kt` con tests; la dirección se verificó en Chromium (cabeza a la izquierda → el sonido pasa al oído derecho, con el volumen total estable).

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
- Iniciar sesión con Google: la app se presenta como Chrome (sin `; wv` ni la marca "Android WebView" en las Client Hints). Si Google aun así muestra su página de rechazo, la app cambia sola a modo Firefox y reinicia el inicio de sesión. Google puede cambiar su detección en cualquier momento, así que no está garantizado. Para volver al modo normal: *Ajustes → Apps → ShieldTube → Almacenamiento → Borrar datos*. Si Google pide la cámara para verificar tu identidad, la app se la presta solo a `accounts.google.com` y solo la cámara (nunca el micrófono), tras el permiso de Android; mientras tanto el seguimiento de cabeza suelta la cámara.
