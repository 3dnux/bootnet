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
- Los enlaces externos se abren en el navegador; los `intent://` a la app oficial se ignoran.

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
- No hay reproducción en segundo plano ni con la pantalla apagada.
- Google puede rechazar el inicio de sesión dentro de WebViews. YouTube funciona sin cuenta.
