# ShieldTube — YouTube con el bloqueador de anuncios de Brave

App de escritorio (Windows, macOS, Linux) y **Android** ([`android/`](android/README.md)) que abre YouTube en su propia ventana, como una PWA instalada, pero con **el mismo motor de bloqueo que Brave Shields**:

- **Motor:** [`adblock-rust`](https://github.com/brave/adblock-rust) (`adblock-rs` en npm), el motor que Brave usa por dentro.
- **Listas:** las que Brave activa por defecto, leídas de su [`list_catalog.json`](https://github.com/brave/adblock-resources/blob/master/filter_lists/list_catalog.json): EasyList, EasyPrivacy, filtros de uBlock Origin (incluido *quick-fixes*, que es donde están los arreglos para YouTube), listas propias de Brave, avisos de cookies y más. Se descargan al arrancar y se guardan en caché 24 h.
- **Scriptlets y redirecciones:** igual que Brave: los scriptlets de uBlock Origin más los recursos propios de Brave ([`brave/adblock-resources`](https://github.com/brave/adblock-resources)). Son los que eliminan los anuncios **dentro del reproductor** (p. ej. `json-prune` sobre `adPlacements`).

## Cómo funciona

| Capa de Brave Shields | En ShieldTube |
|---|---|
| Bloqueo de peticiones de red | `session.webRequest.onBeforeRequest` → `engine.check()`; cancela o redirige a un recurso vacío |
| Scriptlets (`##+js(...)`) | El *preload* los inyecta en el mundo de la página antes que sus propios scripts |
| Ocultación cosmética (`##selector`) | CSS específico del sitio + selectores genéricos según las clases/ids que aparecen |
| Filtros procedurales (`:has-text`) | Soporte básico (`selector:has-text(...)`), usado en YouTube para encuestas y promos |

Navegar por YouTube y por el inicio de sesión de Google ocurre dentro de la app; cualquier otro enlace se abre en el navegador del sistema.

El menú **Shields** permite desactivar el bloqueo, y el título de la ventana muestra cuántas peticiones se han bloqueado en la página actual.

## Requisitos

- [Node.js](https://nodejs.org/) 20 o superior
- [Rust](https://rustup.rs/) (`cargo`): `adblock-rs` se compila al instalar
- `git` (para descargar los scriptlets de uBlock Origin)

## Uso

```bash
cd youtube-shield
npm install        # compila el motor de Brave y genera data/resources.json
npm start          # abre la app
```

Otros comandos:

```bash
npm test                  # tests del motor y utilidades (sin red)
npm run update-resources  # vuelve a descargar los scriptlets
npm run dist              # crea instalador (AppImage/deb, exe o dmg) en dist/
```

La primera vez tarda unos segundos en descargar las listas; después arranca desde caché.

## ¿Por qué no es una PWA "pura"?

Una PWA es una página web: el navegador no le deja modificar ni filtrar lo que carga `youtube.com`, así que no puede quitar anuncios por su cuenta. Brave lo consigue porque el bloqueo está en el propio navegador. Por eso ShieldTube trae su propio navegador (Electron/Chromium) con el motor de Brave dentro: se instala y se ve como una app de YouTube, pero el bloqueo funciona de verdad.

**Alternativa sin código:** si usas Brave, abre `youtube.com` y elige *Menú → Instalar YouTube* (o *Añadir a pantalla de inicio* en Android). Esa PWA se ejecuta dentro de Brave y conserva Shields.

## Limitaciones

- YouTube cambia a menudo su forma de servir anuncios. Las listas se actualizan solas cada 24 h; si aparece un anuncio nuevo, cierra y abre la app al día siguiente o borra la caché (carpeta `filter-lists` en el directorio de datos de la app).
- Google a veces rechaza el inicio de sesión en navegadores integrados. YouTube funciona sin cuenta.
- Los filtros procedurales avanzados (`:upward`, `:matches-css`, etc.) no están implementados.

## Licencias de terceros

`adblock-rs` y `brave/adblock-resources` son MPL-2.0. Los scriptlets de uBlock Origin son GPL-3.0; se descargan durante la instalación y no se incluyen en este repositorio.
