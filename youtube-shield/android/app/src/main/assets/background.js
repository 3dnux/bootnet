// Reproducción en segundo plano. Se inyecta al inicio de cada documento de
// YouTube, antes que sus scripts.
(function () {
  // 1. YouTube pausa el vídeo cuando la página deja de estar visible (pantalla
  //    bloqueada, otra app delante). Se le dice que siempre está visible.
  var visible = function (name, value) {
    try {
      Object.defineProperty(Document.prototype, name, { get: function () { return value; }, configurable: true });
    } catch (e) {}
  };
  visible('hidden', false);
  visible('webkitHidden', false);
  visible('visibilityState', 'visible');
  visible('webkitVisibilityState', 'visible');
  Document.prototype.hasFocus = function () { return true; };
  ['visibilitychange', 'webkitvisibilitychange', 'pagehide', 'freeze', 'blur'].forEach(function (type) {
    window.addEventListener(type, function (event) {
      if (event.target === document || event.target === window) event.stopImmediatePropagation();
    }, true);
  });

  // 2. Informa a la app de qué suena (para la notificación y la pantalla de
  //    bloqueo) y recibe sus órdenes: play, pause, seek:<segundos>.
  if (window !== window.top) return;
  var current = null;
  var port = null;

  // `shieldtubeMedia` lo crea la app (addWebMessageListener).
  function getPort() {
    if (!port && window.shieldtubeMedia) {
      port = window.shieldtubeMedia;
      port.onmessage = onCommand;
    }
    return port;
  }
  var timer = 0;

  function info() {
    var meta = navigator.mediaSession && navigator.mediaSession.metadata;
    var title = (meta && meta.title) || document.title.replace(/ - YouTube$/, '');
    return {
      playing: !!current && !current.paused && !current.ended,
      title: title || 'YouTube',
      artist: (meta && meta.artist) || '',
      position: current ? Math.floor(current.currentTime * 1000) : 0,
      duration: current && isFinite(current.duration) ? Math.floor(current.duration * 1000) : 0,
    };
  }

  function report() {
    clearTimeout(timer);
    // Pequeña espera: YouTube actualiza el título y los metadatos tras el evento.
    timer = setTimeout(function () {
      if (getPort()) port.postMessage(JSON.stringify(info()));
    }, 300);
  }

  ['play', 'playing', 'pause', 'ended', 'seeked', 'durationchange'].forEach(function (type) {
    document.addEventListener(type, function (event) {
      if (event.target instanceof HTMLMediaElement) {
        current = event.target;
        report();
      }
    }, true);
  });

  function onCommand(event) {
    var video = current || document.querySelector('video');
    if (!video) return;
    var command = String(event.data);
    if (command === 'play') video.play();
    else if (command === 'pause') video.pause();
    else if (command.indexOf('seek:') === 0) video.currentTime = parseFloat(command.slice(5));
  }
})();
