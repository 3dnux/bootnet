// Sonido 3D para cualquier audífono: el audio del vídeo pasa por Web Audio y
// se coloca como dos altavoces virtuales delante del oyente (HRTF, el mismo
// filtrado binaural que usan los sistemas de audio espacial) más reflexiones de
// sala desde los lados.
//
// La app fija el nivel en window.__shieldtubeSpatial (0 = apagado, 1 suave,
// 2 media, 3 fuerte) y lo cambia en vivo con window.shieldtubeSetSpatial(n).
// Con seguimiento de cabeza, la app llama a window.shieldtubeSetHeadYaw(grados,
// positivo = cabeza girada a la izquierda): se gira el oyente y los altavoces
// virtuales se quedan quietos, como en el audio espacial de los AirPods.
(function () {
  if (window !== window.top) return;

  // makeup: compensa la subida de volumen del procesado (medida con sonido
  // centrado y lateral) para que activar el 3D no suene más fuerte.
  var PRESETS = [
    null,
    { angle: 30, wet: 0.7, room: 0.18, makeup: 0.77 },
    { angle: 45, wet: 0.9, room: 0.3, makeup: 0.67 },
    { angle: 60, wet: 1.0, room: 0.45, makeup: 0.6 },
  ];
  var DISTANCE = 1.5;
  var graphs = new WeakMap();
  var active = null;
  var headYaw = 0;

  function level() {
    var n = window.__shieldtubeSpatial | 0;
    return n >= 0 && n < PRESETS.length ? n : 0;
  }

  // Solo audio de MediaSource (blob:), que es lo que usa YouTube. Un vídeo de
  // otro origen sin CORS saldría en silencio por Web Audio.
  function canProcess(video) {
    return String(video.currentSrc || video.src).indexOf('blob:') === 0;
  }

  function panner(ctx) {
    var p = ctx.createPanner();
    p.panningModel = 'HRTF';
    p.distanceModel = 'inverse';
    p.refDistance = DISTANCE;
    return p;
  }

  function place(p, degrees) {
    var rad = degrees * Math.PI / 180;
    p.positionX.value = Math.sin(rad) * DISTANCE;
    p.positionY.value = 0;
    p.positionZ.value = -Math.cos(rad) * DISTANCE;
  }

  function build(video) {
    var ctx = new AudioContext({ latencyHint: 'playback' });
    var source = ctx.createMediaElementSource(video);
    var dry = ctx.createGain();
    var wet = ctx.createGain();
    var out = ctx.createGain();
    var limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -2;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.1;

    source.connect(dry);
    dry.connect(out);

    var split = ctx.createChannelSplitter(2);
    source.connect(split);
    var front = [panner(ctx), panner(ctx)];
    var rooms = [];
    for (var ch = 0; ch < 2; ch++) {
      split.connect(front[ch], ch);
      front[ch].connect(wet);
      // Reflexión: copia retrasada y apagada que llega desde el lado contrario.
      var delay = ctx.createDelay(0.1);
      delay.delayTime.value = 0.017 + ch * 0.006;
      var lowpass = ctx.createBiquadFilter();
      lowpass.type = 'lowpass';
      lowpass.frequency.value = 4500;
      var gain = ctx.createGain();
      var side = panner(ctx);
      split.connect(delay, ch);
      delay.connect(lowpass);
      lowpass.connect(gain);
      gain.connect(side);
      side.connect(wet);
      rooms.push({ gain: gain, panner: side, angle: ch === 0 ? 110 : -110 });
    }
    wet.connect(out);
    out.connect(limiter);
    limiter.connect(ctx.destination);
    return { ctx: ctx, dry: dry, wet: wet, out: out, front: front, rooms: rooms };
  }

  // Gira al oyente: con la cabeza a la izquierda, el frente de la escena queda a
  // su derecha. Transición corta (60 ms) para evitar chasquidos.
  function orient(graph) {
    var rad = headYaw * Math.PI / 180;
    var listener = graph.ctx.listener;
    var x = -Math.sin(rad);
    var z = -Math.cos(rad);
    if (listener.forwardX) {
      var t = graph.ctx.currentTime;
      listener.forwardX.setTargetAtTime(x, t, 0.06);
      listener.forwardY.setTargetAtTime(0, t, 0.06);
      listener.forwardZ.setTargetAtTime(z, t, 0.06);
      listener.upX.value = 0;
      listener.upY.value = 1;
      listener.upZ.value = 0;
    } else {
      listener.setOrientation(x, 0, z, 0, 1, 0);
    }
  }

  function apply(graph) {
    var preset = PRESETS[level()];
    if (!preset) {
      graph.dry.gain.value = 1;
      graph.wet.gain.value = 0;
      graph.out.gain.value = 1;
      return;
    }
    graph.dry.gain.value = 1 - preset.wet;
    graph.wet.gain.value = preset.wet;
    graph.out.gain.value = preset.makeup;
    place(graph.front[0], -preset.angle);
    place(graph.front[1], preset.angle);
    graph.rooms.forEach(function (room) {
      room.gain.gain.value = preset.room;
      place(room.panner, room.angle);
    });
    orient(graph);
  }

  // Se engancha al vídeo la primera vez que suena con el 3D activado. Después el
  // grafo se queda (Web Audio no permite soltar un elemento) y "apagado" lo
  // deja pasar sin procesar.
  function attach(video) {
    var graph = graphs.get(video);
    if (!graph) {
      if (!level() || !canProcess(video)) return null;
      graph = build(video);
      graphs.set(video, graph);
    }
    active = graph;
    apply(graph);
    if (graph.ctx.state !== 'running') graph.ctx.resume();
    return graph;
  }

  document.addEventListener('play', function (event) {
    if (event.target instanceof HTMLVideoElement) attach(event.target);
  }, true);

  window.shieldtubeSetHeadYaw = function (degrees) {
    headYaw = Math.max(-90, Math.min(90, Number(degrees) || 0));
    if (active) orient(active);
  };

  window.shieldtubeSetSpatial = function (n) {
    window.__shieldtubeSpatial = n;
    var video = document.querySelector('video');
    if (video && !video.paused) attach(video);
    else if (active) apply(active);
  };
})();
