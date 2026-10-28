// Sonido 3D para cualquier audífono: el audio del vídeo pasa por Web Audio y se
// convierte en un envolvente virtual, como los sistemas de cine en casa para
// audífonos:
//
//   - Graves (< 120 Hz) aparte, sin filtro binaural y con un realce suave: el
//     HRTF "adelgaza" el bajo y los graves no tienen dirección perceptible.
//   - Bocinas frontales izquierda/derecha (el estéreo original).
//   - Bocina central con lo común a los dos canales (voces, diálogos).
//   - Bocinas traseras con lo que difiere entre canales (ambiente, público,
//     efectos), decorrelacionadas con retardos distintos.
//   - Reverberación de sala por convolución con una respuesta generada.
//
// Cada bocina virtual se coloca con un PannerNode HRTF (filtrado binaural). Un
// limitador evita saturación y una compensación medida por nivel hace que
// activarlo no suene más fuerte.
//
// La app fija el nivel en window.__shieldtubeSpatial (0 = apagado, 1 suave,
// 2 media, 3 inmersivo) y lo cambia en vivo con window.shieldtubeSetSpatial(n).
// Con seguimiento de cabeza, la app llama a window.shieldtubeSetHeadYaw(grados,
// positivo = cabeza girada a la izquierda): se gira el oyente y las bocinas
// virtuales se quedan quietas, como en el audio espacial de los AirPods.
(function () {
  if (window !== window.top) return;

  // angle: bocinas frontales (±grados). center/surround/reverb: niveles de la
  // bocina central, las traseras y la sala. bassDb/presenceDb: realces.
  // makeup: compensación de volumen medida con ruido rosa centrado, lateral y
  // estéreo ancho.
  var PRESETS = [
    null,
    { angle: 30, center: 0.35, surround: 0.25, reverb: 0.3, bassDb: 2, presenceDb: 1, makeup: 0.56 },
    { angle: 40, center: 0.45, surround: 0.45, reverb: 0.5, bassDb: 3, presenceDb: 1.5, makeup: 0.59 },
    { angle: 50, center: 0.5, surround: 0.7, reverb: 0.95, bassDb: 4, presenceDb: 2, makeup: 0.48 },
  ];
  var DISTANCE = 1.5;
  var CROSSOVER = 120;
  var REAR_ANGLE = 110;
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

  function filter(ctx, type, frequency, gainDb) {
    var f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = frequency;
    if (gainDb !== undefined) f.gain.value = gainDb;
    return f;
  }

  function gain(ctx, value) {
    var g = ctx.createGain();
    g.gain.value = value;
    return g;
  }

  function chain() {
    for (var i = 0; i < arguments.length - 1; i++) arguments[i].connect(arguments[i + 1]);
    return arguments[arguments.length - 1];
  }

  /**
   * Respuesta al impulso de una sala (~1,8 s): reflexiones tempranas discretas
   * y cola de ruido con caída exponencial que pierde agudos con el tiempo.
   * Ruido distinto por canal para que la sala suene ancha. Cada canal se lleva
   * a energía 1: así el nivel "reverb" es la proporción real de sala.
   */
  function roomImpulse(ctx) {
    var rate = ctx.sampleRate;
    var length = Math.floor(rate * 1.8);
    var buffer = ctx.createBuffer(2, length, rate);
    for (var ch = 0; ch < 2; ch++) {
      var data = buffer.getChannelData(ch);
      for (var r = 0; r < 8; r++) {
        var at = Math.floor(rate * (0.007 + Math.random() * 0.063));
        data[at] += (r % 2 ? -1 : 1) * (0.5 - r * 0.045);
      }
      var smooth = 0;
      var tailStart = Math.floor(rate * 0.02);
      for (var i = tailStart; i < length; i++) {
        var t = (i - tailStart) / rate;
        var envelope = Math.exp(-6.9 * t / 1.6) * Math.min(1, t / 0.01);
        var damping = 0.9 - 0.75 * (i / length);
        smooth += damping * ((Math.random() * 2 - 1) - smooth);
        data[i] += smooth * envelope * 0.6;
      }
      var energy = 0;
      for (var k = 0; k < length; k++) energy += data[k] * data[k];
      var scale = 1 / Math.sqrt(energy);
      for (var j = 0; j < length; j++) data[j] *= scale;
    }
    return buffer;
  }

  // Divisor de frecuencias Linkwitz-Riley de 4.º orden (dos Butterworth en
  // serie): graves y resto vuelven a sumar plano.
  function crossover(ctx, input) {
    var low = chain(input, filter(ctx, 'lowpass', CROSSOVER), filter(ctx, 'lowpass', CROSSOVER));
    var high = chain(input, filter(ctx, 'highpass', CROSSOVER), filter(ctx, 'highpass', CROSSOVER));
    return { low: low, high: high };
  }

  function build(video) {
    var ctx = new AudioContext({ latencyHint: 'playback' });
    var source = ctx.createMediaElementSource(video);
    var dry = gain(ctx, 1);
    var wet = gain(ctx, 0);
    var out = gain(ctx, 1);
    var limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -2;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.1;
    chain(source, dry, out);

    var bus = gain(ctx, 1);
    var presence = filter(ctx, 'highshelf', 8000, 0);
    chain(bus, presence, wet, out);
    chain(out, limiter, ctx.destination);

    var bands = crossover(ctx, source);
    var bass = filter(ctx, 'lowshelf', 80, 0);
    chain(bands.low, bass, bus);

    var split = ctx.createChannelSplitter(2);
    bands.high.connect(split);

    // Frontales.
    var front = [panner(ctx), panner(ctx)];
    split.connect(front[0], 0);
    split.connect(front[1], 1);
    front[0].connect(bus);
    front[1].connect(bus);

    // Central: (L + R) / 2.
    var centerGain = gain(ctx, 0);
    var center = panner(ctx);
    split.connect(gain(ctx, 0.5), 0).connect(centerGain);
    split.connect(gain(ctx, 0.5), 1).connect(centerGain);
    chain(centerGain, center, bus);

    // Traseras: (L − R) / 2, con signo y retardo distintos a cada lado.
    var side = gain(ctx, 1);
    split.connect(gain(ctx, 0.5), 0).connect(side);
    split.connect(gain(ctx, -0.5), 1).connect(side);
    var rears = [];
    [{ angle: -REAR_ANGLE, sign: 1, delay: 0.011 }, { angle: REAR_ANGLE, sign: -1, delay: 0.017 }].forEach(function (r) {
      var delay = ctx.createDelay(0.05);
      delay.delayTime.value = r.delay;
      var amount = gain(ctx, 0);
      var p = panner(ctx);
      chain(side, gain(ctx, r.sign), delay, filter(ctx, 'lowpass', 6500), amount, p, bus);
      rears.push({ gain: amount, panner: p, angle: r.angle });
    });

    // Sala.
    var convolver = ctx.createConvolver();
    convolver.normalize = false;
    convolver.buffer = roomImpulse(ctx);
    var reverb = gain(ctx, 0);
    chain(bands.high, convolver, filter(ctx, 'highpass', 180), reverb, bus);

    return {
      ctx: ctx, dry: dry, wet: wet, out: out, bass: bass, presence: presence,
      front: front, center: center, centerGain: centerGain, rears: rears, reverb: reverb,
    };
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
    var t = graph.ctx.currentTime;
    // Rampa corta para que cambiar de nivel no dé chasquido.
    var ramp = function (param, value) { param.setTargetAtTime(value, t, 0.05); };
    if (!preset) {
      ramp(graph.dry.gain, 1);
      ramp(graph.wet.gain, 0);
      ramp(graph.out.gain, 1);
      return;
    }
    ramp(graph.dry.gain, 0);
    ramp(graph.wet.gain, 1);
    ramp(graph.out.gain, preset.makeup);
    ramp(graph.bass.gain, preset.bassDb);
    ramp(graph.presence.gain, preset.presenceDb);
    ramp(graph.centerGain.gain, preset.center);
    ramp(graph.reverb.gain, preset.reverb);
    place(graph.front[0], -preset.angle);
    place(graph.front[1], preset.angle);
    place(graph.center, 0);
    graph.rears.forEach(function (rear) {
      ramp(rear.gain.gain, preset.surround);
      place(rear.panner, rear.angle);
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
