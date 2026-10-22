function (cfg) {
  // Arranque cosmético de ShieldTube, inyectado al inicio de cada documento de
  // YouTube junto a los scriptlets. Equivale al content script de Brave Shields.
  // `cfg`: { css, generichide, procedural }.
  // `shieldtube` lo crea la app (addWebMessageListener) para pedir filtros genéricos.
  var port = window.shieldtube;

  function addCss(css) {
    if (!css) return;
    try {
      var sheet = new CSSStyleSheet();
      sheet.replaceSync(css);
      document.adoptedStyleSheets = document.adoptedStyleSheets.concat(sheet);
    } catch (e) {
      var style = document.createElement('style');
      style.textContent = css;
      (document.head || document.documentElement).appendChild(style);
    }
  }

  // Llama a `callback` con los nodos nuevos o cambiados, como mucho cada 100 ms.
  function onDomChange(callback) {
    var pending = [];
    var scheduled = false;
    function flush() {
      scheduled = false;
      var nodes = pending;
      pending = [];
      callback(nodes);
    }
    function start() {
      callback([document.documentElement]);
      new MutationObserver(function (mutations) {
        for (var i = 0; i < mutations.length; i++) {
          var m = mutations[i];
          if (m.type === 'attributes') pending.push(m.target);
          else for (var j = 0; j < m.addedNodes.length; j++) pending.push(m.addedNodes[j]);
        }
        if (!scheduled) {
          scheduled = true;
          setTimeout(flush, 100);
        }
      }).observe(document.documentElement, {
        subtree: true, childList: true, attributes: true, attributeFilter: ['class', 'id'],
      });
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', start, { once: true });
    } else {
      start();
    }
  }

  // Filtros genéricos (##.ad-banner): se piden a la app según las clases/ids vistos.
  function watchClassesAndIds() {
    var seenClasses = new Set();
    var seenIds = new Set();
    port.onmessage = function (event) { addCss(event.data); };
    onDomChange(function (nodes) {
      var classes = [];
      var ids = [];
      function collect(el) {
        if (el.id && !seenIds.has(el.id)) { seenIds.add(el.id); ids.push(el.id); }
        var list = el.classList || [];
        for (var i = 0; i < list.length; i++) {
          if (!seenClasses.has(list[i])) { seenClasses.add(list[i]); classes.push(list[i]); }
        }
      }
      for (var i = 0; i < nodes.length; i++) {
        var node = nodes[i];
        if (node.nodeType !== 1) continue;
        collect(node);
        var all = node.querySelectorAll('[id],[class]');
        for (var j = 0; j < all.length; j++) collect(all[j]);
      }
      if (classes.length || ids.length) port.postMessage(JSON.stringify({ classes: classes, ids: ids }));
    });
  }

  // Filtros procedurales sencillos: selector:has-text(...).
  function parseProcedural(actions) {
    var filters = [];
    actions.forEach(function (raw) {
      var filter;
      try { filter = JSON.parse(raw); } catch (e) { return; }
      var ops = filter.selector || [];
      if (!ops.length || ops[0].type !== 'css-selector') return;
      if (!ops.slice(1).every(function (op) { return op.type === 'has-text'; })) return;
      filters.push({
        selector: ops[0].arg,
        texts: ops.slice(1).map(function (op) { return textMatcher(op.arg); }),
        remove: filter.action && filter.action.type === 'remove',
      });
    });
    return filters;
  }

  function textMatcher(arg) {
    var regex = /^\/(.+)\/([imsu]*)$/.exec(arg);
    if (regex) {
      var re = new RegExp(regex[1], regex[2]);
      return function (text) { return re.test(text); };
    }
    return function (text) { return text.indexOf(arg) !== -1; };
  }

  function applyProcedural(filter) {
    var elements;
    try { elements = document.querySelectorAll(filter.selector); } catch (e) { return; }
    for (var i = 0; i < elements.length; i++) {
      var el = elements[i];
      var text = el.textContent || '';
      if (!filter.texts.every(function (matches) { return matches(text); })) continue;
      if (filter.remove) el.remove();
      else el.style.setProperty('display', 'none', 'important');
    }
  }

  addCss(cfg.css);
  if (!cfg.generichide && port) watchClassesAndIds();
  var procedural = parseProcedural(cfg.procedural || []);
  if (procedural.length) onDomChange(function () { procedural.forEach(applyProcedural); });
}
