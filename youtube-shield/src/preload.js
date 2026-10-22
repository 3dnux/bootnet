// Se ejecuta al inicio de cada documento (y de cada iframe), antes que los
// scripts de la página: inyecta scriptlets y CSS cosmético, como Brave Shields.
const { ipcRenderer, webFrame } = require('electron');

const cosmetic = ipcRenderer.sendSync('shield:cosmetic', location.href);

if (cosmetic) {
  if (cosmetic.script) {
    // Mundo principal de la página: los scriptlets necesitan tocar sus objetos.
    webFrame.executeJavaScript(cosmetic.script).catch(() => {});
  }
  if (cosmetic.css) webFrame.insertCSS(cosmetic.css, { cssOrigin: 'user' });
  if (!cosmetic.generichide) watchClassesAndIds(cosmetic.exceptions || []);
  const procedural = parseProcedural(cosmetic.procedural || []);
  if (procedural.length) onDomChange(() => procedural.forEach(applyProcedural));
}

// Llama a `callback` cuando el DOM está listo y, como mucho cada 100 ms,
// cuando cambia. Recibe los nodos añadidos y los que cambiaron de class/id.
function onDomChange(callback) {
  let pending = [];
  let scheduled = false;
  const flush = () => {
    scheduled = false;
    const nodes = pending;
    pending = [];
    callback(nodes);
  };
  const start = () => {
    callback([document.documentElement]);
    new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.type === 'attributes') pending.push(m.target);
        else pending.push(...m.addedNodes);
      }
      if (!scheduled) {
        scheduled = true;
        setTimeout(flush, 100);
      }
    }).observe(document.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['class', 'id'],
    });
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, { once: true });
  } else {
    start();
  }
}

// Filtros genéricos (##.ad-banner, ###sponsor...): se piden al motor según
// las clases e ids que van apareciendo en la página.
function watchClassesAndIds(exceptions) {
  const seenClasses = new Set();
  const seenIds = new Set();

  onDomChange((nodes) => {
    const classes = [];
    const ids = [];
    const collect = (el) => {
      if (el.id && !seenIds.has(el.id)) {
        seenIds.add(el.id);
        ids.push(el.id);
      }
      for (const c of el.classList || []) {
        if (!seenClasses.has(c)) {
          seenClasses.add(c);
          classes.push(c);
        }
      }
    };
    for (const node of nodes) {
      if (node.nodeType !== Node.ELEMENT_NODE) continue;
      collect(node);
      for (const el of node.querySelectorAll('[id],[class]')) collect(el);
    }
    if (!classes.length && !ids.length) return;
    ipcRenderer.invoke('shield:classids', classes, ids, exceptions)
      .then((css) => { if (css) webFrame.insertCSS(css, { cssOrigin: 'user' }); })
      .catch(() => {});
  });
}

// Filtros procedurales sencillos (selector:has-text(...)), que en YouTube
// ocultan encuestas y promociones. Los operadores no soportados se ignoran.
function parseProcedural(actions) {
  const filters = [];
  for (const raw of actions) {
    let filter;
    try {
      filter = JSON.parse(raw);
    } catch {
      continue;
    }
    const [first, ...rest] = filter.selector || [];
    if (first?.type !== 'css-selector' || !rest.every((op) => op.type === 'has-text')) continue;
    filters.push({
      selector: first.arg,
      texts: rest.map((op) => textMatcher(op.arg)),
      remove: filter.action?.type === 'remove',
    });
  }
  return filters;
}

function textMatcher(arg) {
  const regex = /^\/(.+)\/([imsu]*)$/.exec(arg);
  if (regex) {
    const re = new RegExp(regex[1], regex[2]);
    return (text) => re.test(text);
  }
  return (text) => text.includes(arg);
}

function applyProcedural(filter) {
  let elements;
  try {
    elements = document.querySelectorAll(filter.selector);
  } catch {
    return;
  }
  for (const el of elements) {
    const text = el.textContent || '';
    if (!filter.texts.every((matches) => matches(text))) continue;
    if (filter.remove) el.remove();
    else el.style.setProperty('display', 'none', 'important');
  }
}
