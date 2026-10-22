// Motor de bloqueo: adblock-rust (el mismo motor de Brave Shields) cargado con
// las listas que Brave activa por defecto según su list_catalog.json.

const fs = require('node:fs');
const path = require('node:path');
const adblock = require('adblock-rs');

const CATALOG_URL =
  'https://raw.githubusercontent.com/brave/adblock-resources/master/filter_lists/list_catalog.json';
const MAX_AGE_MS = 24 * 60 * 60 * 1000;
const PLATFORM = 'DESKTOP';

// Listas de Brave usadas si no hay red ni caché para descargar el catálogo.
const FALLBACK_CATALOG = [
  {
    title: 'Brave Default Adblock Filters',
    default_enabled: true,
    permission_mask: 1,
    sources: [
      'https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/filters.txt',
      'https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/quick-fixes.txt',
      'https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/unbreak.txt',
      'https://easylist.to/easylist/easylist.txt',
      'https://raw.githubusercontent.com/brave/adblock-lists/master/brave-lists/brave-specific.txt',
      'https://raw.githubusercontent.com/brave/adblock-lists/master/brave-lists/brave-unbreak.txt',
    ].map((url) => ({ url, format: 'Standard' })),
  },
  {
    title: 'Brave Default Privacy Filters',
    default_enabled: true,
    permission_mask: 1,
    sources: [
      'https://easylist.to/easylist/easyprivacy.txt',
      'https://raw.githubusercontent.com/uBlockOrigin/uAssets/master/filters/privacy.txt',
    ].map((url) => ({ url, format: 'Standard' })),
  },
];

function isFresh(file) {
  try {
    return Date.now() - fs.statSync(file).mtimeMs < MAX_AGE_MS;
  } catch {
    return false;
  }
}

function cacheName(url) {
  return url.replace(/^https?:\/\//, '').replace(/[^a-zA-Z0-9.-]+/g, '_');
}

// Devuelve el texto de `url`, usando la copia en caché si es reciente o si la
// descarga falla.
async function cachedFetch(url, cacheDir) {
  const file = path.join(cacheDir, cacheName(url));
  if (isFresh(file)) return fs.readFileSync(file, 'utf8');
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    fs.writeFileSync(file, text);
    return text;
  } catch (err) {
    if (fs.existsSync(file)) return fs.readFileSync(file, 'utf8');
    console.warn(`[shieldtube] No se pudo descargar ${url}: ${err.message}`);
    return null;
  }
}

function enabledLists(catalog) {
  return catalog.filter((list) => list.default_enabled
    && (!list.platforms || list.platforms.includes(PLATFORM)));
}

async function loadCatalog(cacheDir) {
  const text = await cachedFetch(CATALOG_URL, cacheDir);
  if (text) {
    try {
      return JSON.parse(text);
    } catch (err) {
      console.warn(`[shieldtube] Catálogo de listas inválido: ${err.message}`);
    }
  }
  return FALLBACK_CATALOG;
}

async function buildEngine({ cacheDir, resourcesFile }) {
  fs.mkdirSync(cacheDir, { recursive: true });
  const filterSet = new adblock.FilterSet(true);
  const lists = enabledLists(await loadCatalog(cacheDir));
  let loaded = 0;
  for (const list of lists) {
    const texts = await Promise.all(list.sources.map((s) => cachedFetch(s.url, cacheDir)));
    list.sources.forEach((source, i) => {
      if (texts[i] === null) return;
      filterSet.addFilters(texts[i], {
        format: source.format === 'Hosts' ? 'Hosts' : 'Standard',
        permissions: list.permission_mask ?? 0,
      });
      loaded += 1;
    });
  }
  const engine = new adblock.Engine(filterSet);
  if (resourcesFile && fs.existsSync(resourcesFile)) {
    engine.useResources(JSON.parse(fs.readFileSync(resourcesFile, 'utf8')));
  } else {
    console.warn('[shieldtube] Sin data/resources.json: no habrá scriptlets (ejecuta npm run update-resources)');
  }
  console.log(`[shieldtube] Motor listo con ${loaded} listas de filtros de Brave`);
  return engine;
}

module.exports = { buildEngine, enabledLists, cacheName, CATALOG_URL, FALLBACK_CATALOG };
