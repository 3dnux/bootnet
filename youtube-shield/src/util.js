// Funciones puras compartidas (sin dependencias de Electron, para poder testearlas).

// Tipos de recurso de Electron -> tipos de petición que entiende adblock-rust.
const REQUEST_TYPES = {
  mainFrame: 'document',
  subFrame: 'subdocument',
  stylesheet: 'stylesheet',
  script: 'script',
  image: 'image',
  font: 'font',
  object: 'object',
  xhr: 'xmlhttprequest',
  ping: 'ping',
  cspReport: 'csp_report',
  media: 'media',
  webSocket: 'websocket',
};

function toAdblockRequestType(resourceType) {
  return REQUEST_TYPES[resourceType] || 'other';
}

// Dominios que se navegan dentro de la app (YouTube y el inicio de sesión de
// Google). El resto se abre en el navegador del sistema.
const IN_APP_HOSTS = ['youtube.com', 'youtu.be', 'youtube-nocookie.com', 'google.com'];

function isAllowedInApp(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  if (parsed.protocol !== 'https:') return false;
  return IN_APP_HOSTS.some((h) => parsed.hostname === h || parsed.hostname.endsWith(`.${h}`));
}

// Una regla por selector: un selector inválido en una lista anularía la regla entera.
function hideCss(selectors) {
  if (!selectors) return '';
  return [...selectors].map((s) => `${s}{display:none!important}`).join('\n');
}

module.exports = { toAdblockRequestType, isAllowedInApp, hideCss };
