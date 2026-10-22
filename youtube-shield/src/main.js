const { app, BrowserWindow, Menu, ipcMain, session, shell } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { buildEngine } = require('./engine');
const { toAdblockRequestType, isAllowedInApp, hideCss } = require('./util');

const HOME_URL = 'https://www.youtube.com/';
const RESOURCES_FILE = path.join(__dirname, '..', 'data', 'resources.json');
const SETTINGS_FILE = path.join(app.getPath('userData'), 'settings.json');

let engine = null;
let mainWindow = null;
let blockedCount = 0;
const settings = loadSettings();

function loadSettings() {
  try {
    return { shields: true, ...JSON.parse(fs.readFileSync(SETTINGS_FILE, 'utf8')) };
  } catch {
    return { shields: true };
  }
}

function saveSettings() {
  fs.writeFileSync(SETTINGS_FILE, JSON.stringify(settings));
}

function shieldsActive() {
  return settings.shields && engine !== null;
}

function updateTitle() {
  if (!mainWindow) return;
  const state = settings.shields ? `Shields ON · ${blockedCount} bloqueados` : 'Shields OFF';
  mainWindow.setTitle(`ShieldTube — ${state}`);
}

// Bloqueo de red: cada petición pasa por engine.check(), como en Brave Shields.
function installNetworkBlocking(ses) {
  ses.webRequest.onBeforeRequest({ urls: ['http://*/*', 'https://*/*'] }, (details, callback) => {
    if (!shieldsActive()) return callback({});
    const requestType = toAdblockRequestType(details.resourceType);
    const sourceUrl = requestType === 'document'
      ? details.url
      : (details.frame?.url || details.referrer || details.url);
    let result;
    try {
      result = engine.check(details.url, sourceUrl, requestType, details.method, true);
    } catch {
      return callback({});
    }
    if (result.redirect && result.should_block) {
      blockedCount += 1;
      updateTitle();
      return callback({ redirectURL: result.redirect });
    }
    if (result.should_block) {
      blockedCount += 1;
      updateTitle();
      return callback({ cancel: true });
    }
    if (result.rewritten_url && result.rewritten_url !== details.url) {
      return callback({ redirectURL: result.rewritten_url });
    }
    return callback({});
  });
}

// Filtrado cosmético y scriptlets (lo que elimina los anuncios dentro del
// reproductor de YouTube). El preload lo pide al empezar cada documento.
ipcMain.on('shield:cosmetic', (event, url) => {
  if (!shieldsActive() || typeof url !== 'string' || !/^https?:/.test(url)) {
    event.returnValue = null;
    return;
  }
  try {
    const res = engine.urlCosmeticResources(url);
    event.returnValue = {
      css: hideCss(res.hide_selectors),
      script: res.injected_script || '',
      generichide: res.generichide,
      exceptions: res.exceptions,
      procedural: res.procedural_actions,
    };
  } catch {
    event.returnValue = null;
  }
});

ipcMain.handle('shield:classids', (_event, classes, ids, exceptions) => {
  if (!shieldsActive()) return '';
  if (![classes, ids, exceptions].every(Array.isArray)) return '';
  return hideCss(engine.hiddenClassIdSelectors(classes, ids, exceptions));
});

function goHome() {
  mainWindow?.loadURL(HOME_URL);
}

function buildMenu() {
  const template = [
    {
      label: 'Navegación',
      submenu: [
        { label: 'Inicio', accelerator: 'Alt+Home', click: goHome },
        { label: 'Atrás', accelerator: 'Alt+Left', click: () => mainWindow?.webContents.navigationHistory.goBack() },
        { label: 'Adelante', accelerator: 'Alt+Right', click: () => mainWindow?.webContents.navigationHistory.goForward() },
        { role: 'reload', label: 'Recargar' },
        { type: 'separator' },
        { role: 'quit', label: 'Salir' },
      ],
    },
    {
      label: 'Shields',
      submenu: [
        {
          label: 'Bloquear anuncios y rastreadores',
          type: 'checkbox',
          checked: settings.shields,
          click: (item) => {
            settings.shields = item.checked;
            saveSettings();
            updateTitle();
            mainWindow?.webContents.reload();
          },
        },
      ],
    },
    {
      label: 'Ver',
      submenu: [
        { role: 'togglefullscreen', label: 'Pantalla completa' },
        { role: 'zoomIn', label: 'Acercar' },
        { role: 'zoomOut', label: 'Alejar' },
        { role: 'resetZoom', label: 'Tamaño real' },
        { type: 'separator' },
        { role: 'toggleDevTools', label: 'Herramientas de desarrollo' },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function openExternally(url) {
  if (/^https?:\/\//.test(url)) shell.openExternal(url);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: '#0f0f0f',
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      nodeIntegrationInSubFrames: true,
    },
  });
  const { webContents } = mainWindow;

  // YouTube y Google tratan mejor a un Chrome normal que a "Electron".
  webContents.setUserAgent(webContents.getUserAgent()
    .replace(/\s?Electron\/\S+/, '')
    .replace(new RegExp(`\\s?${app.getName()}\\/\\S+`), ''));

  webContents.setWindowOpenHandler(({ url }) => {
    if (isAllowedInApp(url)) {
      webContents.loadURL(url);
    } else {
      openExternally(url);
    }
    return { action: 'deny' };
  });
  webContents.on('will-navigate', (event, url) => {
    if (!isAllowedInApp(url)) {
      event.preventDefault();
      openExternally(url);
    }
  });
  webContents.on('did-start-navigation', ({ isMainFrame, isSameDocument }) => {
    if (isMainFrame && !isSameDocument) {
      blockedCount = 0;
      updateTitle();
    }
  });
  mainWindow.on('page-title-updated', (event) => event.preventDefault());
  mainWindow.on('closed', () => { mainWindow = null; });

  mainWindow.loadFile(path.join(__dirname, 'loading.html'));
  updateTitle();
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  });

  app.whenReady().then(async () => {
    buildMenu();
    createWindow();
    installNetworkBlocking(session.defaultSession);
    try {
      engine = await buildEngine({
        cacheDir: path.join(app.getPath('userData'), 'filter-lists'),
        resourcesFile: RESOURCES_FILE,
      });
    } catch (err) {
      console.error('[shieldtube] No se pudo crear el motor de bloqueo:', err);
    }
    updateTitle();
    goHome();
  });

  app.on('window-all-closed', () => app.quit());
}
