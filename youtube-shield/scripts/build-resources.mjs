// Genera data/resources.json: los recursos (scriptlets y redirecciones) que
// usa el motor adblock-rust de Brave. Brave los arma igual: scriptlets de
// uBlock Origin + recursos propios de brave/adblock-resources.
//
// Uso: node scripts/build-resources.mjs [--force]

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT_FILE = path.join(ROOT, 'data', 'resources.json');
const UBO_REPO = 'https://github.com/gorhill/uBlock';
const BRAVE_RESOURCES_URL =
  'https://raw.githubusercontent.com/brave/adblock-resources/master/dist/resources.json';

// Mismo bit de permiso que Brave asigna a sus listas por defecto
// (permission_mask: 1 en list_catalog.json) para los scriptlets "trusted".
const TRUSTED_PERMISSION = 1;

function cloneUBlock(dir) {
  execFileSync('git', ['clone', '--quiet', '--depth', '1', '--filter=blob:none', '--sparse', UBO_REPO, dir], {
    stdio: 'inherit',
  });
  execFileSync('git', ['-C', dir, 'sparse-checkout', 'set', '--no-cone',
    '/src/js/', '/src/web_accessible_resources/'], {
    stdio: 'inherit',
  });
}

async function scriptletResources(uboDir) {
  // Los scriptlets de uBO son módulos ES sin package.json propio.
  fs.writeFileSync(path.join(uboDir, 'src', 'js', 'package.json'), '{"type":"module"}\n');
  const entry = pathToFileURL(path.join(uboDir, 'src', 'js', 'resources', 'scriptlets.js'));
  const { builtinScriptlets } = await import(entry.href);
  return builtinScriptlets.map((s) => ({
    name: s.name,
    aliases: s.aliases ?? [],
    kind: { mime: s.name.endsWith('.fn') ? 'fn/javascript' : 'application/javascript' },
    content: Buffer.from(s.fn.toString()).toString('base64'),
    dependencies: s.dependencies ?? [],
    permission: s.requiresTrust ? TRUSTED_PERMISSION : 0,
  }));
}

function redirectResources(uboDir) {
  const { uBlockResources } = require('adblock-rs');
  return uBlockResources(
    path.join(uboDir, 'src', 'web_accessible_resources'),
    path.join(uboDir, 'src', 'js', 'redirect-resources.js'),
  );
}

async function braveResources() {
  const res = await fetch(BRAVE_RESOURCES_URL);
  if (!res.ok) throw new Error(`${BRAVE_RESOURCES_URL}: HTTP ${res.status}`);
  return res.json();
}

async function main() {
  if (fs.existsSync(OUT_FILE) && !process.argv.includes('--force')) {
    console.log(`[shieldtube] ${path.relative(ROOT, OUT_FILE)} ya existe (usa --force para regenerar)`);
    return;
  }
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'shieldtube-ubo-'));
  try {
    const uboDir = path.join(tmp, 'uBlock');
    cloneUBlock(uboDir);
    const byName = new Map();
    // Orden de prioridad: los de Brave pisan a los de uBO si comparten nombre.
    for (const r of [...redirectResources(uboDir), ...(await scriptletResources(uboDir)), ...(await braveResources())]) {
      byName.set(r.name, r);
    }
    fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
    fs.writeFileSync(OUT_FILE, JSON.stringify([...byName.values()]));
    console.log(`[shieldtube] ${byName.size} recursos escritos en ${path.relative(ROOT, OUT_FILE)}`);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error('[shieldtube] No se pudieron generar los recursos:', err);
  process.exit(1);
});
