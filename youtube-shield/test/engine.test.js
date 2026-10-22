// Construye el motor sin red: el catálogo y la lista se dejan en la caché.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { buildEngine, enabledLists, cacheName, CATALOG_URL } = require('../src/engine');

const LIST_URL = 'https://lists.example/test.txt';
const IOS_LIST_URL = 'https://lists.example/ios.txt';

async function offlineEngine() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'shieldtube-test-'));
  const cacheDir = path.join(dir, 'cache');
  fs.mkdirSync(cacheDir);
  fs.writeFileSync(path.join(cacheDir, cacheName(CATALOG_URL)), JSON.stringify([
    { title: 'Test', default_enabled: true, permission_mask: 1, sources: [{ url: LIST_URL, format: 'Standard' }] },
    { title: 'iOS', default_enabled: true, platforms: ['IOS'], sources: [{ url: IOS_LIST_URL, format: 'Standard' }] },
  ]));
  fs.writeFileSync(path.join(cacheDir, cacheName(LIST_URL)), [
    '||ads.example^',
    '||tracker.example/lib.js$script,redirect=noopjs',
    'video.example##.banner-ad',
    'video.example##+js(noop-scriptlet)',
  ].join('\n'));
  fs.writeFileSync(path.join(cacheDir, cacheName(IOS_LIST_URL)), '||ios-only.example^');
  const resourcesFile = path.join(dir, 'resources.json');
  fs.writeFileSync(resourcesFile, JSON.stringify([
    {
      name: 'noopjs', aliases: [], kind: { mime: 'application/javascript' },
      content: Buffer.from('(function(){})()').toString('base64'),
    },
    {
      name: 'noop-scriptlet.js', aliases: [], kind: 'template',
      content: Buffer.from('window.__shieldtube = true;').toString('base64'),
    },
  ]));
  return buildEngine({ cacheDir, resourcesFile });
}

test('solo se usan las listas por defecto de escritorio', () => {
  const lists = enabledLists([
    { title: 'a', default_enabled: true },
    { title: 'b', default_enabled: false },
    { title: 'c', default_enabled: true, platforms: ['IOS'] },
    { title: 'd', default_enabled: true, platforms: ['DESKTOP', 'ANDROID'] },
  ]);
  assert.deepEqual(lists.map((l) => l.title), ['a', 'd']);
});

test('bloquea, redirige y genera filtros cosméticos', async () => {
  const engine = await offlineEngine();

  const blocked = engine.check('https://ads.example/banner.png', 'https://video.example/', 'image', 'GET', true);
  assert.equal(blocked.should_block, true);

  const allowed = engine.check('https://video.example/app.js', 'https://video.example/', 'script', 'GET', true);
  assert.equal(allowed.should_block, false);

  const ios = engine.check('https://ios-only.example/x', 'https://video.example/', 'image', 'GET', true);
  assert.equal(ios.should_block, false);

  const redirected = engine.check('https://tracker.example/lib.js', 'https://video.example/', 'script', 'GET', true);
  assert.equal(redirected.should_block, true);
  assert.match(redirected.redirect, /^data:application\/javascript;base64,/);

  const cosmetic = engine.urlCosmeticResources('https://video.example/watch');
  assert.ok(cosmetic.hide_selectors.includes('.banner-ad'));
  assert.match(cosmetic.injected_script, /window\.__shieldtube = true/);
});
