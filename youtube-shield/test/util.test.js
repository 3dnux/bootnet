const test = require('node:test');
const assert = require('node:assert/strict');
const { toAdblockRequestType, isAllowedInApp, hideCss } = require('../src/util');

test('traduce los tipos de recurso de Electron a los de adblock-rust', () => {
  assert.equal(toAdblockRequestType('mainFrame'), 'document');
  assert.equal(toAdblockRequestType('subFrame'), 'subdocument');
  assert.equal(toAdblockRequestType('xhr'), 'xmlhttprequest');
  assert.equal(toAdblockRequestType('desconocido'), 'other');
});

test('solo navega dentro de la app por YouTube y el login de Google (https)', () => {
  assert.ok(isAllowedInApp('https://www.youtube.com/watch?v=x'));
  assert.ok(isAllowedInApp('https://m.youtube.com/'));
  assert.ok(isAllowedInApp('https://youtu.be/x'));
  assert.ok(isAllowedInApp('https://accounts.google.com/ServiceLogin'));
  assert.ok(!isAllowedInApp('http://www.youtube.com/'));
  assert.ok(!isAllowedInApp('https://evil-youtube.com/'));
  assert.ok(!isAllowedInApp('https://youtube.com.evil.net/'));
  assert.ok(!isAllowedInApp('javascript:alert(1)'));
  assert.ok(!isAllowedInApp('no es una url'));
});

test('genera una regla CSS por selector', () => {
  assert.equal(hideCss(['#a', '.b']), '#a{display:none!important}\n.b{display:none!important}');
  assert.equal(hideCss([]), '');
  assert.equal(hideCss(undefined), '');
});
