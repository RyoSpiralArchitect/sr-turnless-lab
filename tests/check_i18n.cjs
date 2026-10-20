// Catalog completeness and presentation-only language switching, without a browser.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const selector = {value: "en"};
const document = {documentElement: {lang: ""}, title: "", querySelectorAll: () => [],
  getElementById: () => selector};
const context = {window: {}, document, localStorage: {getItem: () => null, setItem: () => {}}};
vm.runInNewContext(fs.readFileSync("turnless/web/i18n.js", "utf8"), context);
const i18n = context.window.I18N;
assert.equal(i18n.language, "en");
assert.deepEqual(Object.keys(i18n.messages.en).sort(), Object.keys(i18n.messages.ja).sort());
const html = fs.readFileSync("turnless/web/index.html", "utf8");
for (const [, key] of html.matchAll(/data-i18n(?:-placeholder|-aria)?="([^"]+)"/g)) {
  assert.ok(key in i18n.messages.en, `Missing message ${key}`);
}
for (const language of ["en", "ja"]) {
  i18n.setLanguage(language);
  assert.equal(document.documentElement.lang, language);
  assert.ok(!i18n.t("yieldEvent", {n: 3}).includes("{n}"));
}
console.log("English/Japanese catalogs and HTML translation keys: OK");
