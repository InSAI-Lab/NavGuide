const response = await fetch('/static/locales/zh-CN.json');
if (!response.ok) {
  throw new Error('Unable to load the interface language resources');
}
const messages = await response.json();

export function text(key) {
  const value = messages[key];
  if (typeof value !== 'string') {
    throw new Error(`Unknown interface language key: ${key}`);
  }
  return value;
}

for (const element of document.querySelectorAll('[data-i18n]')) {
  element.textContent = text(element.dataset.i18n);
}
for (const element of document.querySelectorAll('[data-i18n-placeholder]')) {
  element.placeholder = text(element.dataset.i18nPlaceholder);
}
