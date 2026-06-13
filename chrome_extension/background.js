// background.js — Manifest V3 service worker
// Minimal: keeps the extension alive and handles any future background tasks.
chrome.runtime.onInstalled.addListener(() => {
  console.log("Phishpedia extension installed.");
});
