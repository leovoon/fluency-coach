// fluency coach — background worker.
// Grabs text from the page, posts it to the local coach server, opens the
// session tab. The extension is a doorway: the session itself (marks,
// intonation, recording, practice loop) lives in the coach UI.

const MAX_TEXT = 20000;
const DEFAULT_SERVER = "http://127.0.0.1:8080";

async function serverUrl() {
  const { server } = await chrome.storage.sync.get({ server: DEFAULT_SERVER });
  return String(server || DEFAULT_SERVER).replace(/\/+$/, "");
}

// Selection wins; otherwise main content of the page. Runs in the page.
function grabText() {
  const sel = String(window.getSelection() || "").trim();
  if (sel) return sel;
  const main =
    document.querySelector("article") ||
    document.querySelector("main") ||
    document.body;
  return main ? main.innerText : "";
}

// Open the session in a real popup window snapped flush to the right edge
// of the window the click came from. No frames, no CSP — just a window.
async function openBeside(url, tabId) {
  let win = null;
  try {
    const tab = await chrome.tabs.get(tabId);
    win = await chrome.windows.get(tab.windowId);
  } catch (_) { /* no source window — let Chrome place it */ }
  const opts = { url, type: "popup", width: 480, focused: true };
  if (win) {
    opts.left = win.left + win.width - 8; // −8 so the edge stays visible
    opts.top = win.top;
    opts.height = win.height;
  }
  return chrome.windows.create(opts);
}

async function extractFromTab(tabId) {
  const [result] = await chrome.scripting.executeScript({
    target: { tabId },
    func: grabText,
  });
  return (result && result.result ? String(result.result) : "").trim().slice(0, MAX_TEXT);
}

async function startSession(text) {
  const url = await serverUrl();
  const res = await fetch(`${url}/extension/passage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`coach server responded ${res.status} — is it running?`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.url;
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "fc-practice",
    title: "Practice reading this",
    contexts: ["selection", "page"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "fc-practice" || !tab || !tab.id) return;
  try {
    const text = await extractFromTab(tab.id);
    if (!text) throw new Error("no text found on this page");
    await startSession(text);
  } catch (err) {
    console.error("[fluency coach]", err);
  }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "start") {
    const text = String(msg.text || "").trim().slice(0, MAX_TEXT);
    startSession(text)
      .then(async (url) => {
        if (msg.beside && sender.tab && sender.tab.id) {
          // Snapped-window flow: session opens beside the page.
          try {
            await openBeside(url, sender.tab.id);
            return;
          } catch (err) {
            console.error("[fluency coach] snapped window failed, opening tab instead:", err);
          }
        }
        await chrome.tabs.create({ url });
      })
      .then(() => sendResponse({ ok: true }))
      .catch((err) => sendResponse({ ok: false, error: String(err.message || err) }));
    return true; // async sendResponse
  }
  if (msg && msg.type === "grab") {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (!tabs[0] || !tabs[0].id) return sendResponse({ ok: false, error: "no active tab" });
      try {
        const text = await extractFromTab(tabs[0].id);
        sendResponse({ ok: true, text });
      } catch (err) {
        sendResponse({ ok: false, error: String(err.message || err) });
      }
    });
    return true;
  }
});
