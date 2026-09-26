const previewEl = document.getElementById("preview");
const serverEl = document.getElementById("server");
const startBtn = document.getElementById("start");
const errorEl = document.getElementById("error");

let text = "";

chrome.storage.sync.get({ server: "http://127.0.0.1:8080" }, ({ server }) => {
  serverEl.value = server;
});

chrome.runtime.sendMessage({ type: "grab" }, (res) => {
  if (!res || !res.ok) {
    previewEl.textContent = res && res.error ? res.error : "could not read the page";
    startBtn.disabled = true;
    return;
  }
  text = res.text || "";
  if (!text) {
    previewEl.textContent = "no text found on this page";
    startBtn.disabled = true;
    return;
  }
  const words = text.split(/\s+/).length;
  previewEl.textContent = text.slice(0, 600) + (text.length > 600 ? "…" : "");
  startBtn.disabled = false;
  startBtn.textContent = `start session (${words} words)`;
});

startBtn.addEventListener("click", () => {
  chrome.storage.sync.set({ server: serverEl.value.trim() });
  startBtn.disabled = true;
  errorEl.textContent = "";
  chrome.runtime.sendMessage({ type: "start", text }, (res) => {
    if (res && res.ok) {
      window.close();
    } else {
      startBtn.disabled = false;
      errorEl.textContent = (res && res.error) || "failed";
    }
  });
});
