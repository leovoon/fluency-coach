// fluency coach — per-post "practice" buttons on x.com / twitter.com.
// Injects a small button into each tweet's action row; clicking it grabs
// that post's text and opens a coach session in the browser side panel.

const FLAG = "fc-practice-btn";

function grabTweet(article) {
  const el = article.querySelector('[data-testid="tweetText"]');
  return el ? el.innerText.trim() : "";
}

function enhance(article) {
  if (article.querySelector("." + FLAG)) return;
  const row = article.querySelector('[role="group"]');
  if (!row) return;

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = FLAG;
  btn.textContent = "practice";
  btn.setAttribute("aria-label", "Practice reading this post with fluency coach");
  Object.assign(btn.style, {
    border: "0",
    background: "transparent",
    color: "rgb(83, 100, 113)",
    font: "600 12px/1 system-ui, sans-serif",
    cursor: "pointer",
    padding: "0 8px",
    whiteSpace: "nowrap",
  });
  btn.addEventListener("mouseenter", () => { btn.style.color = "#1c5e52"; });
  btn.addEventListener("mouseleave", () => { btn.style.color = "rgb(83, 100, 113)"; });
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const text = grabTweet(article);
    if (text) chrome.runtime.sendMessage({ type: "start", text, beside: true });
  });
  row.appendChild(btn);
}

function scan() {
  document.querySelectorAll('article[data-testid="tweet"]').forEach(enhance);
}

const observer = new MutationObserver(() => scan());
if (document.body) observer.observe(document.body, { childList: true, subtree: true });
scan();
