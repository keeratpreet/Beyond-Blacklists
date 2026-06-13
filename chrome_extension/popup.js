/** Must match ``host_permissions`` in manifest.json and API_PORT on the server. */
const API_BASE = "http://127.0.0.1:8765";

function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  const v = Math.max(0, Math.min(1, Number(x)));
  return `${Math.round(v * 100)}%`;
}

function hostnameFromUrl(u) {
  try {
    return new URL(u).hostname || u;
  } catch {
    return u;
  }
}

function setErr(msg) {
  const el = document.getElementById("err");
  if (!msg) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = msg;
}

function setOut(obj) {
  const el = document.getElementById("out");
  el.hidden = false;
  el.textContent = JSON.stringify(obj, null, 2);
}

function labelFinal(pred) {
  return pred === 1 ? "Suspicious" : "Safe";
}
// function showAlert(message, type = "danger") {
//   const el = document.getElementById("alertBox");

//   el.className = "";
//   el.classList.add(type === "danger" ? "alert-danger" : "alert-safe");

//   el.textContent = message;
//   el.hidden = false;

//   setTimeout(() => {
//     el.hidden = true;
//   }, 4000);
// }

function showAlert(message, type = "danger") {
  const el = document.getElementById("alertBox");

  el.className = "";
  el.classList.add(type === "danger" ? "alert-danger" : "alert-safe");

  el.textContent = message;
  el.hidden = false;
}

function setSummaryView({ url, verdict, brand }) {
  document.getElementById("summary").hidden = false;
  document.getElementById("site").textContent = hostnameFromUrl(url);
  document.getElementById("verdict").textContent = verdict || "—";
  document.getElementById("brand").textContent = brand || "—";
}

function resetUi() {
  setErr("");
  document.getElementById("alertBox").hidden = true;
  document.getElementById("summary").hidden = true;
  document.getElementById("toggleDetails").hidden = true;
  document.getElementById("toggleDetails").textContent = "Show details";
  document.getElementById("details").hidden = true;
  document.getElementById("out").hidden = true;
}

document.getElementById("toggleDetails").addEventListener("click", () => {
  const details = document.getElementById("details");
  const btn = document.getElementById("toggleDetails");
  const showing = !details.hidden;
  details.hidden = showing;
  btn.textContent = showing ? "Show details" : "Hide details";
});

document.getElementById("go").addEventListener("click", async () => {
  const btn = document.getElementById("go");
  resetUi();
  btn.disabled = true;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !tab.url) {
      setErr("No active tab or URL (some pages cannot be analyzed).");
      return;
    }
    if (!tab.url.startsWith("http://") && !tab.url.startsWith("https://")) {
      setErr("This page URL is not http(s); capture/analysis is skipped.");
      return;
    }

    const dataUrl = await chrome.tabs.captureVisibleTab(undefined, { format: "png" });
    const blob = await (await fetch(dataUrl)).blob();

    const form = new FormData();
    form.append("url", tab.url);
    form.append("screenshot", blob, "shot.png");

    const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
    const text = await res.text();
    let body;
    try {
      body = JSON.parse(text);
    } catch {
      body = { raw: text };
    }
    if (!res.ok) {
      setErr(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
      if (body && Object.keys(body).length) setOut(body);
      return;
    }

    const verdict = labelFinal(body.final_pred);
    const brand = body.predicted_brand || null;

    // ── Alert banner ─────────────────────────────────────────────────────────
    // "domain verified" is ONLY safe to say when the full visual pipeline ran
    // AND the visual branch itself cleared the page (visual_pred==0 with a brand
    // match means _is_domain_consistent returned true — real brand domain).
    // In URL-only fallback mode (mode=="url_only") or when brand came purely from
    // the URL heuristic, we must NOT say "domain verified".
    const domainVerified =
      body.mode !== "url_only" &&
      body.visual_pred === 0 &&
      brand !== null;

    if (body.final_pred === 1) {
      // Phishing: brand impersonation confirmed
      showAlert(
        brand ? `⚠️ Suspicious — impersonating ${brand}` : "⚠️ Suspicious site detected",
        "danger"
      );
    } else if (brand) {
      // Benign with brand detected
      const safeMsg = domainVerified
        ? `✅ Safe — ${brand} (domain verified)`
        : `✅ Safe — ${brand} detected`;
      showAlert(safeMsg, "safe");
    } else {
      // Benign, no brand recognised
      showAlert("✅ No phishing indicators found", "safe");
    }

    // ── Summary card ─────────────────────────────────────────────────────────
    // Show a "✓" suffix only when domain was actually verified by visual branch
    const brandLabel =
      brand && domainVerified
        ? `${brand} ✓`
        : brand || null;

    setSummaryView({
      url: body.url || tab.url,
      verdict,
      brand: brandLabel,
    });

    // Warn when running in degraded URL-only mode (visual branch offline)
    if (body.mode === "url_only") {
      setErr("⚠️ Visual branch offline — URL analysis only (reduced accuracy)");
    }

    document.getElementById("toggleDetails").hidden = false;
    setOut({
      details: {
        case_id: body.case_id,
        matched_domain: body.matched_domain,
        final_pred: body.final_pred,
        url_pred: body.url_pred,
        visual_pred: body.visual_pred,
        url_risk: body.url_risk,
        confidence: body.confidence,
        timing_s: { url: body.url_time, visual: body.visual_time },
      },
      raw_response: body,
    });
  } catch (e) {
    setErr(e?.message || String(e));
  } finally {
    btn.disabled = false;
  }
});
