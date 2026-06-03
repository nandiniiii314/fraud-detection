/* ── Navigation ─────────────────────────────────────── */
document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
    item.classList.add("active");
    const sec = document.getElementById(`section-${item.dataset.section}`);
    if (sec) sec.classList.add("active");
  });
});

/* ── Status ──────────────────────────────────────────── */
const dot      = document.getElementById("statusDot");
const statusTx = document.getElementById("statusText");
function setStatus(cls, msg) { dot.className = `status-dot ${cls}`; statusTx.textContent = msg; }

function markPipe(...ids) {
  ids.forEach(id => { const el = document.getElementById(id); if (el) el.classList.add("done"); });
}

const MODEL_ORDER = ["Logistic Regression", "Random Forest"];
const LOCAL_API_ORIGIN = "http://127.0.0.1:5000";
const API_ORIGINS = [];

if (/^https?:$/i.test(window.location.protocol) && window.location.origin) {
  API_ORIGINS.push(window.location.origin);
}
if (!API_ORIGINS.includes(LOCAL_API_ORIGIN)) {
  API_ORIGINS.push(LOCAL_API_ORIGIN);
}

function buildApiUrl(path, origin = "") {
  return origin ? `${origin}${path}` : path;
}

function getPrimaryApiUrl(path) {
  return buildApiUrl(path, API_ORIGINS[0] || LOCAL_API_ORIGIN);
}

async function apiFetchJson(path, options = {}) {
  let lastError = null;

  for (const origin of API_ORIGINS) {
    try {
      const res = await fetch(buildApiUrl(path, origin), options);
      const text = await res.text();
      const data = text ? JSON.parse(text) : {};

      if (!res.ok) {
        throw new Error(data.error || data.message || `Request failed with status ${res.status}.`);
      }
      return data;
    } catch (error) {
      lastError = error;
    }
  }

  if (lastError instanceof SyntaxError) {
    throw new Error("The app returned an invalid response. Restart the Flask server and try again.");
  }
  throw lastError || new Error("Could not reach the local Flask server at http://127.0.0.1:5000.");
}

function orderedModelEntries(results = {}) {
  const entries = Object.entries(results || {});
  const map = new Map(entries);
  const ordered = MODEL_ORDER.filter(name => map.has(name)).map(name => [name, map.get(name)]);
  const extras = entries.filter(([name]) => !MODEL_ORDER.includes(name));
  return [...ordered, ...extras];
}

/* ── Simplified Fields ───────────────────────────────────── */
/* Removed V1-V28 fields for user-friendliness. Only Amount and Time now. */

/* ── Load Dataset ────────────────────────────────────── */
document.getElementById("btnLoad").addEventListener("click", async () => {
  const btn = document.getElementById("btnLoad");
  btn.disabled = true; btn.innerHTML = '<span class="spin">↻</span> Loading…';
  try {
    const data = await apiFetchJson("/api/load_dataset", { method: "POST" });
    if (!data.success) { alert("Error: " + data.error); return; }

    document.getElementById("vRows").textContent  = data.rows.toLocaleString();
    document.getElementById("vFraud").textContent = data.fraud.toLocaleString();
    document.getElementById("vFraudPct").textContent = data.fraud_pct + "% of total";
    document.getElementById("vLegit").textContent = data.legit.toLocaleString();
    document.getElementById("vNulls").textContent = data.nulls;
    document.getElementById("vAmt").textContent   = "$" + data.amount_mean;
    document.getElementById("vCols").textContent  = data.columns;

    const cleanBtn = document.getElementById("btnClean");
    cleanBtn.disabled = false;

    setStatus("loaded", "Dataset loaded");
    markPipe("ps1");
    document.getElementById("ps2").classList.add("active");
    await syncTrainingState();
  } catch(e) { alert("Failed: " + e.message); }
  finally { btn.disabled = false; btn.innerHTML = '<span class="btn-icon">⬇</span> Load Dataset'; }
});

/* ── Clean Data ─────────────────────────────────────── */
document.getElementById("btnClean").addEventListener("click", async () => {
  const btn = document.getElementById("btnClean");
  let cleaned = false;
  btn.disabled = true; btn.innerHTML = '<span class="spin">...</span> Cleaning...';
  try {
    const data = await apiFetchJson("/api/clean_data", { method: "POST" });
    if (!data.success) { alert("Error: " + data.error); return; }

    // Update stats with cleaned data
    document.getElementById("vRows").textContent = data.cleaned_rows.toLocaleString();
    document.getElementById("vNulls").textContent = "0"; // Cleaned data has no nulls

    // Disable clean button after cleaning and show saved file name
    cleaned = true;
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">OK</span> Cleaned';
    alert(data.message + "\nSaved file: " + data.cleaned_path);

    setStatus("success", "Data cleaned and saved");
    markPipe("ps2");
    await syncTrainingState();
  } catch(e) { alert("Error: " + e.message); }
  finally {
    if (!cleaned) {
      btn.disabled = false;
      btn.innerHTML = '<span class="btn-icon">CLR</span> Clean Data';
    }
  }
});

/* ── Sliders ─────────────────────────────────────────── */
document.getElementById("pcaSlider").addEventListener("input", e => {
  document.getElementById("pcaVal").textContent = e.target.value;
});
document.getElementById("splitSlider").addEventListener("input", e => {
  document.getElementById("splitVal").textContent = e.target.value;
});
document.getElementById("smoteToggle").addEventListener("change", e => {
  document.getElementById("togLabel").textContent = e.target.checked ? "Enabled" : "Disabled";
});

/* ── Train ───────────────────────────────────────────── */
let modelsTrained = false;

async function syncTrainingState() {
  try {
    const data = await apiFetchJson("/api/state");
    if (!data.success) return false;

    if (data.trained && Object.keys(data.results || {}).length > 0) {
      renderResults(data.results, data.pca_variance || [], data.pca_total_variance || 0);
      document.getElementById("trainResults").classList.remove("hidden");
      buildReportPreview(data.results);
      setStatus("trained", "Models trained");
      markPipe("ps3", "ps4", "ps5");
      modelsTrained = true;
      return true;
    }

    modelsTrained = false;
    return false;
  } catch (_) {
    return false;
  }
}

const trainSteps = [
  [8,  "Loading dataset…"],
  [18, "Applying StandardScaler…"],
  [30, "Running PCA…"],
  [45, "Splitting train/test…"],
  [58, "Applying SMOTE…"],
  [72, "Training Logistic Regression…"],
  [88, "Training Random Forest (100 trees)…"],
  [96, "Evaluating metrics…"],
  [100,"✓ Complete!"]
];

document.getElementById("btnTrain").addEventListener("click", async () => {
  const btn = document.getElementById("btnTrain");
  btn.disabled = true; btn.innerHTML = '<span class="spin">↻</span> Training…';

  const pw = document.getElementById("trainProgress");
  const pb = document.getElementById("progBar");
  const pm = document.getElementById("progMsg");
  pw.classList.remove("hidden");
  document.getElementById("trainResults").classList.add("hidden");

  let si = 0;
  const iv = setInterval(() => {
    if (si < trainSteps.length) {
      pb.style.width  = trainSteps[si][0] + "%";
      pm.textContent  = trainSteps[si][1];
      si++;
    }
  }, 700);

  const payload = {
    pca_components: parseInt(document.getElementById("pcaSlider").value),
    test_size:      parseInt(document.getElementById("splitSlider").value) / 100,
    smote:          document.getElementById("smoteToggle").checked
  };

  try {
    const data = await apiFetchJson("/api/train", {
      method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload)
    });
    clearInterval(iv);
    if (!data.success) { alert("Error: " + data.error); return; }

    pb.style.width = "100%"; pm.textContent = "✓ Training complete!";
    renderResults(data.results, data.pca_variance, data.pca_total_variance);
    document.getElementById("trainResults").classList.remove("hidden");

    markPipe("ps2","ps3","ps4","ps5");
    setStatus("trained", "Models trained");
    modelsTrained = true;
    buildReportPreview(data.results);
  } catch(e) { clearInterval(iv); alert("Error: " + e.message); }
  finally { btn.disabled = false; btn.innerHTML = '<span class="btn-icon">▶</span> Start Training'; }
});

function metaBadge(val, high=90, mid=75) {
  const cls = val >= high ? "mb-good" : val >= mid ? "mb-warn" : "mb-bad";
  return `<span class="mbadge ${cls}">${val}%</span>`;
}

function renderResults(results, pcaVar, totalVar) {
  const tbody = document.getElementById("rBody");
  tbody.innerHTML = "";
  for (const [name, r] of orderedModelEntries(results)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="mbadge mb-model">${name}</span></td>
      <td>${metaBadge(r.accuracy)}</td>
      <td>${metaBadge(r.precision, 80, 60)}</td>
      <td>${metaBadge(r.recall, 80, 60)}</td>
      <td>${metaBadge(r.f1, 80, 60)}</td>
      <td>${metaBadge(r.roc_auc)}</td>
    `;
    tbody.appendChild(tr);
  }
  document.getElementById("pcaBadge").textContent =
    `PCA: ${pcaVar.length} components → ${totalVar}% variance retained | ` +
    pcaVar.map((v,i)=>`PC${i+1}:${v}%`).join("  ");

  const cmCards = document.getElementById("cmCards");
  cmCards.innerHTML = "";
  for (const [name, r] of orderedModelEntries(results)) {
    const card = document.createElement("div");
    card.className = "cm-card";
    card.innerHTML = `
      <h4>${name}</h4>
      <div class="cm-grid">
        <div class="cm-cell tn"><div class="cm-num">${r.tn.toLocaleString()}</div><div class="cm-lbl">True Neg</div></div>
        <div class="cm-cell fp"><div class="cm-num">${r.fp.toLocaleString()}</div><div class="cm-lbl">False Pos</div></div>
        <div class="cm-cell fn"><div class="cm-num">${r.fn.toLocaleString()}</div><div class="cm-lbl">False Neg</div></div>
        <div class="cm-cell tp"><div class="cm-num">${r.tp.toLocaleString()}</div><div class="cm-lbl">True Pos</div></div>
      </div>`;
    cmCards.appendChild(card);
  }
}

/* ── Python Charts ───────────────────────────────────── */
const chartEndpoints = {
  distribution: "distribution",
  amount:       "amount",
  time:         "time_fraud",
  heatmap:      "correlation",
  pca:          "pca",
  roc:          "roc"
};

const chartImage = document.getElementById("chartImage");
const chartPlaceholder = document.getElementById("chartPlaceholder");
const chartTabs = document.querySelectorAll(".ctab");

chartTabs.forEach(tab => {
  tab.addEventListener("click", async () => {
    chartTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    await loadChart(tab.dataset.chart);
  });
});

async function loadChart(type) {
  if (!chartImage || !chartPlaceholder) return;
  chartImage.style.display = "none";
  chartPlaceholder.style.display = "block";
  chartPlaceholder.innerHTML = '<div class="spin" style="font-size:26px">↻</div>';

  const endpoint = chartEndpoints[type];
  if (!endpoint) {
    chartPlaceholder.innerHTML = `<p style="color:var(--yellow)">Unknown chart type.</p>`;
    return;
  }

  try {
    const data = await apiFetchJson(`/api/charts/${encodeURIComponent(endpoint)}`);
    if (!data.image) {
      chartPlaceholder.innerHTML = `<p style="color:var(--yellow)">${data.error || "Chart not available"}</p>`;
      return;
    }
    chartImage.src = `data:image/png;base64,${data.image}`;
    chartImage.style.display = "block";
    chartPlaceholder.style.display = "none";
  } catch (e) {
    chartPlaceholder.innerHTML = `<p style="color:var(--red)">Failed to load chart.</p>`;
  }
}

loadChart("distribution");
syncTrainingState();

/* ── R Status Check ──────────────────────────────────── */
const rButtons = document.querySelectorAll(".btn-r");
let rAvailable = false;

async function checkR() {
  const bar  = document.getElementById("rStatusBar");
  const icon = document.getElementById("rStatusIcon");
  const msg  = document.getElementById("rStatusMsg");
  try {
    const data = await apiFetchJson("/api/r/check");
    if (data.available) {
      icon.textContent = "✅";
      msg.textContent = `R is installed: ${data.version}${data.path ? ` (${data.path})` : ""}`;
      bar.style.borderColor = "var(--green)";
      rButtons.forEach(btn => { btn.disabled = false; });
      rAvailable = true;
    } else {
      icon.textContent = "ℹ️";
      msg.textContent = "R not found (optional). Python ML tools are fully functional. To use R scripts, install R from https://www.r-project.org/";
      bar.style.borderColor = "var(--blue)";
      rButtons.forEach(btn => { btn.disabled = true; btn.title = "R is not installed. Install R to enable this feature."; });
      rAvailable = false;
    }
  } catch(e) {
    icon.textContent = "ℹ️";
    msg.textContent = "R status check failed (optional). Python ML tools are still fully functional.";
    bar.style.borderColor = "var(--blue)";
    rButtons.forEach(btn => { btn.disabled = true; btn.title = "R not available."; });
    rAvailable = false;
  }
}
checkR();

/* ── R Scripts ───────────────────────────────────────── */
document.querySelectorAll(".btn-r").forEach(btn => {
  btn.addEventListener("click", async () => {
    if (btn.disabled) return;
    const script = btn.dataset.script;
    const orig   = btn.textContent;
    btn.disabled = true; btn.innerHTML = '<span class="spin">↻</span> Running R…';

    const console_el  = document.getElementById("rConsole");
    const console_txt = document.getElementById("rConsoleText");
    const imagesWrap  = document.getElementById("rImagesWrap");
    const imgGrid     = document.getElementById("rImgGrid");

    console_el.classList.remove("hidden");
    console_txt.textContent = `Running r_scripts/${script}.R …\n(this may take 1–3 minutes for first run while R packages install)\n`;

    try {
      const data = await apiFetchJson("/api/r/run", {
        method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({script})
      });

      if (!data.success) {
        console_txt.textContent += "\n❌ ERROR: " + data.error;
        if (data.output) {
          console_txt.textContent += "\n\n" + data.output;
        }
        return;
      }

      console_txt.textContent += data.output || "(no output)";
      if (data.generated_files && data.generated_files.length > 0) {
        console_txt.textContent += `\n\nGenerated files:\n- ${data.generated_files.join("\n- ")}`;
      }
      if (data.report_url) {
        console_txt.textContent += `\n\nReport ready: ${getPrimaryApiUrl(data.report_url)}`;
      }

      // Show images if returned
      if (data.images && Object.keys(data.images).length > 0) {
        imagesWrap.classList.remove("hidden");
        imgGrid.innerHTML = "";
        for (const [fname, b64] of Object.entries(data.images)) {
          const div = document.createElement("div");
          div.className = "r-img-item";
          div.innerHTML = `<div class="r-img-name">${fname}</div><img src="data:image/png;base64,${b64}" alt="${fname}">`;
          imgGrid.appendChild(div);
        }
        document.getElementById("ps6").classList.add("done");
      }
    } catch(e) {
      console_txt.textContent += "\n❌ " + e.message;
    } finally {
      btn.disabled = false; btn.textContent = orig;
    }
  });
});

document.getElementById("rConsoleClose").addEventListener("click", () => {
  document.getElementById("rConsole").classList.add("hidden");
});

/* ── Predict ─────────────────────────────────────────── */
document.getElementById("btnPredict").addEventListener("click", async () => {
  const btn = document.getElementById("btnPredict");
  btn.disabled = true; btn.innerHTML = '<span class="spin">↻</span> Classifying…';

  const inputs   = document.querySelectorAll(".finput");
  const features = {};
  inputs.forEach(inp => {
    const value = parseFloat(inp.value) || 0;
    features[inp.dataset.col] = inp.dataset.col === "Time" ? value * 3600 : value;
  });
  const model = document.getElementById("modelSel").value;

  try {
    const requestPrediction = () => apiFetchJson("/api/predict", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({features, model})
    });

    let data = await requestPrediction();
    if (!data.success && data.error === "Train models first.") {
      await syncTrainingState();
      data = await requestPrediction();
    }
    if (!data.success) { alert(data.error); return; }

    const box = document.getElementById("resultBox");
    box.className = `result-box ${data.prediction === 1 ? "fraud":"legit"}`;
    document.getElementById("resIcon").textContent  = data.prediction === 1 ? "🚨" : "✅";
    document.getElementById("resLabel").textContent = data.label;
    document.getElementById("resProb").textContent  = `Fraud probability: ${data.probability}%`;
    document.getElementById("resModel").textContent = `via ${model}`;
    drawGauge(data.probability);
    document.getElementById("gaugeSub").textContent = `${data.probability}% fraud probability`;
  } catch(e) { alert("Error: " + e.message); }
  finally { btn.disabled = false; btn.innerHTML = '<span class="btn-icon">◎</span> Classify Transaction'; }
});

function drawGauge(pct) {
  const canvas = document.getElementById("gaugeCanvas");
  const ctx = canvas.getContext("2d");
  const cx = canvas.width / 2, cy = canvas.height - 8, r = 95;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.lineWidth = 14; ctx.strokeStyle = "#21262d"; ctx.stroke();

  const angle = Math.PI + (pct / 100) * Math.PI;
  const color = pct > 70 ? "#f85149" : pct > 40 ? "#e3b341" : "#3fb950";
  ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, angle, false);
  ctx.lineWidth = 14; ctx.strokeStyle = color; ctx.lineCap = "round"; ctx.stroke();

  // Needle
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(angle);
  ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(-r+22,0);
  ctx.lineWidth=2; ctx.strokeStyle="#e6edf3"; ctx.stroke(); ctx.restore();

  ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI*2);
  ctx.fillStyle = "#e6edf3"; ctx.fill();

  ctx.font = "bold 10px JetBrains Mono, monospace";
  ctx.fillStyle = "#8b949e";
  ctx.textAlign = "left";  ctx.fillText("0%",   cx - r - 8,  cy + 15);
  ctx.textAlign = "right"; ctx.fillText("100%", cx + r + 8,  cy + 15);
}

/* ── Report ──────────────────────────────────────────── */
function buildReportPreview(results) {
  const lines = [
    "=".repeat(52),
    "  CREDIT CARD FRAUD DETECTION - REPORT",
    "=".repeat(52), ""
  ];
  for (const [name, r] of orderedModelEntries(results)) {
    lines.push(`Model     : ${name}`, "-".repeat(35),
      `  Accuracy  : ${r.accuracy}%`,
      `  Precision : ${r.precision}%`,
      `  Recall    : ${r.recall}%`,
      `  F1-Score  : ${r.f1}%`,
      `  ROC-AUC   : ${r.roc_auc}%`,
      `  TP:${r.tp}  FP:${r.fp}  FN:${r.fn}  TN:${r.tn}`, ""
    );
  }
  document.getElementById("reportPreview").textContent = lines.join("\n");
}

document.getElementById("btnReport").addEventListener("click", () => {
  if (!modelsTrained) { alert("Train models first."); return; }
  window.location.href = getPrimaryApiUrl("/api/report");
});
