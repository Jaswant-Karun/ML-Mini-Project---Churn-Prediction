const form = document.getElementById("predictForm");
const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("resultBox");
const historyBody = document.getElementById("historyBody");
const historyInfo = document.getElementById("historyInfo");
const refreshHistoryBtn = document.getElementById("refreshHistory");
const apiStatus = document.getElementById("apiStatus");

const API_BASE = "http://127.0.0.1:8000";

let featureSchema = {};

function setStatus(ok, text) {
  apiStatus.textContent = text;
  apiStatus.className = `badge ${ok ? "success" : "warn"}`;
}

function buildForm(schema) {
  form.innerHTML = "";

  Object.entries(schema).forEach(([feature, config]) => {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.setAttribute("for", feature);
    label.textContent = feature;

    let input;
    if (config.type === "categorical") {
      input = document.createElement("select");
      (config.options || []).forEach((optionValue) => {
        const option = document.createElement("option");
        option.value = optionValue;
        option.textContent = optionValue;
        input.appendChild(option);
      });
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = String(config.example ?? 0);
    }

    input.id = feature;
    input.name = feature;

    wrapper.appendChild(label);
    wrapper.appendChild(input);
    form.appendChild(wrapper);
  });
}

function collectFormData() {
  const payload = {};

  Object.keys(featureSchema).forEach((feature) => {
    const input = document.getElementById(feature);
    if (!input) {
      return;
    }

    if (featureSchema[feature].type === "number") {
      payload[feature] = Number(input.value);
    } else {
      payload[feature] = input.value;
    }
  });

  return payload;
}

function renderPrediction(data) {
  const prob = data.churn_probability == null ? "N/A" : `${(data.churn_probability * 100).toFixed(2)}%`;
  const tone = data.label === "Churn" ? "#ff9f9f" : "#88ffd6";

  resultBox.innerHTML = `
    <div>
      <div style="font-size: 1.1rem; margin-bottom: 8px;">Prediction: <strong style="color:${tone}">${data.label}</strong></div>
      <div class="muted">Model Output: ${data.prediction}</div>
      <div class="muted">Churn Probability: ${prob}</div>
    </div>
  `;
}

function renderHistory(rows) {
  historyBody.innerHTML = "";

  if (!rows.length) {
    historyBody.innerHTML = '<tr><td colspan="3" class="muted">No history available.</td></tr>';
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const created = row.created_at ? new Date(row.created_at).toLocaleString() : "-";
    const probability = row.churn_probability == null ? "-" : `${(row.churn_probability * 100).toFixed(2)}%`;

    tr.innerHTML = `
      <td>${created}</td>
      <td>${row.label ?? "-"}</td>
      <td>${probability}</td>
    `;
    historyBody.appendChild(tr);
  });
}

async function loadHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (res.ok) {
      const supa = data.supabase_enabled ? "Supabase On" : "Supabase Off";
      setStatus(true, `API Ready | ${supa}`);
    } else {
      setStatus(false, "API issue");
    }
  } catch {
    setStatus(false, "Backend not reachable");
  }
}

async function loadSchema() {
  const res = await fetch(`${API_BASE}/api/feature-schema`);
  if (!res.ok) {
    throw new Error("Unable to load feature schema");
  }
  const data = await res.json();
  featureSchema = data.features;
  buildForm(featureSchema);
}

async function loadHistory() {
  const res = await fetch(`${API_BASE}/api/history?limit=10`);
  const data = await res.json();
  historyInfo.textContent = data.supabase_enabled
    ? "Showing latest prediction logs from Supabase."
    : "Supabase not configured. History requires SUPABASE_URL and SUPABASE_KEY.";
  renderHistory(data.rows || []);
}

predictBtn.addEventListener("click", async () => {
  try {
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    const features = collectFormData();
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });

    const data = await res.json();
    if (!res.ok) {
      resultBox.textContent = data.detail || "Prediction failed.";
      return;
    }

    renderPrediction(data);
    await loadHistory();
  } catch (err) {
    resultBox.textContent = `Error: ${err.message}`;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Run Prediction";
  }
});

refreshHistoryBtn.addEventListener("click", loadHistory);

(async function init() {
  await loadHealth();
  await loadSchema();
  await loadHistory();
})();
