const panelGrid = document.getElementById("panel-grid");
const eventFeed = document.getElementById("event-feed");

const panelSpecs = new Map();
const panelData = new Map();
const panelModes = new Map();

function addEventLine(event) {
  const div = document.createElement("div");
  div.className = "event-item";
  div.textContent = `${event.ts_utc || ""}  ${event.event_type || "event"}  ${event.domain || ""}`;
  eventFeed.prepend(div);
  while (eventFeed.children.length > 100) {
    eventFeed.removeChild(eventFeed.lastChild);
  }
}

function stateColor(state) {
  if (!state) return "#9aa6b2";
  if (state === "online" || state.includes("completed") || state === "running") return "#2e7d32";
  if (state === "registered") return "#0a6fb8";
  if (state === "offline" || state === "unregistered") return "#b26a00";
  if (state.includes("failed") || state === "timeout") return "#b00020";
  if (state.includes("stopped")) return "#8f5a00";
  return "#9aa6b2";
}

function renderTopology(payload) {
  const nodes = payload.nodes || [];
  const edges = payload.edges || [];

  const xByType = {
    superlink: 80,
    supernode: 240,
  };

  const yCounters = { superlink: 80, supernode: 30 };
  const positions = {};

  nodes.forEach((node) => {
    const x = xByType[node.type] || 80;
    const y = yCounters[node.type] || 80;
    positions[node.id] = { x, y };
    yCounters[node.type] = y + 45;
  });

  const edgeLines = edges
    .map((e) => {
      const s = positions[e.source];
      const t = positions[e.target];
      if (!s || !t) return "";
      return `<line x1="${s.x}" y1="${s.y}" x2="${t.x}" y2="${t.y}" stroke="#9eb3c7" stroke-width="1.5"/>`;
    })
    .join("\n");

  const nodeCircles = nodes
    .map((n) => {
      const p = positions[n.id];
      const color = stateColor(n.state);
      const label = n.type === "supernode" ? `node-${n.node_id ?? n.id}` : n.id;
      return `
        <circle cx="${p.x}" cy="${p.y}" r="10" fill="${color}"/>
        <text x="${p.x + 14}" y="${p.y + 4}" fill="#1b2430" font-size="11">${label}</text>
      `;
    })
    .join("\n");

  return `<div class="topology"><svg viewBox="0 0 520 180">${edgeLines}${nodeCircles}</svg></div>
    <div style="font-size:12px;color:#516174;margin-top:6px;">online: ${payload.online_count ?? 0} / total: ${payload.total_count ?? 0}</div>`;
}

function renderTimeline(payload) {
  const items = (payload.events || []).slice(-30).reverse();
  const lines = items
    .map((e) => {
      const dom = e.domain ? ` [${e.domain}]` : "";
      return `<div class="timeline-item"><strong>${e.event_type}</strong>${dom}<br/><span>${e.ts_utc}</span></div>`;
    })
    .join("");
  return `<div class="timeline-list">${lines || "No timeline events yet."}</div>`;
}

function buildPath(points, width, height) {
  const valid = points.filter((p) => typeof p.y === "number");
  if (valid.length < 2) return "";
  let minX = valid[0].x;
  let maxX = valid[valid.length - 1].x;
  if (maxX === minX) maxX = minX + 1;
  const minY = 0.0;
  const maxY = 1.0;

  return valid
    .map((p, i) => {
      const x = ((p.x - minX) / (maxX - minX)) * (width - 20) + 10;
      const y = height - (((p.y - minY) / (maxY - minY)) * (height - 20) + 10);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function renderQuality(panelId, payload) {
  const mode = panelModes.get(panelId) || payload.default_view || "gated";
  const metrics = payload.metrics || [];
  const domains = payload.domains || {};

  const controls = `<div class="controls">
      <button class="${mode === "gated" ? "active" : ""}" data-panel="${panelId}" data-mode="gated">Gated</button>
      <button class="${mode === "raw" ? "active" : ""}" data-panel="${panelId}" data-mode="raw">Raw</button>
    </div>`;

  const rows = Object.entries(domains)
    .map(([domain, data]) => {
      const series = data[mode] || {};
      const macroPath = buildPath(series.macro_f1 || [], 520, 150);
      const peakPath = buildPath(series.event_peak_macro_f1 || [], 520, 150);
      const aurocPath = buildPath(series.anomaly_auroc || [], 520, 150);
      return `
        <div style="margin-bottom:8px; font-size:12px;"><strong>${domain}</strong></div>
        <svg viewBox="0 0 520 150">
          <path d="${macroPath}" fill="none" stroke="#0057b8" stroke-width="2"/>
          <path d="${peakPath}" fill="none" stroke="#0d8f61" stroke-width="2"/>
          <path d="${aurocPath}" fill="none" stroke="#9f6700" stroke-width="2"/>
          <text x="12" y="16" font-size="11" fill="#0057b8">macro_f1</text>
          <text x="95" y="16" font-size="11" fill="#0d8f61">event_peak_macro_f1</text>
          <text x="255" y="16" font-size="11" fill="#9f6700">anomaly_auroc</text>
        </svg>
      `;
    })
    .join("");

  return `<div class="quality">${controls}${rows || "No quality metrics yet."}</div>`;
}

function renderStub(payload) {
  return `<div class="stub-box">${payload.message}\n\nExpected schema:\n${JSON.stringify(payload.expected_schema || {}, null, 2)}</div>`;
}

function renderPanelCard(spec, payload) {
  const card = document.getElementById(`panel-${spec.id}`);
  if (!card) return;
  const body = card.querySelector(".panel-body");

  if (spec.id === "federation_topology") {
    body.innerHTML = renderTopology(payload);
  } else if (spec.id === "round_timeline") {
    body.innerHTML = renderTimeline(payload);
  } else if (spec.id === "global_quality_trends") {
    body.innerHTML = renderQuality(spec.id, payload);
  } else {
    body.innerHTML = renderStub(payload);
  }

  card.querySelectorAll("button[data-mode]").forEach((btn) => {
    btn.onclick = () => {
      panelModes.set(spec.id, btn.dataset.mode);
      renderPanelCard(spec, panelData.get(spec.id) || {});
    };
  });
}

async function refreshPanel(panelId) {
  const res = await fetch(`/api/v1/panels/${panelId}`);
  if (!res.ok) return;
  const payload = await res.json();
  panelData.set(panelId, payload);
  const spec = panelSpecs.get(panelId);
  if (spec) renderPanelCard(spec, payload);
}

async function bootstrap() {
  const layoutRes = await fetch("/api/v1/layout");
  const layout = await layoutRes.json();

  layout.panels.forEach((spec) => {
    panelSpecs.set(spec.id, spec);
    const div = document.createElement("article");
    div.className = "panel";
    div.id = `panel-${spec.id}`;
    div.innerHTML = `
      <div class="panel-head">
        <h3>${spec.title}</h3>
        <span class="badge ${spec.implemented ? "impl" : "stub"}">${spec.implemented ? "implemented" : "stub"}</span>
      </div>
      <p>${spec.description}</p>
      <div class="panel-body">Loading...</div>
    `;
    panelGrid.appendChild(div);
  });

  await Promise.all(layout.panels.map((p) => refreshPanel(p.id)));
  connectWs();

  setInterval(() => {
    layout.panels.forEach((p) => refreshPanel(p.id));
  }, 2000);
}

function connectWs() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/api/v1/events`);
  ws.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      addEventLine(data);
      if (data.domain && panelSpecs.has("round_timeline")) {
        refreshPanel("round_timeline");
      }
      if (data.event_type === "metrics.updated") {
        refreshPanel("global_quality_trends");
      }
      if (data.event_type.startsWith("runtime.") || data.event_type.startsWith("domain.")) {
        refreshPanel("federation_topology");
      }
    } catch (_err) {
      // Ignore malformed event payloads
    }
  };
  ws.onclose = () => {
    setTimeout(connectWs, 2000);
  };
}

bootstrap();
