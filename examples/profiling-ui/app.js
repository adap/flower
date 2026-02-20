const jsonInput = document.getElementById("json-input");
const fetchButton = document.getElementById("fetch-button");
const parseButton = document.getElementById("parse-button");
const runIdInput = document.getElementById("run-id-input");
const liveToggle = document.getElementById("live-toggle");
const dropZone = document.getElementById("drop-zone");

const runIdEl = document.getElementById("run-id");
const generatedAtEl = document.getElementById("generated-at");
const eventCountEl = document.getElementById("event-count");
const taskCountEl = document.getElementById("task-count");
const nodeCountEl = document.getElementById("node-count");
const roundCountEl = document.getElementById("round-count");

const summaryTableBody = document.querySelector("#summary-table tbody");
const networkTableBody = document.querySelector("#network-table tbody");

const memScope = document.getElementById("mem-scope");
const memTask = document.getElementById("mem-task");
const diskScope = document.getElementById("disk-scope");
const diskTask = document.getElementById("disk-task");

const memCanvas = document.getElementById("mem-chart");
const diskCanvas = document.getElementById("disk-chart");

const networkTasks = new Set([
  "network_upstream",
  "network_downstream",
  "send_and_receive",
]);

let liveSource = null;

function parseProfile(text) {
  const summary = JSON.parse(text);
  summary.entries = summary.entries || [];
  summary.events = summary.events || [];
  return summary;
}

function unique(values) {
  return Array.from(new Set(values));
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return Number(value).toFixed(digits);
}

function toDisplayNode(entry) {
  if (entry.node_id === null || entry.node_id === undefined) {
    if (entry.scope === "server") {
      return "server";
    }
    return "N/A";
  }
  return String(entry.node_id);
}

function renderCards(summary) {
  runIdEl.textContent = summary.run_id ?? "—";
  generatedAtEl.textContent = summary.generated_at ?? "—";

  const taskSet = new Set(
    summary.entries
      .filter((entry) => entry.task !== "total")
      .map((entry) => `${entry.scope}:${entry.task}`)
  );
  const nodes = unique(
    summary.entries
      .map((entry) => toDisplayNode(entry))
      .filter((node) => node !== "N/A")
  );
  const rounds = unique(
    summary.entries
      .map((entry) => entry.round)
      .filter((round) => round !== null && round !== undefined)
  );

  eventCountEl.textContent = summary.events.length.toLocaleString();
  taskCountEl.textContent = `${taskSet.size} tasks`;
  nodeCountEl.textContent = nodes.length.toString();
  roundCountEl.textContent = `${rounds.length} rounds`;
}

function renderTables(summary) {
  summaryTableBody.innerHTML = "";
  networkTableBody.innerHTML = "";

  const entries = summary.entries.filter(
    (entry) => !(entry.scope === "client" && entry.task === "total")
  );
  const events = summary.events || [];

  const eventDiskIndex = new Map();
  events.forEach((event) => {
    const key = [
      event.scope,
      event.task,
      event.round ?? "N/A",
      event.node_id ?? "N/A",
    ].join("|");
    const existing = eventDiskIndex.get(key) || {
      readSum: 0,
      readCount: 0,
      writeSum: 0,
      writeCount: 0,
    };
    if (event.disk_read_mb !== null && event.disk_read_mb !== undefined) {
      existing.readSum += event.disk_read_mb;
      existing.readCount += 1;
    }
    if (event.disk_write_mb !== null && event.disk_write_mb !== undefined) {
      existing.writeSum += event.disk_write_mb;
      existing.writeCount += 1;
    }
    eventDiskIndex.set(key, existing);
  });

  const networkEntries = entries.filter(
    (entry) => entry.scope === "server" && networkTasks.has(entry.task)
  );
  const mainEntries = entries.filter(
    (entry) => !(entry.scope === "server" && networkTasks.has(entry.task))
  );

  const mapping = {
    network_upstream: "upstream",
    network_downstream: "downstream",
    send_and_receive: "combined",
  };

  const renderMainRow = (entry) => {
    const tr = document.createElement("tr");
    const key = [
      entry.scope,
      entry.task,
      entry.round ?? "N/A",
      entry.node_id ?? "N/A",
    ].join("|");
    const fallback = eventDiskIndex.get(key);
    const avgDiskRead =
      entry.avg_disk_read_mb ??
      (fallback && fallback.readCount
        ? fallback.readSum / fallback.readCount
        : null);
    const avgDiskWrite =
      entry.avg_disk_write_mb ??
      (fallback && fallback.writeCount
        ? fallback.writeSum / fallback.writeCount
        : null);
    const cells = [
      entry.task,
      entry.scope,
      entry.round ?? "N/A",
      toDisplayNode(entry),
      formatNumber(entry.avg_ms),
      formatNumber(entry.max_ms),
      formatNumber(entry.avg_mem_mb),
      formatNumber(entry.avg_mem_delta_mb),
      formatNumber(avgDiskRead),
      formatNumber(avgDiskWrite),
      entry.disk_source ?? "—",
      entry.count ?? 0,
    ];
    cells.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = String(cell);
      tr.appendChild(td);
    });
    summaryTableBody.appendChild(tr);
  };

  const renderNetworkRow = (entry) => {
    const tr = document.createElement("tr");
    const taskLabel = mapping[entry.task] || entry.task;
    const cells = [
      taskLabel,
      entry.round ?? "N/A",
      toDisplayNode(entry),
      formatNumber(entry.avg_ms),
      formatNumber(entry.max_ms),
      entry.count ?? 0,
    ];
    cells.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = String(cell);
      tr.appendChild(td);
    });
    networkTableBody.appendChild(tr);
  };

  mainEntries.forEach(renderMainRow);
  networkEntries.forEach(renderNetworkRow);
}

function renderSelectOptions(selectEl, values) {
  selectEl.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectEl.appendChild(option);
  });
}

function downsample(points, maxPoints = 260) {
  if (points.length <= maxPoints) {
    return points;
  }
  const step = Math.ceil(points.length / maxPoints);
  return points.filter((_, index) => index % step === 0);
}

function drawLineChart(canvas, points, color, fillColor) {
  const ratio = resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);

  if (points.length === 0) {
    ctx.fillStyle = "#c2c6cc";
    ctx.font = "14px Montserrat";
    ctx.fillText("No event samples available", 16, height / 2);
    return;
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, 1);

  const padLeft = 44;
  const padBottom = 32;
  const padTop = 16;
  const padRight = 16;
  const scaleX = (value) =>
    padLeft +
    ((value - xs[0]) / (xs[xs.length - 1] - xs[0] || 1)) *
      (width - padLeft - padRight);
  const scaleY = (value) =>
    height -
    padBottom -
    ((value - minY) / (maxY - minY || 1)) * (height - padTop - padBottom);

  ctx.strokeStyle = "#e0dcd1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, height - padBottom);
  ctx.lineTo(width - padRight, height - padBottom);
  ctx.stroke();

  const ticks = 4;
  ctx.fillStyle = "#68788a";
  ctx.font = "11px Montserrat";
  for (let i = 0; i <= ticks; i += 1) {
    const t = minY + ((maxY - minY) * i) / ticks;
    const y = scaleY(t);
    ctx.strokeStyle = "rgba(104, 120, 138, 0.15)";
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);
    ctx.stroke();
    ctx.fillStyle = "#68788a";
    ctx.fillText(t.toFixed(1), 6, y + 4);
  }

  ctx.beginPath();
  points.forEach((point, idx) => {
    const x = scaleX(point.x);
    const y = scaleY(point.y);
    if (idx === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.lineTo(scaleX(points[points.length - 1].x), height - padBottom);
  ctx.lineTo(scaleX(points[0].x), height - padBottom);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  ctx.fillStyle = "#68788a";
  ctx.font = "12px Montserrat";
  ctx.fillText("MB", padLeft + 4, padTop - 2);
  ctx.fillText("Time", width - 36, height - 8);
}

function drawDiskChart(canvas, points) {
  const ratio = resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);

  if (points.length === 0) {
    ctx.fillStyle = "#c2c6cc";
    ctx.font = "14px Montserrat";
    ctx.fillText("No disk IO samples available", 16, height / 2);
    return;
  }

  const padLeft = 44;
  const padBottom = 32;
  const padTop = 16;
  const padRight = 16;
  const maxY = Math.max(
    1,
    ...points.map((p) => Math.max(p.read, 0) + Math.max(p.write, 0))
  );
  const barWidth = Math.max(
    2,
    (width - padLeft - padRight) / points.length
  );

  ctx.strokeStyle = "#e0dcd1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, height - padBottom);
  ctx.lineTo(width - padRight, height - padBottom);
  ctx.stroke();

  ctx.fillStyle = "#68788a";
  ctx.font = "12px Montserrat";
  ctx.fillText("MB", padLeft + 4, padTop - 2);
  ctx.fillText("Events", width - 52, height - 8);

  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const t = (maxY * i) / ticks;
    const y =
      height -
      padBottom -
      (t / maxY) * (height - padTop - padBottom);
    ctx.strokeStyle = "rgba(104, 120, 138, 0.15)";
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);
    ctx.stroke();
    ctx.fillStyle = "#68788a";
    ctx.fillText(t.toFixed(1), 6, y + 4);
  }

  points.forEach((point, index) => {
    const x = padLeft + index * barWidth;
    const readHeight =
      (Math.max(point.read, 0) / maxY) * (height - padTop - padBottom);
    const writeHeight =
      (Math.max(point.write, 0) / maxY) * (height - padTop - padBottom);
    ctx.fillStyle = "#f2b705";
    ctx.fillRect(
      x,
      height - padBottom - readHeight,
      barWidth * 0.8,
      readHeight
    );
    ctx.fillStyle = "#f78819";
    ctx.fillRect(
      x,
      height - padBottom - readHeight - writeHeight,
      barWidth * 0.8,
      writeHeight
    );
  });
}

function buildEventFilters(summary) {
  const events = summary.events.filter((event) => event.task !== "total");
  const scopes = unique(events.map((event) => event.scope));
  renderSelectOptions(memScope, scopes);
  renderSelectOptions(diskScope, scopes);

  const updateTasks = (scopeSelect, taskSelect) => {
    const scopeValue = scopeSelect.value;
    const tasks = unique(
      events
        .filter((event) => event.scope === scopeValue)
        .map((event) => event.task)
    );
    renderSelectOptions(taskSelect, tasks);
  };

  updateTasks(memScope, memTask);
  updateTasks(diskScope, diskTask);

  memScope.onchange = () => {
    updateTasks(memScope, memTask);
    renderCharts(summary);
  };
  diskScope.onchange = () => {
    updateTasks(diskScope, diskTask);
    renderCharts(summary);
  };
  memTask.onchange = () => renderCharts(summary);
  diskTask.onchange = () => renderCharts(summary);
}

function renderCharts(summary) {
  const events = summary.events.filter((event) => event.task !== "total");
  const memEvents = events.filter(
    (event) =>
      event.scope === memScope.value &&
      event.task === memTask.value &&
      event.memory_delta_mb !== null &&
      event.memory_delta_mb !== undefined
  );
  const memPoints = downsample(
    memEvents.map((event) => ({
      x: event.timestamp_ms,
      y: event.memory_delta_mb,
    }))
  );
  drawLineChart(memCanvas, memPoints, "#f2b705", "rgba(242, 183, 5, 0.18)");

  const diskEvents = events.filter(
    (event) =>
      event.scope === diskScope.value &&
      event.task === diskTask.value &&
      (event.disk_read_mb !== null ||
        event.disk_write_mb !== null ||
        event.disk_read_mb !== undefined ||
        event.disk_write_mb !== undefined)
  );
  const diskPoints = downsample(
    diskEvents.map((event) => ({
      read: event.disk_read_mb || 0,
      write: event.disk_write_mb || 0,
    }))
  );
  drawDiskChart(diskCanvas, diskPoints);
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  const nextWidth = Math.floor(width * ratio);
  const nextHeight = Math.floor(height * ratio);
  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth;
    canvas.height = nextHeight;
  }
  return ratio;
}

window.addEventListener("resize", () => {
  if (memCanvas && diskCanvas) {
    resizeCanvas(memCanvas);
    resizeCanvas(diskCanvas);
  }
});

function applySummary(summary) {
  renderCards(summary);
  renderTables(summary);
  buildEventFilters(summary);
  renderCharts(summary);
}

function loadFromText(text) {
  try {
    const summary = parseProfile(text);
    applySummary(summary);
  } catch (err) {
    alert(`Failed to parse JSON: ${err.message}`);
  }
}

async function fetchProfile(runId) {
  const response = await fetch(`/api/profile?run_id=${encodeURIComponent(runId)}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function startLive(runId) {
  if (liveSource) {
    liveSource.close();
  }
  liveSource = new EventSource(
    `/api/profile?run_id=${encodeURIComponent(runId)}&live=1`
  );
  liveSource.onmessage = (event) => {
    try {
      const summary = parseProfile(event.data);
      applySummary(summary);
    } catch (err) {
      console.error(err);
    }
  };
  liveSource.onerror = () => {
    liveSource.close();
    liveSource = null;
  };
}

fetchButton.addEventListener("click", async () => {
  const runId = runIdInput.value.trim();
  if (!runId) {
    alert("Enter a run ID.");
    return;
  }
  if (liveToggle.checked) {
    startLive(runId);
    return;
  }
  if (liveSource) {
    liveSource.close();
    liveSource = null;
  }
  try {
    const summary = await fetchProfile(runId);
    jsonInput.value = JSON.stringify(summary, null, 2);
    applySummary(summary);
  } catch (err) {
    alert(`Failed to fetch: ${err.message}`);
  }
});

parseButton.addEventListener("click", () => {
  if (jsonInput.value.trim().length === 0) {
    alert("Paste JSON first.");
    return;
  }
  loadFromText(jsonInput.value.trim());
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("active");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    jsonInput.value = reader.result;
    loadFromText(reader.result);
  };
  reader.readAsText(file);
});
