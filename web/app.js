'use strict';
const $ = id => document.getElementById(id);
const state = {running: false, busy: false, timer: null, offset: 0, rows: [], model: null, selected: null, sequence: 0};
const money = new Intl.NumberFormat('en-IN', {style: 'currency', currency: 'INR', maximumFractionDigits: 0});
const percent = number => `${(number * 100).toFixed(1)}%`;
const error = message => { $('error-banner').textContent = message || ''; $('error-banner').hidden = !message; };

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const data = await response.json();
    $('connection').className = 'connection online'; $('connection').lastChild.textContent = 'API online';
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map(item => `${item.loc.slice(1).join('.')}: ${item.msg}`).join('; ') : data.detail;
      throw new Error(detail || `Request failed (${response.status})`);
    }
    return data;
  } catch (failure) {
    if (failure instanceof TypeError || failure.name === 'AbortError') {
      $('connection').className = 'connection offline'; $('connection').lastChild.textContent = 'API unavailable';
    }
    if (failure.name === 'AbortError') throw new Error('The API took too long to respond. Check the server terminal, then try again.');
    throw failure;
  } finally { clearTimeout(timeout); }
}
const post = (path, body = {}) => api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});

function controls() {
  $('start-label').textContent = state.running ? 'Pause traffic' : 'Start traffic';
  $('play-icon').textContent = state.running ? 'Ⅱ' : '▶';
  $('start').disabled = !state.model || (state.busy && !state.running) || state.rows.length >= 1000;
  $('step').disabled = !state.model || state.running || state.busy || state.rows.length >= 1000;
  $('predict-button').disabled = !state.model || state.busy || state.rows.length >= 1000;
  $('reset').disabled = state.busy;
  $('reload').disabled = state.busy;
  $('traffic-mode').disabled = state.busy;
  $('download').disabled = !state.rows.length;
  $('session-status').textContent = state.running ? `Running · ${$('traffic-speed').value} transactions/sec target` : state.rows.length ? 'Traffic paused' : 'Ready when you are';
}
function stop() { state.running = false; clearTimeout(state.timer); controls(); }
function tab(which) {
  for (const name of ['send', 'inspect']) {
    const active = name === which;
    $(`tab-${name}`).classList.toggle('active', active);
    $(`tab-${name}`).setAttribute('aria-selected', String(active));
    $(`${name}-panel`).hidden = !active;
  }
}
function inspect(row) {
  state.selected = row.id;
  $('inspection-empty').hidden = true;
  $('inspection-content').hidden = false;
  $('inspection-decision').textContent = `${row.result.decision === 'review' ? 'Flagged for review' : 'Approved'} · score ${row.result.score.toFixed(3)}`;
  $('request-json').textContent = JSON.stringify(row.transaction, null, 2);
  $('response-json').textContent = JSON.stringify(row.result, null, 2);
  $('inspection-truth').textContent = row.truth === null ? 'Manual request: actual outcome unknown; excluded from precision and recall.' : `Known synthetic label: ${row.truth ? 'fraud' : 'legitimate'}. The label was not sent to /predict.`;
  tab('inspect'); renderFeed();
}
function element(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}
function renderFeed() {
  const filter = $('feed-filter').value;
  const rows = state.rows.slice(-60).reverse().filter(row => filter === 'all' || row.result.decision === filter);
  const body = $('feed'); body.replaceChildren();
  for (const row of rows) {
    const tr = element('tr'); tr.classList.toggle('selected', row.id === state.selected);
    tr.tabIndex = 0; tr.setAttribute('aria-label', `Inspect ${row.transaction.txn_id}`);
    tr.addEventListener('click', () => inspect(row));
    tr.addEventListener('keydown', event => { if (event.key === 'Enter') inspect(row); });
    const identity = element('td'); identity.append(element('strong', row.transaction.txn_id));
    identity.append(element('small', `${row.transaction.city} · ${row.source}`));
    const amount = element('td', money.format(row.transaction.amount_inr));
    const score = element('td'); const scoreCell = element('div', undefined, 'score-cell');
    const track = element('div', undefined, 'mini-track'); const fill = element('div', undefined, `mini-fill ${row.result.decision}`);
    fill.style.width = `${row.result.score * 100}%`; track.append(fill);
    scoreCell.append(element('span', row.result.score.toFixed(3)), track); score.append(scoreCell);
    const decision = element('td'); decision.append(element('span', row.result.decision === 'review' ? 'Review' : 'Approve', `pill ${row.result.decision}`));
    const truth = element('td'); truth.append(element('span', row.truth === null ? 'Unknown' : row.truth ? 'Fraud' : 'Legitimate', `truth ${row.truth ? 'fraud' : 'legit'}`));
    tr.append(identity, amount, score, decision, truth, element('td', `${row.result.latency_ms.toFixed(1)} ms`));
    body.append(tr);
  }
  $('feed-empty').hidden = Boolean(rows.length);
  $('feed-empty').querySelector('strong').textContent = state.rows.length ? 'No matching decisions' : 'No transactions yet';
  $('feed-empty').querySelector('p').textContent = state.rows.length ? 'Choose another filter or send more transactions.' : 'Start the simulator or send a transaction of your own.';
  $('feed-count').textContent = state.rows.length.toLocaleString();
}
function renderChart() {
  const svg = $('score-chart'); svg.replaceChildren();
  const ns = 'http://www.w3.org/2000/svg';
  function shape(tag, attributes, text) {
    const item = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) item.setAttribute(key, value);
    if (text !== undefined) item.textContent = text;
    svg.append(item); return item;
  }
  const left = 31, top = 15, width = 750, height = 150;
  for (const value of [0, 0.5, 1]) {
    const y = top + (1 - value) * height;
    shape('line', {x1: left, x2: left + width, y1: y, y2: y, stroke: '#eaf0f6'});
    shape('text', {x: 0, y: y + 4, fill: '#93a0b3', 'font-size': 11}, value.toFixed(1));
  }
  const points = state.rows.slice(-40);
  const slot = width / 40;
  points.forEach((row, index) => {
    const h = Math.max(2, row.result.score * height);
    shape('rect', {x: left + index * slot + 3, y: top + height - h, width: slot - 6, height: h, rx: 2, fill: row.result.decision === 'review' ? '#da6672' : '#40b397', opacity: 0.9});
  });
  const threshold = state.model?.threshold ?? 0.6;
  const y = top + (1 - threshold) * height;
  shape('line', {x1: left, x2: left + width, y1: y, y2: y, stroke: '#7491bd', 'stroke-width': 1.2, 'stroke-dasharray': '5 5'});
  shape('text', {x: left, y: 187, fill: '#93a0b3', 'font-size': 11}, 'Earlier');
  shape('text', {x: left + width, y: 187, 'text-anchor': 'end', fill: '#93a0b3', 'font-size': 11}, 'Latest requests');
  $('chart-empty').hidden = Boolean(points.length);
}
function renderMetrics() {
  const labelled = state.rows.filter(row => row.truth !== null);
  const flagged = state.rows.filter(row => row.result.decision === 'review').length;
  const tp = labelled.filter(row => row.truth && row.result.decision === 'review').length;
  const fp = labelled.filter(row => !row.truth && row.result.decision === 'review').length;
  const fn = labelled.filter(row => row.truth && row.result.decision === 'approve').length;
  $('total').textContent = state.rows.length.toLocaleString();
  $('flagged').textContent = flagged.toLocaleString();
  $('precision').textContent = tp + fp ? percent(tp / (tp + fp)) : '—';
  $('recall').textContent = tp + fn ? percent(tp / (tp + fn)) : '—';
  $('total-note').textContent = state.rows.length ? `${labelled.length} labelled replay · ${state.rows.length - labelled.length} manual` : 'Waiting for the first request';
  $('flag-note').textContent = state.rows.length ? `${percent(flagged / state.rows.length)} of this session sent for review` : 'Model score at or above threshold';
  $('precision-note').textContent = tp + fp ? `${tp} true fraud / ${tp + fp} labelled alerts` : 'Needs at least one labelled alert';
  $('recall-note').textContent = tp + fn ? `${tp} caught / ${tp + fn} known fraud` : 'Needs at least one known fraud';
}
function render() { renderMetrics(); renderFeed(); renderChart(); controls(); }
function reset() {
  stop(); state.rows = []; state.offset = 0; state.selected = null;
  $('inspection-empty').hidden = false; $('inspection-content').hidden = true;
  $('feed-filter').value = 'all'; render();
}
async function send(transaction, truth, source) {
  const result = await post('/predict', transaction);
  const row = {id: ++state.sequence, transaction, truth, source, result};
  state.rows.push(row); render();
  if (state.rows.length >= 1000) { stop(); error('Session reached 1,000 transactions. Export the CSV, then reset to start a new session.'); }
  return row;
}
async function replayOne() {
  if (state.busy || state.rows.length >= 1000) return;
  state.busy = true; controls(); error('');
  try {
    const batch = await api(`/api/samples?mode=${$('traffic-mode').value}&offset=${state.offset}&count=1`);
    const sample = batch.samples[0];
    await send(sample.transaction, sample.actual_fraud, 'replay');
    state.offset = batch.next_offset;
  } catch (failure) { stop(); error(failure.message); }
  finally { state.busy = false; controls(); }
}
async function tick() {
  if (!state.running) return;
  const start = performance.now();
  await replayOne();
  if (state.running) state.timer = setTimeout(tick, Math.max(0, 1000 / Number($('traffic-speed').value) - (performance.now() - start)));
}
function preset(risky) {
  $('amount').value = risky ? 7200 : 500; $('average').value = 600;
  $('distance').value = risky ? 950 : 5; $('hour').value = risky ? 2 : 14;
  $('category').value = risky ? 'electronics' : 'grocery'; $('device').value = risky ? 'desktop' : 'mobile';
  $('city').value = risky ? 'Delhi' : 'Mumbai'; $('velocity').value = risky ? 3 : 0;
  $('new-merchant').checked = risky;
  $('routine-preset').classList.toggle('active', !risky); $('risk-preset').classList.toggle('active', risky);
}
async function loadModel() {
  const model = await api('/api/model'); state.model = model;
  $('model-name').textContent = model.model;
  document.querySelector('.model-icon').textContent = ({'Random Forest': 'RF', 'Logistic Regression': 'LR', 'LightGBM': 'GB'})[model.model] || 'ML';
  $('model-version').textContent = `${model.release} · threshold ${model.threshold.toFixed(2)}`;
  $('release').textContent = model.release; $('data-version').textContent = model.data_version;
  $('threshold').textContent = model.threshold.toFixed(2);
  $('threshold-label').textContent = `Review threshold ${model.threshold.toFixed(2)}`;
  $('val-precision').textContent = percent(model.validation.precision);
  $('val-recall').textContent = percent(model.validation.recall);
  $('connection').className = 'connection online'; $('connection').lastChild.textContent = 'API online';
  updateMixNote(); render();
}
function updateMixNote() {
  const mode = $('traffic-mode').value;
  $('mix-note').textContent = mode === 'balanced' ? 'Balanced replay makes both classes visible. Session metrics reflect this mix.' : mode === 'natural' ? `Natural replay · ${percent(state.model.replay.fraud_rate)} fraud. Session metrics use only requests seen so far.` : 'Single-class replay is for inspecting decisions, not measuring overall model performance.';
}
$('start').addEventListener('click', () => { if (state.running) stop(); else { state.running = true; controls(); tick(); } });
$('step').addEventListener('click', replayOne);
$('reset').addEventListener('click', () => { reset(); error(''); });
$('traffic-mode').addEventListener('change', () => { reset(); updateMixNote(); });
$('traffic-speed').addEventListener('change', controls);
$('feed-filter').addEventListener('change', renderFeed);
$('tab-send').addEventListener('click', () => tab('send'));
$('tab-inspect').addEventListener('click', () => tab('inspect'));
$('routine-preset').addEventListener('click', () => preset(false));
$('risk-preset').addEventListener('click', () => preset(true));
$('transaction-form').addEventListener('submit', async event => {
  event.preventDefault(); if (state.busy || state.rows.length >= 1000) return;
  stop(); state.busy = true; controls(); error('');
  const transaction = {txn_id: `MANUAL-${String(state.sequence + 1).padStart(4, '0')}`, timestamp: `2025-03-10T${String($('hour').value).padStart(2, '0')}:00:00`, customer_id: 100,
    amount_inr: Number($('amount').value), customer_avg_amount_inr: Number($('average').value),
    distance_from_home_km: $('distance').value === '' ? null : Number($('distance').value),
    merchant_category: $('category').value, device_type: $('device').value, city: $('city').value,
    is_new_merchant: Number($('new-merchant').checked), txns_last_1h: Number($('velocity').value)};
  try { inspect(await send(transaction, null, 'manual')); }
  catch (failure) { error(failure.message); }
  finally { state.busy = false; controls(); }
});
$('reload').addEventListener('click', async () => {
  stop(); state.busy = true; controls(); error('');
  try { await post('/api/reload'); await loadModel(); reset(); }
  catch (failure) { error(failure.message); }
  finally { state.busy = false; controls(); }
});
$('download').addEventListener('click', () => {
  const fields = ['txn_id', 'amount_inr', 'source', 'actual_fraud', 'score', 'threshold', 'decision', 'release', 'latency_ms'];
  const safe = value => { let text = String(value ?? ''); if (/^[=+@\-]/.test(text)) text = `'${text}`; return `"${text.replaceAll('"', '""')}"`; };
  const lines = state.rows.map(row => [row.transaction.txn_id, row.transaction.amount_inr, row.source, row.truth, row.result.score, row.result.threshold, row.result.decision, row.result.release, row.result.latency_ms].map(safe).join(','));
  const url = URL.createObjectURL(new Blob([[fields.join(','), ...lines].join('\r\n')], {type: 'text/csv;charset=utf-8'}));
  const link = element('a'); link.href = url; link.download = 'fraud-lab-session.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
render();
loadModel().catch(failure => { $('connection').className = 'connection offline'; $('connection').lastChild.textContent = 'API unavailable'; error(`Could not load the model: ${failure.message}. Check the server terminal and refresh.`); });
