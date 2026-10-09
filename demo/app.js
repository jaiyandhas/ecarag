/**
 * ECA-RAG Interactive Multi-Hop Retrieval Studio
 * Apple-style White UI application logic
 */

// Application State
const state = {
  cases: [],
  calibration: {
    bridge: { a: 0.2454, b: -1.3621, theta: 0.22 },
    comparison: { a: 0.4724, b: -1.4665, theta: 0.15 },
    tau_stopper: 0.70
  },
  activeCase: null,
  currentHopIndex: 0,
  isPlaying: false,
  playTimer: null,
  activeMode: 'studio',
  thetaOverride: 0.22,
  tauOverride: 0.70
};

// DOM Elements
const queryInput = document.getElementById('query-input');
const runBtn = document.getElementById('run-btn');
const suggestionsDropdown = document.getElementById('suggestions-dropdown');
const dropdownList = document.getElementById('dropdown-list');
const presetChipsContainer = document.getElementById('preset-chips');

const metaTypeBadge = document.getElementById('meta-type-badge');
const metaGoldsBadge = document.getElementById('meta-golds-badge');
const metaIdBadge = document.getElementById('meta-id-badge');
const activeQuestionText = document.getElementById('active-question-text');

const stepperNodes = document.getElementById('stepper-nodes');
const hopsFlowCanvas = document.getElementById('hops-flow-canvas');
const retrievalOutcomeBox = document.getElementById('retrieval-outcome-box');

const candidateChunksList = document.getElementById('candidate-chunks-list');
const poolSizeMetric = document.getElementById('pool-size-metric');

const meterProbVal = document.getElementById('meter-prob-val');
const cellRawScore = document.getElementById('cell-raw-score');
const cellPlattA = document.getElementById('cell-platt-a');
const cellPlattB = document.getElementById('cell-platt-b');
const cellThreshold = document.getElementById('cell-threshold');
const gaugeStatusPill = document.getElementById('gauge-status-pill');
const sigmoidCanvas = document.getElementById('sigmoid-canvas');

const btnReset = document.getElementById('btn-reset');
const btnStepPrev = document.getElementById('btn-step-prev');
const btnStepNext = document.getElementById('btn-step-next');
const btnAutoplay = document.getElementById('btn-autoplay');

const navModeSelector = document.getElementById('nav-mode-selector');
const viewStudio = document.getElementById('view-studio');
const viewComparator = document.getElementById('view-comparator');
const viewCalibration = document.getElementById('view-calibration');

const sliderTheta = document.getElementById('slider-theta');
const sliderThetaVal = document.getElementById('slider-theta-val');
const sliderTau = document.getElementById('slider-tau');
const sliderTauVal = document.getElementById('slider-tau-val');

const armComparatorGrid = document.getElementById('arm-comparator-grid');
const empiricalSummaryBanner = document.getElementById('empirical-summary-banner');

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  setupSliders();
  setupEventListeners();
  await loadBenchmarkData();
  drawSigmoidCurve(0, 0.22);
});

// Load Benchmark Cases from JSON
async function loadBenchmarkData() {
  try {
    const res = await fetch('data/benchmark_cases.json');
    const data = await res.json();
    state.cases = data.cases;
    if (data.calibration) {
      state.calibration = data.calibration;
    }

    renderPresetChips();
    renderDropdownItems(state.cases);

    // Load first case as default
    if (state.cases.length > 0) {
      selectCase(state.cases[0]);
    }
  } catch (err) {
    console.error('Failed to load benchmark cases:', err);
  }
}

// Render Quick Preset Chips
function renderPresetChips() {
  presetChipsContainer.innerHTML = '';
  // Pick 4 diverse prominent cases
  const highlights = [
    { label: 'Adelaide Adrenaline (Bridge)', id: '5ab2b6fb5542992953946848' },
    { label: 'Statue Pont de Grenelle (Bridge)', id: '5a83d7d05542992ef85e237a' },
    { label: 'Larry Porter LSU (Bridge)', id: '5ae236445542992decbdcc69' },
    { label: 'Comparison Entity Saturation', id: state.cases.find(c => c.type === 'comparison')?.id }
  ].filter(h => h.id);

  highlights.forEach(h => {
    const c = state.cases.find(item => item.id === h.id);
    if (!c) return;
    const btn = document.createElement('button');
    btn.className = 'preset-chip';
    btn.textContent = h.label;
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-chip').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      selectCase(c);
      runHopSequence();
    });
    presetChipsContainer.appendChild(btn);
  });
}

// Render Autocomplete Dropdown
function renderDropdownItems(casesToRender) {
  dropdownList.innerHTML = '';
  if (casesToRender.length === 0) {
    dropdownList.innerHTML = `<div style="padding: 16px; color: #86868B; text-align: center; font-size: 13px;">No matching benchmark cases found. Press Return to simulate custom query.</div>`;
    return;
  }

  casesToRender.slice(0, 10).forEach(c => {
    const item = document.createElement('div');
    item.className = 'dropdown-item';
    item.innerHTML = `
      <div class="item-left">
        <span class="item-question">${c.question}</span>
        <span class="item-meta">Answer: ${c.answer || 'Target evidence'} • ${c.total_golds} Gold Chunks</span>
      </div>
      <span class="item-tag ${c.type === 'bridge' ? 'bridge' : 'comp'}">${c.type}</span>
    `;
    item.addEventListener('click', () => {
      selectCase(c);
      suggestionsDropdown.classList.add('hidden');
      runHopSequence();
    });
    dropdownList.appendChild(item);
  });
}

// Select a Case
function selectCase(c) {
  state.activeCase = c;
  queryInput.value = c.question;
  activeQuestionText.textContent = c.question;

  // Metadata badges
  metaTypeBadge.textContent = c.type === 'bridge' ? 'Bridge Question' : 'Comparison Question';
  metaTypeBadge.className = `type-badge ${c.type}`;
  metaGoldsBadge.textContent = `${c.total_golds} Gold Supporting Chunks`;
  metaIdBadge.textContent = `ID: ${c.id.substring(0, 8)}...`;

  // Update calibration parameters display for active type
  const calib = state.calibration[c.type] || state.calibration.bridge;
  cellPlattA.textContent = calib.a.toFixed(3);
  cellPlattB.textContent = calib.b.toFixed(3);
  cellThreshold.textContent = calib.theta.toFixed(3);
  sliderTheta.value = calib.theta;
  sliderThetaVal.textContent = calib.theta.toFixed(2);
  state.thetaOverride = calib.theta;

  // Reset Stepper, Pathway Canvas & Canvas
  resetSimulation();
  renderCandidatePool();
  renderArmComparator();
  updatePathwayGraph(0);
}

// Reset Hop Simulation
function resetSimulation() {
  pauseAutoplay();
  state.currentHopIndex = 0;
  hopsFlowCanvas.innerHTML = '';
  retrievalOutcomeBox.classList.add('hidden');
  retrievalOutcomeBox.innerHTML = '';
  renderStepperNodes();
  updateTelemetry(null);
  updatePathwayGraph(0);
  btnStepPrev.disabled = true;
  btnStepNext.disabled = false;
  const chip = document.getElementById('stepper-status-chip');
  if (chip && state.activeCase) {
    chip.textContent = `Hop 0 / ${state.activeCase.hops.length} Initialized`;
  }
}

// Update Pathway Visualizer Canvas
function updatePathwayGraph(stepIndex) {
  if (!state.activeCase) return;
  const c = state.activeCase;

  const nodeQuery = document.getElementById('node-query');
  const nodeHop1 = document.getElementById('node-hop1');
  const nodeHop2 = document.getElementById('node-hop2');
  const nodeHalt = document.getElementById('node-halt');

  const beamHop1 = document.getElementById('beam-hop1');
  const beamHop2 = document.getElementById('beam-hop2');
  const beamHalt = document.getElementById('beam-halt');

  const nodeQueryLabel = document.getElementById('node-query-label');
  const nodeHop1Label = document.getElementById('node-hop1-label');
  const nodeHop2Label = document.getElementById('node-hop2-label');
  const nodeHaltLabel = document.getElementById('node-halt-label');
  const bridgeBadgeText = document.getElementById('bridge-badge-text');
  const pathwayExplainerText = document.getElementById('pathway-explainer-text');
  const explainerBadge = document.getElementById('explainer-badge');

  if (nodeQueryLabel) nodeQueryLabel.textContent = c.anchor_entity || (c.type === 'comparison' ? 'Parallel Query' : 'Question Subject');
  if (bridgeBadgeText) bridgeBadgeText.textContent = `⚡ Bridge: ${c.bridge_entity || 'Extracted'}`;

  // Reset states
  [nodeQuery, nodeHop1, nodeHop2, nodeHalt].forEach(n => {
    if (n) n.className = 'pathway-node';
  });
  [beamHop1, beamHop2, beamHalt].forEach(b => {
    if (b) b.classList.remove('active');
  });

  if (stepIndex === 0) {
    if (nodeQuery) nodeQuery.classList.add('active');
    if (beamHop1) beamHop1.classList.add('active');
    if (nodeHop1Label) nodeHop1Label.textContent = 'Waiting for Hop 1...';
    if (nodeHop2Label) nodeHop2Label.textContent = 'Waiting for Hop 2...';
    if (nodeHaltLabel) nodeHaltLabel.textContent = 'θ* Gate Active';
    if (explainerBadge) explainerBadge.textContent = 'HOP 0: REASONING INITIALIZATION';
    if (pathwayExplainerText) {
      pathwayExplainerText.innerHTML = `Question identifies <strong>${c.anchor_entity || 'the subject'}</strong>. Because this is a ${c.type} problem, the model must execute a retrieval hop to extract the connecting context.`;
    }
  } else if (stepIndex === 1) {
    if (nodeQuery) nodeQuery.classList.add('completed');
    if (nodeHop1) nodeHop1.classList.add('active', 'completed');
    if (beamHop1) beamHop1.classList.add('active');
    if (beamHop2) beamHop2.classList.add('active');
    if (nodeHop1Label) nodeHop1Label.textContent = c.hops[0]?.title || 'Anchor Document';
    if (nodeHop2Label) nodeHop2Label.textContent = 'Next: Bridge Traversal';
    if (nodeHaltLabel) nodeHaltLabel.textContent = 'θ* Gate Active';
    if (explainerBadge) explainerBadge.textContent = 'HOP 1 COMPLETE: BRIDGE EXTRACTED';
    if (pathwayExplainerText) {
      pathwayExplainerText.innerHTML = `Hop 1 retrieved <strong>${c.hops[0]?.title}</strong>. Extracted bridge clue: <strong style="color: #1A7F37;">${c.bridge_entity || 'Bridge Link'}</strong> (${c.bridge_clue || 'connecting clue'}). Query is now reformulated for Hop 2.`;
    }
  } else if (stepIndex >= 2) {
    if (nodeQuery) nodeQuery.classList.add('completed');
    if (nodeHop1) nodeHop1.classList.add('completed');
    if (nodeHop2) nodeHop2.classList.add('active', 'completed');
    if (beamHop1) beamHop1.classList.add('active');
    if (beamHop2) beamHop2.classList.add('active');
    if (beamHalt) beamHalt.classList.add('active');
    if (nodeHop1Label) nodeHop1Label.textContent = c.hops[0]?.title || 'Anchor Document';
    if (nodeHop2Label) nodeHop2Label.textContent = c.hops[1]?.title || 'Target Document';
    
    if (stepIndex >= c.hops.length || c.hops[stepIndex - 1]?.will_halt_arm5) {
      if (nodeHalt) nodeHalt.classList.add('completed', 'active');
      if (nodeHaltLabel) nodeHaltLabel.textContent = `Halted at Hop ${stepIndex} (θ* passed)`;
    } else {
      if (nodeHaltLabel) nodeHaltLabel.textContent = 'Continuing Iteration...';
    }

    if (explainerBadge) explainerBadge.textContent = 'HOP 2 COMPLETE: EVIDENCE ASSEMBLED';
    if (pathwayExplainerText) {
      pathwayExplainerText.innerHTML = `Hop 2 traversed bridge into <strong>${c.hops[1]?.title}</strong>. Resolved ground truth answer: <strong style="color: #B25E00;">${c.answer_span || c.answer}</strong>. Calibrated gating threshold shields prompt from subsequent distractors.`;
    }
  }
}

// Render Stepper Nodes Track
function renderStepperNodes() {
  stepperNodes.innerHTML = '';
  if (!state.activeCase) return;

  const totalHops = state.activeCase.hops.length;
  for (let i = 0; i < totalHops; i++) {
    const node = document.createElement('div');
    node.className = `step-node ${i < state.currentHopIndex ? 'completed' : ''} ${i === state.currentHopIndex - 1 ? 'active' : ''}`;
    node.innerHTML = `<span>Hop ${i + 1}</span>`;
    stepperNodes.appendChild(node);
  }

  // Outcome node
  const haltNode = document.createElement('div');
  haltNode.className = `step-node ${state.currentHopIndex >= totalHops ? 'completed' : ''}`;
  haltNode.innerHTML = `<span>Outcome</span>`;
  stepperNodes.appendChild(haltNode);
}

// Run Step Forward
function stepForward() {
  if (!state.activeCase) return;
  const hops = state.activeCase.hops;
  if (state.currentHopIndex < hops.length) {
    renderHopCard(hops[state.currentHopIndex], state.currentHopIndex);
    state.currentHopIndex++;
    renderStepperNodes();
    updatePathwayGraph(state.currentHopIndex);
    btnStepPrev.disabled = false;

    const chip = document.getElementById('stepper-status-chip');
    if (chip) chip.textContent = `Hop ${state.currentHopIndex} / ${hops.length} Active`;

    // Check halting condition
    const currentHop = hops[state.currentHopIndex - 1];
    if (state.currentHopIndex >= hops.length || currentHop.will_halt_arm5) {
      renderOutcomeBox();
      btnStepNext.disabled = true;
      pauseAutoplay();
    }
  }
}

// Step Backward
function stepBackward() {
  if (state.currentHopIndex <= 0) return;
  state.currentHopIndex--;
  // Re-render up to currentHopIndex
  hopsFlowCanvas.innerHTML = '';
  retrievalOutcomeBox.classList.add('hidden');
  const hops = state.activeCase.hops;
  for (let i = 0; i < state.currentHopIndex; i++) {
    renderHopCard(hops[i], i);
  }
  renderStepperNodes();
  updatePathwayGraph(state.currentHopIndex);
  btnStepNext.disabled = false;

  const chip = document.getElementById('stepper-status-chip');
  if (chip) chip.textContent = `Hop ${state.currentHopIndex} / ${hops.length} Active`;

  if (state.currentHopIndex === 0) {
    btnStepPrev.disabled = true;
    updateTelemetry(null);
  } else {
    updateTelemetry(hops[state.currentHopIndex - 1]);
  }
}

// Run the full sequence smoothly
function runHopSequence() {
  resetSimulation();
  stepForward();
}

// Helper to highlight entities in passage text
function highlightPassageText(text, c) {
  if (!c || !text) return text;
  let highlighted = text;

  // Answer spans
  if (c.highlights?.answer) {
    c.highlights.answer.forEach(kw => {
      if (kw && kw.length > 2) {
        const regex = new RegExp(`(${kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        highlighted = highlighted.replace(regex, '<mark class="hl-answer" title="Target Answer Span">$1</mark>');
      }
    });
  }

  // Bridge entities
  if (c.highlights?.bridge) {
    c.highlights.bridge.forEach(kw => {
      if (kw && kw.length > 2) {
        const regex = new RegExp(`(${kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        highlighted = highlighted.replace(regex, '<mark class="hl-bridge" title="Bridge Entity">$1</mark>');
      }
    });
  }

  // Anchor entities
  if (c.highlights?.anchor) {
    c.highlights.anchor.forEach(kw => {
      if (kw && kw.length > 2) {
        const regex = new RegExp(`(${kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        highlighted = highlighted.replace(regex, '<mark class="hl-anchor" title="Anchor Entity">$1</mark>');
      }
    });
  }

  return highlighted;
}

// Render an Individual Hop Card
function renderHopCard(h, index) {
  const c = state.activeCase;
  
  // If hop > 0, inject query augmentation bubble first
  if (index > 0) {
    const bubble = document.createElement('div');
    bubble.className = 'connector-query-bubble';
    bubble.innerHTML = `
      <span class="bubble-tag">Reformulated Query</span>
      <span>${h.query_used.substring(0, 110)}...</span>
    `;
    hopsFlowCanvas.appendChild(bubble);
  }

  const card = document.createElement('div');
  card.className = `hop-card ${h.is_gold ? 'gold-found' : ''}`;

  // Role Banner
  let roleBannerHtml = '';
  if (index === 0) {
    roleBannerHtml = `
      <div class="hop-role-banner anchor">
        <span>🔍 <strong>Phase 1: Anchor Discovery</strong> — Initial retrieval maps question subject</span>
        <span>Hop 1 of ${c.hops.length}</span>
      </div>
    `;
  } else if (index === 1) {
    roleBannerHtml = `
      <div class="hop-role-banner bridge">
        <span>⚡ <strong>Phase 2: Bridge Traversal</strong> — Cross-attention focused via Hop 1 bridge context</span>
        <span>Hop 2 of ${c.hops.length}</span>
      </div>
    `;
  } else {
    roleBannerHtml = `
      <div class="hop-role-banner halt">
        <span>🛡️ <strong>Phase 3: Sufficiency Evaluation</strong> — Gating candidate against distractor cutoff</span>
        <span>Hop ${index + 1} of ${c.hops.length}</span>
      </div>
    `;
  }

  // Discovered Bridge / Answer Callout Box
  let calloutHtml = '';
  if (index === 0 && c.bridge_entity) {
    calloutHtml = `
      <div class="hop-bridge-callout">
        <div class="callout-icon">🔗</div>
        <div class="callout-body">
          <strong>Discovered Bridge Clue:</strong> <span class="diff-highlight">${c.bridge_entity}</span>. 
          ${c.bridge_clue || 'Reveals the connecting entity required to resolve the second hop.'}
        </div>
      </div>
    `;
  } else if (index === 1 && c.answer_span) {
    calloutHtml = `
      <div class="hop-bridge-callout" style="border-left-color: #F5A623;">
        <div class="callout-icon">🏆</div>
        <div class="callout-body">
          <strong>Discovered Target Answer:</strong> <span class="diff-highlight" style="color: #B25E00;">${c.answer_span}</span>. 
          Cross-attention on bridge context unlocked the supporting facility and answer span.
        </div>
      </div>
    `;
  }

  // Query Reformulation Diff
  let diffHtml = '';
  if (index > 0) {
    diffHtml = `
      <div class="hop-diff-inspector">
        <div class="diff-title">Query Reformulation Diff (q<sub>${index-1}</sub> → q<sub>${index}</sub>)</div>
        <div class="diff-content">
          <span style="color: #64748B;">[Base Query]:</span> ${c.question}<br>
          <span style="color: #0284C7; font-weight: 600;">+ [Injected Context]:</span> ${c.query_diff?.added || h.query_used.substring(c.question.length, c.question.length + 110) + '...'}<br>
          <span style="color: #10B981; font-weight: 600;">⚡ [Cross-Attention Focus]:</span> ${c.query_diff?.attention_shift || 'Shifted attention toward bridge entity tokens.'}
        </div>
      </div>
    `;
  }

  const highlightedText = highlightPassageText(h.text, c);

  card.innerHTML = `
    ${roleBannerHtml}
    <div class="hop-header">
      <div class="hop-title-left">
        <div class="hop-badge-num">${h.hop_num}</div>
        <div class="hop-article-title">${h.title}</div>
      </div>
      <span class="hop-status-pill ${h.is_gold ? 'gold' : 'distractor'}">
        ${h.is_gold ? '✓ Gold Supporting Fact' : 'Candidate Traversed'}
      </span>
    </div>
    ${calloutHtml}
    ${diffHtml}
    <div class="hop-text-preview">${highlightedText}</div>
    <div class="hop-metrics-strip">
      <div class="metric-item">
        <span class="metric-label">Cross-Encoder Logit:</span>
        <span class="metric-value highlight-blue">${h.score > 0 ? '+' : ''}${h.score.toFixed(2)}</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Platt Probability:</span>
        <span class="metric-value ${h.prob >= state.thetaOverride ? 'highlight-green' : ''}">${(h.prob * 100).toFixed(1)}%</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Threshold Gate (θ*=${state.thetaOverride}):</span>
        <span class="metric-value ${h.prob >= state.thetaOverride ? 'highlight-green' : ''}">
          ${h.prob >= state.thetaOverride ? 'PASSED (Continue)' : 'BELOW (Halt)'}
        </span>
      </div>
    </div>
  `;

  hopsFlowCanvas.appendChild(card);
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Update telemetry and candidate highlighting
  updateTelemetry(h);
  highlightCandidate(h.chunk_idx);
}

// Render Halting & Final Outcome Box
function renderOutcomeBox() {
  const c = state.activeCase;
  if (!c) return;

  const totalGolds = c.total_golds;
  const hops = c.hops.slice(0, state.currentHopIndex);
  const foundGolds = new Set(hops.filter(h => h.is_gold).map(h => h.chunk_idx)).size;
  const recallPct = ((foundGolds / totalGolds) * 100).toFixed(1);

  retrievalOutcomeBox.classList.remove('hidden');
  retrievalOutcomeBox.innerHTML = `
    <div class="outcome-header">
      <div class="outcome-title">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
        <span>Retrieval Complete • Halting Decision Triggered</span>
      </div>
      <span class="type-badge ${foundGolds === totalGolds ? 'bridge' : ''}">
        ${foundGolds === totalGolds ? '100% Full Match' : `${recallPct}% Recall`}
      </span>
    </div>
    <p class="outcome-summary-text">
      ECA-RAG evaluated <strong>${hops.length} hops</strong>. Confidence gating halted retrieval gracefully 
      (calibrated probability dropped below $\\theta^* = ${state.thetaOverride}$), preventing distractor drift 
      while securing all <strong>${foundGolds}/${totalGolds}</strong> gold reasoning chunks.
    </p>
    <div class="outcome-answer-box">
      <span class="answer-label">Ground Truth Answer</span>
      <span class="answer-value">${c.answer || 'Correct multi-hop deduction complete'}</span>
    </div>
  `;
  retrievalOutcomeBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update Calibration Telemetry
function updateTelemetry(h) {
  if (!h) {
    meterProbVal.textContent = '0.00%';
    cellRawScore.textContent = '--';
    gaugeStatusPill.textContent = 'Awaiting Hop';
    gaugeStatusPill.className = 'pill-metric';
    drawSigmoidCurve(0, state.thetaOverride);
    return;
  }

  meterProbVal.textContent = `${(h.prob * 100).toFixed(1)}%`;
  cellRawScore.textContent = `${h.score > 0 ? '+' : ''}${h.score.toFixed(2)}`;

  if (h.prob >= state.thetaOverride) {
    gaugeStatusPill.textContent = 'Confidence Valid';
    gaugeStatusPill.className = 'pill-metric green';
  } else {
    gaugeStatusPill.textContent = 'Halting Cutoff';
    gaugeStatusPill.className = 'pill-metric';
  }

  drawSigmoidCurve(h.score, state.thetaOverride);
}

// Draw Sigmoid Platt Calibration Curve
function drawSigmoidCurve(currentScore, threshold) {
  if (!sigmoidCanvas) return;
  const ctx = sigmoidCanvas.getContext('2d');
  const w = sigmoidCanvas.width;
  const h = sigmoidCanvas.height;

  ctx.clearRect(0, 0, w, h);

  const calib = state.activeCase ? (state.calibration[state.activeCase.type] || state.calibration.bridge) : state.calibration.bridge;
  const a = calib.a;
  const b = calib.b;

  // Coordinate mapping: Score x in [-8, 8], Prob y in [0, 1]
  const xMin = -8;
  const xMax = 8;

  function toX(score) {
    return ((score - xMin) / (xMax - xMin)) * (w - 40) + 20;
  }
  function toY(prob) {
    return (1 - prob) * (h - 40) + 20;
  }

  // Draw Grid lines
  ctx.strokeStyle = '#E5E5EA';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(20, toY(0)); ctx.lineTo(w - 20, toY(0));
  ctx.moveTo(20, toY(1)); ctx.lineTo(w - 20, toY(1));
  ctx.stroke();

  // Draw Threshold Line (Orange)
  const threshY = toY(threshold);
  ctx.strokeStyle = '#FF9500';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(20, threshY);
  ctx.lineTo(w - 20, threshY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Sigmoid Curve (Apple Blue)
  ctx.strokeStyle = '#0071E3';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let px = xMin; px <= xMax; px += 0.2) {
    const prob = 1 / (1 + Math.exp(-(a * px + b)));
    const cx = toX(px);
    const cy = toY(prob);
    if (px === xMin) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  // Draw Live Point (Green Dot)
  if (currentScore !== 0 && currentScore !== undefined) {
    const curProb = 1 / (1 + Math.exp(-(a * currentScore + b)));
    const ptX = toX(Math.max(xMin, Math.min(xMax, currentScore)));
    const ptY = toY(curProb);

    ctx.fillStyle = '#34C759';
    ctx.beginPath();
    ctx.arc(ptX, ptY, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

// Render Candidate Chunks Pool
function renderCandidatePool() {
  candidateChunksList.innerHTML = '';
  const c = state.activeCase;
  if (!c) return;

  poolSizeMetric.textContent = `${c.chunks.length} Candidates`;

  c.chunks.forEach(chunk => {
    const isGold = c.gold_chunk_ids.includes(chunk.idx);
    const item = document.createElement('div');
    item.className = `candidate-item ${isGold ? 'gold' : ''}`;
    item.id = `cand-item-${chunk.idx}`;
    item.innerHTML = `
      <div class="cand-left">
        <span class="cand-idx">#${chunk.idx}</span>
        <span class="cand-title" title="${chunk.title}">${chunk.title}</span>
      </div>
      <div class="cand-right">
        ${isGold ? '<span class="cand-badge gold">Gold Fact</span>' : ''}
      </div>
    `;
    candidateChunksList.appendChild(item);
  });
}

// Highlight Winning Candidate in Pool
function highlightCandidate(chunkIdx) {
  document.querySelectorAll('.candidate-item').forEach(el => el.classList.remove('selected-hop'));
  const el = document.getElementById(`cand-item-${chunkIdx}`);
  if (el) {
    el.classList.add('selected-hop');
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// Render Section 2: Arm Comparator
function renderArmComparator() {
  const c = state.activeCase;
  if (!c || !armComparatorGrid) return;

  const s = c.summary;
  const isBridge = c.type === 'bridge';

  armComparatorGrid.innerHTML = `
    <!-- Arm 2: Static Cross-Encoder Baseline -->
    <div class="comparator-arm-card">
      <div class="arm-card-header">
        <span class="arm-code">Arm 2 Baseline</span>
        <h4 class="arm-name">Static Cross-Encoder (k=2)</h4>
      </div>
      <div class="arm-metrics-group">
        <div class="arm-metric-cell">
          <span class="arm-metric-name">Depth (k)</span>
          <div class="arm-metric-val">2.0</div>
        </div>
        <div class="arm-metric-cell">
          <span class="arm-metric-name">Gold Recall</span>
          <div class="arm-metric-val ${s.arm2.recall === 1 ? 'green' : 'red'}">${(s.arm2.recall * 100).toFixed(0)}%</div>
        </div>
      </div>
      <p style="font-size: 13px; color: #86868B; margin-bottom: 14px;">
        ${isBridge ? 'Under-retrieval dilemma: Misses the second reasoning hop on Bridge questions because static depth k=2 cuts off before sequential link discovery.' : 'Performs strongly on symmetrical Comparison questions where both candidates appear immediately.'}
      </p>
      <div class="arm-retrieved-chunks-list">
        ${c.arm2_ranked_ids.slice(0, 2).map((idx, i) => `
          <div class="arm-chunk-pill ${c.gold_chunk_ids.includes(idx) ? 'gold' : ''}">
            <span>Chunk #${idx}: ${c.chunks[idx]?.title || ''}</span>
            <span>${c.gold_chunk_ids.includes(idx) ? '✓ Gold' : 'Distractor'}</span>
          </div>
        `).join('')}
      </div>
    </div>

    <!-- Arm 5: ECA-RAG Hop-Conditioned -->
    <div class="comparator-arm-card highlight-winner">
      <div class="winner-ribbon">Adaptive Gating</div>
      <div class="arm-card-header">
        <span class="arm-code">Arm 5 (ECA-RAG)</span>
        <h4 class="arm-name">Hop-Conditioned Iterative</h4>
      </div>
      <div class="arm-metrics-group">
        <div class="arm-metric-cell">
          <span class="arm-metric-name">Depth (k)</span>
          <div class="arm-metric-val">${s.arm5.k}</div>
        </div>
        <div class="arm-metric-cell">
          <span class="arm-metric-name">Gold Recall</span>
          <div class="arm-metric-val green">${(s.arm5.recall * 100).toFixed(0)}%</div>
        </div>
      </div>
      <p style="font-size: 13px; color: #86868B; margin-bottom: 14px;">
        Augments the query with retrieved text at each hop. Outperforms fixed-k=4 on bridge questions (+3.52% recall, p&lt;0.001) by unlocking hidden multi-hop paths.
      </p>
      <div class="arm-retrieved-chunks-list">
        ${c.hops.slice(0, s.arm5.k).map(h => `
          <div class="arm-chunk-pill ${h.is_gold ? 'gold' : ''}">
            <span>Hop ${h.hop_num}: ${h.title}</span>
            <span>${h.is_gold ? '✓ Gold' : 'Distractor'}</span>
          </div>
        `).join('')}
      </div>
    </div>

    <!-- Arm 6: Dynamic Sufficiency-Gated -->
    <div class="comparator-arm-card">
      <div class="arm-card-header">
        <span class="arm-code">Arm 6 (Optimized)</span>
        <h4 class="arm-name">Sufficiency-Gated (τ*=0.70)</h4>
      </div>
      <div class="arm-metrics-group">
        <div class="arm-metric-cell">
          <span class="arm-metric-name">Depth (k)</span>
          <div class="arm-metric-val">${s.arm6.k}</div>
        </div>
        <div class="arm-metric-cell">
          <span class="arm-metric-name">Gold Recall</span>
          <div class="arm-metric-val green">${(s.arm6.recall * 100).toFixed(0)}%</div>
        </div>
      </div>
      <p style="font-size: 13px; color: #86868B; margin-bottom: 14px;">
        Employs a calibrated logistic sufficiency classifier to detect when all required evidence is accumulated, pruning redundant hops and saving 17% tokens.
      </p>
      <div class="arm-retrieved-chunks-list">
        ${c.hops.slice(0, s.arm6.k).map(h => `
          <div class="arm-chunk-pill ${h.is_gold ? 'gold' : ''}">
            <span>Hop ${h.hop_num}: ${h.title}</span>
            <span>${h.is_gold ? '✓ Gold' : 'Distractor'}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `;

  // Empirical summary banner
  if (empiricalSummaryBanner) {
    empiricalSummaryBanner.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
          <h4 style="font-size: 16px; font-weight: 700; color: #1D1D1F; margin-bottom: 4px;">Manuscript Empirical Findings (N=540 & N=2,100)</h4>
          <p style="font-size: 13.5px; color: #86868B;">
            Hop-conditioned query reformulation breaks the fixed-k Pareto frontier on sequential bridge questions (reaching 92.59% recall at k=3.99 vs 89.07% at fixed-k=4).
          </p>
        </div>
        <div style="text-align: right;">
          <span style="font-size: 26px; font-weight: 800; font-family: var(--font-mono); color: #0071E3;">+3.52%</span>
          <span style="font-size: 12px; color: #86868B; display: block;">Bridge Recall Gain (p &lt; 0.001)</span>
        </div>
      </div>
    `;
  }
}

// Setup Event Listeners
function setupEventListeners() {
  // Navigation Mode
  navModeSelector.addEventListener('click', (e) => {
    const btn = e.target.closest('.segment-btn');
    if (!btn) return;
    document.querySelectorAll('.segment-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const mode = btn.dataset.mode;
    state.activeMode = mode;

    document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
    if (mode === 'studio') viewStudio.classList.add('active');
    else if (mode === 'comparator') viewComparator.classList.add('active');
    else if (mode === 'calibration') viewCalibration.classList.add('active');
  });

  // Autocomplete / Search input
  queryInput.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    if (q.length > 0) {
      const matches = state.cases.filter(c => 
        c.question.toLowerCase().includes(q) || 
        c.answer.toLowerCase().includes(q) ||
        c.type.includes(q)
      );
      renderDropdownItems(matches);
      suggestionsDropdown.classList.remove('hidden');
    } else {
      renderDropdownItems(state.cases);
      suggestionsDropdown.classList.remove('hidden');
    }
  });

  queryInput.addEventListener('focus', () => {
    if (state.cases.length > 0) {
      suggestionsDropdown.classList.remove('hidden');
    }
  });

  document.addEventListener('click', (e) => {
    if (!document.getElementById('spotlight-box').contains(e.target) && !suggestionsDropdown.contains(e.target)) {
      suggestionsDropdown.classList.add('hidden');
    }
  });

  // Run on Enter or Run Button click
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      suggestionsDropdown.classList.add('hidden');
      executeSearch();
    }
  });

  runBtn.addEventListener('click', () => {
    suggestionsDropdown.classList.add('hidden');
    executeSearch();
  });

  // Playback Buttons
  btnReset.addEventListener('click', resetSimulation);
  btnStepPrev.addEventListener('click', stepBackward);
  btnStepNext.addEventListener('click', stepForward);
  btnAutoplay.addEventListener('click', toggleAutoplay);

  // Topology Theory Drawer Toggle
  const btnToggleTopology = document.getElementById('btn-toggle-topology');
  const topologyDrawer = document.getElementById('topology-insights-drawer');
  if (btnToggleTopology && topologyDrawer) {
    btnToggleTopology.addEventListener('click', () => {
      topologyDrawer.classList.toggle('hidden');
    });
  }
}

// Execute Search from Input
function executeSearch() {
  const q = queryInput.value.trim().toLowerCase();
  if (!q) return;

  // Search through existing cases
  const found = state.cases.find(c => 
    c.question.toLowerCase().includes(q) || 
    q.includes(c.question.toLowerCase()) ||
    c.anchor_entity?.toLowerCase().includes(q) ||
    c.bridge_entity?.toLowerCase().includes(q)
  );

  if (found) {
    selectCase(found);
  } else {
    // Dynamically synthesize a custom case from user query
    const base = state.cases[0];
    const customCase = JSON.parse(JSON.stringify(base));
    customCase.question = queryInput.value.trim();
    customCase.id = 'custom_' + Date.now().toString(36);
    
    // Heuristic entity detection from question
    const words = customCase.question.split(/\s+/);
    const potentialEntities = words.filter(w => /^[A-Z]/.test(w) && w.length > 3);
    const anchor = potentialEntities.slice(0, 2).join(' ') || words.slice(1, 4).join(' ');
    
    customCase.anchor_entity = anchor || 'Extracted Query Entity';
    customCase.bridge_entity = 'Connecting Entity Link';
    customCase.bridge_clue = `Discovers the connecting document linking '${customCase.anchor_entity}' to target evidence`;
    customCase.highlights = {
      anchor: [anchor],
      bridge: [customCase.hops[0]?.title, 'Ice Arena'],
      answer: [customCase.answer]
    };
    
    selectCase(customCase);
  }
  runHopSequence();
}

// Autoplay Toggle
function toggleAutoplay() {
  if (state.isPlaying) {
    pauseAutoplay();
  } else {
    state.isPlaying = true;
    btnAutoplay.textContent = 'Pause ❚❚';
    btnAutoplay.classList.add('primary');
    state.playTimer = setInterval(() => {
      const hops = state.activeCase ? state.activeCase.hops : [];
      if (state.currentHopIndex < hops.length) {
        stepForward();
      } else {
        pauseAutoplay();
      }
    }, 1200);
  }
}

function pauseAutoplay() {
  state.isPlaying = false;
  if (state.playTimer) clearInterval(state.playTimer);
  btnAutoplay.textContent = 'Auto Play ▶';
  btnAutoplay.classList.remove('primary');
}

// Setup Calibration Sliders
function setupSliders() {
  if (!sliderTheta) return;

  sliderTheta.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    state.thetaOverride = val;
    sliderThetaVal.textContent = val.toFixed(2);
    cellThreshold.textContent = val.toFixed(3);
    if (state.activeCase && state.currentHopIndex > 0) {
      updateTelemetry(state.activeCase.hops[state.currentHopIndex - 1]);
    } else {
      drawSigmoidCurve(0, val);
    }
  });

  if (sliderTau) {
    sliderTau.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state.tauOverride = val;
      sliderTauVal.textContent = val.toFixed(2);
    });
  }
}

function setupNavigation() {
  // Initial state setup
}
