// 추천 화면과 분석 화면이 함께 쓰는 설정·호출·표시 도우미.

const MODEL_PRESETS = {
  ollama: [
    { value: 'qwen3:14b', label: 'Qwen3 14B' },
    { value: 'deepseek-r1:8b', label: 'DeepSeek R1 8B' },
    { value: 'eeve-korean-instruct:10.8b', label: 'EEVE Korean Instruct 10.8B' },
    { value: 'batiai/qwen3.6-27b:iq4', label: 'Qwen3.6 27B (BatiAI IQ4)' },
    { value: 'qwen3.6:35b-a3b', label: 'Qwen3.6 35B-A3B (official)' },
    { value: 'batiai/qwen3.6-35b:iq4', label: 'Qwen3.6 35B-A3B (BatiAI IQ4)' },
    { value: 'gemma4:e2b', label: 'Gemma 4 E2B' },
    { value: 'gemma4:e4b', label: 'Gemma 4 E4B' },
    { value: 'exaone3.5:7.8b', label: 'EXAONE 3.5 7.8B' },
    { value: 'exaone3.5:32b', label: 'EXAONE 3.5 32B' },
  ],
  omniserve: [
    { value: 'track_a_model', label: 'HyperCLOVAX SEED Think 32B (Track A)' },
    { value: 'naver-hyperclovax/HyperCLOVAX-SEED-Think-32B', label: 'HyperCLOVAX SEED Think 32B (HF id)' },
    { value: 'HyperCLOVAX-SEED-Think-32B', label: 'HyperCLOVAX SEED Think 32B (short)' },
  ],
  'local-openai': [
    { value: 'Qwen/Qwen3-14B', label: 'Qwen3 14B' },
    { value: 'Qwen/Qwen3-32B', label: 'Qwen3 32B' },
  ],
};

const CONFIG_DEFAULTS = {
  provider: 'ollama',
  model: 'qwen3:14b',
  ollamaUrl: '/ollama',
  omniserveUrl: '/omniserve/a/v1',
  openAIBaseUrl: 'http://localhost:8000/v1',
};

const RESULT_KEY = 'gr2-last-run';

function readConfig() {
  const config = {};
  for (const [key, fallback] of Object.entries(CONFIG_DEFAULTS)) {
    config[key] = localStorage.getItem(`gr2-${key}`) || fallback;
  }
  return config;
}

function writeConfig(patch) {
  for (const [key, value] of Object.entries(patch)) {
    localStorage.setItem(`gr2-${key}`, value);
  }
}

function engineConfig() {
  return { engine: 'llm-question-pipeline', topK: 30, embeddingModel: 'bge-m3', ...readConfig() };
}

async function postJson(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return { ok: res.ok, status: res.status, data };
}

function isOutOfScopeLabel(stage, subCategory) {
  const norm = value => String(value || '').trim().toLowerCase().replace(/\s+/g, '');
  return norm(stage) === '기타' || norm(subCategory) === 'other' || norm(subCategory) === '기타';
}

// 질문 하나를 분류부터 표제어까지 끝까지 돌린다. 사람 확인 단계는 예측값을 그대로
// 쓰고, 서버가 분류 충돌(409)을 알려주면 제안된 분류로 한 번만 다시 시도한다.
async function runPipeline(question, override = null) {
  const config = engineConfig();
  let classify = null;

  if (override) {
    classify = override.classify || null;
  } else {
    const res = await postJson('/api/llm-pipeline/classify', { ...config, question, maxTokens: 96, numCtx: 4096 });
    if (!res.ok) throw new Error(res.data?.error || `분류 실패 (HTTP ${res.status})`);
    classify = res.data;
  }

  const stage = override?.stage || classify?.predictedStage || '';
  const subCategory = override?.subCategory || classify?.predictedSubCategory || '';

  if (!override && (classify?.outOfScope || isOutOfScopeLabel(stage, subCategory))) {
    return { question, classify, recommend: null, outOfScope: true, stage, subCategory };
  }

  const ask = (useStage, useSub) => postJson('/api/llm-pipeline/recommend', {
    ...config,
    question,
    confirmedStage: useStage,
    confirmedSubCategory: useSub,
    answerCount: 8,
    maxTokens: 768,
    numCtx: 4096,
  });

  let res = await ask(stage, subCategory);
  let repaired = null;

  if (!res.ok && res.status === 409 && res.data?.labelValidation?.suggestedSubCategory) {
    const v = res.data.labelValidation;
    const retry = await ask(v.suggestedStage || stage, v.suggestedSubCategory);
    if (retry.ok) {
      res = retry;
      repaired = { from: `${stage} / ${subCategory}`, to: `${v.suggestedStage || stage} / ${v.suggestedSubCategory}`, reason: v.reason || '' };
    }
  }
  if (!res.ok) throw new Error(res.data?.error || `표제어 생성 실패 (HTTP ${res.status})`);

  return {
    question,
    classify,
    recommend: res.data,
    outOfScope: false,
    repaired,
    stage: res.data.stage || stage,
    subCategory: res.data.subCategory || subCategory,
  };
}

function saveRun(run) {
  try {
    sessionStorage.setItem(RESULT_KEY, JSON.stringify({ ...run, savedAt: Date.now() }));
  } catch {
    // 결과가 너무 커서 못 담아도 추천 화면 자체는 계속 동작해야 한다.
  }
}

function loadRun() {
  try {
    return JSON.parse(sessionStorage.getItem(RESULT_KEY) || 'null');
  } catch {
    return null;
  }
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatLatency(value) {
  const ms = Number(value || 0);
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

function keywordStatistics(recommend) {
  const items = recommend?.keywordCandidates || [];
  const coverage = recommend?.keywordCoverage || {};
  return {
    total: items.length,
    evidence: items.filter(item => item.source === 'answer_evidence').length,
    expanded: items.filter(item => item.source === 'intent_expansion').length,
    primary: items.filter(item => item.intentRole === 'primary').length,
    secondary: items.filter(item => item.intentRole === 'secondary').length,
    dropped: Number(coverage.droppedCount || 0),
    answers: Number(coverage.answers || 0),
    resolvedAnswers: Number(coverage.resolved || 0),
    latency: Number(recommend?.latencyMs?.total || 0),
  };
}

function keywordStatsHtml(recommend) {
  const stats = keywordStatistics(recommend);
  const max = Math.max(stats.total, stats.dropped, 1);
  const bar = (label, value, className) => `
    <div class="stat-bar-row"><span>${escapeHtml(label)}</span>
      <div class="stat-bar-track"><i class="${className}" style="width:${Math.max(0, Math.min(100, value / max * 100))}%"></i></div>
      <strong>${value}</strong></div>`;
  return `<div class="stat-grid">
      <div class="stat-card"><span>최종 표제어</span><strong>${stats.total}</strong><small>개</small></div>
      <div class="stat-card"><span>답변 근거</span><strong>${stats.evidence}</strong><small>개</small></div>
      <div class="stat-card"><span>의도 확장</span><strong>${stats.expanded}</strong><small>개</small></div>
      <div class="stat-card"><span>필터 제외</span><strong>${stats.dropped}</strong><small>개</small></div>
      <div class="stat-card"><span>답변 매핑</span><strong>${stats.resolvedAnswers}</strong><small>/${stats.answers}</small></div>
      <div class="stat-card"><span>전체 응답시간</span><strong>${formatLatency(stats.latency)}</strong></div>
    </div>
    <div class="stat-chart" aria-label="표제어 구성 통계">
      ${bar('질문 대상', stats.primary, 'bar-primary')}
      ${bar('보조 부류', stats.secondary, 'bar-secondary')}
      ${bar('답변 근거', stats.evidence, 'bar-evidence')}
      ${bar('의도 확장', stats.expanded, 'bar-expanded')}
      ${bar('필터 제외', stats.dropped, 'bar-dropped')}
    </div>`;
}

function keywordResultTableHtml(items, { showEvidence = false } = {}) {
  const roleText = { primary: '질문 대상', secondary: '보조 부류', related: '관련 부류', none: '부류 밖' };
  const rows = (items || []).map((item, index) => {
    const names = String(item.gloss || item.name || '').split(',').map(value => value.trim()).filter(Boolean);
    const source = item.source === 'answer_evidence' ? '답변 근거' : '의도 확장';
    const detail = showEvidence
      ? (item.answers || []).map(sentence => `<div class="evidence-sent">${highlightKeywords(sentence, item.keywords)}</div>`).join('')
      : escapeHtml((item.keywords || []).join(', ') || item.intentLabel || '-');
    return `<tr>
      <td class="row-number">${index + 1}</td>
      <td class="gl"><strong>${escapeHtml(names[0] || '-')}</strong>${names.length > 1 ? `<small>${escapeHtml(names.slice(1).join(', '))}</small>` : ''}</td>
      <td>${escapeHtml(String(item.glossIndex ?? item.origin ?? '-'))}</td>
      <td>${escapeHtml(item.category || '-')}</td>
      <td><span class="source-badge ${item.source === 'answer_evidence' ? 'evidence-source' : 'expanded-source'}">${source}</span></td>
      <td>${escapeHtml(roleText[item.intentRole] || item.intentRole || '-')} · ${escapeHtml(item.intentLabel || '-')}</td>
      <td>${detail || '-'}</td>
    </tr>`;
  }).join('');
  return `<div class="table-scroll keyword-table-scroll"><table class="table keyword-table">
    <thead><tr><th>#</th><th>표제어</th><th>인덱스</th><th>분류</th><th>출처</th><th>의미 부류</th><th>${showEvidence ? '근거 문장' : '근거 키워드'}</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="7">추천 표제어 없음</td></tr>'}</tbody>
  </table></div>`;
}

// 표제어가 어느 대목에서 나왔는지 보이도록 근거 키워드에 표시를 넣는다.
// 긴 키워드부터 처리해야 짧은 것이 먼저 걸려 겹치는 일이 없다.
function highlightKeywords(sentence, keywords) {
  let html = escapeHtml(sentence);
  const targets = [...new Set((keywords || []).filter(Boolean))]
    .sort((a, b) => b.length - a.length)
    .map(escapeHtml);
  for (const target of targets) {
    if (!target || html.includes(`<mark>${target}</mark>`)) continue;
    html = html.split(target).join(`<mark>${target}</mark>`);
  }
  return html;
}

// 답변에서 실제로 관찰된 표제어만 근거 문장을 가진다. 부류 확장으로 들어온
// 표제어는 근거 문장이 없는 게 정상이다.
function evidenceRowsHtml(candidates) {
  const rows = (candidates || [])
    .filter(item => item.source === 'answer_evidence' && (item.answers || []).length);
  if (!rows.length) return '';
  return rows.map(item => `
    <div class="evidence-row">
      <span class="evidence-gloss">${escapeHtml(item.gloss)}</span>
      <span class="evidence-sents">${item.answers
        .map(sentence => `<span class="evidence-sent">${highlightKeywords(sentence, item.keywords)}</span>`)
        .join('')}</span>
    </div>`).join('');
}
