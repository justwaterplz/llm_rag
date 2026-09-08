const express = require('express');
const XLSX = require('xlsx');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

let dataset = [];
let embeddingIndex = null;
let questionAnswerPool = null;
let llmGlossVectorDb = null;
let questionRetrievalIndex = null;
let keywordGlossMap = null;
let intentGlossSets = null;

const EMBEDDING_INDEX_FILE = process.env.EMBEDDING_RAG_INDEX ||
  path.join(__dirname, '..', 'dataset-builder', 'data', 'embedding_rag_index.json');
const QUESTION_ANSWER_POOL_FILE = process.env.QUESTION_ANSWER_POOL ||
  path.join(__dirname, '..', 'dataset-builder', 'data', 'llm_pipeline', 'question_answer_pool.json');
const LLM_GLOSS_VECTOR_DB_FILE = process.env.LLM_GLOSS_VECTOR_DB ||
  path.join(__dirname, '..', 'dataset-builder', 'data', 'llm_pipeline', 'test_658_gloss_vector_db.json');
const QUESTION_RETRIEVAL_INDEX_FILE = process.env.QUESTION_RETRIEVAL_INDEX ||
  path.join(__dirname, '..', 'dataset-builder', 'data', 'llm_pipeline', 'question_retrieval_index.json');
const KEYWORD_GLOSS_MAP_FILE = process.env.KEYWORD_GLOSS_MAP ||
  path.join(__dirname, '..', 'dataset-builder', 'data', 'llm_pipeline', 'keyword_gloss_map.json');
const INTENT_GLOSS_SETS_FILE = process.env.INTENT_GLOSS_SETS ||
  path.join(__dirname, '..', 'dataset-builder', 'data', 'llm_pipeline', 'intent_gloss_sets.json');
const OUT_OF_SCOPE_LABEL = {
  stage: '기타',
  subCategory: 'other',
  examples: ['진료와 관계없는 질문', '일상 대화', '시스템 사용법 질문'],
};

// ── 엑셀 로드 ──────────────────────────────────────────────────
function loadDataset(filePath) {
  const wb = XLSX.readFile(filePath);
  const ws = wb.Sheets['Dataset'];
  if (!ws) throw new Error('"Dataset" 시트를 찾을 수 없습니다.');
  const rows = XLSX.utils.sheet_to_json(ws, { defval: '' });
  return rows.filter(r => String(r['gloss_names'] || '').trim());
}

// ── 유사 행 검색: 질문 + 답변 모두 고려 ──────────────────────
function findSimilarRows(question, answer, rows, topN = 15) {
  const qWords = question.split(/\s+/).filter(w => w.length > 1);
  const aWords = answer.split(/\s+/).filter(w => w.length > 1);

  const scored = rows.map(r => {
    const rQ = String(r['의사 질문(개별)'] || '');
    const rA = String(r['환자 답변'] || '');
    const rK = String(r['대표 환자키워드'] || '');

    // 답변 유사도 가중치 2, 질문 유사도 가중치 1
    const qScore = qWords.filter(w => rQ.includes(w)).length;
    const aScore = aWords.filter(w => (rA + rK).includes(w)).length;
    return { r, score: qScore + aScore * 2 };
  });

  const matched = scored.filter(s => s.score > 0).sort((a, b) => b.score - a.score);
  if (matched.length > 0) return matched.slice(0, topN).map(s => s.r);

  // 매칭 없으면 세부분류 다양성 샘플
  const seen = new Set();
  return rows.filter(r => {
    if (seen.has(r['세부분류'])) return false;
    seen.add(r['세부분류']); return true;
  }).slice(0, topN);
}

function splitValues(value) {
  if (Array.isArray(value)) return value.flatMap(splitValues);
  return String(value || '')
    .split(/[,，、/|]+/)
    .map(v => v.trim())
    .filter(Boolean);
}

function normalizeToken(value) {
  return String(value || '').trim().toLowerCase().replace(/\s+/g, '');
}

function uniqueValues(values, keyFn = normalizeToken) {
  const seen = new Set();
  return values.filter(value => {
    const key = normalizeToken(keyFn(value));
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function loadQuestionAnswerPool() {
  if (questionAnswerPool) return questionAnswerPool;
  if (!fs.existsSync(QUESTION_ANSWER_POOL_FILE)) {
    throw new Error(`Question-answer pool not found: ${QUESTION_ANSWER_POOL_FILE}`);
  }
  const pool = JSON.parse(fs.readFileSync(QUESTION_ANSWER_POOL_FILE, 'utf8'));
  if (!Array.isArray(pool.labels) || !Array.isArray(pool.questions)) {
    throw new Error(`Invalid question-answer pool: ${QUESTION_ANSWER_POOL_FILE}`);
  }
  questionAnswerPool = pool;
  return questionAnswerPool;
}

function loadLlmGlossVectorDb() {
  if (llmGlossVectorDb) return llmGlossVectorDb;
  if (!fs.existsSync(LLM_GLOSS_VECTOR_DB_FILE)) {
    throw new Error(`LLM gloss vector DB not found: ${LLM_GLOSS_VECTOR_DB_FILE}`);
  }
  llmGlossVectorDb = JSON.parse(fs.readFileSync(LLM_GLOSS_VECTOR_DB_FILE, 'utf8'));
  return llmGlossVectorDb;
}

function loadQuestionRetrievalIndex() {
  if (questionRetrievalIndex) return questionRetrievalIndex;
  if (!fs.existsSync(QUESTION_RETRIEVAL_INDEX_FILE)) {
    throw new Error(`Question retrieval index not found: ${QUESTION_RETRIEVAL_INDEX_FILE}`);
  }
  const index = JSON.parse(fs.readFileSync(QUESTION_RETRIEVAL_INDEX_FILE, 'utf8'));
  if (!Array.isArray(index.documents) || !index.documents.every(doc => Array.isArray(doc.embedding))) {
    throw new Error(`Invalid question retrieval index: ${QUESTION_RETRIEVAL_INDEX_FILE}`);
  }
  questionRetrievalIndex = index;
  return questionRetrievalIndex;
}

function stageLabelsFromQaPool(pool) {
  const labels = (pool.labels || []).map(label => ({
    stage: label.stage || '',
    subCategory: label.subCategory || '',
    examples: label.questionExamples || [],
  }));
  if (!labels.some(label => isOutOfScopeLabel(label.stage, label.subCategory))) {
    labels.push(OUT_OF_SCOPE_LABEL);
  }
  return labels.sort((a, b) => a.subCategory.localeCompare(b.subCategory));
}

function isOutOfScopeLabel(stage, subCategory) {
  return normalizeToken(stage) === normalizeToken(OUT_OF_SCOPE_LABEL.stage) ||
    normalizeToken(subCategory) === normalizeToken(OUT_OF_SCOPE_LABEL.subCategory) ||
    normalizeToken(subCategory) === '기타';
}

function scoreLabelAgainstQuestion(question, label) {
  const queryTokens = textTokens(question);
  if (!queryTokens.length) return 0;
  const text = normalizeToken(`${label.stage} ${label.subCategory} ${(label.examples || []).join(' ')}`);
  let hits = 0;
  for (const token of queryTokens) {
    if (text.includes(normalizeToken(token))) hits += 1;
  }
  return hits / queryTokens.length;
}

function bestLabelForStage(question, labels, stage) {
  const stageKey = normalizeToken(stage);
  const candidates = labels.filter(label => normalizeToken(label.stage) === stageKey);
  if (!candidates.length) return null;
  return candidates
    .map(label => ({ label, score: scoreLabelAgainstQuestion(question, label) }))
    .sort((a, b) => b.score - a.score)[0].label;
}

function resolveStageLabel(parsed, labels, question = '') {
  const stage = String(parsed?.stage || '');
  const subCategory = String(parsed?.subCategory || '');
  const combined = normalizeToken(`${stage} ${subCategory}`);
  const exact = labels.find(label =>
    normalizeToken(label.stage) === normalizeToken(stage) &&
    normalizeToken(label.subCategory) === normalizeToken(subCategory)
  );
  if (exact) return { ...exact, repaired: false };
  const bySub = labels.find(label => normalizeToken(label.subCategory) === normalizeToken(subCategory));
  if (bySub) return { ...bySub, repaired: true };
  // Some models echo the JSON template's field descriptions as values, which shifts
  // the hierarchy one level and leaves the real stage sitting in subCategory.
  const slippedStage = labels.find(label => normalizeToken(label.stage) === normalizeToken(subCategory));
  if (slippedStage) {
    return { ...(bestLabelForStage(question, labels, subCategory) || slippedStage), repaired: true };
  }
  const included = labels.find(label =>
    combined.includes(normalizeToken(label.subCategory)) ||
    combined.includes(normalizeToken(`${label.stage}${label.subCategory}`))
  );
  if (included) return { ...included, repaired: true };
  const byStage = labels.find(label => normalizeToken(label.stage) === normalizeToken(stage));
  if (byStage) {
    return { ...(bestLabelForStage(question, labels, stage) || byStage), repaired: true };
  }
  return { stage, subCategory, repaired: false, unresolved: true };
}

function buildLlmClassifyPrompt(question, labels) {
  const labelText = labels.map(label => {
    const examples = (label.examples || []).join(' | ');
    return `- ${label.stage} / ${label.subCategory}${examples ? ` / 예시: ${examples}` : ''}`;
  }).join('\n');
  const system = [
    '당신은 통증의학과 초진 의사 질문을 의료문진단계와 세부분류로 분류하는 전문가입니다.',
    '반드시 제공된 세부분류 후보 중 하나만 선택하세요.',
    '통증의학과 진료 문진과 관계없는 질문만 기타 / other를 선택하세요.',
    '체중 변화, 식욕 부진, 발열, 오한, 대소변 조절, 마비, 암 이력 같은 red flag 질문은 진료 문진 질문이므로 기타가 아닙니다.',
    '술, 담배, 직업, 운동 같은 생활습관 질문도 진료 문진 질문이므로 기타가 아닙니다.',
    '환자 답변, 키워드, 글로스는 생성하지 마세요.',
    '반드시 JSON 객체 하나로만 응답하세요.',
  ].join('\n');
  const sample = labels.find(label => !isOutOfScopeLabel(label.stage, label.subCategory)) || labels[0];
  const user = [
    '[세부분류 후보]',
    labelText,
    '',
    '[출력 형식]',
    '{ "stage": "<후보 줄 왼쪽 한글 단계명>", "subCategory": "<후보 줄 오른쪽 영문 세부분류>", "reason": "짧은 근거" }',
    '',
    '[출력 예시]',
    `{ "stage": "${sample?.stage || '진료개시'}", "subCategory": "${sample?.subCategory || 'chief_complaint'}", "reason": "짧은 근거" }`,
    '',
    '주의:',
    '- stage에는 후보 줄의 왼쪽 한글 단계명만 그대로 쓰세요.',
    '- subCategory에는 후보 줄의 오른쪽 영문 세부분류만 그대로 쓰세요.',
    '- "의료문진단계", "세부분류" 같은 형식 설명어를 값으로 쓰지 마세요.',
    '- "진료개시 / chief_complaint" 처럼 합친 문자열도 금지입니다.',
    '',
    '[입력 의사 질문]',
    question,
  ].join('\n');
  return { system, user };
}

function textTokens(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}_]+/gu, ' ')
    .split(/\s+/)
    .map(v => v.trim())
    .filter(v => v.length >= 2);
}

function scorePoolQuestion(question, item) {
  const queryTokens = textTokens(question);
  const rowText = `${item.question || ''} ${item.stage || ''} ${item.subCategory || ''}`;
  const normalized = normalizeToken(rowText);
  if (!queryTokens.length) return 0;
  let hits = 0;
  for (const token of queryTokens) {
    if (normalized.includes(normalizeToken(token))) hits += 1;
  }
  return hits / queryTokens.length;
}

function poolQuestionsForCategory(pool, stage, subCategory) {
  const stageKey = normalizeToken(stage);
  const subKey = normalizeToken(subCategory);
  return (pool.questions || []).filter(item =>
    normalizeToken(item.stage) === stageKey &&
    normalizeToken(item.subCategory) === subKey
  );
}

function knownAnswerPool(question, pool, stage, subCategory, minScore = 0.15) {
  const categoryQuestions = poolQuestionsForCategory(pool, stage, subCategory);
  const scored = categoryQuestions
    .map(item => ({ item, score: scorePoolQuestion(question, item) }))
    .sort((a, b) => b.score - a.score);
  const best = scored[0]?.score || 0;
  const sourceQuestions = best >= minScore
    ? scored.filter(item => item.score >= minScore).map(item => item.item)
    : [];
  return {
    known: best >= minScore,
    bestQuestionScore: Number(best.toFixed(4)),
    questions: sourceQuestions,
    answers: uniqueValues(sourceQuestions.flatMap(item => item.answerPool || []).filter(Boolean)),
  };
}

function ruleClassifyQuestion(question) {
  const text = normalizeToken(question);
  const hasVisitVerb = /(오셨|왔|방문|내원)/.test(text);
  const hasPainPlace = /(어디|부위|곳|쪽).*(아프|불편|통증)|(?:아프|불편|통증).*(어디|부위|곳|쪽)/.test(text);

  if (isClearlyOutOfScopeQuestion(question)) {
    return {
      ...OUT_OF_SCOPE_LABEL,
      reason: '통증의학과 진료 문진과 관계없는 질문으로 판단했습니다.',
      classifySource: 'rule_out_of_scope',
      outOfScope: true,
    };
  }
  if (isAllergyQuestion(question)) {
    return {
      stage: '알레르기 및 약물반응',
      subCategory: 'adverse_reaction',
      reason: '알레르기 또는 약물반응 여부를 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  if (/(체중|몸무게|살이|식욕|입맛|발열|열이나|열이|오한|식은땀|야간발한|소변|대변|대소변|회음부|마비|힘이빠|힘빠|암|종양|감염|고열|redflag)/.test(text)) {
    return {
      stage: '통증 동반증상',
      subCategory: 'red_flag',
      reason: '체중 변화, 식욕 부진, 발열, 마비 등 red flag 증상을 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  if (/(술|담배|흡연|음주|소주|맥주|금연|생활습관|태우|피우|마시)/.test(text)) {
    return {
      stage: '생활습관',
      subCategory: 'lifestyle',
      reason: '음주 또는 흡연 습관을 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  if (/(약|진통제|처방|처방전).*(드셨|드시|먹|복용|남|다드|다먹|챙겨)|(?:드셨|드시|먹|복용|남|다드|다먹|챙겨).*(약|진통제|처방)/.test(text)) {
    return {
      stage: '약물복용',
      subCategory: 'medication_use',
      reason: '약 복용 여부 또는 복용 이행을 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  if (/(이전|예전|과거|전에|비슷한증상|같은증상)/.test(text)) {
    return {
      stage: '과거질환',
      subCategory: 'history',
      reason: '경험/과거 여부를 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  if (hasVisitVerb && /(어디|무엇|뭐|증상|아프|불편|통증)/.test(text)) {
    return {
      stage: '진료개시',
      subCategory: 'chief_complaint',
      reason: '내원 이유 또는 주호소를 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  if (hasPainPlace) {
    return {
      stage: '통증 부위',
      subCategory: 'location',
      reason: '통증 위치를 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  if (/(언제부터|몇일|며칠|얼마나오래|시작)/.test(text)) {
    return {
      stage: '통증 발생시점',
      subCategory: 'onset',
      reason: '통증 시작 시점을 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  if (/(몇점|점수|정도|얼마나아프|많이아프|심하)/.test(text)) {
    return {
      stage: '통증강도',
      subCategory: 'severity',
      reason: '통증 강도를 묻는 대표 패턴입니다.',
      classifySource: 'rule_fast_match',
    };
  }
  return null;
}

function isClearlyOutOfScopeQuestion(question) {
  const text = normalizeToken(question);
  if (!text) return false;
  const medicalHints = /(아프|통증|불편|증상|치료|병원|진료|의사|환자|약|주사|수술|검사|엑스레이|mri|ct|디스크|협착|저리|마비|붓|열감|잠|술|담배|성함|이름|입원|알레르기|부작용|고혈압|당뇨|간염|결핵|허리|목|어깨|무릎|팔|다리|손|발|치아|턱|체중|몸무게|식욕|입맛|발열|오한|식은땀|소변|대변|대소변|회음부|암|종양|감염|고열)/.test(text);
  if (medicalHints) return false;
  return /(날씨|주식|코딩|프로그래밍|맛집|점심|저녁메뉴|여행|영화|음악|노래|뉴스|스포츠|축구|야구|게임|번역|계산|수학|농담|자기소개|챗봇|인공지능|ai|llm|컴퓨터|핸드폰|자동차|부동산|환율|비트코인)/.test(text);
}

function fastClassifyFromQaPool(question, pool, minScore = 0.75) {
  const query = normalizeToken(question);
  const scored = (pool.questions || [])
    .map(item => {
      const itemQuestion = normalizeToken(item.question || '');
      const exact = itemQuestion && itemQuestion === query;
      const contained = itemQuestion && (itemQuestion.includes(query) || query.includes(itemQuestion));
      const score = exact ? 1 : contained ? 0.95 : scorePoolQuestion(question, item);
      return { item, score };
    })
    .filter(row => row.score > 0)
    .sort((a, b) => b.score - a.score);

  const best = scored[0];
  if (!best || best.score < minScore) return null;
  const second = scored.find(row =>
    normalizeToken(row.item.subCategory) !== normalizeToken(best.item.subCategory) ||
    normalizeToken(row.item.stage) !== normalizeToken(best.item.stage)
  );
  const margin = best.score - (second?.score || 0);
  if (best.score < 0.95 && margin < 0.2) return null;

  return {
    stage: best.item.stage || '',
    subCategory: best.item.subCategory || '',
    score: Number(best.score.toFixed(4)),
    margin: Number(margin.toFixed(4)),
    matchedQuestion: best.item.question || '',
  };
}

function validateConfirmedLabel(question, qaPool, stage, subCategory) {
  const rule = ruleClassifyQuestion(question);
  const fast = fastClassifyFromQaPool(question, qaPool, 0.8);
  const agrees = label => Boolean(label) &&
    normalizeToken(label.stage) === normalizeToken(stage) &&
    normalizeToken(label.subCategory) === normalizeToken(subCategory);

  const rejectByRule = () => ({
    ok: false,
    reason: `질문은 "${rule.stage} / ${rule.subCategory}"에 더 가깝습니다.`,
    suggestedStage: rule.stage,
    suggestedSubCategory: rule.subCategory,
    source: rule.classifySource || 'rule_fast_match',
  });
  const rejectByPool = () => ({
    ok: false,
    reason: `기존 질문 pool은 "${fast.stage} / ${fast.subCategory}"를 더 강하게 지지합니다. 매칭 질문: "${fast.matchedQuestion}"`,
    suggestedStage: fast.stage,
    suggestedSubCategory: fast.subCategory,
    source: 'qa_pool_fast_match',
    score: fast.score,
    margin: fast.margin,
  });

  // 사실상 같은 질문이 이미 pool에 있으면 그 라벨이 정답이다. 규칙 분류기가
  // 다른 의견을 내도 여기서는 pool을 따른다.
  if (fast && Number(fast.score) >= 0.95) {
    return agrees(fast) ? { ok: true } : rejectByPool();
  }

  // 두 검증기가 서로 다른 답을 낼 때 둘 다 만족하는 값을 요구하면 어떤 라벨도
  // 통과할 수 없다. 한쪽이라도 지지하면 통과시킨다.
  const ruleActive = Boolean(rule) && !rule.outOfScope;
  if ((ruleActive && agrees(rule)) || agrees(fast)) return { ok: true };

  if (ruleActive) return rejectByRule();
  if (fast) return rejectByPool();
  return { ok: true };
}

function medicationUseAnswerPool(question, glossVectorDb) {
  const text = normalizeToken(question);
  if (!/(약|진통제|처방|처방전).*(드셨|드시|먹|복용|남|다드|다먹|챙겨)|(?:드셨|드시|먹|복용|남|다드|다먹|챙겨).*(약|진통제|처방)/.test(text)) {
    return [];
  }

  const drugHint = docHint(glossVectorDb, 11990) || '약_11990';
  const eatHint = docHint(glossVectorDb, 10419) || '먹다,식사_10419';
  const existHint = docHint(glossVectorDb, 8966) || '있다_8966';
  const noneHint = docHint(glossVectorDb, 2722) || '(존재가)없다_2722';
  const unableHint = docHint(glossVectorDb, 10440) || '못하다,할 수 없다_10440';
  const allHint = docHint(glossVectorDb, 7497) || '모두,온통,전부,전체,제반,모든,온,다,모조리,몽땅,죄다_7497';

  return [
    {
      answer: '네, 지난번에 받은 약은 다 먹었어요.',
      keywords: ['약', '먹다', '다'],
      glossHints: [drugHint, eatHint, allHint],
      source: 'rule_medication_use_completion',
    },
    {
      answer: '아니요, 약이 조금 남았어요.',
      keywords: ['약', '남다', '있다'],
      glossHints: [drugHint, existHint],
      source: 'rule_medication_use_completion',
    },
    {
      answer: '중간에 약을 못 먹었어요.',
      keywords: ['약', '먹다', '못하다'],
      glossHints: [drugHint, eatHint, unableHint],
      source: 'rule_medication_use_completion',
    },
    {
      answer: '약은 따로 안 먹었어요.',
      keywords: ['약', '없다'],
      glossHints: [drugHint, noneHint],
      source: 'rule_medication_use_completion',
    },
  ];
}

function docByGlossPattern(glossVectorDb, pattern) {
  const re = pattern instanceof RegExp ? pattern : new RegExp(String(pattern));
  return (glossVectorDb.documents || []).find(doc =>
    re.test(`${doc.gloss || ''} ${doc.name || ''} ${(doc.synonyms || []).join(' ')}`)
  );
}

function glossHintFromDoc(doc) {
  if (!doc) return null;
  return `${doc.gloss || doc.name}_${Number(doc.glossIndex ?? doc.origin)}`;
}

function allergyDisplayName(doc) {
  const idx = Number(doc?.glossIndex ?? doc?.origin);
  if (idx === 10952) return '계란';
  if (idx === 11990) return '약';
  if (idx === 9525) return '주사';
  if (idx === 6765) return '가려운';
  return String(doc?.gloss || doc?.name || '').split(',')[0].trim();
}

function isAllergyQuestion(question, stage = '', subCategory = '') {
  const text = normalizeToken(`${question} ${stage} ${subCategory}`);
  return /(알레르|부작용|두드러기|가려움|가렵|숨이차|숨차|호흡곤란|약물반응|adversereaction)/.test(text);
}

function isRedFlagQuestion(question, stage = '', subCategory = '') {
  const text = normalizeToken(`${question} ${stage} ${subCategory}`);
  return /(redflag|체중|몸무게|살이|식욕|입맛|발열|열이나|열이|오한|식은땀|야간발한|소변|대변|대소변|회음부|마비|힘이빠|힘빠|암|종양|감염|고열)/.test(text);
}

function allergyAnswerPool(question, glossVectorDb) {
  if (!isAllergyQuestion(question)) return [];

  const triggerDocs = [
    docByGlossPattern(glossVectorDb, /알,계란,달걀|계란|달걀/),
    docByGlossPattern(glossVectorDb, /(^|\s)약(\s|$)/),
    docByGlossPattern(glossVectorDb, /주사|접종/),
  ].filter(Boolean);
  const symptomDocs = [
    docByGlossPattern(glossVectorDb, /가렵다|긁다/),
    docByGlossPattern(glossVectorDb, /호흡|숨쉬다|숨/),
  ].filter(Boolean);
  const existHint = docHint(glossVectorDb, 8966) || '있다_8966';
  const noneHint = docHint(glossVectorDb, 2722) || '(존재가)없다_2722';

  const answers = triggerDocs.map((doc, index) => ({
    answer: `${allergyDisplayName(doc)} 알레르기가 있어요.`,
    keywords: [allergyDisplayName(doc), '알레르기', '있다'],
    glossHints: [glossHintFromDoc(doc), existHint].filter(Boolean),
    source: 'rule_allergy_domain_completion',
    displayPriority: index + 1,
  }));

  if (symptomDocs.length) {
    const first = symptomDocs[0];
    answers.push({
      answer: `${allergyDisplayName(first)} 증상이 있었어요.`,
      keywords: [allergyDisplayName(first), '알레르기', '있다'],
      glossHints: [glossHintFromDoc(first), existHint].filter(Boolean),
      source: 'rule_allergy_domain_completion',
      displayPriority: 40,
    });
  }

  answers.push({
    answer: '알레르기는 없어요.',
    keywords: ['알레르기', '없다'],
    glossHints: [noneHint],
    source: 'rule_allergy_domain_completion',
    displayPriority: 90,
  });

  return uniqueValues(answers, item => item.answer);
}

function redFlagAnswerPool(question, glossVectorDb, stage = '', subCategory = '') {
  if (!isRedFlagQuestion(question, stage, subCategory)) return [];
  const text = normalizeToken(question);
  const recentHint = docHint(glossVectorDb, 3261) || '요즘,요사이,요새,요즈음,최근_3261';
  const bodyHint = docHint(glossVectorDb, 5655) || '몸,동체,몸뚱이,몸체,신체_5655';
  const changeHint = docHint(glossVectorDb, 5504) || '변하다,변질,변화,변환,돌아서다_5504';
  const decreaseHint = docHint(glossVectorDb, 6001) || '줄이다,감소,절감,축소,줄다_6001';
  const existHint = docHint(glossVectorDb, 8966) || '있다_8966';
  const noneHint = docHint(glossVectorDb, 2722) || '(존재가)없다_2722';
  const urineHint = docHint(glossVectorDb, 651) || '방뇨,소변,오줌_651';
  const stoolHint = docHint(glossVectorDb, 11053) || '똥,대변_11053';
  const paralysisHint = docHint(glossVectorDb, 6275) || '마비_6275';

  if (/(소변|대변|대소변)/.test(text)) {
    return [
      {
        answer: '소변 조절이 어려워졌어요.',
        keywords: ['소변', '어렵다', '있다'],
        glossHints: [urineHint, existHint],
        source: 'rule_red_flag_completion',
        displayPriority: 1,
      },
      {
        answer: '대변 조절이 어려워졌어요.',
        keywords: ['대변', '어렵다', '있다'],
        glossHints: [stoolHint, existHint],
        source: 'rule_red_flag_completion',
        displayPriority: 2,
      },
      {
        answer: '소변이나 대변 문제는 없어요.',
        keywords: ['소변', '대변', '없다'],
        glossHints: [urineHint, stoolHint, noneHint],
        source: 'rule_red_flag_completion',
        displayPriority: 90,
      },
    ];
  }

  if (/(마비|힘이빠|힘빠)/.test(text)) {
    return [
      {
        answer: '다리에 힘이 빠졌어요.',
        keywords: ['다리', '힘', '있다'],
        glossHints: [docHint(glossVectorDb, 11238) || '기운,힘_11238', existHint],
        source: 'rule_red_flag_completion',
        displayPriority: 1,
      },
      {
        answer: '마비 증상이 있어요.',
        keywords: ['마비', '있다'],
        glossHints: [paralysisHint, existHint],
        source: 'rule_red_flag_completion',
        displayPriority: 2,
      },
      {
        answer: '마비나 힘 빠짐은 없어요.',
        keywords: ['마비', '없다'],
        glossHints: [paralysisHint, noneHint],
        source: 'rule_red_flag_completion',
        displayPriority: 90,
      },
    ];
  }

  return [
    {
      answer: '최근 몸 상태가 변했어요.',
      keywords: ['최근', '몸', '변화'],
      glossHints: [recentHint, bodyHint, changeHint],
      source: 'rule_red_flag_completion',
      displayPriority: 1,
    },
    {
      answer: '최근 몸 상태가 줄어든 느낌이에요.',
      keywords: ['최근', '몸', '줄다'],
      glossHints: [recentHint, bodyHint, decreaseHint],
      source: 'rule_red_flag_completion',
      displayPriority: 2,
    },
    {
      answer: '최근 몸 상태 변화는 없어요.',
      keywords: ['최근', '몸', '없다'],
      glossHints: [recentHint, bodyHint, noneHint],
      source: 'rule_red_flag_completion',
      displayPriority: 90,
    },
  ];
}

function categoryFallbackAnswerPool(pool, stage, subCategory, limit = 8) {
  const questions = poolQuestionsForCategory(pool, stage, subCategory);
  return uniqueValues(
    questions.flatMap(item => item.answerPool || [])
      .filter(Boolean)
      .map(answer => ({
        answer,
        keywords: [],
        glossHints: [],
        source: 'category_fallback_answer_pool',
      })),
    item => item.answer
  ).slice(0, limit);
}

function ruleAnswerPool(question, glossVectorDb) {
  const allergyAnswers = allergyAnswerPool(question, glossVectorDb);
  if (allergyAnswers.length) return allergyAnswers;
  const redFlagAnswers = redFlagAnswerPool(question, glossVectorDb);
  if (redFlagAnswers.length) return redFlagAnswers;
  const medicationAnswers = medicationUseAnswerPool(question, glossVectorDb);
  if (medicationAnswers.length) return medicationAnswers;
  if (!questionNeedsPolarityCoverage(question)) return [];
  return ensurePolarityAnswerCoverage(question, [], glossVectorDb);
}

function prioritizeAnswerPool(question, answers, glossVectorDb, stage = '', subCategory = '') {
  const list = uniqueValues(answers || [], item => typeof item === 'string' ? item : item.answer);
  let merged = list;
  if (isAllergyQuestion(question, stage, subCategory)) {
    merged = uniqueValues([...allergyAnswerPool(question, glossVectorDb), ...merged], item => typeof item === 'string' ? item : item.answer);
  }
  if (isRedFlagQuestion(question, stage, subCategory)) {
    merged = uniqueValues([...redFlagAnswerPool(question, glossVectorDb, stage, subCategory), ...merged], item => typeof item === 'string' ? item : item.answer);
  }

  return merged
    .map((item, index) => ({ item, index, priority: answerDisplayPriority(question, item, stage, subCategory) }))
    .sort((a, b) => a.priority - b.priority || a.index - b.index)
    .map(row => row.item);
}

function answerText(answer) {
  return typeof answer === 'string' ? answer : String(answer?.answer || '');
}

function answerDisplayPriority(question, answer, stage = '', subCategory = '') {
  if (typeof answer !== 'string' && Number.isFinite(Number(answer.displayPriority))) {
    return Number(answer.displayPriority);
  }

  const q = normalizeToken(`${question} ${stage} ${subCategory}`);
  const text = normalizeToken(answerText(answer));
  let score = 50;

  const asksQuantity = /(어느정도|얼마|몇|횟수|양|빈도|하루|일주일|한달|매일|갑|병|잔|개비|생활습관|lifestyle|occupation)/.test(q);
  const asksSpecific = /(어디|무엇|무슨|어떤|언제|부위|이름|종류|검사|치료|수술|약|알레르)/.test(q);
  const asksPresence = /(있|없|하세|하시|나요|습니까|적|경험|유무)/.test(q) && !asksQuantity && !asksSpecific;

  if (asksQuantity) {
    if (/[0-9０-９]/.test(text)) score -= 26;
    if (/(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|반|두세|서너)/.test(text)) score -= 16;
    if (/(갑|병|잔|개비|번|회|일|주|달|개월|년|시간|분|매일|하루|일주일|한달|정도)/.test(text)) score -= 16;
    if (/(소주|맥주|담배|음주|흡연|술)/.test(text)) score -= 8;
    if (/(안|않|없|끊|금연|거의안|둘다안)/.test(text)) score += 22;
    if (/(가끔|조금|별로|거의)/.test(text) && !/[0-9０-９]|한달|일주일|하루|갑|병|잔|번/.test(text)) score += 10;
  } else if (asksSpecific) {
    if (/[0-9０-９]|(한|두|세|네|반|작년|어제|오늘|개월|년|달|주|일|MRI|CT|엑스레이)/i.test(text)) score -= 12;
    if (/(허리|목|무릎|어깨|발목|손목|디스크|주사|약|계란|치료|수술|입원|검사|물리치료|도수치료|MRI|CT|엑스레이)/i.test(text)) score -= 10;
    if (/^(없어요|아니요|안해요|없습니다|아니요없어요)$/.test(text)) score += 18;
  } else if (asksPresence) {
    if (/(있|했|받|먹|피|마시|아프|증상)/.test(text)) score -= 6;
    if (/(없|안|아니|않)/.test(text)) score += 2;
  }

  if (/(잘모르|기억안|모르겠|애매)/.test(text)) score += 18;
  if (text.length < 4) score += 12;
  return score;
}

function glossaryByIndex(glossVectorDb) {
  return new Map((glossVectorDb.documents || []).map(doc => [Number(doc.glossIndex ?? doc.origin), doc]));
}

function scoreGlossesFromQaQuestions(questions, glossVectorDb) {
  const allowed = glossaryByIndex(glossVectorDb);
  const counts = new Map();
  const add = (origin, source, weight) => {
    const idx = Number(origin);
    if (!Number.isFinite(idx) || !allowed.has(idx)) return;
    const prev = counts.get(idx) || { count: 0, sources: new Set() };
    prev.count += weight;
    prev.sources.add(source);
    counts.set(idx, prev);
  };
  for (const item of questions) {
    splitValues(item.glossOrigins).forEach(origin => add(origin, 'answer_pool', 1));
  }
  const max = Math.max(1, ...[...counts.values()].map(item => item.count));
  return [...counts.entries()]
    .map(([idx, item]) => {
      const doc = allowed.get(idx);
      return {
        gloss: doc.gloss || doc.name,
        name: doc.gloss || doc.name,
        glossIndex: idx,
        origin: idx,
        score: Number((item.count / max).toFixed(6)),
        source: [...item.sources].join('+'),
        count: item.count,
      };
    })
    .sort((a, b) => b.score - a.score || a.gloss.localeCompare(b.gloss));
}

function buildUnknownAnswerPrompt(question, stage, subCategory, answerCount, glossPool) {
  const system = [
    '당신은 통증의학과 초진 문진에서 환자가 할 법한 답변 pool을 예측하는 전문가입니다.',
    '환자 답변은 짧은 1인칭 문장으로만 작성하세요.',
    '진단, 검사 결과, 치료법을 임의로 만들지 마세요.',
    '가능하면 제공된 gloss pool의 단어를 사용하세요.',
    '약에 대한 답변은 "마시다"가 아니라 "먹다" 또는 "복용하다"로 표현하세요.',
    '알레르기 유무 질문은 "있어요/없어요"만 쓰지 말고, 제공된 gloss pool에 있는 알레르기 원인 후보를 먼저 답변으로 나열하세요.',
    '있다/없다, 예/아니오, 경험 여부를 묻는 질문은 긍정 답변과 부정 답변을 모두 포함하세요.',
    '반드시 JSON 객체 하나로만 응답하세요.',
  ].join('\n');
  const user = [
    `의사 질문: ${question}`,
    `확정 의료문진단계: ${stage}`,
    `확정 세부분류: ${subCategory}`,
    `필요 답변 후보 수: ${answerCount}`,
    '',
    '사용 가능한 gloss pool:',
    glossPool.slice(0, 220).map(item => `${item.gloss}_${item.glossIndex}`).join(', '),
    '',
    '[출력 JSON]',
    `{
  "answers": [
    {
      "answer": "환자 답변",
      "keywords": ["핵심 키워드"],
      "glossHints": ["사용한 gloss 표제어"]
    }
  ]
}`,
  ].join('\n');
  return { system, user };
}

function cleanPatientAnswerText(text) {
  return String(text || '')
    .replace(/약을\s*마/g, '약을 먹')
    .replace(/약은\s*마/g, '약은 먹')
    .replace(/약도\s*마/g, '약도 먹')
    .replace(/약\s*마/g, '약 먹');
}

function normalizeGeneratedAnswers(parsed) {
  return uniqueValues((parsed?.answers || [])
    .map(item => ({
      answer: cleanPatientAnswerText(item.answer).trim(),
      keywords: uniqueValues(splitValues(item.keywords || [])),
      glossHints: uniqueValues(splitValues(item.glossHints || [])),
    }))
    .filter(item => item.answer), item => item.answer);
}

function docHint(glossVectorDb, origin) {
  const doc = glossaryByIndex(glossVectorDb).get(Number(origin));
  if (!doc) return null;
  return `${doc.gloss || doc.name}_${Number(doc.glossIndex ?? doc.origin)}`;
}

function questionNeedsPolarityCoverage(question) {
  const text = normalizeToken(question);
  if (/(해주세요|설명|어떻게|무엇|무슨|어디|언제|얼마나|몇)/.test(text)) return false;
  return /(있었|있나|있습|없었|없나|없습|합니까|됩니까|인가|입니까|하셨나요|되나요|비슷한증상|경험.*나요)/.test(text);
}

function hasPositivePolarity(answer) {
  const text = normalizeToken([
    answer.answer,
    ...(answer.keywords || []),
    ...(answer.glossHints || []),
  ].join(' '));
  return /(있었|있어|있다|있음|경험|예전|과거)/.test(text) && !hasNegativePolarity(answer);
}

function hasNegativePolarity(answer) {
  const text = normalizeToken([
    answer.answer,
    ...(answer.keywords || []),
    ...(answer.glossHints || []),
  ].join(' '));
  return /(없었|없어|없다|없음|아니|안했|하지않|못했)/.test(text);
}

function ensurePolarityAnswerCoverage(question, answers, glossVectorDb) {
  if (!questionNeedsPolarityCoverage(question)) return answers;

  const result = [...answers];
  const pastHint = docHint(glossVectorDb, 19582) || '과거_19582';
  const positiveHint = docHint(glossVectorDb, 8966) || '있다_8966';
  const negativeHint = docHint(glossVectorDb, 2722) || '(존재가)없다_2722';

  if (!result.some(hasPositivePolarity)) {
    result.push({
      answer: '예전에 비슷한 증상이 있었어요.',
      keywords: ['과거', '있다'],
      glossHints: [pastHint, positiveHint],
      source: 'rule_polarity_completion',
    });
  }

  if (!result.some(hasNegativePolarity)) {
    result.push({
      answer: '예전에 비슷한 증상은 없었어요.',
      keywords: ['과거', '없다'],
      glossHints: [pastHint, negativeHint],
      source: 'rule_polarity_completion',
    });
  }

  return uniqueValues(result, item => item.answer);
}

function scoreGlossesFromGeneratedAnswers(answers, glossVectorDb) {
  const docs = glossVectorDb.documents || [];
  const candidates = new Map();
  const answerText = answers.flatMap(answer => [answer.answer, ...(answer.keywords || []), ...(answer.glossHints || [])]).join(' ');
  const tokens = textTokens(answerText).map(normalizeToken);
  for (const doc of docs) {
    const fields = [doc.gloss, doc.name, ...(doc.synonyms || [])].map(normalizeToken);
    let score = 0;
    for (const token of tokens) {
      if (!token) continue;
      if (fields.some(field => field === token)) score += 3;
      else if (fields.some(field => field.length >= 2 && token.length >= 2 && (field.includes(token) || token.includes(field)))) score += 1;
    }
    if (score > 0) {
      candidates.set(Number(doc.glossIndex ?? doc.origin), {
        gloss: doc.gloss || doc.name,
        name: doc.gloss || doc.name,
        glossIndex: Number(doc.glossIndex ?? doc.origin),
        origin: Number(doc.glossIndex ?? doc.origin),
        score,
        source: 'llm_answer_pool',
      });
    }
  }
  const max = Math.max(1, ...[...candidates.values()].map(item => item.score));
  return [...candidates.values()]
    .map(item => ({ ...item, score: Number((item.score / max).toFixed(6)) }))
    .sort((a, b) => b.score - a.score || a.gloss.localeCompare(b.gloss));
}

function prioritizeGlossCandidates(question, candidates, stage = '', subCategory = '') {
  if (!isAllergyQuestion(question, stage, subCategory)) return candidates;
  const priority = gloss => {
    const text = normalizeToken(gloss);
    if (/대략|대강|대충|약간|약소|약하다|약속/.test(text)) return 9;
    if (/계란|달걀|^약$|주사|접종/.test(text)) return 0;
    if (/가렵|호흡|숨/.test(text)) return 1;
    if (/있다|없다|아니/.test(text)) return 2;
    return 3;
  };
  return [...candidates].sort((a, b) =>
    priority(a.gloss || a.name) - priority(b.gloss || b.name) ||
    b.score - a.score ||
    String(a.gloss || a.name).localeCompare(String(b.gloss || b.name))
  );
}

// ── 답변별 핵심 표제어 1개 선택 ────────────────────────────────
// 답변 문장에 등장하는 모든 글로스를 나열하면 후속 모듈이 모든 조합을 따져야 한다.
// "무릎이 아파서 왔어요"에서 부위/아프다/오다가 아니라 부위 하나만 남기는 것이 목표다.
// 형태소 분석 결과는 build_keyword_gloss_map.py가 미리 계산해 둔다.

const JOSA_PATTERN = /(이라고|이라는|으로는|에서는|에게서|한테서|이랑은|까지는|부터는|으로도|에서도|이라도|이나마|이야말로|으로|에서|에게|한테|께서|이랑|까지|부터|처럼|만큼|보다|밖에|조차|마저|대로|같이|이라|라고|라는|이며|이나|이든|이고|와는|과는|도는|은|는|이|가|을|를|에|의|와|과|도|만|랑|께|나|든|고|요)$/;

function stripJosa(token) {
  const text = String(token || '');
  if (text.length < 2) return text;
  const stripped = text.replace(JOSA_PATTERN, '');
  return stripped.length >= 1 ? stripped : text;
}

function loadKeywordGlossMap() {
  if (keywordGlossMap) return keywordGlossMap;
  if (!fs.existsSync(KEYWORD_GLOSS_MAP_FILE)) {
    throw new Error(`Keyword gloss map not found: ${KEYWORD_GLOSS_MAP_FILE}`);
  }
  keywordGlossMap = JSON.parse(fs.readFileSync(KEYWORD_GLOSS_MAP_FILE, 'utf8'));
  return keywordGlossMap;
}

function glossPriority(map, glossIndex) {
  const meta = map.glossMeta?.[String(glossIndex)];
  return Number.isFinite(Number(meta?.priority)) ? Number(meta.priority) : Number(map.defaultPriority ?? 7);
}

function surfaceCandidates(surface) {
  const base = normalizeToken(surface);
  if (!base) return [];
  const forms = [base, stripJosa(base)];
  const digits = base.match(/^(\d+)/);
  if (digits) forms.push(digits[1]);
  return uniqueValues(forms, value => value);
}

// 후보는 655개 글로스 전체다. 검색된 질문의 allowedGloss로 좁히지 않는다.
function pickCoreGloss(text, map) {
  const tokens = String(text || '')
    .split(/[^\p{L}\p{N}]+/u)
    .map(token => token.trim())
    .filter(Boolean);
  const scored = [];
  tokens.forEach((token, order) => {
    for (const form of surfaceCandidates(token)) {
      for (const glossIndex of map.surfaceToGloss?.[form] || []) {
        scored.push({ glossIndex, order, matchedLemma: form, priority: glossPriority(map, glossIndex) });
      }
    }
  });
  if (!scored.length) return null;
  scored.sort((a, b) => a.priority - b.priority || a.order - b.order || a.glossIndex - b.glossIndex);
  return scored[0];
}

function resolveKeywordSurface(surface, map) {
  const key = normalizeToken(surface);
  if (!key) return null;
  // 기존 답변의 keyword는 형태소 분석까지 끝난 결과가 표에 들어 있다.
  if (Object.prototype.hasOwnProperty.call(map.keywordToGloss || {}, key)) {
    const hit = map.keywordToGloss[key];
    if (hit) return { ...hit, priority: glossPriority(map, hit.glossIndex) };
    return null;
  }
  return pickCoreGloss(surface, map);
}

function selectAnswerKeyword(answer, map, glossVectorDb) {
  const glossary = glossaryByIndex(glossVectorDb);
  const finalize = (surface, hit, matchType) => {
    const doc = glossary.get(Number(hit.glossIndex));
    if (!doc) return null;
    return {
      keyword: String(surface || '').trim(),
      matchedLemma: hit.matchedLemma,
      matchType,
      gloss: doc.gloss || doc.name,
      glossIndex: Number(hit.glossIndex),
      category: doc.category || '',
      priority: Number.isFinite(Number(hit.priority)) ? Number(hit.priority) : glossPriority(map, hit.glossIndex),
    };
  };

  if (typeof answer !== 'string') {
    // 원본 keyword 주석이 있으면 그것만 근거로 삼는다. 주석이 글로스로 표현되지
    // 않을 때 문장 전체로 되돌아가면 '무릎이' 대신 '병원'이 뽑혀, 키워드가 아닌
    // 값이 출력된다. 표현할 글로스가 없으면 없다고 보고하는 편이 맞다.
    if (answer?.keyword) {
      const hit = resolveKeywordSurface(answer.keyword, map);
      return hit ? finalize(answer.keyword, hit, 'annotated_keyword') : null;
    }
    for (const item of answer?.keywords || []) {
      const hit = resolveKeywordSurface(item, map);
      if (hit) return finalize(item, hit, 'llm_keyword');
    }
  }

  // 주석이 아예 없는 답변(LLM 생성분)만 문장에서 핵심어를 추정한다.
  const hit = pickCoreGloss(answerText(answer), map);
  return hit ? finalize(hit.matchedLemma, hit, 'answer_text') : null;
}

function buildKeywordCandidates(answers, glossVectorDb) {
  const map = loadKeywordGlossMap();
  const byGloss = new Map();
  const unresolved = [];

  for (const answer of answers || []) {
    const picked = selectAnswerKeyword(answer, map, glossVectorDb);
    if (!picked) {
      unresolved.push({ answer: answerText(answer), keyword: typeof answer === 'string' ? '' : (answer?.keyword || '') });
      continue;
    }
    const prev = byGloss.get(picked.glossIndex);
    if (prev) {
      prev.count += 1;
      if (!prev.keywords.includes(picked.keyword)) prev.keywords.push(picked.keyword);
      if (!prev.answers.includes(answerText(answer))) prev.answers.push(answerText(answer));
    } else {
      byGloss.set(picked.glossIndex, {
        gloss: picked.gloss,
        name: picked.gloss,
        glossIndex: picked.glossIndex,
        origin: picked.glossIndex,
        category: picked.category,
        priority: picked.priority,
        matchType: picked.matchType,
        keywords: [picked.keyword],
        answers: [answerText(answer)],
        count: 1,
        source: 'answer_core_keyword',
      });
    }
  }

  const max = Math.max(1, ...[...byGloss.values()].map(item => item.count));
  const candidates = [...byGloss.values()]
    .map(item => ({ ...item, score: Number((item.count / max).toFixed(6)) }))
    .sort((a, b) =>
      a.priority - b.priority ||
      b.count - a.count ||
      String(a.gloss).localeCompare(String(b.gloss))
    );

  return { candidates, unresolved };
}

// ── 질문 의도 기반 표제어 확장 ─────────────────────────────────
// 답변 pool에서 관찰된 키워드만 내보내면 "어디가 아파서 오셨어요?"에 대해
// pool에 우연히 들어온 부위 몇 개만 나온다. 질문이 부위를 묻고 있으면 사전에
// 있는 신체 부위 표제어 전체가 후보가 되어야 후속 모듈이 답을 놓치지 않는다.

function loadIntentGlossSets() {
  if (intentGlossSets) return intentGlossSets;
  if (!fs.existsSync(INTENT_GLOSS_SETS_FILE)) {
    throw new Error(`Intent gloss sets not found: ${INTENT_GLOSS_SETS_FILE}`);
  }
  intentGlossSets = JSON.parse(fs.readFileSync(INTENT_GLOSS_SETS_FILE, 'utf8'));
  return intentGlossSets;
}

// 질문이 직접 묻는 부류(primary)와 답변에 곁들여질 수 있는 부류(secondary)를 나눈다.
// 둘을 섞어 통째로 내보내면 "예전에 앓으신 병" 하나에 시점·기간 51개가 따라붙어
// 후속 모듈이 걸러야 할 노이즈가 된다.
function resolveIntentSets(question, subCategory) {
  const config = loadIntentGlossSets();
  const primary = [];
  const secondary = [];
  const focusedSecondary = [];
  let hasFocusedSecondary = false;
  const push = (list, name) => {
    if (name && config.sets?.[name] && !list.includes(name)) list.push(name);
  };

  const bySub = config.subCategorySets?.[String(subCategory || '').trim()];
  (bySub?.primary || []).forEach(name => push(primary, name));
  (bySub?.secondary || []).forEach(name => push(secondary, name));

  for (const rule of config.questionPatterns || []) {
    let regex;
    try {
      regex = new RegExp(rule.pattern);
    } catch {
      continue;
    }
    if (regex.test(String(question || ''))) {
      (rule.sets || []).forEach(name => push(primary, name));
      if (Array.isArray(rule.secondarySets)) {
        hasFocusedSecondary = true;
        rule.secondarySets.forEach(name => push(focusedSecondary, name));
      }
    }
  }

  const resolvedSecondary = hasFocusedSecondary ? focusedSecondary : secondary;
  return { primary, secondary: resolvedSecondary.filter(name => !primary.includes(name)) };
}

// 스코어는 연속값이 아니라 근거의 종류를 나타내는 단계값이다. 값이 촘촘하면
// 어디까지가 쓸 만한 후보인지 후속 모듈이 판단할 수 없다.
const KEYWORD_TIERS = {
  primary_evidence: 1,      // 질문이 묻는 부류이면서 실제 답변에 나온 표제어
  primary_expansion: 0.8,   // 질문이 묻는 부류. 답변에 없어도 답이 될 수 있다
  secondary_evidence: 0.5,  // 질문에 곁들여지는 부류이면서 실제 답변에 나온 표제어
  related_evidence: 0.4,    // 질문 의도와 무관한 다른 부류에서 관찰된 표제어
  secondary_expansion: 0.3, // 곁들여지는 부류. 실제 답변에서 쓰인 적 있는 것만
  unclassified_evidence: 0.2, // 어느 부류에도 속하지 않는 표제어
};

// 이 값 미만은 최종 리스트에서 뺀다. 질문의 primary/secondary 부류가 정해졌다면
// 단순히 답변에 등장한 다른 부류(교통사고, 계속, 계단 등)는 키워드가 아니다.
const KEYWORD_SCORE_MIN = Number(process.env.KEYWORD_SCORE_MIN ?? KEYWORD_TIERS.secondary_evidence);

function buildKeywordOutput(question, subCategory, evidenceCandidates, glossVectorDb) {
  const config = loadIntentGlossSets();
  const { primary, secondary } = resolveIntentSets(question, subCategory);
  const glossary = glossaryByIndex(glossVectorDb);

  // 실제 답변에서 그 표제어가 쓰인 횟수. 부류 안 정렬과 보조 부류 선별에 쓴다.
  const map = loadKeywordGlossMap();
  const subPrior = map.glossPriorBySubCategory?.[String(subCategory || '').trim()] || {};
  const globalPrior = map.glossPrior || {};
  const priorOf = glossIndex => {
    const key = String(glossIndex);
    return Number(subPrior[key] || 0) * 1000 + Number(globalPrior[key] || 0);
  };

  // 글로스 -> 소속 부류. 앞선 부류가 이긴다. 질문이 묻지 않는 부류라도 답변에서
  // 실제로 나왔다면 키워드로 쓸 수 있으므로, 나머지 부류도 related로 붙여둔다.
  const roleOf = new Map();
  const labelOf = new Map();
  const assign = (names, role) => {
    for (const setName of names) {
      for (const glossIndex of config.sets[setName].glossIds || []) {
        const id = Number(glossIndex);
        if (roleOf.has(id)) continue;
        roleOf.set(id, role);
        labelOf.set(id, config.sets[setName].label);
      }
    }
  };
  assign(primary, 'primary');
  assign(secondary, 'secondary');
  assign(Object.keys(config.sets).filter(name => !primary.includes(name) && !secondary.includes(name)), 'related');

  const merged = new Map();
  for (const item of evidenceCandidates) {
    const role = roleOf.get(item.glossIndex) || 'none';
    merged.set(item.glossIndex, {
      ...item,
      source: 'answer_evidence',
      intentRole: role,
      intentLabel: labelOf.get(item.glossIndex) || '',
      prior: priorOf(item.glossIndex),
      score: role === 'primary' ? KEYWORD_TIERS.primary_evidence
        : role === 'secondary' ? KEYWORD_TIERS.secondary_evidence
        : role === 'none' ? KEYWORD_TIERS.unclassified_evidence
          : KEYWORD_TIERS.related_evidence,
    });
  }

  // primary는 부류 전체를 낸다. secondary를 통째로 내면 노이즈가 되므로,
  // 문진 답변에서 실제로 쓰인 적 있는 표제어로 제한한다.
  const expand = (names, role, keep) => {
    for (const setName of names) {
      for (const glossIndex of config.sets[setName].glossIds || []) {
        const id = Number(glossIndex);
        if (merged.has(id) || roleOf.get(id) !== role) continue;
        const doc = glossary.get(id);
        if (!doc || !keep(id)) continue;
        merged.set(id, {
          gloss: doc.gloss || doc.name,
          name: doc.gloss || doc.name,
          glossIndex: id,
          origin: id,
          category: doc.category || '',
          intentRole: role,
          intentLabel: labelOf.get(id) || config.sets[setName].label,
          keywords: [],
          count: 0,
          source: 'intent_expansion',
          prior: priorOf(id),
          score: role === 'primary' ? KEYWORD_TIERS.primary_expansion : KEYWORD_TIERS.secondary_expansion,
        });
      }
    }
  };
  expand(primary, 'primary', () => true);
  expand(secondary, 'secondary', id => priorOf(id) > 0);

  const ranked = [...merged.values()].sort((a, b) =>
    b.score - a.score ||
    (b.count || 0) - (a.count || 0) ||
    (b.prior || 0) - (a.prior || 0) ||
    String(a.gloss).localeCompare(String(b.gloss))
  );

  const candidates = ranked.filter(item => item.score >= KEYWORD_SCORE_MIN);
  const dropped = ranked.filter(item => item.score < KEYWORD_SCORE_MIN);

  // 확장이 하나도 안 걸린 질문(인사·신원 확인 등)까지 빈 리스트가 되면 후속
  // 모듈이 아무것도 받지 못한다. 이때는 근거 있는 표제어라도 남긴다.
  const finalCandidates = candidates.length
    ? candidates
    : ranked.filter(item => item.source === 'answer_evidence');

  const describe = (names, role) => names.map(name => ({
    key: name,
    role,
    label: config.sets[name].label,
    glossCount: (config.sets[name].glossIds || []).length,
  }));

  // 집계는 걸러내고 남은 것 기준이어야 화면에 보이는 수와 맞는다.
  const kept = tier => finalCandidates.filter(item => item.score === tier).length;

  return {
    candidates: finalCandidates,
    evidenceCount: finalCandidates.filter(item => item.source === 'answer_evidence').length,
    expandedCount: finalCandidates.filter(item => item.source === 'intent_expansion').length,
    primaryExpandedCount: kept(KEYWORD_TIERS.primary_expansion),
    secondaryExpandedCount: kept(KEYWORD_TIERS.secondary_expansion),
    scoreMin: KEYWORD_SCORE_MIN,
    droppedCount: candidates.length ? dropped.length : 0,
    droppedSamples: (candidates.length ? dropped : []).slice(0, 8)
      .map(item => ({ gloss: item.gloss, glossIndex: item.glossIndex, score: item.score })),
    activeSets: [...describe(primary, 'primary'), ...describe(secondary, 'secondary')],
  };
}

function mergeGlossCandidates(primary, supplemental) {
  const merged = new Map();
  const add = (item, preferredSource = '') => {
    const key = Number(item.glossIndex ?? item.origin);
    if (!Number.isFinite(key)) return;
    const source = uniqueValues([item.source, preferredSource].filter(Boolean)).join('+');
    const prev = merged.get(key);
    if (!prev || Number(item.score || 0) > Number(prev.score || 0)) {
      merged.set(key, { ...item, source: source || item.source });
    }
  };
  (primary || []).forEach(item => add(item));
  (supplemental || []).forEach(item => add(item, 'display_answer_pool'));
  return [...merged.values()];
}

function stageLabels(rows) {
  const map = new Map();
  for (const row of rows) {
    const stage = row['단계'] || '';
    const subCategory = row['세부분류'] || '';
    const key = `${stage}::${subCategory}`;
    if (!stage || !subCategory) continue;
    if (!map.has(key)) map.set(key, { stage, subCategory, examples: [] });
    const label = map.get(key);
    const question = row['의사 질문(개별)'];
    if (question && label.examples.length < 3 && !label.examples.includes(question)) {
      label.examples.push(question);
    }
  }
  return [...map.values()].sort((a, b) => a.subCategory.localeCompare(b.subCategory));
}

function buildStageCandidates(rows, stage, subCategory, fallbackRows) {
  const stageKey = normalizeToken(stage);
  const subKey = normalizeToken(subCategory);
  const matched = rows.filter(r =>
    normalizeToken(r['단계']) === stageKey &&
    normalizeToken(r['세부분류']) === subKey
  );
  const subCategoryMatched = matched.length ? [] : rows.filter(r => normalizeToken(r['세부분류']) === subKey);
  const sourceRows = matched.length ? matched : subCategoryMatched.length ? subCategoryMatched : fallbackRows;
  return {
    mappingFound: matched.length > 0 || subCategoryMatched.length > 0,
    rows: sourceRows,
    keywords: uniqueValues(sourceRows.map(r => r['대표 환자키워드']).filter(Boolean)).slice(0, 8),
    glosses: uniqueValues(sourceRows.flatMap(r => splitValues(r['gloss_names']))).slice(0, 8),
  };
}

function loadEmbeddingIndex() {
  if (embeddingIndex) return embeddingIndex;
  if (!fs.existsSync(EMBEDDING_INDEX_FILE)) {
    throw new Error(`Embedding RAG index not found: ${EMBEDDING_INDEX_FILE}`);
  }
  embeddingIndex = JSON.parse(fs.readFileSync(EMBEDDING_INDEX_FILE, 'utf8'));
  return embeddingIndex;
}

function normalizeVector(vector) {
  const values = (vector || []).map(Number);
  const norm = Math.sqrt(values.reduce((sum, v) => sum + v * v, 0)) || 1;
  return values.map(v => v / norm);
}

function dot(a, b) {
  let sum = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

async function embedOllama(baseUrl, model, input) {
  const res = await fetch(`${baseUrl.replace(/\/$/, '')}/api/embed`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, input }),
  }).catch(() => {
    throw new Error(`Ollama embedding 서버(${baseUrl})에 연결할 수 없습니다.`);
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error || data?.message || `Ollama embed 오류 ${res.status}`);
  if (Array.isArray(data.embeddings?.[0])) return normalizeVector(data.embeddings[0]);
  if (Array.isArray(data.embedding)) return normalizeVector(data.embedding);
  throw new Error('Ollama embedding 응답에 embedding 값이 없습니다.');
}

function searchByEmbedding(queryEmbedding, docs, topK, excludeFn = () => false) {
  return docs
    .filter(doc => !excludeFn(doc))
    .map(doc => ({ ...doc, score: dot(queryEmbedding, doc.embedding || []) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

async function retrieveQuestionTopK(question, stage, subCategory, options = {}) {
  const index = loadQuestionRetrievalIndex();
  const topK = Math.max(1, Math.min(Number(options.topK || 5), 20));
  const highThreshold = Number.isFinite(Number(options.highThreshold)) ? Number(options.highThreshold) : 0.86;
  const mediumThreshold = Number.isFinite(Number(options.mediumThreshold)) ? Number(options.mediumThreshold) : 0.70;
  const model = options.embeddingModel || index.embedding?.model || 'bge-m3';
  const baseUrl = options.ollamaUrl || process.env.OLLAMA_URL || 'http://localhost:11434';
  const queryEmbedding = await embedOllama(baseUrl, model, `의사 질문: ${question}`);
  const sameCategory = (index.documents || []).filter(doc =>
    normalizeToken(doc.stage) === normalizeToken(stage) &&
    normalizeToken(doc.subCategory) === normalizeToken(subCategory)
  );
  const candidates = sameCategory.length
    ? sameCategory
    : (index.documents || []).filter(doc => normalizeToken(doc.subCategory) === normalizeToken(subCategory));
  const hits = searchByEmbedding(queryEmbedding, candidates, topK).map(hit => ({
    ...hit,
    score: Number(hit.score.toFixed(6)),
    exact: normalizeToken(hit.question) === normalizeToken(question),
  }));
  const best = hits[0];
  const bestScore = best?.exact ? 1 : Number(best?.score || 0);
  const confidence = bestScore >= highThreshold ? 'high' : bestScore >= mediumThreshold ? 'medium' : 'low';
  return {
    confidence,
    bestScore,
    highThreshold,
    mediumThreshold,
    topK,
    embeddingModel: model,
    categoryFiltered: sameCategory.length > 0,
    hits,
  };
}

function requiredAnswerSlots(question, stage, subCategory) {
  const text = normalizeToken(`${question} ${stage} ${subCategory}`);
  if (/(언제부터|얼마나오래|발생시점|onset)/.test(text)) {
    return ['recent', 'days', 'weeks_months', 'long_term', 'uncertain'];
  }
  if (/(몇점|점수|강도|severity|pain_score|얼마나아프|심하)/.test(text)) {
    return ['low', 'medium', 'high', 'numeric', 'uncertain'];
  }
  if (/(빈도|횟수|얼마나자주|일주일|한달|매일|frequency|lifestyle|occupation)/.test(text)) {
    return ['none', 'occasional', 'regular', 'frequent', 'uncertain'];
  }
  if (/(어디|부위|location|surgery_site)/.test(text)) {
    return ['specific', 'multiple_or_side', 'uncertain'];
  }
  if (questionNeedsPolarityCoverage(question) || /(history|adverse_reaction|red_flag|medication_use)/.test(text)) {
    return ['positive', 'negative', 'uncertain'];
  }
  return ['representative', 'negative_or_none', 'uncertain'];
}

function classifyAnswerSlot(answer, requiredSlots) {
  const text = normalizeToken(answerText(answer));
  const has = slot => requiredSlots.includes(slot);
  if (has('uncertain') && /(모르|기억안|기억나지않|기억나지못|잘모르|확실하지|애매)/.test(text)) return 'uncertain';
  if (has('negative') && /(없|아니|안했|하지않|못했)/.test(text)) return 'negative';
  if (has('positive') && /(있|했|받|먹|복용|경험)/.test(text)) return 'positive';
  if (has('none') && /(없|안하|안마|끊|금연|전혀)/.test(text)) return 'none';
  if (has('numeric') && /\d+\s*점/.test(text)) return 'numeric';
  if (has('high') && /(매우|너무|심하|극심|참기힘|[78910]\s*점)/.test(text)) return 'high';
  if (has('medium') && /(중간|보통|그럭저럭|[456]\s*점)/.test(text)) return 'medium';
  if (has('low') && /(조금|약간|경미|참을만|[0123]\s*점)/.test(text)) return 'low';
  if (has('recent') && /(오늘|어제|방금|최근|몇시간)/.test(text)) return 'recent';
  if (has('days') && /(\d+|한|두|세|며칠)\s*일/.test(text)) return 'days';
  if (has('weeks_months') && /(주|주일|달|개월)/.test(text)) return 'weeks_months';
  if (has('long_term') && /(년|오래|예전부터)/.test(text)) return 'long_term';
  if (has('frequent') && /(매일|거의매일|하루|자주)/.test(text)) return 'frequent';
  if (has('regular') && /(매주|일주일|한달에|정기)/.test(text)) return 'regular';
  if (has('occasional') && /(가끔|어쩌다|한두번|드물)/.test(text)) return 'occasional';
  if (has('multiple_or_side') && /(양쪽|여러|오른|왼|한쪽|두군데)/.test(text)) return 'multiple_or_side';
  if (has('specific') && text.length >= 2) return 'specific';
  if (has('negative_or_none') && /(없|아니|안|않)/.test(text)) return 'negative_or_none';
  return requiredSlots.includes('representative') ? 'representative' : '';
}

function analyzeAnswerCoverage(question, stage, subCategory, answers) {
  const requiredSlots = requiredAnswerSlots(question, stage, subCategory);
  const covered = new Set();
  const annotatedAnswers = (answers || []).map(answer => {
    const slot = classifyAnswerSlot(answer, requiredSlots);
    if (slot) covered.add(slot);
    return typeof answer === 'string' ? { answer, slot, source: 'existing_qa' } : { ...answer, slot: answer.slot || slot };
  });
  return {
    requiredSlots,
    coveredSlots: requiredSlots.filter(slot => covered.has(slot)),
    missingSlots: requiredSlots.filter(slot => !covered.has(slot)),
    answers: annotatedAnswers,
  };
}

function allowedGlossesForRetrieval(retrieval, index, limit = 80) {
  const direct = retrieval.hits.flatMap(hit => hit.allowedGlosses || []);
  const best = retrieval.hits[0];
  const category = best
    ? (index.documents || []).filter(doc => normalizeToken(doc.subCategory) === normalizeToken(best.subCategory))
    : [];
  return uniqueValues(
    [...direct, ...category.flatMap(doc => doc.allowedGlosses || [])],
    item => String(item.glossId)
  ).slice(0, limit);
}

function buildMissingSlotPrompt(question, stage, subCategory, retrieval, coverage, allowedGlosses, count) {
  const system = [
    '당신은 통증의학과 초진 문진의 예상 환자 답변을 보완합니다.',
    '검색된 기존 질문과 답변을 근거로, 요청된 누락 슬롯만 생성하세요.',
    '진단, 검사 결과, 치료 이력을 새로 지어내지 마세요.',
    'glossIds에는 반드시 허용 Gloss 목록의 숫자 ID만 사용하세요.',
    '허용 Gloss로 표현할 수 없는 답변은 생성하지 마세요.',
    '기존 답변을 그대로 반복하지 말고 짧은 1인칭 환자 답변으로 작성하세요.',
    'JSON 객체 하나로만 응답하세요.',
  ].join('\n');
  const examples = retrieval.hits.slice(0, 5).map(hit => ({
    questionId: hit.questionId,
    question: hit.question,
    similarity: hit.score,
    answers: (hit.answers || []).slice(0, 12).map(item => item.answer || item),
  }));
  const user = [
    `입력 질문: ${question}`,
    `확정 분류: ${stage} / ${subCategory}`,
    `검색 신뢰도: ${retrieval.confidence}`,
    `생성할 누락 슬롯: ${coverage.missingSlots.slice(0, count).join(', ')}`,
    '',
    `기존 근거: ${JSON.stringify(examples)}`,
    `이미 있는 답변: ${JSON.stringify(coverage.answers.map(item => item.answer).slice(0, 30))}`,
    `허용 Gloss: ${JSON.stringify(allowedGlosses.map(item => ({ glossId: item.glossId, name: item.name })))}`,
    '',
    '[출력 JSON]',
    '{"answers":[{"slot":"누락 슬롯 중 하나","answer":"환자 답변","glossIds":[숫자]}]}',
  ].join('\n');
  return { system, user };
}

function validateGeneratedSlotAnswers(parsed, coverage, allowedGlosses, existingAnswers) {
  const missing = new Set(coverage.missingSlots);
  const allowed = new Map(allowedGlosses.map(item => [Number(item.glossId), item]));
  const existing = new Set(existingAnswers.map(answer => normalizeToken(answerText(answer))));
  const rejected = [];
  const accepted = [];
  for (const item of parsed?.answers || []) {
    const slot = String(item.slot || '').trim();
    const answer = cleanPatientAnswerText(item.answer).trim();
    const glossIds = uniqueValues(splitValues(item.glossIds).map(Number).filter(Number.isFinite), String);
    const errors = [];
    if (!missing.has(slot)) errors.push('slot_not_requested');
    if (!answer || answer.length > 120) errors.push('invalid_answer');
    if (existing.has(normalizeToken(answer))) errors.push('duplicate_answer');
    if (!glossIds.length) errors.push('missing_gloss_ids');
    if (glossIds.some(id => !allowed.has(id))) errors.push('gloss_id_not_allowed');
    if (classifyAnswerSlot(answer, coverage.requiredSlots) !== slot && slot !== 'representative') errors.push('slot_mismatch');
    if (errors.length) {
      rejected.push({ slot, answer, glossIds, errors });
      continue;
    }
    const glossHints = glossIds.map(id => `${allowed.get(id).name}_${id}`);
    accepted.push({ answer, slot, glossIds, glossHints, keywords: [], source: 'llm_slot_variation' });
    existing.add(normalizeToken(answer));
    missing.delete(slot);
  }
  return { accepted, rejected };
}

function voteStage(rowHits) {
  const votes = new Map();
  for (const hit of rowHits) {
    const key = `${hit.stage}::${hit.subCategory}`;
    const prev = votes.get(key) || { stage: hit.stage, subCategory: hit.subCategory, score: 0, count: 0 };
    prev.score += Math.max(0, hit.score);
    prev.count += 1;
    votes.set(key, prev);
  }
  return [...votes.values()]
    .sort((a, b) => b.score - a.score || b.count - a.count)[0] || { stage: '', subCategory: '', score: 0, count: 0 };
}

function addGlossCandidate(candidates, name, origin, source, score) {
  const cleanName = String(name || '').trim();
  const numericOrigin = Number(origin);
  if (!cleanName) return;
  if (candidates.some(item =>
    normalizeToken(item.name) === normalizeToken(cleanName) ||
    (Number.isFinite(numericOrigin) && item.origin === numericOrigin)
  )) return;
  candidates.push({
    name: cleanName,
    origin: Number.isFinite(numericOrigin) ? numericOrigin : null,
    source,
    score: Number(Number(score || 0).toFixed(6)),
  });
}

function buildEmbeddingGlossCandidates(rowHits, glossHits, topK) {
  const candidates = [];
  for (const hit of rowHits) {
    (hit.glosses || []).forEach((name, i) => {
      addGlossCandidate(candidates, name, (hit.glossOrigins || [])[i], 'similar_row', hit.score);
    });
  }
  for (const hit of glossHits) {
    addGlossCandidate(candidates, hit.name, hit.origin, 'gloss_vector', hit.score);
  }
  return candidates
    .sort((a, b) => b.score - a.score || a.source.localeCompare(b.source))
    .slice(0, topK);
}

function makeEmbeddingQuery(question, answer) {
  return [
    `의사 질문: ${question || ''}`,
    answer ? `환자 답변: ${answer}` : '',
  ].filter(Boolean).join('\n');
}

function makeEmbeddingGlossQuery(question, rowHits) {
  return [
    `의사 질문: ${question || ''}`,
    ...rowHits.slice(0, 5).map(hit => `예상 환자 답변: ${hit.answer}`),
    ...rowHits.slice(0, 8).map(hit => `키워드: ${hit.keyword}`),
  ].filter(Boolean).join('\n');
}

async function recommendEmbeddingRag(body) {
  const index = loadEmbeddingIndex();
  const topK = Math.max(1, Math.min(Number(body.topK || body.outputTopK || 30), 200));
  const rowTopK = Math.max(1, Math.min(Number(body.rowTopK || 12), 100));
  const glossTopK = Math.max(1, Math.min(Number(body.glossTopK || topK), 200));
  const model = body.embeddingModel || index.embedding?.model || 'bge-m3';
  const baseUrl = body.ollamaUrl || index.embedding?.baseUrl || DEFAULT_LLM.ollamaUrl;
  const useAnswerInQuery = body.queryMode === 'question-answer' && body.answer;
  const queryEmbedding = await embedOllama(baseUrl, model, makeEmbeddingQuery(body.question, useAnswerInQuery ? body.answer : ''));
  const rowHits = searchByEmbedding(queryEmbedding, index.questionDocs || [], rowTopK);
  const voted = voteStage(rowHits);
  const glossQueryEmbedding = await embedOllama(baseUrl, model, makeEmbeddingGlossQuery(body.question, rowHits));
  const glossHits = searchByEmbedding(glossQueryEmbedding, index.glossDocs || [], glossTopK);
  const glossCandidates = buildEmbeddingGlossCandidates(rowHits, glossHits, topK);
  const keywords = uniqueValues(rowHits.map(hit => hit.keyword).filter(Boolean)).slice(0, 20);

  return {
    engine: 'embedding-rag',
    glossSet: body.glossSet || index.source?.ragIndex || 'embedding_rag_index',
    embeddingModel: model,
    topK,
    stage: voted.stage,
    subCategory: voted.subCategory,
    patientAnswer: rowHits[0]?.answer || body.answer || '',
    patientAnswerCandidates: uniqueValues(rowHits.map(hit => hit.answer).filter(Boolean)).slice(0, 10),
    keyword: keywords.join(', '),
    keywords,
    glosses: glossCandidates.map(item => item.name),
    glossCandidates,
    reason: `embedding rowTopK=${rowTopK}, glossTopK=${glossTopK}, outputTopK=${topK}`,
    mappingFound: rowHits.length > 0,
    recommendationSource: 'embedding_rows+embedding_gloss',
    similarCount: rowHits.length,
    examples: rowHits.slice(0, 8).map(hit => ({
      stage: hit.stage,
      subCategory: hit.subCategory,
      question: hit.question,
      answer: hit.answer,
      keyword: hit.keyword,
      glosses: (hit.glosses || []).join(', '),
      score: Number(hit.score.toFixed(6)),
    })),
    ragHits: glossHits.slice(0, Math.min(glossTopK, 20)).map(hit => ({
      origin: hit.origin,
      name: hit.name,
      score: Number(hit.score.toFixed(6)),
    })),
  };
}

// ── JSON 파싱 (제어문자 안전) ──────────────────────────────────
function safeParseJson(text) {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;
  try { return JSON.parse(match[0]); } catch (_) {}
  let clean = '', inStr = false, esc = false;
  for (const c of match[0]) {
    const code = c.charCodeAt(0);
    if (esc) { clean += c; esc = false; }
    else if (c === '\\' && inStr) { clean += c; esc = true; }
    else if (c === '"') { clean += c; inStr = !inStr; }
    else if (inStr && code < 0x20) { clean += '\\u' + code.toString(16).padStart(4, '0'); }
    else { clean += c; }
  }
  try { return JSON.parse(clean); } catch (_) { return null; }
}

const DEFAULT_LLM = {
  provider: 'ollama',
  model: 'qwen3:14b',
  ollamaUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
  openAIBaseUrl: process.env.OPENAI_COMPATIBLE_URL || 'http://localhost:8000/v1',
};

function normalizeBackendUrl(value, fallback) {
  const raw = String(value || '').trim();
  if (!raw) return fallback;
  if (raw.startsWith('/ollama')) return process.env.OLLAMA_URL || 'http://ollama:11434';
  if (raw.startsWith('/omniserve')) {
    const suffix = raw.replace(/^\/omniserve\/?/, '');
    return `http://host.docker.internal:8000/${suffix}`.replace(/\/$/, '');
  }
  return raw;
}

function normalizeLlmConfig(body = {}) {
  const think = body.think === true || body.think === 'true' || body.enableThinking === true;
  return {
    provider: body.provider || DEFAULT_LLM.provider,
    model: body.model || DEFAULT_LLM.model,
    ollamaUrl: normalizeBackendUrl(body.ollamaUrl, DEFAULT_LLM.ollamaUrl),
    omniserveUrl: normalizeBackendUrl(body.omniserveUrl, process.env.OMNISERVE_URL || 'http://host.docker.internal:8000/a/v1'),
    openAIBaseUrl: normalizeBackendUrl(body.openAIBaseUrl, DEFAULT_LLM.openAIBaseUrl),
    maxTokens: Number.isFinite(Number(body.maxTokens)) ? Number(body.maxTokens) : 384,
    numCtx: Number.isFinite(Number(body.numCtx)) ? Number(body.numCtx) : 4096,
    keepAlive: body.keepAlive || '30m',
    think,
  };
}

function isEeveStyleModel(model) {
  const name = String(model || '').toLowerCase();
  return name.includes('eeve-korean-instruct') ||
    name.includes('yanolja-eeve') ||
    name === 'jmpark333/eeve:latest' ||
    name.endsWith('/eeve:latest');
}

async function callOllama(config, system, user) {
  const model = config.model || DEFAULT_LLM.model;
  const think = config.think === true;
  if (isEeveStyleModel(model)) {
    const res = await fetch(`${config.ollamaUrl.replace(/\/$/, '')}/api/generate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        raw: true,
        prompt: `### System:\n${system}\n\n### User:\n${user}\n\n### Assistant:\n`,
        stream: false,
        keep_alive: config.keepAlive,
        options: {
          temperature: 0.1,
          num_ctx: Math.min(config.numCtx || 4096, 4096),
          num_predict: Math.min(config.maxTokens || 384, 384),
          stop: ['### System:', '### User:'],
        },
      }),
    }).catch(() => {
      throw new Error(`Ollama 서버(${config.ollamaUrl})에 연결할 수 없습니다.`);
    });
    if (!res.ok) throw new Error(`Ollama 오류 ${res.status}: 모델 "${model}"이 설치되어 있는지 확인하세요.`);
    const data = await res.json();
    return data.response || '';
  }
  const res = await fetch(`${config.ollamaUrl.replace(/\/$/, '')}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
      stream: false,
      think,
      keep_alive: config.keepAlive,
      options: {
        temperature: 0.1,
        num_ctx: config.numCtx || 4096,
        num_predict: config.maxTokens || 384,
      },
    }),
  }).catch(() => {
    throw new Error(`Ollama 서버(${config.ollamaUrl})에 연결할 수 없습니다.`);
  });
  if (!res.ok) throw new Error(`Ollama 오류 ${res.status}: 모델 "${model}"이 설치되어 있는지 확인하세요.`);
  const data = await res.json();
  return data.message?.content || data.message?.thinking || '';
}

async function callOpenAICompatible(config, system, user) {
  const baseUrl = config.provider === 'omniserve'
    ? (config.omniserveUrl || 'http://localhost:8000/a/v1')
    : config.openAIBaseUrl;
  const endpoint = `${baseUrl.replace(/\/$/, '')}/chat/completions`;
  const headers = { 'content-type': 'application/json' };

  const res = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: config.model || 'Qwen/Qwen3-14B',
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
      temperature: 0.1,
      max_tokens: 512,
    }),
  }).catch(() => {
    throw new Error(`로컬 OpenAI-compatible 서버(${baseUrl})에 연결할 수 없습니다.`);
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);
  return data.choices?.[0]?.message?.content || '';
}

async function callLocalLLM(config, system, user) {
  if (config.provider === 'local-openai' || config.provider === 'omniserve') return callOpenAICompatible(config, system, user);
  return callOllama(config, system, user);
}

// ── API ────────────────────────────────────────────────────────
app.get('/api/gloss-catalog', (_req, res) => {
  try {
    const glossDb = loadLlmGlossVectorDb();
    const retrieval = loadQuestionRetrievalIndex();
    const intents = loadIntentGlossSets();
    const glosses = (glossDb.documents || []).map(doc => ({
      glossId: Number(doc.glossIndex ?? doc.origin),
      gloss: doc.gloss || doc.name || '',
      synonyms: Array.isArray(doc.synonyms) ? doc.synonyms : [],
      category: doc.category || '',
    })).sort((a, b) => a.glossId - b.glossId);
    const assigned = new Set(Object.values(intents.sets || {})
      .flatMap(set => set.glossIds || []).map(Number));
    const stageMap = new Map();
    for (const doc of retrieval.documents || []) {
      const stage = doc.stage || '미분류';
      const subCategory = doc.subCategory || '미분류';
      const key = `${stage}::${subCategory}`;
      const row = stageMap.get(key) || { stage, subCategory, questionCount: 0, answerCount: 0 };
      row.questionCount += 1;
      row.answerCount += (doc.answers || []).length;
      stageMap.set(key, row);
    }
    const stages = [...stageMap.values()].map(row => {
      const spec = intents.subCategorySets?.[row.subCategory] || {};
      return { ...row, primarySets: spec.primary || [], secondarySets: spec.secondary || [] };
    }).sort((a, b) => a.stage.localeCompare(b.stage) || a.subCategory.localeCompare(b.subCategory));
    res.json({
      glossSet: 'test_658',
      totalGlosses: glosses.length,
      assignedGlosses: assigned.size,
      unassignedGlosses: glosses.length - assigned.size,
      totalStages: new Set(stages.map(row => row.stage)).size,
      totalSubCategories: stages.length,
      totalQuestions: (retrieval.documents || []).length,
      totalAnswers: stages.reduce((sum, row) => sum + row.answerCount, 0),
      stages,
      glosses,
    });
  } catch (e) {
    console.error('[gloss catalog error]', e.message);
    res.status(500).json({ error: e.message || String(e) });
  }
});

app.post('/api/load', (req, res) => {
  const { filePath } = req.body;
  if (!filePath) return res.status(400).json({ error: '파일 경로를 입력하세요.' });
  try {
    dataset = loadDataset(filePath);
    res.json({ count: dataset.length });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/status', (_req, res) => {
  res.json({
    loaded: dataset.length > 0,
    count: dataset.length,
    embeddingRag: {
      indexFile: EMBEDDING_INDEX_FILE,
      available: fs.existsSync(EMBEDDING_INDEX_FILE),
      loaded: Boolean(embeddingIndex),
      questionDocs: embeddingIndex?.questionDocs?.length || 0,
      glossDocs: embeddingIndex?.glossDocs?.length || 0,
    },
    llmPipeline: {
      questionAnswerPoolFile: QUESTION_ANSWER_POOL_FILE,
      questionAnswerPoolAvailable: fs.existsSync(QUESTION_ANSWER_POOL_FILE),
      glossVectorDbFile: LLM_GLOSS_VECTOR_DB_FILE,
      glossVectorDbAvailable: fs.existsSync(LLM_GLOSS_VECTOR_DB_FILE),
      questionRetrievalIndexFile: QUESTION_RETRIEVAL_INDEX_FILE,
      questionRetrievalIndexAvailable: fs.existsSync(QUESTION_RETRIEVAL_INDEX_FILE),
      questionRetrievalIndexLoaded: Boolean(questionRetrievalIndex),
      questionRetrievalDocs: questionRetrievalIndex?.documents?.length || 0,
      labels: questionAnswerPool?.labels?.length || 0,
      questions: questionAnswerPool?.questions?.length || 0,
      answers: questionAnswerPool?.counts?.answers || 0,
      glossDocs: llmGlossVectorDb?.documents?.length || 0,
    },
  });
});

app.post('/api/llm-pipeline/classify', async (req, res) => {
  const startedAt = Date.now();
  const { question } = req.body;
  if (!question) return res.status(400).json({ error: '의사 질문을 입력하세요.' });
  try {
    const qaPool = loadQuestionAnswerPool();
    const labels = stageLabelsFromQaPool(qaPool);
    const fast = req.body.fastClassify === false
      ? null
      : (ruleClassifyQuestion(question) || fastClassifyFromQaPool(question, qaPool));
    if (fast) {
      const outOfScope = Boolean(fast.outOfScope || isOutOfScopeLabel(fast.stage, fast.subCategory));
      return res.json({
        engine: 'llm-question-pipeline',
        mode: 'classify',
        question,
        predictedStage: fast.stage,
        predictedSubCategory: fast.subCategory,
        stage: fast.stage,
        subCategory: fast.subCategory,
        reason: fast.reason || `기존 질문 pool 빠른 매칭: "${fast.matchedQuestion}" (score ${fast.score}, margin ${fast.margin})`,
        labelRepaired: false,
        requiresHumanConfirmation: true,
        nextStep: outOfScope
          ? '진료 문진 질문이 아니므로 답변/표제어 추천을 중단합니다. 필요하면 사람이 기존 세부분류로 수동 변경하세요.'
          : '분류 결과를 사람이 확인한 뒤 추천 단계로 진행하세요.',
        labels,
        raw: '',
        classifySource: fast.classifySource || 'qa_pool_fast_match',
        outOfScope,
        latencyMs: {
          total: Date.now() - startedAt,
          llm: 0,
        },
      });
    }
    const llmConfig = normalizeLlmConfig({ ...req.body, maxTokens: req.body.maxTokens || 96, numCtx: req.body.numCtx || 4096 });
    const llmStartedAt = Date.now();
    const raw = await callLocalLLM(llmConfig, ...Object.values(buildLlmClassifyPrompt(question, labels)));
    const llmLatency = Date.now() - llmStartedAt;
    const parsed = safeParseJson(raw);
    const resolved = parsed
      ? resolveStageLabel(parsed, labels, question)
      : { stage: '', subCategory: '', repaired: false, unresolved: true };
    const outOfScope = isOutOfScopeLabel(resolved.stage, resolved.subCategory);
    res.json({
      engine: 'llm-question-pipeline',
      mode: 'classify',
      question,
      predictedStage: resolved.stage || '',
      predictedSubCategory: resolved.subCategory || '',
      stage: resolved.stage || '',
      subCategory: resolved.subCategory || '',
      reason: parsed?.reason || raw,
      labelRepaired: Boolean(resolved.repaired),
      labelUnresolved: Boolean(resolved.unresolved),
      requiresHumanConfirmation: true,
      nextStep: outOfScope
        ? '진료 문진 질문이 아니므로 답변/표제어 추천을 중단합니다. 필요하면 사람이 기존 세부분류로 수동 변경하세요.'
        : '분류 결과를 사람이 확인한 뒤 추천 단계로 진행하세요.',
      labels,
      raw,
      outOfScope,
      latencyMs: {
        total: Date.now() - startedAt,
        llm: llmLatency,
      },
    });
  } catch (e) {
    console.error('[llm pipeline classify error]', e.message);
    res.status(500).json({ error: e.message || String(e) });
  }
});

app.post('/api/llm-pipeline/recommend', async (req, res) => {
  const startedAt = Date.now();
  const { question, confirmedStage, confirmedSubCategory } = req.body;
  if (!question) return res.status(400).json({ error: '의사 질문을 입력하세요.' });
  if (!confirmedStage || !confirmedSubCategory) {
    return res.status(400).json({ error: '추천 전 사람이 확인한 stage/subCategory가 필요합니다.' });
  }
  if (isOutOfScopeLabel(confirmedStage, confirmedSubCategory)) {
    return res.status(422).json({
      error: '기타/진료 외 질문은 답변 pool과 표제어 추천을 생성하지 않습니다. 기존 세부분류로 수동 변경한 뒤 다시 진행하세요.',
      outOfScope: true,
      answerPool: [],
      keywordCandidates: [],
      glossCandidates: [],
      outputTuples: [],
    });
  }

  try {
    const qaPool = loadQuestionAnswerPool();
    const glossVectorDb = loadLlmGlossVectorDb();
    const labelValidation = validateConfirmedLabel(question, qaPool, confirmedStage, confirmedSubCategory);
    const allowMismatch = req.body.allowMismatchedLabel === true;
    if (!labelValidation.ok && !allowMismatch) {
      return res.status(409).json({
        error: '확정 세부분류가 입력 질문과 맞지 않는 것으로 보입니다. 추천 분류를 사용하거나, 의도적으로 진행하려면 다시 확인이 필요합니다.',
        labelValidation,
        answerPool: [],
        keywordCandidates: [],
        glossCandidates: [],
        outputTuples: [],
      });
    }
    const answerCount = Math.max(1, Math.min(Number(req.body.answerCount || 8), 100));
    const knownSearchStartedAt = Date.now();
    const retrievalIndex = loadQuestionRetrievalIndex();
    const retrieval = await retrieveQuestionTopK(question, confirmedStage, confirmedSubCategory, {
      topK: req.body.questionTopK || 5,
      highThreshold: req.body.questionHighThreshold,
      mediumThreshold: req.body.questionMediumThreshold,
      embeddingModel: req.body.embeddingModel,
      ollamaUrl: normalizeBackendUrl(req.body.ollamaUrl, DEFAULT_LLM.ollamaUrl),
    });
    const knownQuestionSearchLatency = Date.now() - knownSearchStartedAt;
    const selectedHits = retrieval.confidence === 'high'
      ? retrieval.hits.filter(hit => hit.exact || hit.score >= Math.max(retrieval.mediumThreshold, retrieval.bestScore - 0.08))
      : retrieval.confidence === 'medium'
        ? retrieval.hits.filter(hit => hit.score >= retrieval.mediumThreshold)
        : [];
    const selectedQuestionIds = new Set(selectedHits.map(hit => normalizeToken(hit.questionId)));
    const matchedQaQuestions = (qaPool.questions || []).filter(item => selectedQuestionIds.has(normalizeToken(item.questionId)));
    const pool = {
      known: retrieval.confidence !== 'low',
      bestQuestionScore: retrieval.bestScore,
      questions: matchedQaQuestions,
      // 원본 답변의 keyword 주석을 버리지 않는다. 핵심 표제어 선택의 1순위 근거다.
      answers: uniqueValues(
        selectedHits
          .flatMap(hit => hit.answers || [])
          .map(item => (typeof item === 'string'
            ? { answer: item, source: 'existing_qa' }
            : { answer: item.answer, keyword: item.keyword || '', answerId: item.answerId, source: 'existing_qa' }))
          .filter(item => item.answer),
        item => item.answer
      ),
    };
    let answerPool = pool.answers;
    let answerPoolSource = retrieval.confidence === 'high'
      ? 'embedding_existing_question_pool'
      : retrieval.confidence === 'medium'
        ? 'embedding_similar_question_pool'
        : 'llm_grounded_low_similarity_pool';
    let raw = '';
    let llmAnswerGenerationLatency = 0;
    let generationValidation = { accepted: [], rejected: [] };

    const ruleAnswers = ruleAnswerPool(question, glossVectorDb);
    const domainRuleAnswers = ruleAnswers.filter(item => item.source !== 'rule_polarity_completion');
    if (domainRuleAnswers.length) {
      answerPool = domainRuleAnswers;
      answerPoolSource = 'rule_fallback_pool';
    } else if (!answerPool.length && ruleAnswers.length) {
      answerPool = ruleAnswers;
    }

    let coverage = analyzeAnswerCoverage(question, confirmedStage, confirmedSubCategory, answerPool);
    answerPool = coverage.answers;
    const allowedGlosses = allowedGlossesForRetrieval(retrieval, retrievalIndex, Number(req.body.allowedGlossLimit || 80));
    const maxGeneratedSlots = Math.max(0, Math.min(Number(req.body.maxGeneratedSlots ?? 4), 10));
    // 데이터셋 질문과 정확히 일치하면 검증된 기존 답변 pool을 그대로 사용한다.
    // 새로운 질문(정확 일치 없음)은 검색 신뢰도가 높더라도 누락 슬롯을 LLM이 보완한다.
    const exactKnownQuestion = retrieval.hits.some(hit => hit.exact);
    const shouldGenerate = req.body.generateMissingSlots !== false &&
      !exactKnownQuestion &&
      answerPoolSource !== 'rule_fallback_pool' &&
      coverage.missingSlots.length > 0 &&
      maxGeneratedSlots > 0 &&
      allowedGlosses.length > 0;

    if (shouldGenerate) {
      const prompt = buildMissingSlotPrompt(
        question,
        confirmedStage,
        confirmedSubCategory,
        retrieval,
        coverage,
        allowedGlosses,
        maxGeneratedSlots
      );
      const llmConfig = normalizeLlmConfig({
        ...req.body,
        maxTokens: Math.max(
          Number(req.body.maxTokens || 0),
          Math.min(768, Math.max(384, answerCount * 90))
        ),
        numCtx: req.body.numCtx || 4096,
      });
      const llmStartedAt = Date.now();
      raw = await callLocalLLM(llmConfig, prompt.system, prompt.user);
      llmAnswerGenerationLatency = Date.now() - llmStartedAt;
      const parsed = safeParseJson(raw);
      generationValidation = validateGeneratedSlotAnswers(parsed, coverage, allowedGlosses, answerPool);
      if (generationValidation.accepted.length) {
        answerPool = uniqueValues([...answerPool, ...generationValidation.accepted], item => answerText(item));
        if (retrieval.confidence !== 'low') answerPoolSource = 'embedding_existing_plus_llm_variation';
      } else if (!answerPool.length) {
        answerPoolSource = 'llm_empty_answer_pool';
      }
    }

    if (answerPoolSource !== 'rule_fallback_pool' && !answerPool.some(answer => answer.source === 'rule_medication_use_completion')) {
      const limit = questionNeedsPolarityCoverage(question) ? Math.max(answerCount, 2) : answerCount;
      answerPool = ensurePolarityAnswerCoverage(question, answerPool, glossVectorDb).slice(0, Math.max(limit, answerPool.length));
    }
    if (!answerPool.length) {
      answerPool = categoryFallbackAnswerPool(qaPool, confirmedStage, confirmedSubCategory, answerCount);
      answerPoolSource = 'category_fallback_answer_pool';
    }

    answerPool = prioritizeAnswerPool(question, answerPool, glossVectorDb, confirmedStage, confirmedSubCategory);
    coverage = analyzeAnswerCoverage(question, confirmedStage, confirmedSubCategory, answerPool);

    const glossScoringStartedAt = Date.now();
    const qaGlossCandidates = matchedQaQuestions.length
      ? scoreGlossesFromQaQuestions(pool.questions, glossVectorDb)
      : [];
    const answerGlossCandidates = scoreGlossesFromGeneratedAnswers(answerPool, glossVectorDb);
    const glossCandidates = prioritizeGlossCandidates(
      question,
      matchedQaQuestions.length
        ? mergeGlossCandidates(qaGlossCandidates, answerGlossCandidates)
        : answerGlossCandidates,
      confirmedStage,
      confirmedSubCategory
    );
    const keywordResult = buildKeywordCandidates(answerPool, glossVectorDb);
    const keywordOutput = buildKeywordOutput(question, confirmedSubCategory, keywordResult.candidates, glossVectorDb);
    const keywordCandidates = keywordOutput.candidates;
    const annotatedAnswerPool = answerPool.map(item => {
      const picked = selectAnswerKeyword(item, loadKeywordGlossMap(), glossVectorDb);
      const base = typeof item === 'string' ? { answer: item } : item;
      return {
        ...base,
        coreKeyword: picked ? picked.keyword : null,
        coreGloss: picked ? picked.gloss : null,
        coreGlossIndex: picked ? picked.glossIndex : null,
        coreKeywordSource: picked ? picked.matchType : 'unresolved',
      };
    });
    const glossScoringLatency = Date.now() - glossScoringStartedAt;

    res.json({
      engine: 'llm-question-pipeline',
      mode: 'recommend',
      question,
      stage: confirmedStage,
      subCategory: confirmedSubCategory,
      requiresHumanConfirmation: false,
      confirmedByHuman: true,
      answerPoolSource,
      bestKnownQuestionScore: pool.bestQuestionScore,
      matchedQuestionCount: pool.questions?.length || 0,
      retrieval: {
        confidence: retrieval.confidence,
        bestScore: retrieval.bestScore,
        highThreshold: retrieval.highThreshold,
        mediumThreshold: retrieval.mediumThreshold,
        embeddingModel: retrieval.embeddingModel,
        topK: retrieval.hits.map(hit => ({
          questionId: hit.questionId,
          question: hit.question,
          stage: hit.stage,
          subCategory: hit.subCategory,
          score: hit.score,
          exact: hit.exact,
        })),
      },
      coverage: {
        requiredSlots: coverage.requiredSlots,
        coveredSlots: coverage.coveredSlots,
        missingSlots: coverage.missingSlots,
      },
      allowedGlosses,
      generationValidation,
      answerPool: annotatedAnswerPool,
      keywordCandidates,
      keywordCoverage: {
        answers: answerPool.length,
        resolved: answerPool.length - keywordResult.unresolved.length,
        unresolved: keywordResult.unresolved.length,
        unresolvedSamples: keywordResult.unresolved.slice(0, 10),
        glossPoolSize: (glossVectorDb.documents || []).length,
        evidenceCount: keywordOutput.evidenceCount,
        intentExpandedCount: keywordOutput.expandedCount,
        primaryExpandedCount: keywordOutput.primaryExpandedCount,
        secondaryExpandedCount: keywordOutput.secondaryExpandedCount,
        scoreMin: keywordOutput.scoreMin,
        droppedCount: keywordOutput.droppedCount,
        droppedSamples: keywordOutput.droppedSamples,
        intentSets: keywordOutput.activeSets,
      },
      outputTuples: keywordCandidates.map(item => [item.gloss, item.glossIndex, item.score]),
      glossCandidates,
      labelValidation,
      glossSet: 'test_658',
      glossDocCount: (glossVectorDb.documents || []).length,
      latencyMs: {
        total: Date.now() - startedAt,
        knownQuestionSearch: knownQuestionSearchLatency,
        llmAnswerGeneration: llmAnswerGenerationLatency,
        glossScoring: glossScoringLatency,
      },
      raw,
    });
  } catch (e) {
    console.error('[llm pipeline recommend error]', e.message);
    res.status(500).json({ error: e.message || String(e) });
  }
});

// 외부 시스템용 단일 호출 API. 분류와 추천을 서버 내부에서 연속 실행하고
// 최종 글로스-스코어 pair만 간단한 형태로 함께 반환한다.
app.post('/api/keywords', async (req, res) => {
  const startedAt = Date.now();
  const question = String(req.body.question || req.body.sentence || '').trim();
  if (!question) return res.status(400).json({ error: 'question 또는 sentence를 입력하세요.' });

  const internalPost = async (pathName, body) => {
    const response = await fetch(`http://127.0.0.1:${PORT}${pathName}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    return { ok: response.ok, status: response.status, data };
  };

  try {
    const options = {
      provider: req.body.provider,
      model: req.body.model,
      ollamaUrl: req.body.ollamaUrl,
      omniserveUrl: req.body.omniserveUrl,
      openAIBaseUrl: req.body.openAIBaseUrl,
      embeddingModel: req.body.embeddingModel || 'bge-m3',
      generateMissingSlots: req.body.generateMissingSlots,
      answerCount: req.body.answerCount || 8,
      maxGeneratedSlots: req.body.maxGeneratedSlots,
      maxTokens: req.body.maxTokens,
      numCtx: req.body.numCtx,
    };
    let stage = String(req.body.stage || req.body.confirmedStage || '').trim();
    let subCategory = String(req.body.subCategory || req.body.confirmedSubCategory || '').trim();
    let classification = null;
    if (!stage || !subCategory) {
      const classified = await internalPost('/api/llm-pipeline/classify', { ...options, question });
      if (!classified.ok) return res.status(classified.status).json(classified.data);
      classification = classified.data;
      stage = classification.predictedStage || classification.stage || '';
      subCategory = classification.predictedSubCategory || classification.subCategory || '';
      if (classification.outOfScope || isOutOfScopeLabel(stage, subCategory)) {
        return res.json({ question, stage, subCategory, outOfScope: true, count: 0, keywords: [], pairs: [], classification });
      }
    }

    const recommendBody = { ...options, question, confirmedStage: stage, confirmedSubCategory: subCategory };
    let recommended = await internalPost('/api/llm-pipeline/recommend', recommendBody);
    let repaired = null;
    if (recommended.status === 409 && recommended.data?.labelValidation?.suggestedSubCategory) {
      const validation = recommended.data.labelValidation;
      const nextStage = validation.suggestedStage || stage;
      const nextSubCategory = validation.suggestedSubCategory;
      recommended = await internalPost('/api/llm-pipeline/recommend', {
        ...recommendBody,
        confirmedStage: nextStage,
        confirmedSubCategory: nextSubCategory,
      });
      if (recommended.ok) {
        repaired = { from: { stage, subCategory }, to: { stage: nextStage, subCategory: nextSubCategory } };
        stage = nextStage;
        subCategory = nextSubCategory;
      }
    }
    if (!recommended.ok) return res.status(recommended.status).json(recommended.data);

    const keywords = (recommended.data.keywordCandidates || []).map(item => ({
      keyword: item.gloss || item.name,
      glossId: Number(item.glossIndex ?? item.origin),
      score: Number(item.score),
      source: item.source,
      intentRole: item.intentRole,
      intentLabel: item.intentLabel || '',
    }));
    res.json({
      question,
      stage,
      subCategory,
      outOfScope: false,
      count: keywords.length,
      keywords,
      output: keywords.map(item => [`${item.keyword}_${item.glossId}`, item.score]),
      tuples: keywords.map(item => [item.glossId, item.keyword, item.score]),
      pairs: keywords.map(item => [item.keyword, item.score]),
      idPairs: keywords.map(item => [item.glossId, item.score]),
      latencyMs: {
        total: Date.now() - startedAt,
        classify: classification?.latencyMs?.total || 0,
        recommend: recommended.data?.latencyMs?.total || 0,
      },
      repaired,
    });
  } catch (e) {
    console.error('[keywords api error]', e.message);
    res.status(500).json({ error: e.message || String(e) });
  }
});

app.post('/api/recommend', async (req, res) => {
  const { question, answer } = req.body;
  const engine = req.body.engine || 'llm-rag';
  const llmConfig = normalizeLlmConfig(req.body);
  if (!question && !answer) return res.status(400).json({ error: '의사 질문 또는 환자 답변을 입력하세요.' });

  if (engine === 'embedding-rag') {
    try {
      const result = await recommendEmbeddingRag(req.body);
      return res.json(result);
    } catch (e) {
      console.error('[embedding-rag recommend error]', e.message);
      return res.status(500).json({ error: e.message || String(e) });
    }
  }

  if (!answer) return res.status(400).json({ error: '환자 답변을 입력하세요.' });
  if (dataset.length === 0) return res.status(400).json({ error: '먼저 데이터셋을 로드하세요.' });

  const similar = findSimilarRows(question, answer, dataset);

  // 중복 제거된 예시
  const seen = new Set();
  const examples = similar.filter(r => {
    const key = `${r['환자 답변']}::${r['gloss_names']}`;
    if (seen.has(key)) return false;
    seen.add(key); return true;
  }).slice(0, 12);

  const labelText = stageLabels(dataset).map(l => {
    const examples = (l.examples || []).slice(0, 3).join(' | ');
    return `- ${l.stage} / ${l.subCategory}${examples ? ` / 예시: ${examples}` : ''}`;
  }).join('\n');

  const system = `당신은 통증의학과 문진 데이터셋과 한국 수어(KSL) 글로스 전문가입니다.
새 파이프라인에서 LLM의 역할은 의사 질문을 의료문진단계(stage)와 세부분류(subCategory)로 분류하는 것입니다.
키워드와 글로스는 서버가 로드된 데이터셋 후보 안에서만 선택합니다.
세부분류는 제공된 후보 목록에서만 선택하세요.
후보 목록의 예시 질문은 분류 기준입니다. 입력 질문과 가장 의미가 가까운 예시가 있는 세부분류를 선택하세요.
"어떻게 오셨어요", "왜 오셨어요", "어디가 불편해서 오셨어요"처럼 방문 이유나 주증상을 묻는 질문은 인사가 아니라 chief_complaint입니다.
greeting은 "안녕하세요"처럼 순수한 인사 표현에만 사용하세요.
반드시 JSON 형식으로만 응답하세요.
형식: { "stage": "의료문진단계", "subCategory": "세부분류", "reason": "선택 근거" }`;

  const user = `[세부분류 후보]
${labelText}

[추천 요청]
의사 질문: ${question || '(없음)'}
환자 답변: ${answer}

의사 질문의 의료문진단계와 세부분류만 분류하세요. 키워드와 글로스는 출력하지 마세요.`;

  try {
    const raw = await callLocalLLM(llmConfig, system, user);
    const parsed = safeParseJson(raw);
    const candidates = buildStageCandidates(dataset, parsed?.stage, parsed?.subCategory, examples);
    res.json({
      stage: parsed?.stage || '',
      subCategory: parsed?.subCategory || '',
      patientAnswer: answer,
      keyword: candidates.keywords.join(', '),
      glosses: candidates.glosses,
      reason: [
        parsed?.reason || raw,
        candidates.mappingFound ? `단계 매핑 ${candidates.rows.length}건 사용` : '예측 단계 매핑 없음: 유사 사례 후보 사용',
      ].join(' / '),
      mappingFound: candidates.mappingFound,
      recommendationSource: candidates.mappingFound ? 'stage_map' : 'similarity_fallback',
      similarCount: similar.length,
      examples: examples.slice(0, 6).map(r => ({
        subCategory: r['세부분류'],
        question: r['의사 질문(개별)'],
        answer: r['환자 답변'],
        glosses: r['gloss_names'],
      })),
    });
  } catch (e) {
    console.error('[recommend error]', e.message);
    res.status(500).json({ error: e.message || String(e) });
  }
});

app.post('/api/test-llm', async (req, res) => {
  const llmConfig = normalizeLlmConfig(req.body);
  try {
    const result = await callLocalLLM(llmConfig, 'You are a helpful assistant.', '"OK"라고만 답하세요.');
    res.json({ ok: true, response: result });
  } catch (e) {
    console.error('[test-llm error]', e.message);
    res.status(401).json({ error: e.message });
  }
});

const PORT = Number(process.env.PORT || 3100);
app.listen(PORT, () => {
  console.log(`\n✅  Gloss Recommender →  http://localhost:${PORT}\n`);
});
