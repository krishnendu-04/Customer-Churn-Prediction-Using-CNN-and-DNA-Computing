import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
} from "firebase/auth";
import {
  doc, setDoc, getDoc, serverTimestamp,
} from "firebase/firestore";
import { auth, db } from "./firebase";

/* ── FONTS ───────────────────────────────────────────────────────────────────── */
const _fl = document.createElement("link");
_fl.rel = "stylesheet";
_fl.href = "https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&family=DM+Sans:wght@400;500;600&display=swap";
document.head.appendChild(_fl);

/* ── TOKENS ──────────────────────────────────────────────────────────────────── */
const C = {
  // Backgrounds
  bg: "#f5f2ee",   // warm off-white
  paper: "#faf9f7",   // pure paper
  panel: "#ede9e3",   // slightly warm grey
  card: "#ffffff",   // pure white cards
  rule: "#e2ddd6",   // warm border
  dim: "#c8c2b8",   // muted border
  // Typography
  ink: "#1a1714",   // near-black
  body: "#4a4540",   // warm body text
  muted: "#8a837a",   // muted text
  faint: "#b8b0a7",   // very faint
  white: "#ffffff",
  // Accents
  coral: "#e8442a",   // primary accent — vermillion coral
  amber: "#d97706",   // amber warning
  teal: "#0d7377",   // safe/positive
  blue: "#1d4ed8",   // info
  violet: "#7c3aed",   // secondary
  // Fonts
  D: "'Fraunces',serif",        // display — elegant serif
  M: "'DM Mono',monospace",     // mono
  S: "'DM Sans',sans-serif",    // body sans
};

/* Firebase auth + Firestore — config in src/firebase.js */

/* ── GLOBAL CSS ──────────────────────────────────────────────────────────────── */
const _gs = document.createElement("style");
_gs.textContent = `
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html,body{background:${C.bg};color:${C.body};font-family:${C.S};min-height:100vh;overflow-x:hidden;cursor:none}
  ::selection{background:${C.coral};color:${C.white}}
  ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:${C.dim}}::-webkit-scrollbar-track{background:${C.panel}}
  select option{background:${C.paper};color:${C.ink}}

  @keyframes ticker   {from{transform:translateX(0)}to{transform:translateX(-50%)}}
  @keyframes blink    {0%,49%,100%{opacity:1}50%,99%{opacity:0}}
  @keyframes fadeUp   {from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
  @keyframes fadeIn   {from{opacity:0}to{opacity:1}}
  @keyframes countUp  {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
  @keyframes shimmer  {0%{left:-80%}100%{left:160%}}
  @keyframes pulse    {0%,100%{opacity:.6;transform:scale(1)}50%{opacity:1;transform:scale(1.04)}}
  @keyframes float    {0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
  @keyframes ripple   {0%{transform:scale(0);opacity:.4}100%{transform:scale(5);opacity:0}}
  @keyframes glitch   {
    0%,90%,100%{transform:none;clip-path:none}
    91%{transform:translate(-3px,1px);clip-path:inset(10% 0 75% 0)}
    93%{transform:translate(3px,-1px);clip-path:inset(70% 0 10% 0)}
    95%{transform:translate(-1px,3px);clip-path:inset(42% 0 42% 0)}
    96%{transform:none;clip-path:none}
  }
  @keyframes scanline {0%{top:-10%}100%{top:110%}}
  @keyframes drawLine {from{stroke-dashoffset:1000}to{stroke-dashoffset:0}}
  @keyframes cursorBlink{0%,100%{opacity:1}50%{opacity:0}}
  @keyframes dataRain {0%{transform:translateY(-100px);opacity:0}8%{opacity:.7}88%{opacity:.5}100%{transform:translateY(110vh);opacity:0}}
  @keyframes borderPulse{0%,100%{box-shadow:0 0 0 0 ${C.coral}00}50%{box-shadow:0 0 0 2px ${C.coral}22}}

  .fu  {animation:fadeUp  .5s cubic-bezier(.22,1,.36,1) both}
  .fi  {animation:fadeIn  .4s ease both}
  .cu  {animation:countUp .7s cubic-bezier(.22,1,.36,1) both}

  .tilt3d{transition:transform .12s ease,box-shadow .12s ease;transform-style:preserve-3d;will-change:transform}

  .pill{
    display:flex;align-items:center;gap:6px;padding:6px 12px;
    cursor:none;transition:all .16s cubic-bezier(.22,1,.36,1);
    user-select:none;border:1.5px solid ${C.rule};background:${C.paper};
    font-family:${C.M};font-size:10px;position:relative;overflow:hidden;
    box-shadow:0 1px 3px rgba(0,0,0,.04);
  }
  .pill:hover{border-color:${C.dim};transform:translateY(-1px);box-shadow:0 3px 8px rgba(0,0,0,.08)}
  .pill:active{transform:scale(.97)}

  .glow-btn{
    font-family:${C.M};font-weight:500;letter-spacing:.1em;cursor:none;
    border:none;position:relative;overflow:hidden;
    transition:all .2s cubic-bezier(.22,1,.36,1);
  }
  .glow-btn::before{
    content:'';position:absolute;top:0;left:-80%;width:40%;height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,.6),transparent);
    animation:shimmer 2.4s 1.2s infinite;
  }
  .glow-btn:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(232,68,42,.3)}
  .glow-btn:active{transform:translateY(0)}

  .row-tr{transition:background .1s}
  .row-tr:hover{background:${C.panel} !important}

  .nav-btn{
    font-family:${C.M};font-size:10px;letter-spacing:.08em;cursor:none;
    background:transparent;transition:all .16s;padding:6px 13px;
  }
  .nav-btn:hover{color:${C.ink} !important;border-color:${C.dim} !important}

  /* Noise texture overlay */
  body::after{
    content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
    background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.03'/%3E%3C/svg%3E");
    background-size:200px 200px;opacity:.4;mix-blend-mode:multiply;
  }
`;
document.head.appendChild(_gs);

/* ── FEATURES ────────────────────────────────────────────────────────────────── */
// These are the 6 semantic roles the user maps their CSV columns to.
// churnOutput = the model's prediction column (probability, label, score — whatever CNN produced)
// Everything else = context/feature columns displayed as supporting analytics
const FEATURES = [
  {
    key: "customerID", label: "Customer ID", tag: "ID", color: C.blue, required: true, multi: false,
    desc: "Unique row identifier per customer", hint: "First selected column is used as the ID.",
    aliases: ["customerid", "customer_id", "userid", "user_id", "custid", "cust_id", "clientid", "client_id", "accountid", "account_id", "memberid", "member_id", "subscriberid", "subscriber_id", "contactid", "contact_id", "uuid", "recordid", "record_id", "patientid", "patient_id", "employeeid", "employee_id", "id"],
    keywords: ["customerid", "userid", "clientid", "accountid", "memberid", "contactid", "custid", "subscriberid", "id"]
  },

  {
    key: "churnOutput", label: "Churn Output", tag: "CHURN", color: C.coral, required: false, multi: false,
    desc: "The model's prediction — probability (0–1), score, or label (Yes/No/High/Low). If absent, score is derived from other mapped features.",
    hint: "One column only. Leave unmapped to use the built-in feature-based scorer.",
    aliases: ["churn", "churned", "churn_probability", "churnprobability", "churn_prob", "churnprob", "churn_score", "churnscore", "churn_prediction", "churnprediction", "churn_label", "churnlabel", "churn_flag", "churnflag", "churn_risk", "churnrisk", "churn_status", "churnstatus", "churn_value", "churnvalue", "predicted_churn", "predictedchurn", "is_churned", "ischurned", "attrition", "attrition_risk", "attritionrisk", "attrition_score", "attritionscore", "risk_score", "riskscore", "risk_level", "risklevel", "exit_probability", "exitprobability", "leave_probability", "leaveprobability", "flight_risk", "flightrisk", "cancellation_risk", "cancellationrisk", "propensity", "propensity_score", "churn_output", "churnoutput"],
    keywords: ["churn", "attrition", "propensity", "cancel", "exit", "leave", "flight", "dropout", "predict", "probability", "score", "risk", "flag", "label", "output", "status", "tier"]
  },

  {
    key: "tenure", label: "Tenure", tag: "TEN", color: C.amber, required: false, multi: true,
    desc: "How long the customer has been active", hint: "Multiple columns averaged.",
    aliases: ["tenure", "months_active", "account_age", "customer_age", "age", "months", "days_active", "subscription_age", "length_of_stay", "customer_tenure", "relationship_length", "membership_duration", "time_as_customer", "active_months", "active_days", "days_since_signup", "months_since_signup", "account_duration", "contract_length", "years_active", "years_as_customer", "customer_lifetime"],
    keywords: ["tenure", "age", "month", "day", "year", "duration", "length", "period", "lifetime", "seniority", "since", "active", "membership", "contract", "subscription", "signup", "join", "start"]
  },

  {
    key: "engagement", label: "Engagement", tag: "ENG", color: C.teal, required: false, multi: true,
    desc: "Customer activity level", hint: "Multiple columns summed.",
    aliases: ["engagement", "engagement_score", "activity", "activity_score", "logins", "login_count", "sessions", "session_count", "page_views", "pageviews", "clicks", "visits", "visit_count", "actions", "events", "interactions", "nps", "nps_score", "csat", "health_score", "healthscore", "product_usage", "usage_score", "usage_count", "dau", "mau", "wau"],
    keywords: ["engage", "login", "session", "visit", "click", "view", "action", "event", "interact", "usage", "use", "active", "activity", "frequency", "nps", "satisfaction", "csat", "health", "count", "rate", "page", "watch"]
  },

  {
    key: "inactivity", label: "Inactivity", tag: "INACT", color: C.violet, required: false, multi: true,
    desc: "Days/periods without activity", hint: "Higher values = more dormant.",
    aliases: ["inactivity", "inactive_days", "days_inactive", "days_since_last_login", "days_since_login", "days_since_last_activity", "days_since_activity", "last_active_days", "days_since_last_visit", "days_since_visit", "recency_days", "recency", "last_seen", "dormant_days", "idle_days", "absence_days", "days_since_purchase", "days_since_order"],
    keywords: ["inactiv", "dormant", "idle", "absent", "recency", "offline", "days_since", "last_login", "last_visit", "last_active"]
  },

  {
    key: "supportIssues", label: "Support Issues", tag: "SUP", color: C.amber, required: false, multi: true,
    desc: "Complaint or ticket volume", hint: "Higher = more friction.",
    aliases: ["support_tickets", "supporttickets", "ticket_count", "ticketcount", "complaints", "complaint_count", "complaintcount", "issues", "issue_count", "cases", "case_count", "escalations", "escalation_count", "open_tickets", "disputes", "refund_requests", "chargebacks", "negative_feedback", "helpdesk_tickets", "service_requests"],
    keywords: ["ticket", "complaint", "issue", "case", "support", "escalat", "dispute", "refund", "chargeback", "feedback", "helpdesk", "service", "problem"]
  },
];

/* ── LOGIC HELPERS ───────────────────────────────────────────────────────────── */
function norm(h) { return h.toLowerCase().replace(/[\s\-\.\/\\()\[\]#%*&^@!?:;"'`~]+/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, ""); }
function segs(h) { return h.replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase().split(/[_\s\-\.]+/).filter(Boolean); }
function flat(h) { return norm(h).replace(/_/g, ""); }

function matchScore(header, feat) {
  const n = norm(header), f = flat(header), sg = segs(header);
  if (feat.aliases.includes(n)) return 100;
  if (feat.aliases.some(a => a.length >= 4 && (n.includes(a) || a.includes(n)))) return 80;
  if (feat.aliases.some(a => a.length >= 4 && (f.includes(a.replace(/_/g, "")) || a.replace(/_/g, "").includes(f)))) return 70;
  if (feat.keywords.some(kw => sg.includes(kw))) return 60;
  if (feat.keywords.some(kw => kw.length >= 4 && (f.includes(kw) || kw.includes(f)))) return 40;
  if (feat.keywords.some(kw => kw.length >= 4 && sg.some(s => s.length >= 4 && (s.includes(kw) || kw.includes(s))))) return 20;
  return 0;
}
function relevantCols(headers, feat) { return headers.filter(h => matchScore(h, feat) > 0); }

function autoDetect(headers) {
  const m = {};
  FEATURES.forEach(f => {
    const scored = headers
      .map(h => ({ h, score: matchScore(h, f) }))
      .filter(x => x.score >= 40)
      .sort((a, b) => b.score - a.score);
    if (scored.length) m[f.key] = scored.map(x => x.h);
  });
  return m;
}

function parseCSV(text) {
  const records = [];
  let row = [], cell = "", quoted = false;
  const src = String(text || "").replace(/^\uFEFF/, "");
  for (let i = 0; i < src.length; i++) {
    const ch = src[i], next = src[i + 1];
    if (ch === '"') {
      if (quoted && next === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === "," && !quoted) {
      row.push(cell.trim()); cell = "";
    } else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && next === "\n") i++;
      row.push(cell.trim()); cell = "";
      if (row.some(v => v !== "")) records.push(row);
      row = [];
    } else {
      cell += ch;
    }
  }
  row.push(cell.trim());
  if (row.some(v => v !== "")) records.push(row);
  if (records.length < 2) return { headers: [], rows: [] };
  const headers = records[0].map(h => h.trim());
  const rows = records.slice(1).map(vals => {
    const obj = {}; headers.forEach((h, i) => { obj[h] = vals[i] ?? ""; }); return obj;
  }).filter(r => Object.values(r).some(v => String(v).trim() !== ""));
  return { headers, rows };
}

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

async function api(path, { token, method = "GET", body } = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({ detail: "Request failed" }))).detail || "Request failed");
  return res.json();
}

/* ── MONGODB BACKEND ─────────────────────────────────────────────────────────── */
const MONGO_API_URL = "http://localhost:5000";

/**
 * Sends the raw CSV File object to the backend via multipart/form-data.
 * The backend stores it in MongoDB GridFS — no row parsing on the server.
 */
async function saveToMongo({ csvFile, userName, userEmail, mapping, stats }) {
  try {
    const formData = new FormData();
    formData.append("csv",       csvFile, csvFile.name);
    formData.append("userName",  userName  || "");
    formData.append("userEmail", userEmail || "");
    formData.append("mapping",   JSON.stringify(mapping || {}));
    formData.append("stats",     JSON.stringify(stats   || {}));
    // NOTE: do NOT set Content-Type — the browser adds the multipart boundary automatically
    const res = await fetch(`${MONGO_API_URL}/api/upload-file`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("MongoDB save skipped (backend offline?):", err.message);
    return null;
  }
}

function toArr(v) { if (!v) return []; if (Array.isArray(v)) return v; if (v instanceof Set) return [...v]; return [v]; }
function avgC(row, cols) { const v = cols.map(c => parseFloat(row[c])).filter(n => !isNaN(n)); return v.length ? v.reduce((a, b) => a + b) / v.length : null }
function sumC(row, cols) { const v = cols.map(c => parseFloat(row[c])).filter(n => !isNaN(n)); return v.length ? v.reduce((a, b) => a + b) : null }

// Read model churn output from a row — handles probabilities, 0/1, Yes/No, High/Medium/Low labels
function readChurnOutput(val) {
  if (val === undefined || val === null || val === "") return null;
  const s = String(val).trim().toLowerCase();
  // Boolean-style labels
  if (s === "yes" || s === "true" || s === "1" || s === "churned" || s === "churn") return 1.0;
  if (s === "no" || s === "false" || s === "0" || s === "not churned" || s === "no churn") return 0.0;
  // Text risk tiers
  if (s === "high" || s === "high risk") return 0.85;
  if (s === "medium" || s === "med" || s === "medium risk" || s === "moderate") return 0.50;
  if (s === "low" || s === "low risk" || s === "safe") return 0.15;
  // Numeric — could be 0–1 probability or 0–100 score
  const n = parseFloat(val);
  if (isNaN(n)) return null;
  return n > 1 ? n / 100 : n; // normalise 0–100 to 0–1
}

// Classify a churn score into tier
function scoreTier(s) {
  if (s === null) return null;
  if (s >= 0.65) return "HIGH";
  if (s >= 0.35) return "MEDIUM";
  return "SAFE";
}

function retentionStrategy(score, row, mapping) {
  const g = k => toArr(mapping[k]);
  const support = sumC(row, g("supportIssues")) ?? 0;
  const inactivity = avgC(row, g("inactivity")) ?? 0;
  const engagement = sumC(row, g("engagement")) ?? 0;
  const tenure = avgC(row, g("tenure")) ?? 0;
  const tier = scoreTier(score);
  if (tier === "HIGH") {
    if (support >= 3) return "Priority save call, resolve open support issues, and offer a service-credit retention package.";
    if (inactivity >= 30) return "Trigger win-back outreach, usage coaching, and a time-boxed reactivation offer.";
    return "Assign an account owner, review pricing fit, and offer a targeted renewal incentive within 48 hours.";
  }
  if (tier === "MEDIUM") {
    if (engagement <= 2) return "Send product education, recommend underused features, and schedule a health-check touchpoint.";
    if (tenure <= 3) return "Strengthen onboarding with milestone nudges, setup support, and early-success check-ins.";
    return "Monitor weekly, send loyalty benefits, and test a personalized plan adjustment.";
  }
  return "Keep in nurture: reinforce value, ask for feedback, and continue normal lifecycle engagement.";
}

// ── DATASET-RELATIVE CHURN SCORER ─────────────────────────────────────────────
// Normalises each feature to [0,1] using the uploaded dataset's own min/max.
// This makes scores meaningful for ANY CSV regardless of column units or scale.
// Weights: engagement (inverted) 45 %, support issues 40 %, inactivity 15 %.
function localModelRows(rows, mapping) {
  const g = k => toArr(mapping[k]);
  const churnCol = toArr(mapping.churnOutput)[0];

  // If CSV already has a churn/probability column — read it directly, no prediction needed
  if (churnCol) {
    return rows.map(r => {
      const score = readChurnOutput(r[churnCol]) ?? 0.35;
      return {
        ...r,
        "Model Churn Score": score,
        "Risk Tier": scoreTier(score),
        "Retention Strategy": retentionStrategy(score, r, mapping),
        churn_score: score,
        risk_tier: scoreTier(score),
        retention_strategy: retentionStrategy(score, r, mapping),
      };
    });
  }

  // ── No churn column: derive from behavioural features ──────────────────────
  const engCols   = g("engagement");
  const supCols   = g("supportIssues");
  const inactCols = g("inactivity");
  const tenCols   = g("tenure");

  // Pass 1: compute min/max across ALL rows for each feature slot
  const featureStats = (cols, fn) => {
    if (!cols.length) return null;
    const vals = rows.map(r => fn(r, cols)).filter(v => v !== null && !isNaN(v));
    if (vals.length < 2) return null;
    const mn = Math.min(...vals), mx = Math.max(...vals);
    return mn === mx ? null : { min: mn, max: mx };
  };

  const eStats = featureStats(engCols,   (r, c) => sumC(r, c));
  const sStats = featureStats(supCols,   (r, c) => sumC(r, c));
  const iStats = featureStats(inactCols, (r, c) => avgC(r, c));
  const tStats = featureStats(tenCols,   (r, c) => avgC(r, c));

  // Min-max normalise: returns 0–1 where 0=min, 1=max
  const mm = (val, st) => {
    if (val === null || !st) return null;
    return (val - st.min) / (st.max - st.min);
  };

  // Pass 2: score each row
  return rows.map(r => {
    const eng   = engCols.length   ? sumC(r, engCols)   : null;
    const sup   = supCols.length   ? sumC(r, supCols)   : null;
    const inact = inactCols.length ? avgC(r, inactCols) : null;
    const ten   = tenCols.length   ? avgC(r, tenCols)   : null;

    const components = [];

    const nEng   = mm(eng,   eStats);
    const nSup   = mm(sup,   sStats);
    const nInact = mm(inact, iStats);
    const nTen   = mm(ten,   tStats);

    // HIGH engagement   → LOW risk  (inverted)
    if (nEng   !== null) components.push({ w: 0.45, v: 1 - nEng });
    // HIGH support tickets → HIGH risk
    if (nSup   !== null) components.push({ w: 0.40, v: nSup });
    // HIGH inactivity   → HIGH risk
    if (nInact !== null) components.push({ w: 0.15, v: nInact });
    // Tenure: very new = higher risk (use non-linear map, not raw percentile)
    if (nTen !== null && ten !== null) {
      const tenRisk = ten <= 3 ? 0.80 : ten <= 6 ? 0.60 : ten <= 12 ? 0.40 : 0.15;
      components.push({ w: 0.10, v: tenRisk });
    }

    let score = 0.35; // neutral default
    if (components.length) {
      const totalW = components.reduce((a, c) => a + c.w, 0);
      score = components.reduce((a, c) => a + c.v * c.w, 0) / totalW;
    }

    return {
      ...r,
      "Model Churn Score": score,
      "Risk Tier": scoreTier(score),
      "Retention Strategy": retentionStrategy(score, r, mapping),
      churn_score: score,
      risk_tier: scoreTier(score),
      retention_strategy: retentionStrategy(score, r, mapping),
    };
  });
}

// Build all analytics from mapped columns — reads model output directly, no derivation
function buildStats(rows, mapping) {
  const total = rows.length;
  const cc = toArr(mapping.churnOutput);  // the model's output column(s)
  let high = 0, medium = 0, safe = 0, unscored = 0;
  const scores = [];

  if (cc.length) {
    rows.forEach(r => {
      // Use first churnOutput column as the primary prediction
      const raw = r[cc[0]];
      const s = readChurnOutput(raw);
      if (s === null) { unscored++; return; }
      scores.push(s);
      const t = scoreTier(s);
      if (t === "HIGH") high++; else if (t === "MEDIUM") medium++; else safe++;
    });
  }

  const avgChurn = scores.length
    ? (scores.reduce((a, b) => a + b) / scores.length * 100).toFixed(1)
    : null;

  // Tenure distribution
  const tc = toArr(mapping.tenure);
  let tenureBuckets = null;
  if (tc.length) {
    const b = { "0–3": 0, "3–6": 0, "6–12": 0, "12–24": 0, "24+": 0 };
    rows.forEach(r => {
      const v = avgC(r, tc); if (v === null) return;
      if (v <= 3) b["0–3"]++; else if (v <= 6) b["3–6"]++; else if (v <= 12) b["6–12"]++; else if (v <= 24) b["12–24"]++; else b["24+"]++;
    });
    tenureBuckets = Object.entries(b).map(([l, v]) => ({ label: l, value: v }));
  }

  // Engagement distribution — split into Low/Medium/High by percentile
  const ec = toArr(mapping.engagement);
  let engBuckets = null;
  if (ec.length) {
    const vs = rows.map(r => sumC(r, ec)).filter(v => v !== null);
    if (vs.length) {
      vs.sort((a, b) => a - b);
      const p33 = vs[Math.floor(vs.length * .33)], p66 = vs[Math.floor(vs.length * .66)];
      const b = { Low: 0, Medium: 0, High: 0 };
      vs.forEach(v => { if (v <= p33) b.Low++; else if (v <= p66) b.Medium++; else b.High++; });
      engBuckets = Object.entries(b).map(([l, v]) => ({ label: l, value: v }));
    }
  }

  // Inactivity distribution
  const ic = toArr(mapping.inactivity);
  let inactBuckets = null;
  if (ic.length) {
    const vs = rows.map(r => avgC(r, ic)).filter(v => v !== null);
    if (vs.length) {
      vs.sort((a, b) => a - b);
      const p33 = vs[Math.floor(vs.length * .33)], p66 = vs[Math.floor(vs.length * .66)];
      const b = { "Active": 0, "Dormant": 0, "Gone Cold": 0 };
      vs.forEach(v => { if (v <= p33) b["Active"]++; else if (v <= p66) b["Dormant"]++; else b["Gone Cold"]++; });
      inactBuckets = Object.entries(b).map(([l, v]) => ({ label: l, value: v }));
    }
  }

  // Support issues distribution
  const sc = toArr(mapping.supportIssues);
  let supBuckets = null;
  if (sc.length) {
    const vs = rows.map(r => sumC(r, sc)).filter(v => v !== null);
    if (vs.length) {
      vs.sort((a, b) => a - b);
      const p33 = vs[Math.floor(vs.length * .33)], p66 = vs[Math.floor(vs.length * .66)];
      const b = { "None": 0, "Moderate": 0, "High": 0 };
      vs.forEach(v => { if (v <= p33) b["None"]++; else if (v <= p66) b["Moderate"]++; else b["High"]++; });
      supBuckets = Object.entries(b).map(([l, v]) => ({ label: l, value: v }));
    }
  }

  // Churn score distribution — only when numeric probabilities present
  let scoreBuckets = null;
  if (scores.length && scores.some(s => s > 0 && s < 1)) {
    const b = { "0–20%": 0, "20–40%": 0, "40–60%": 0, "60–80%": 0, "80–100%": 0 };
    scores.forEach(s => {
      if (s < .2) b["0–20%"]++; else if (s < .4) b["20–40%"]++; else if (s < .6) b["40–60%"]++; else if (s < .8) b["60–80%"]++; else b["80–100%"]++;
    });
    scoreBuckets = Object.entries(b).map(([l, v]) => ({ label: l, value: v }));
  }

  const hasRisk = cc.length > 0 && (high + medium + safe) > 0;

  return { total, high, medium, safe, unscored, avgChurn, hasRisk, tenureBuckets, engBuckets, inactBuckets, supBuckets, scoreBuckets };
}

/* ── CURSOR ──────────────────────────────────────────────────────────────────── */
function Cursor() {
  const dot = useRef(), ring = useRef(), pos = useRef({ x: 0, y: 0 }), rp = useRef({ x: 0, y: 0 }), clicked = useRef(false);
  useEffect(() => {
    const mv = e => { pos.current = { x: e.clientX, y: e.clientY }; };
    const dn = () => { clicked.current = true; }; const up = () => { clicked.current = false; };
    window.addEventListener("mousemove", mv); window.addEventListener("mousedown", dn); window.addEventListener("mouseup", up);
    let raf;
    const animate = () => {
      rp.current.x += (pos.current.x - rp.current.x) * .13; rp.current.y += (pos.current.y - rp.current.y) * .13;
      if (dot.current) dot.current.style.transform = `translate(${pos.current.x - 4}px,${pos.current.y - 4}px) scale(${clicked.current ? .4 : 1})`;
      if (ring.current) ring.current.style.transform = `translate(${rp.current.x - 16}px,${rp.current.y - 16}px) scale(${clicked.current ? 1.5 : 1})`;
      raf = requestAnimationFrame(animate);
    }; animate();
    return () => { window.removeEventListener("mousemove", mv); window.removeEventListener("mousedown", dn); window.removeEventListener("mouseup", up); cancelAnimationFrame(raf); };
  }, []);
  return (
    <>
      <div ref={dot} style={{ position: "fixed", top: 0, left: 0, width: 8, height: 8, borderRadius: "50%", background: C.coral, pointerEvents: "none", zIndex: 99999, transition: "transform .05s" }} />
      <div ref={ring} style={{ position: "fixed", top: 0, left: 0, width: 32, height: 32, borderRadius: "50%", border: `2px solid ${C.coral}`, pointerEvents: "none", zIndex: 99998, transition: "transform .06s", opacity: .7 }} />
    </>
  );
}

/* ── LIGHT MESH BACKGROUND ───────────────────────────────────────────────────── */
function LightMesh({ intensity = 1 }) {
  const canvasRef = useRef();
  const mouse = useRef({ x: .5, y: .5 });
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let W, H, raf;
    const orbs = [
      { x: .1, y: .2, r: .5, color: "#e8442a", vx: .0002, vy: .00015 },
      { x: .85, y: .15, r: .45, color: "#d97706", vx: -.00015, vy: .0002 },
      { x: .5, y: .8, r: .55, color: "#0d7377", vx: .00018, vy: -.00018 },
      { x: .25, y: .65, r: .4, color: "#7c3aed", vx: -.0002, vy: -.00015 },
      { x: .78, y: .6, r: .38, color: "#1d4ed8", vx: .00015, vy: .0002 },
    ];
    const resize = () => { W = canvas.width = canvas.offsetWidth; H = canvas.height = canvas.offsetHeight; };
    resize(); window.addEventListener("resize", resize);
    const onMouse = e => { mouse.current = { x: e.clientX / window.innerWidth, y: e.clientY / window.innerHeight }; };
    window.addEventListener("mousemove", onMouse);
    let t = 0;
    const draw = () => {
      t += .003; ctx.clearRect(0, 0, W, H);
      // warm paper base
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
      orbs.forEach((o, i) => {
        o.x += o.vx + Math.sin(t + i * .8) * .00015; o.y += o.vy + Math.cos(t + i * .6) * .00015;
        if (o.x < 0 || o.x > 1) o.vx *= -1; if (o.y < 0 || o.y > 1) o.vy *= -1;
        o.x += (mouse.current.x - o.x) * .0002; o.y += (mouse.current.y - o.y) * .0002;
        const grd = ctx.createRadialGradient(o.x * W, o.y * H, 0, o.x * W, o.y * H, o.r * Math.min(W, H) * .7);
        grd.addColorStop(0, o.color + "1a"); grd.addColorStop(.5, o.color + "0a"); grd.addColorStop(1, "transparent");
        ctx.beginPath(); ctx.ellipse(o.x * W, o.y * H, o.r * W * .45, o.r * H * .45, t * .08 + i, 0, Math.PI * 2);
        ctx.fillStyle = grd; ctx.fill();
      });
      // subtle grid
      ctx.strokeStyle = `${C.rule}`; ctx.lineWidth = .5;
      for (let x = 0; x < W; x += 80) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      for (let y = 0; y < H; y += 80) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
      raf = requestAnimationFrame(draw);
    }; draw();
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize); window.removeEventListener("mousemove", onMouse); };
  }, []);
  return <canvas ref={canvasRef} style={{ position: "fixed", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 0, opacity: intensity }} />;
}

/* ── PARTICLE NET (light) ────────────────────────────────────────────────────── */
function ParticleNet() {
  const canvasRef = useRef();
  const mouse = useRef({ x: -999, y: -999 });
  useEffect(() => {
    const canvas = canvasRef.current; const ctx = canvas.getContext("2d");
    let W, H, raf;
    const pts = [];
    const resize = () => { W = canvas.width = canvas.offsetWidth; H = canvas.height = canvas.offsetHeight; };
    resize(); window.addEventListener("resize", resize);
    const onMouse = e => { const r = canvas.getBoundingClientRect(); mouse.current = { x: e.clientX - r.left, y: e.clientY - r.top }; };
    window.addEventListener("mousemove", onMouse);
    for (let i = 0; i < 70; i++)pts.push({ x: Math.random() * 1400, y: Math.random() * 900, vx: (Math.random() - .5) * .25, vy: (Math.random() - .5) * .25, r: Math.random() * .8 + .3, alpha: Math.random() * .2 + .05 });
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      const { x: mx, y: my } = mouse.current;
      pts.forEach(p => {
        const dx = p.x - mx, dy = p.y - my, d = Math.sqrt(dx * dx + dy * dy);
        if (d < 100) { const f = (100 - d) / 100 * .4; p.vx += dx / d * f * .08; p.vy += dy / d * f * .08; }
        p.vx *= .97; p.vy *= .97; p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = W; if (p.x > W) p.x = 0; if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      });
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y, d = Math.sqrt(dx * dx + dy * dy);
          if (d < 100) { ctx.strokeStyle = `rgba(232,68,42,${(1 - d / 100) * .1})`; ctx.lineWidth = .8; ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke(); }
        }
        const d = Math.sqrt((pts[i].x - mx) ** 2 + (pts[i].y - my) ** 2);
        const a = d < 140 ? (pts[i].alpha + (1 - d / 140) * .4) : pts[i].alpha;
        ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, pts[i].r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(232,68,42,${a})`; ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    }; draw();
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize); window.removeEventListener("mousemove", onMouse); };
  }, []);
  return <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 1 }} />;
}

/* ── RIPPLE ──────────────────────────────────────────────────────────────────── */
function useRipple() {
  const [ripples, setRipples] = useState([]);
  const addRipple = useCallback(e => {
    const r = e.currentTarget.getBoundingClientRect(), id = Date.now();
    setRipples(p => [...p, { id, x: e.clientX - r.left, y: e.clientY - r.top }]);
    setTimeout(() => setRipples(p => p.filter(r => r.id !== id)), 800);
  }, []);
  const rippleEls = ripples.map(r => (
    <div key={r.id} style={{ position: "absolute", left: r.x - 20, top: r.y - 20, width: 40, height: 40, borderRadius: "50%", background: C.coral, animation: "ripple .8s ease-out forwards", pointerEvents: "none", zIndex: 10, opacity: .15 }} />
  ));
  return { addRipple, rippleEls };
}

/* ── TILT CARD ───────────────────────────────────────────────────────────────── */
function TiltCard({ children, style = {}, color = C.coral, className = "" }) {
  const ref = useRef();
  const onMove = useCallback(e => {
    if (!ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5;
    ref.current.style.transform = `perspective(800px) rotateX(${-y * 6}deg) rotateY(${x * 6}deg) translateY(-2px)`;
    ref.current.style.boxShadow = `${-x * 16}px ${-y * 16}px 32px rgba(0,0,0,.08), 0 4px 24px rgba(0,0,0,.06)`;
  }, []);
  const onLeave = useCallback(() => {
    if (!ref.current) return;
    ref.current.style.transform = "perspective(800px) rotateX(0) rotateY(0) translateY(0)";
    ref.current.style.boxShadow = "0 1px 4px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.06)";
  }, []);
  return (
    <div ref={ref} onMouseMove={onMove} onMouseLeave={onLeave} className={`tilt3d ${className}`}
      style={{ position: "relative", overflow: "hidden", boxShadow: "0 1px 4px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.06)", ...style }}>
      {children}
    </div>
  );
}

/* ── TYPEWRITER ──────────────────────────────────────────────────────────────── */
function Typewriter({ words, speed = 80 }) {
  const [idx, setIdx] = useState(0), [chars, setChars] = useState(0), [del, setDel] = useState(false);
  useEffect(() => {
    const word = words[idx % words.length];
    const t = setTimeout(() => {
      if (!del && chars < word.length) setChars(c => c + 1);
      else if (!del && chars === word.length) setTimeout(() => setDel(true), 1600);
      else if (del && chars > 0) setChars(c => c - 1);
      else { setDel(false); setIdx(i => (i + 1) % words.length); }
    }, del ? 35 : speed);
    return () => clearTimeout(t);
  }, [chars, del, idx, words, speed]);
  return (
    <span style={{ color: C.coral, fontStyle: "italic" }}>
      {words[idx % words.length].slice(0, chars)}
      <span style={{ animation: "cursorBlink 1s infinite", borderRight: `2px solid ${C.coral}` }}>&nbsp;</span>
    </span>
  );
}

/* ── TICKER ──────────────────────────────────────────────────────────────────── */
function Ticker({ to, prefix = "", suffix = "", decimals = 0 }) {
  const num = parseFloat(String(to).replace(/[^0-9.]/g, "")) || 0;
  const [val, setVal] = useState(0);
  useEffect(() => {
    setVal(0); let i = 0; const steps = 60, dur = 1200;
    const t = setInterval(() => { i++; const p = 1 - Math.pow(1 - i / steps, 4); setVal(num * p); if (i >= steps) { setVal(num); clearInterval(t); } }, dur / steps);
    return () => clearInterval(t);
  }, [num]);
  return <>{prefix}{decimals > 0 ? val.toFixed(decimals) : Math.round(val).toLocaleString()}{suffix}</>;
}

/* ── MARQUEE ─────────────────────────────────────────────────────────────────── */
function Marquee({ stats, fileName }) {
  const items = [
    `File: ${fileName || "—"}`, `Customers: ${stats?.total?.toLocaleString() || "—"}`,
    `High Risk: ${stats?.high?.toLocaleString() || "—"}`, `Medium: ${stats?.medium?.toLocaleString() || "—"}`,
    `Safe: ${stats?.safe?.toLocaleString() || "—"}`,
    stats?.avgChurn ? `Avg Churn Score: ${stats.avgChurn}%` : "Avg Churn: —",
  ].join("   ·   ");
  return (
    <div style={{ background: C.coral, color: C.white, overflow: "hidden", height: 28, display: "flex", alignItems: "center", fontFamily: C.M, fontSize: 10, letterSpacing: ".06em", whiteSpace: "nowrap" }}>
      <div style={{ display: "flex", animation: "ticker 24s linear infinite" }}>
        {[items, items].map((t, i) => <span key={i} style={{ paddingRight: 100 }}>{t}</span>)}
      </div>
    </div>
  );
}

/* ── DONUT ───────────────────────────────────────────────────────────────────── */
function Donut({ high, medium, safe, total }) {
  const [on, setOn] = useState(false), [hov, setHov] = useState(null);
  useEffect(() => { setTimeout(() => setOn(true), 400); }, []);
  const sz = 180, cx = 90, cy = 90, r = 66, sw = 18, circ = 2 * Math.PI * r;
  const segsData = [
    { pct: total ? high / total : 0, off: 0, color: C.coral, label: "High" },
    { pct: total ? medium / total : 0, off: total ? high / total : 0, color: C.amber, label: "Medium" },
    { pct: total ? safe / total : 0, off: total ? (high + medium) / total : 0, color: C.teal, label: "Safe" },
  ];
  return (
    <svg width={sz} height={sz} style={{ overflow: "visible", flexShrink: 0 }}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke={C.rule} strokeWidth={sw} />
      {segsData.map((s, i) => (
        <circle key={i} cx={cx} cy={cy} r={hov === i ? r + 3 : r} fill="none"
          stroke={s.color} strokeWidth={hov === i ? sw + 4 : sw}
          strokeDasharray={`${(on ? s.pct : 0) * circ} ${circ}`}
          strokeDashoffset={-s.off * circ}
          style={{ transform: "rotate(-90deg)", transformOrigin: `${cx}px ${cy}px`, transition: "stroke-dasharray 1.3s cubic-bezier(.22,1,.36,1)", cursor: "none", filter: hov === i ? `drop-shadow(0 0 4px ${s.color}66)` : "none" }}
          onMouseEnter={() => setHov(i)} onMouseLeave={() => setHov(null)} />
      ))}
      <circle cx={cx} cy={cy} r={r - sw / 2 - 3} fill={C.paper} />
      <text x={cx} y={cy - 6} textAnchor="middle" fill={C.muted} fontSize="8" fontFamily={C.M} letterSpacing=".12em">TOTAL</text>
      <text x={cx} y={cy + 14} textAnchor="middle" fill={hov !== null ? segsData[hov].color : C.ink} fontSize="24" fontFamily={C.D} style={{ transition: "fill .2s" }}>
        {hov !== null ? (total ? ((segsData[hov].pct) * 100).toFixed(1) + "%" : "0%") : (total >= 1000 ? (total / 1000).toFixed(0) + "K" : total)}
      </text>
      {hov !== null && <text x={cx} y={cy + 28} textAnchor="middle" fill={segsData[hov].color} fontSize="8" fontFamily={C.M}>{segsData[hov].label}</text>}
    </svg>
  );
}

/* ── BAR ─────────────────────────────────────────────────────────────────────── */
function Bar({ items, color }) {
  const [on, setOn] = useState(false), [hov, setHov] = useState(null);
  useEffect(() => { setTimeout(() => setOn(true), 250); }, []);
  const max = Math.max(...items.map(i => i.value), 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {items.map((it, i) => (
        <div key={i} className="fu" style={{ animationDelay: `${i * 70}ms` }} onMouseEnter={() => setHov(i)} onMouseLeave={() => setHov(null)}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
            <span style={{ fontFamily: C.S, fontSize: 12, color: hov === i ? C.ink : C.body, transition: "color .15s", fontWeight: 500 }}>{it.label}</span>
            <span style={{ fontFamily: C.M, fontSize: 10, color: hov === i ? color : C.muted, transition: "color .15s" }}>{it.value.toLocaleString()}</span>
          </div>
          <div style={{ height: hov === i ? 8 : 5, background: C.rule, position: "relative", overflow: "hidden", borderRadius: 99, transition: "height .2s cubic-bezier(.22,1,.36,1)" }}>
            <div style={{
              position: "absolute", top: 0, left: 0, height: "100%", borderRadius: 99,
              background: hov === i ? color : `${color}99`,
              width: on ? `${(it.value / max) * 100}%` : "0%",
              transition: `width 1s cubic-bezier(.22,1,.36,1) ${i * 70}ms, background .2s`,
            }} />
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── STAT CARD ───────────────────────────────────────────────────────────────── */
function StatCard({ label, value, sub, color, index = 0 }) {
  const isNum = !isNaN(parseFloat(String(value).replace(/[^0-9.]/g, "")));
  const { addRipple, rippleEls } = useRipple();
  return (
    <TiltCard color={color} className="fu" style={{
      animationDelay: `${index * 60}ms`, background: C.card,
      border: `1px solid ${C.rule}`, borderTop: `3px solid ${color}`,
      padding: "22px 20px", cursor: "none",
    }} onClick={addRipple}>
      {rippleEls}
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 60, background: `linear-gradient(to top,${color}06,transparent)`, pointerEvents: "none" }} />
      <div style={{ fontFamily: C.M, fontSize: 9, letterSpacing: ".16em", color: C.muted, marginBottom: 12, fontWeight: 500, textTransform: "uppercase" }}>{label}</div>
      <div className="cu" style={{ animationDelay: `${index * 60 + 100}ms`, fontFamily: C.D, fontSize: 42, color: C.ink, lineHeight: 1, letterSpacing: ".01em", fontWeight: 700 }}>
        {isNum
          ? <Ticker to={value} prefix={String(value).startsWith("$") ? "$" : ""} suffix={String(value).endsWith("%") ? "%" : String(value).endsWith("K") ? "K" : String(value).endsWith("M") ? "M" : ""} decimals={String(value).includes(".") ? 1 : 0} />
          : value}
      </div>
      {sub && <div style={{ fontFamily: C.S, fontSize: 12, color: color, marginTop: 8, fontWeight: 500 }}>{sub} of total</div>}
    </TiltCard>
  );
}

/* ── DATA DOTS (upload BG) ───────────────────────────────────────────────────── */
function DataDots() {
  const chars = "01◆◈·×+—";
  const cols = useMemo(() => Array.from({ length: 22 }, (_, i) => ({
    id: i, left: `${(i / 22) * 100 + Math.random() * 3}%`,
    delay: `${Math.random() * 6}s`, dur: `${3 + Math.random() * 4}s`,
    chars: Array.from({ length: 16 }, () => chars[Math.floor(Math.random() * chars.length)]),
    opacity: .04 + Math.random() * .06,
    color: [C.coral, C.amber, C.teal, C.violet][Math.floor(Math.random() * 4)],
    fontSize: 9 + Math.random() * 3,
  })), []);
  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden", pointerEvents: "none", zIndex: 1 }}>
      {cols.map(col => (
        <div key={col.id} style={{ position: "absolute", top: 0, left: col.left, fontFamily: C.M, fontSize: col.fontSize, color: col.color, opacity: col.opacity, display: "flex", flexDirection: "column", gap: 4, animation: `dataRain ${col.dur} ${col.delay} infinite linear`, letterSpacing: ".05em" }}>
          {col.chars.map((ch, i) => <span key={i}>{ch}</span>)}
        </div>
      ))}
    </div>
  );
}

/* ── UPLOAD ──────────────────────────────────────────────────────────────────── */
function Upload({ onFile }) {
  const ref = useRef();
  const [drag, setDrag] = useState(false), [scanning, setScanning] = useState(false);
  const { addRipple, rippleEls } = useRipple();
  const handle = useCallback(file => {
    if (!file || !file.name.endsWith(".csv")) return;
    setScanning(true);
    const r = new FileReader();
    // Pass the raw File object so the parent can send it to GridFS later
    r.onload = e => { setTimeout(() => { setScanning(false); onFile(file, e.target.result); }, 900); };
    r.readAsText(file);
  }, [onFile]);

  return (
    <div style={{ minHeight: "calc(100vh - 52px)", display: "flex", position: "relative", overflow: "hidden" }}>
      <LightMesh intensity={1} />
      <ParticleNet />

      {scanning && (
        <div style={{ position: "fixed", inset: 0, zIndex: 200, background: "rgba(245,242,238,.88)", backdropFilter: "blur(12px)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <div style={{ animation: "float 1.4s ease-in-out infinite", textAlign: "center" }}>
            <div style={{ fontFamily: C.D, fontSize: 64, color: C.coral, letterSpacing: ".02em", marginBottom: 8, fontWeight: 300, fontStyle: "italic" }}>Reading file</div>
            <div style={{ fontFamily: C.M, fontSize: 11, color: C.muted, letterSpacing: ".2em" }}>
              PROCESSING<span style={{ animation: "blink 1s infinite" }}>_</span>
            </div>
            <div style={{ marginTop: 24, display: "flex", justifyContent: "center", gap: 6 }}>
              {[0, 1, 2, 3, 4].map(i => (
                <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: C.coral, animation: `pulse .8s ${i * .12}s infinite` }} />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Left hero */}
      <div style={{ flex: "0 0 50%", display: "flex", flexDirection: "column", justifyContent: "center", padding: "60px 56px", borderRight: `1px solid ${C.rule}`, position: "relative", zIndex: 10 }}>
        <div className="fu" style={{ fontFamily: C.M, fontSize: 9, letterSpacing: ".28em", color: C.coral, marginBottom: 20, textTransform: "uppercase" }}>
          Churn Risk Analytics
        </div>
        <h1 className="fu" style={{ fontFamily: C.D, fontSize: "clamp(60px,8vw,100px)", color: C.ink, lineHeight: .88, letterSpacing: "-.01em", marginBottom: 20, animationDelay: "70ms", fontWeight: 300 }}>
          Know<br />who's<br /><em style={{ display: "inline-block", animation: "glitch 7s infinite", color: C.coral, fontStyle: "italic" }}>leaving.</em>
        </h1>
        <div className="fu" style={{ fontFamily: C.D, fontSize: 24, color: C.body, marginBottom: 28, animationDelay: "140ms", fontWeight: 300 }}>
          <Typewriter words={["Predict churn.", "Retain customers.", "Analyse risk.", "Act faster."]} speed={85} />
        </div>
        <p className="fu" style={{ fontFamily: C.S, fontSize: 14, color: C.muted, maxWidth: 340, lineHeight: 1.75, marginBottom: 36, animationDelay: "200ms" }}>
          Upload any customer CSV, map your columns, get instant risk intelligence — no configuration required.
        </p>
        <div className="fu" style={{ display: "flex", gap: 8, flexWrap: "wrap", animationDelay: "280ms" }}>
          {["CSV Upload", "Column Mapping", "Risk Scoring", "Visual Analytics"].map((f, i) => (
            <div key={i} style={{ fontFamily: C.M, fontSize: 9, color: C.muted, border: `1px solid ${C.rule}`, padding: "4px 10px", letterSpacing: ".08em", background: C.paper, animation: "borderPulse 3s infinite", animationDelay: `${i * 600}ms` }}>
              {f}
            </div>
          ))}
        </div>
      </div>

      {/* Right drop zone */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "60px 48px", position: "relative", zIndex: 10 }}>
        <div
          onDragOver={e => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }}
          onClick={e => { if (!scanning) { addRipple(e); ref.current.click(); } }}
          style={{
            border: `2px ${drag ? "solid" : "dashed"} ${drag ? C.coral : C.dim}`,
            padding: "56px 36px", textAlign: "center", cursor: "none",
            background: drag ? `${C.coral}08` : C.paper,
            transition: "all .3s cubic-bezier(.22,1,.36,1)",
            boxShadow: drag ? `0 0 40px ${C.coral}18,inset 0 0 40px ${C.coral}06` : "0 2px 8px rgba(0,0,0,.04)",
            transform: drag ? "scale(1.02)" : "scale(1)",
            position: "relative", overflow: "hidden",
          }}>
          <input ref={ref} type="file" accept=".csv" style={{ display: "none" }} onChange={e => handle(e.target.files[0])} />
          {rippleEls}
          <div style={{ fontFamily: C.D, fontSize: 56, color: drag ? C.coral : C.ink, marginBottom: 10, letterSpacing: "-.01em", transition: "color .2s", animation: drag ? "pulse .7s infinite" : "float 5s ease-in-out infinite", fontWeight: drag ? 700 : 300, fontStyle: "italic" }}>
            {drag ? "Drop it!" : "Upload"}
          </div>
          <div style={{ fontFamily: C.M, fontSize: 11, color: C.muted, marginBottom: 28, letterSpacing: ".04em" }}>
            {drag ? "Release to import" : "Drag CSV here or click to browse"}
          </div>
          <button className="glow-btn" style={{ fontSize: 12, background: C.coral, color: C.white, padding: "12px 36px", letterSpacing: ".1em", borderRadius: 2, boxShadow: `0 4px 20px ${C.coral}33` }}
            onClick={e => { e.stopPropagation(); ref.current.click(); }}>
            Select File
          </button>
        </div>

        <div className="fu" style={{ marginTop: 20, padding: "14px 18px", border: `1px solid ${C.rule}`, background: C.paper, animationDelay: "320ms", boxShadow: "0 1px 4px rgba(0,0,0,.04)" }}>
          <div style={{ fontFamily: C.M, fontSize: 9, letterSpacing: ".16em", color: C.muted, marginBottom: 8, textTransform: "uppercase" }}>Recognised Columns</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {["CustomerID", "Tenure", "MonthlyCharges", "LoginCount", "DaysSinceLastLogin", "SupportTickets", "Churn"].map((col, i) => (
              <span key={col} style={{ fontFamily: C.M, fontSize: 9, color: C.muted, border: `1px solid ${C.rule}`, padding: "3px 8px", letterSpacing: ".04em", background: C.bg, transition: "all .18s", cursor: "default" }}
                onMouseEnter={e => { e.target.style.borderColor = C.coral; e.target.style.color = C.coral; }}
                onMouseLeave={e => { e.target.style.borderColor = C.rule; e.target.style.color = C.muted; }}>
                {col}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── FEATURE ROW ─────────────────────────────────────────────────────────────── */
function FeatureRow({ feat, fi, headers, checked, isNone, toggle, toggleNone }) {
  const [showRest, setShowRest] = useState(false);
  const suggested = relevantCols(headers, feat);
  const suggestedSet = new Set(suggested);
  const rest = headers.filter(h => !suggestedSet.has(h));
  const done = checked.size > 0 || isNone;
  return (
    <div className="fu" style={{
      animationDelay: `${fi * 80}ms`,
      borderTop: `1px solid ${C.rule}`, padding: "22px 0",
      borderLeft: done ? `3px solid ${feat.color}` : `3px solid transparent`,
      paddingLeft: done ? 20 : 0,
      transition: "border-color .4s, padding-left .4s, background .4s",
      background: done ? `${feat.color}05` : "transparent",
    }}>
      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 24, alignItems: "flex-start" }}>
        {/* Left label */}
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
            <span style={{ fontFamily: C.M, fontSize: 9, letterSpacing: ".08em", background: feat.color, color: C.white, padding: "2px 8px", fontWeight: 500 }}>{feat.tag}</span>
            {feat.required && <span style={{ fontFamily: C.M, fontSize: 9, color: C.coral }}>required</span>}
            {done && <span style={{ fontFamily: C.M, fontSize: 9, color: feat.color, animation: "pulse 2.5s infinite" }}>✓</span>}
          </div>
          <div style={{ fontFamily: C.D, fontSize: 20, color: done ? feat.color : C.ink, letterSpacing: "-.01em", marginBottom: 3, transition: "color .3s", fontWeight: done ? 700 : 300, fontStyle: "italic" }}>{feat.label}</div>
          <div style={{ fontFamily: C.S, fontSize: 11, color: C.muted, lineHeight: 1.5 }}>{feat.desc}</div>
          {checked.size > 1 && <div style={{ fontFamily: C.M, fontSize: 9, color: feat.color, marginTop: 4 }}>{feat.hint}</div>}
          {suggested.length > 0 && (
            <div style={{ fontFamily: C.M, fontSize: 9, color: feat.color, marginTop: 8, display: "flex", alignItems: "center", gap: 5, opacity: .75 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: feat.color, display: "inline-block", flexShrink: 0 }} />
              {suggested.length} auto-detected
            </div>
          )}
        </div>

        {/* Right pills */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingTop: 4 }}>

          {/* Suggested — always visible, highlighted */}
          {suggested.length > 0 && (
            <div>
              <div style={{ fontFamily: C.M, fontSize: 8, letterSpacing: ".12em", color: feat.color, marginBottom: 6, opacity: .8 }}>
                SUGGESTED — {suggested.length} MATCHED
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {suggested.map(col => {
                  const ck = checked.has(col);
                  return (
                    <div key={col} className="pill" onClick={() => !isNone && toggle(feat.key, col)}
                      style={{ border: `1.5px solid ${ck ? feat.color : feat.color + "55"}`, background: ck ? `${feat.color}15` : `${feat.color}06`, opacity: isNone ? .25 : 1, boxShadow: ck ? `0 2px 10px ${feat.color}25` : "none" }}>
                      <div style={{ width: 9, height: 9, border: `1.5px solid ${ck ? feat.color : feat.color + "77"}`, background: ck ? feat.color : "transparent", borderRadius: 2, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, transition: "all .15s" }}>
                        {ck && <span style={{ color: C.white, fontSize: 7, fontWeight: 900, lineHeight: 1 }}>✓</span>}
                      </div>
                      <span style={{ color: ck ? feat.color : C.body, transition: "color .12s", fontWeight: ck ? 600 : 400 }}>{col}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* All other columns */}
          <div>
            <div style={{ fontFamily: C.M, fontSize: 8, letterSpacing: ".12em", color: C.muted, marginBottom: 6, display: "flex", alignItems: "center", gap: 8 }}>
              {suggested.length > 0
                ? <>
                  <span>ALL OTHER COLUMNS — {rest.length}</span>
                  <button onClick={() => setShowRest(p => !p)}
                    style={{ fontFamily: C.M, fontSize: 8, color: C.muted, background: "transparent", border: `1px solid ${C.rule}`, padding: "1px 8px", cursor: "none", letterSpacing: ".06em", transition: "all .15s", borderRadius: 2 }}
                    onMouseEnter={e => e.currentTarget.style.color = feat.color}
                    onMouseLeave={e => e.currentTarget.style.color = C.muted}>
                    {showRest ? "HIDE" : "SHOW"}
                  </button>
                </>
                : <span style={{ color: C.amber }}>ALL COLUMNS — {headers.length} &nbsp;·&nbsp; No auto-match, select manually</span>
              }
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
              {/* nothing pill */}
              <div className="pill" onClick={() => toggleNone(feat.key)}
                style={{ border: `1.5px solid ${isNone ? C.muted : C.rule}`, background: isNone ? C.panel : C.paper, opacity: checked.size > 0 ? .4 : 1 }}>
                <div style={{ width: 9, height: 9, border: `1.5px solid ${isNone ? C.muted : C.dim}`, background: isNone ? C.muted : "transparent", borderRadius: 2, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                  {isNone && <span style={{ color: C.white, fontSize: 7, fontWeight: 900, lineHeight: 1 }}>✓</span>}
                </div>
                <span style={{ color: isNone ? C.body : C.muted, fontStyle: "italic" }}>none</span>
              </div>
              {/* show rest: always if no suggestions, toggled if suggestions exist */}
              {(suggested.length === 0 || showRest) && rest.map(col => {
                const ck = checked.has(col);
                return (
                  <div key={col} className="pill" onClick={() => !isNone && toggle(feat.key, col)}
                    style={{ border: `1.5px solid ${ck ? feat.color : C.rule}`, background: ck ? `${feat.color}12` : C.paper, opacity: isNone ? .25 : 1, boxShadow: ck ? `0 2px 8px ${feat.color}22` : "none" }}>
                    <div style={{ width: 9, height: 9, border: `1.5px solid ${ck ? feat.color : C.dim}`, background: ck ? feat.color : "transparent", borderRadius: 2, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, transition: "all .15s" }}>
                      {ck && <span style={{ color: C.white, fontSize: 7, fontWeight: 900, lineHeight: 1 }}>✓</span>}
                    </div>
                    <span style={{ color: ck ? feat.color : C.muted, transition: "color .12s" }}>{col}</span>
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

/* ── MAP STEP ────────────────────────────────────────────────────────────────── */
function MapStep({ headers, initialMapping, fileName, onConfirm }) {
  const [mapping, setMapping] = useState(() => { const m = {}; FEATURES.forEach(f => { m[f.key] = new Set(initialMapping[f.key] || []); }); return m; });
  const [nothing, setNothing] = useState(() => new Set());
  const toggle = (fk, col) => {
    setMapping(p => { const n = { ...p }; const s = new Set(n[fk]); s.has(col) ? s.delete(col) : s.add(col); n[fk] = s; return n; });
    setNothing(p => { const n = new Set(p); n.delete(fk); return n; });
  };
  const toggleNone = fk => {
    setNothing(p => { const n = new Set(p); if (n.has(fk)) n.delete(fk); else { n.add(fk); setMapping(pm => ({ ...pm, [fk]: new Set() })); } return n; });
  };
  const ok = FEATURES.filter(f => f.required).every(f => mapping[f.key].size > 0 || nothing.has(f.key));
  const total = Object.values(mapping).reduce((a, s) => a + s.size, 0);

  return (
    <div style={{ maxWidth: 940, margin: "0 auto", padding: "44px 24px 64px", position: "relative", background: C.bg, minHeight: "calc(100vh - 52px)" }}>
      <LightMesh intensity={.4} />
      <div style={{ position: "relative", zIndex: 1 }}>
        <div className="fu" style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 36, paddingBottom: 20, borderBottom: `1px solid ${C.rule}` }}>
          <div>
            <div style={{ fontFamily: C.M, fontSize: 9, letterSpacing: ".22em", color: C.coral, marginBottom: 8, textTransform: "uppercase" }}>Step 02 / 03</div>
            <h2 style={{ fontFamily: C.D, fontSize: 52, color: C.ink, lineHeight: .95, letterSpacing: "-.01em", fontWeight: 300, fontStyle: "italic" }}>Map Columns</h2>
            <div style={{ fontFamily: C.M, fontSize: 11, color: C.muted, marginTop: 8 }}>
              <span style={{ color: C.blue, fontWeight: 500 }}>{fileName}</span> — {headers.length} columns detected
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontFamily: C.D, fontSize: 36, color: total > 0 ? C.coral : C.muted, lineHeight: 1, transition: "color .3s", fontWeight: 700 }}>{total}</div>
            <div style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".1em" }}>MAPPED</div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {FEATURES.map((feat, fi) => (
            <FeatureRow key={feat.key} feat={feat} fi={fi} headers={headers}
              checked={mapping[feat.key]} isNone={nothing.has(feat.key)}
              toggle={toggle} toggleNone={toggleNone} />
          ))}
        </div>

        <div style={{ marginTop: 36, paddingTop: 24, borderTop: `1px solid ${C.rule}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontFamily: C.M, fontSize: 11, color: ok ? C.teal : C.coral }}>
            {ok ? "✓ Ready to analyse" : `Missing: ${FEATURES.filter(f => f.required && mapping[f.key].size === 0 && !nothing.has(f.key)).map(f => f.label).join(", ")}`}
          </div>
          <button className="glow-btn" onClick={() => ok && onConfirm(mapping, nothing)}
            style={{ fontSize: 12, background: ok ? C.coral : C.rule, color: ok ? C.white : C.muted, padding: "13px 40px", letterSpacing: ".1em", borderRadius: 2, boxShadow: ok ? `0 4px 24px ${C.coral}33` : "none", cursor: ok ? "none" : "not-allowed" }}>
            Run Analysis →
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── TABLE ───────────────────────────────────────────────────────────────────── */
function Table({ rows, mapping }) {
  const [search, setSearch] = useState(""), [filter, setFilter] = useState("ALL"), [page, setPage] = useState(0);
  const PER = 10;
  const g = k => toArr(mapping[k]);
  const g1 = k => { const v = g(k); return v.length ? v[0] : null; };
  const churnCols = g("churnOutput");
  const idCol = g1("customerID");
  const tenCols = g("tenure"), engCols = g("engagement"), inactCols = g("inactivity"), supCols = g("supportIssues");

  // Read churn score directly from the mapped model output column
  const getScore = r => {
    if (!churnCols.length) return null;
    return readChurnOutput(r[churnCols[0]]);
  };
  const getTier = r => {
    const s = getScore(r);
    return scoreTier(s) || "SAFE";
  };
  const showRisk = churnCols.length > 0;
  const tierColor = t => t === "HIGH" ? C.coral : t === "MEDIUM" ? C.amber : C.teal;

  const filtered = rows.filter(r => {
    const matchSearch = !search || (idCol && (r[idCol] || "").toLowerCase().includes(search.toLowerCase()));
    const matchFilter = filter === "ALL" || getTier(r) === filter;
    return matchSearch && matchFilter;
  });
  const pages = Math.ceil(filtered.length / PER);
  const visible = filtered.slice(page * PER, (page + 1) * PER);

  const cols = [
    idCol && { label: "Customer ID", render: r => <span style={{ fontFamily: C.M, fontSize: 11, color: C.blue, fontWeight: 500 }}>{r[idCol] || "—"}</span> },
    showRisk && { label: "Risk", render: r => { const t = getTier(r), col = tierColor(t); return <span style={{ fontFamily: C.M, fontSize: 9, color: col, background: `${col}12`, border: `1px solid ${col}33`, padding: "2px 8px", borderRadius: 2, letterSpacing: ".04em" }}>{t}</span>; } },
    showRisk && { label: "Score", render: r => { const s = getScore(r), col = s === null ? C.muted : s >= .65 ? C.coral : s >= .35 ? C.amber : C.teal; return <span style={{ fontFamily: C.M, fontSize: 11, color: col, fontWeight: 500 }}>{s === null ? "—" : (s * 100).toFixed(1) + "%"}</span>; } },
    tenCols.length && { label: "Tenure", render: r => { const v = avgC(r, tenCols); return <span style={{ fontFamily: C.M, fontSize: 11, color: C.amber }}>{v === null ? "—" : v.toFixed(1)}</span>; } },
    engCols.length && { label: "Engagement", render: r => { const v = sumC(r, engCols); return <span style={{ fontFamily: C.M, fontSize: 11, color: C.teal }}>{v === null ? "—" : v.toFixed(1)}</span>; } },
    inactCols.length && { label: "Inactivity", render: r => { const v = avgC(r, inactCols); return <span style={{ fontFamily: C.M, fontSize: 11, color: C.violet }}>{v === null ? "—" : v.toFixed(1)}</span>; } },
    supCols.length && { label: "Support", render: r => { const v = sumC(r, supCols); return <span style={{ fontFamily: C.M, fontSize: 11, color: C.amber }}>{v === null ? "—" : v.toFixed(0)}</span>; } },
  ].filter(Boolean);
  if (rows.some(r => r.retention_strategy || r["Retention Strategy"])) {
    cols.push({ label: "Strategy", render: r => <span style={{ display: "block", minWidth: 260, fontSize: 11, lineHeight: 1.45, color: C.body }}>{r.retention_strategy || r["Retention Strategy"] || "â€”"}</span> });
  }

  const TH = { fontFamily: C.M, fontSize: 9, letterSpacing: ".1em", color: C.muted, padding: "10px 16px", textAlign: "left", borderBottom: `1px solid ${C.rule}`, background: C.panel, fontWeight: 500 };
  const TD = { padding: "11px 16px", borderBottom: `1px solid ${C.rule}`, fontSize: 13, color: C.body };

  return (
    <TiltCard color={C.coral} className="fu" style={{ border: `1px solid ${C.rule}`, background: C.card, borderRadius: 2 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 20px", borderBottom: `1px solid ${C.rule}` }}>
        <div style={{ fontFamily: C.D, fontSize: 20, color: C.ink, letterSpacing: "-.01em", fontWeight: 300, fontStyle: "italic" }}>Customer Records</div>
        <div style={{ display: "flex", gap: 8 }}>
          {idCol && (
            <input value={search} onChange={e => { setSearch(e.target.value); setPage(0); }} placeholder="Search ID..."
              style={{ fontFamily: C.M, fontSize: 11, background: C.bg, border: `1px solid ${C.rule}`, color: C.ink, padding: "7px 12px", outline: "none", letterSpacing: ".02em", width: 180, borderRadius: 2 }}
              onFocus={e => { e.target.style.borderColor = C.coral; }} onBlur={e => { e.target.style.borderColor = C.rule; }} />
          )}
          {showRisk && (
            <select value={filter} onChange={e => { setFilter(e.target.value); setPage(0); }}
              style={{ fontFamily: C.M, fontSize: 11, background: C.bg, border: `1px solid ${C.rule}`, color: C.ink, padding: "7px 12px", outline: "none", cursor: "none", borderRadius: 2 }}>
              {["ALL", "HIGH", "MEDIUM", "SAFE"].map(o => <option key={o}>{o}</option>)}
            </select>
          )}
        </div>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead><tr>{cols.map((c, i) => <th key={i} style={TH}>{c.label}</th>)}</tr></thead>
          <tbody>
            {visible.length === 0
              ? <tr><td colSpan={cols.length} style={{ ...TD, textAlign: "center", padding: 44, color: C.muted, fontFamily: C.M, fontSize: 11, letterSpacing: ".1em" }}>No records found</td></tr>
              : visible.map((r, i) => (
                <tr key={i} className="row-tr" style={{ background: i % 2 === 0 ? C.card : C.bg }}>
                  {cols.map((c, j) => <td key={j} style={TD}>{c.render(r)}</td>)}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 20px", borderTop: `1px solid ${C.rule}`, background: C.panel }}>
          <span style={{ fontFamily: C.M, fontSize: 10, color: C.muted, letterSpacing: ".04em" }}>{filtered.length.toLocaleString()} records · page {page + 1} of {pages}</span>
          <div style={{ display: "flex", gap: 3 }}>
            <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
              style={{ fontFamily: C.M, fontSize: 11, padding: "4px 10px", background: "transparent", border: `1px solid ${C.rule}`, color: page === 0 ? C.faint : C.body, cursor: page === 0 ? "not-allowed" : "none", borderRadius: 2 }}>‹</button>
            {[...Array(Math.min(pages, 7))].map((_, i) => {
              const t = Math.max(0, Math.min(pages - 7, page - 3)) + i;
              return <button key={t} onClick={() => setPage(t)} style={{ fontFamily: C.M, fontSize: 10, padding: "4px 9px", background: t === page ? C.coral : "transparent", border: `1px solid ${t === page ? C.coral : C.rule}`, color: t === page ? C.white : C.body, cursor: "none", borderRadius: 2, transition: "all .15s" }}>{t + 1}</button>;
            })}
            <button onClick={() => setPage(p => Math.min(pages - 1, p + 1))} disabled={page === pages - 1}
              style={{ fontFamily: C.M, fontSize: 11, padding: "4px 10px", background: "transparent", border: `1px solid ${C.rule}`, color: page === pages - 1 ? C.faint : C.body, cursor: page === pages - 1 ? "not-allowed" : "none", borderRadius: 2 }}>›</button>
          </div>
        </div>
      )}
    </TiltCard>
  );
}

/* ── ANALYTICS ───────────────────────────────────────────────────────────────── */
function Analytics({ rows, mapping }) {
  const mappingKey = useMemo(() => JSON.stringify(mapping), [mapping]);
  const stats = useMemo(() => buildStats(rows, mapping), [rows, mappingKey]);
  const hasTen = !!stats.tenureBuckets, hasEng = !!stats.engBuckets;
  const hasInact = !!stats.inactBuckets, hasSup = !!stats.supBuckets, hasScore = !!stats.scoreBuckets;
  const charts = [stats.hasRisk, hasTen, hasEng, hasInact, hasSup, hasScore].filter(Boolean).length;
  const chartCols = charts >= 3 ? "1fr 1fr 1fr" : charts === 2 ? "1fr 1fr" : "1fr";

  const cards = [
    { label: "Total Customers", value: stats.total, color: C.blue },
    stats.hasRisk && { label: "High Risk", value: stats.high, color: C.coral, sub: `${stats.total ? ((stats.high / stats.total) * 100).toFixed(1) : 0}%` },
    stats.hasRisk && { label: "Medium Risk", value: stats.medium, color: C.amber, sub: `${stats.total ? ((stats.medium / stats.total) * 100).toFixed(1) : 0}%` },
    stats.hasRisk && { label: "Safe", value: stats.safe, color: C.teal, sub: `${stats.total ? ((stats.safe / stats.total) * 100).toFixed(1) : 0}%` },
    stats.avgChurn != null && { label: "Avg Churn Score", value: stats.avgChurn + "%", color: C.coral },
  ].filter(Boolean);

  const ChartCard = ({ color, label, children, delay = "0ms" }) => (
    <TiltCard color={color} className="fu" style={{ background: C.card, padding: 24, borderTop: `3px solid ${color}`, border: `1px solid ${C.rule}`, animationDelay: delay, borderRadius: 2 }}>
      <div style={{ fontFamily: C.M, fontSize: 9, letterSpacing: ".14em", color: C.muted, marginBottom: 16, textTransform: "uppercase" }}>{label}</div>
      {children}
    </TiltCard>
  );
  const strategyRows = ["HIGH", "MEDIUM", "SAFE"].map(t => {
    const match = rows.find(r => (r.risk_tier || r["Risk Tier"]) === t && (r.retention_strategy || r["Retention Strategy"]));
    return match && { tier: t, strategy: match.retention_strategy || match["Retention Strategy"], color: t === "HIGH" ? C.coral : t === "MEDIUM" ? C.amber : C.teal };
  }).filter(Boolean);

  return (
    <div style={{ padding: "0 0 64px" }}>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(cards.length, 3)},1fr)`, gap: 1, marginBottom: 1 }}>
        {cards.map((c, i) => <StatCard key={i} {...c} index={i} />)}
      </div>
      {strategyRows.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${strategyRows.length},1fr)`, gap: 1, marginBottom: 1 }}>
          {strategyRows.map((s, i) => (
            <TiltCard key={s.tier} color={s.color} className="fu" style={{ background: C.card, padding: 18, border: `1px solid ${C.rule}`, borderTop: `3px solid ${s.color}`, borderRadius: 2, animationDelay: `${i * 70}ms` }}>
              <div style={{ fontFamily: C.M, fontSize: 9, letterSpacing: ".14em", color: s.color, marginBottom: 8 }}>{s.tier} RISK PLAYBOOK</div>
              <div style={{ fontSize: 12, lineHeight: 1.6, color: C.body }}>{s.strategy}</div>
            </TiltCard>
          ))}
        </div>
      )}
      {charts > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: chartCols, gap: 1, marginBottom: 1 }}>
          {stats.hasRisk && (
            <ChartCard color={C.coral} label="Risk Distribution">
              <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
                <Donut high={stats.high} medium={stats.medium} safe={stats.safe} total={stats.total} />
                <div style={{ flex: 1 }}>
                  {[{ l: "High", c: C.coral, n: stats.high }, { l: "Medium", c: C.amber, n: stats.medium }, { l: "Safe", c: C.teal, n: stats.safe }].map((x, i) => (
                    <div key={i} style={{ marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontFamily: C.S, fontSize: 12, color: C.body, fontWeight: 500 }}>{x.l}</span>
                        <span style={{ fontFamily: C.M, fontSize: 10, color: x.c }}>{stats.total ? ((x.n / stats.total) * 100).toFixed(1) : 0}%</span>
                      </div>
                      <div style={{ height: 4, background: C.rule, overflow: "hidden", borderRadius: 99 }}>
                        <div style={{ height: "100%", width: `${stats.total ? (x.n / stats.total) * 100 : 0}%`, background: x.c, transition: "width 1.2s cubic-bezier(.22,1,.36,1)", borderRadius: 99 }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </ChartCard>
          )}
          {hasScore && <ChartCard color={C.coral} label="Churn Score Distribution"><Bar items={stats.scoreBuckets} color={C.coral} /></ChartCard>}
          {hasTen && <ChartCard color={C.amber} label="Tenure Distribution" delay="80ms"><Bar items={stats.tenureBuckets} color={C.amber} /></ChartCard>}
          {hasEng && <ChartCard color={C.teal} label="Engagement Distribution" delay="160ms"><Bar items={stats.engBuckets} color={C.teal} /></ChartCard>}
          {hasInact && <ChartCard color={C.violet} label="Inactivity Levels" delay="240ms"><Bar items={stats.inactBuckets} color={C.violet} /></ChartCard>}
          {hasSup && <ChartCard color={C.amber} label="Support Issue Load" delay="320ms"><Bar items={stats.supBuckets} color={C.amber} /></ChartCard>}
        </div>
      )}
      <div style={{ marginTop: 1 }}><Table rows={rows} mapping={mapping} /></div>
    </div>
  );
}

/* ── APP ─────────────────────────────────────────────────────────────────────── */
function LoginScreen({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [industry, setIndustry] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async e => {
    e.preventDefault(); setLoading(true); setError("");
    try {
      if (mode === "signup") {
        // 1. Create Firebase Auth user
        const cred = await createUserWithEmailAndPassword(auth, email, password);
        const uid = cred.user.uid;
        // 2. Save profile to Firestore users/{uid}
        await setDoc(doc(db, "users", uid), {
          name, companyName, industry, email,
          role: "company",
          createdAt: serverTimestamp(),
        });
        onLogin({ uid, user: { uid, email, name, companyName, industry, role: "company" } });
      } else {
        // 1. Sign in with Firebase
        const cred = await signInWithEmailAndPassword(auth, email, password);
        const uid = cred.user.uid;
        // 2. Fetch profile from Firestore
        const snap = await getDoc(doc(db, "users", uid));
        const profile = snap.exists() ? snap.data() : {};
        onLogin({ uid, user: { uid, email, ...profile } });
      }
    } catch (err) {
      const msg = err.code === "auth/user-not-found" || err.code === "auth/wrong-password"
        ? "Incorrect email or password."
        : err.code === "auth/email-already-in-use"
          ? "This email is already registered. Log in instead."
          : err.code === "auth/weak-password"
            ? "Password must be at least 6 characters."
            : err.message || "Something went wrong.";
      setError(msg);
    } finally { setLoading(false); }
  };

  const inp = { fontSize: 15, padding: "13px 14px", border: `1px solid ${C.rule}`, background: C.paper, color: C.ink, outline: "none", borderRadius: 2, transition: "border-color .18s" };
  const focus = e => e.target.style.borderColor = C.coral;
  const blur = e => e.target.style.borderColor = C.rule;

  return (
    <div style={{ minHeight: "100vh", display: "grid", gridTemplateColumns: "1.05fr .95fr", background: C.bg }}>
      <Cursor /><LightMesh intensity={.9} /><ParticleNet />
      {/* Left hero */}
      <div style={{ position: "relative", zIndex: 1, padding: "72px 64px", display: "flex", flexDirection: "column", justifyContent: "center", borderRight: `1px solid ${C.rule}` }}>
        <div style={{ fontFamily: C.M, fontSize: 10, letterSpacing: ".24em", color: C.coral, marginBottom: 18 }}>COMMUNO CHURN OS</div>
        <h1 style={{ fontFamily: C.D, fontSize: "clamp(58px,8vw,104px)", lineHeight: .9, color: C.ink, fontWeight: 300, letterSpacing: 0 }}>
          {mode === "signup" ? "Join" : "Welcome"}<br />
          <em style={{ color: C.coral, fontWeight: 700 }}>{mode === "signup" ? "us." : "back."}</em>
        </h1>
        <p style={{ fontSize: 15, lineHeight: 1.8, color: C.muted, maxWidth: 430, marginTop: 24 }}>
          Upload customer data, run model-backed churn scoring, and get retention strategies for every risk tier.
        </p>
        {mode === "signup" && (
          <div style={{ marginTop: 28, padding: "16px 20px", border: `1px solid ${C.rule}`, background: C.paper, borderRadius: 2 }}>
            <div style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".14em", marginBottom: 10 }}>YOUR PROFILE IS SAVED TO</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {["Company Name", "Industry", "Your Name", "Email"].map(f => (
                <span key={f} style={{ fontFamily: C.M, fontSize: 9, color: C.teal, border: `1px solid ${C.teal}44`, padding: "3px 8px", borderRadius: 2 }}>✓ {f}</span>
              ))}
            </div>
            <div style={{ fontFamily: C.M, fontSize: 9, color: C.muted, marginTop: 8 }}>Stored securely in Firestore</div>
          </div>
        )}
      </div>

      {/* Right form */}
      <form onSubmit={submit} style={{ position: "relative", zIndex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "72px 64px", gap: 14 }}>
        <button type="button" onClick={() => { setMode(mode === "signup" ? "login" : "signup"); setError(""); setEmail(""); setPassword(""); setName(""); setCompanyName(""); setIndustry(""); }}
          style={{ fontFamily: C.M, fontSize: 10, color: C.coral, background: "transparent", border: `1px solid ${C.coral}44`, padding: "9px 12px", borderRadius: 2, cursor: "none", transition: "background .18s", marginBottom: 4 }}
          onMouseEnter={e => e.currentTarget.style.background = `${C.coral}0a`}
          onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
          {mode === "signup" ? "Already have an account? Sign in" : "New company? Create account →"}
        </button>

        {mode === "signup" && (
          <>
            <label style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".12em" }}>COMPANY NAME *</label>
            <input value={companyName} onChange={e => setCompanyName(e.target.value)} required placeholder="Acme Corp" style={inp} onFocus={focus} onBlur={blur} />
            <label style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".12em" }}>INDUSTRY</label>
            <input value={industry} onChange={e => setIndustry(e.target.value)} placeholder="Telecom · SaaS · Banking · OTT..." style={inp} onFocus={focus} onBlur={blur} />
            <label style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".12em" }}>YOUR NAME *</label>
            <input value={name} onChange={e => setName(e.target.value)} required placeholder="Jane Smith" style={inp} onFocus={focus} onBlur={blur} />
          </>
        )}

        <label style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".12em" }}>EMAIL *</label>
        <input type="email" value={email} onChange={e => setEmail(e.target.value)} required placeholder="you@company.com" style={inp} onFocus={focus} onBlur={blur} />

        <label style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".12em" }}>PASSWORD *</label>
        <input type="password" value={password} onChange={e => setPassword(e.target.value)} required minLength={6} placeholder="Min 6 characters" style={inp} onFocus={focus} onBlur={blur} />

        {error && (
          <div style={{ fontSize: 12, color: C.coral, lineHeight: 1.5, padding: "10px 14px", background: `${C.coral}0a`, border: `1px solid ${C.coral}33`, borderRadius: 2 }}>
            ⚠ {error}
          </div>
        )}

        <button className="glow-btn" disabled={loading}
          style={{ marginTop: 4, fontSize: 12, background: loading ? C.dim : C.coral, color: C.white, padding: "14px 18px", borderRadius: 2, transition: "background .2s" }}>
          {loading ? "Working..." : mode === "signup" ? "Create Account & Enter" : "Sign In to Workspace"}
        </button>
      </form>
    </div>
  );
}

function AdminDashboard({ session, onLogout }) {
  const MONGO = "http://localhost:5000";

  // ── aggregate stats ──────────────────────────────────────────────────────────
  const [stats, setStats]       = useState(null);
  const [statsErr, setStatsErr] = useState("");

  // ── uploads list ─────────────────────────────────────────────────────────────
  const [uploads, setUploads]   = useState([]);
  const [total, setTotal]       = useState(0);
  const [page, setPage]         = useState(0);
  const [search, setSearch]     = useState("");
  const [listLoading, setListLoading] = useState(false);
  const [listErr, setListErr]   = useState("");

  // ── detail view ──────────────────────────────────────────────────────────────
  const [detail, setDetail]     = useState(null);   // full upload doc with rows
  const [detailLoading, setDetailLoading] = useState(false);

  const PER = 20;

  const loadStats = useCallback(async () => {
    try {
      const r = await fetch(`${MONGO}/api/admin/stats`);
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      setStats(j); setStatsErr("");
    } catch (e) { setStatsErr(e.message); }
  }, []);

  const loadList = useCallback(async (pg, srch) => {
    setListLoading(true); setListErr("");
    try {
      const q = new URLSearchParams({ page: pg, limit: PER, ...(srch ? { email: srch } : {}) });
      const r = await fetch(`${MONGO}/api/admin/uploads?${q}`);
      const j = await r.json();
      if (j.error) throw new Error(j.error);
      setUploads(j.uploads); setTotal(j.total);
    } catch (e) { setListErr(e.message); } finally { setListLoading(false); }
  }, []);

  useEffect(() => { loadStats(); }, [loadStats]);
  useEffect(() => { loadList(page, search); }, [loadList, page, search]);

  const openDetail = async (id) => {
    setDetailLoading(true); setDetail(null);
    try {
      const r = await fetch(`${MONGO}/api/uploads/${id}`);
      const j = await r.json();
      setDetail(j);
    } catch { setDetail(null); } finally { setDetailLoading(false); }
  };

  const deleteUpload = async (id) => {
    if (!window.confirm("Delete this upload permanently?")) return;
    await fetch(`${MONGO}/api/uploads/${id}`, { method: "DELETE" });
    setUploads(u => u.filter(x => x._id !== id));
    setTotal(t => t - 1);
    if (detail?._id === id) setDetail(null);
    loadStats();
  };

  const tierColor = t => t === "HIGH" ? C.coral : t === "MEDIUM" ? C.amber : C.teal;
  const fmtDate  = s => s ? new Date(s).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }) : "—";
  const pages = Math.ceil(total / PER);

  // ── Styles ───────────────────────────────────────────────────────────────────
  const TH = { fontFamily: C.M, fontSize: 9, letterSpacing: ".1em", color: C.muted, padding: "10px 14px", textAlign: "left", borderBottom: `1px solid ${C.rule}`, background: C.panel, whiteSpace: "nowrap" };
  const TD = { padding: "11px 14px", borderBottom: `1px solid ${C.rule}`, fontSize: 12, color: C.body, verticalAlign: "middle" };

  return (
    <div style={{ minHeight: "100vh", background: C.bg }}>
      <Cursor />

      {/* NAV */}
      <nav style={{ display: "flex", justifyContent: "space-between", alignItems: "center", height: 56, padding: "0 32px", background: `${C.paper}ee`, backdropFilter: "blur(16px)", borderBottom: `1px solid ${C.rule}`, position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ fontFamily: C.D, fontSize: 22, color: C.ink }}>
            Admin <span style={{ color: C.coral, fontStyle: "italic" }}>Console</span>
          </div>
          <div style={{ width: 1, height: 16, background: C.rule }} />
          <div style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".12em" }}>MongoDB Uploads Panel</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontFamily: C.M, fontSize: 9, color: C.muted }}>{session.user.email}</div>
          <button className="nav-btn" onClick={onLogout} style={{ border: `1px solid ${C.rule}`, color: C.muted, borderRadius: 2 }}>Logout</button>
        </div>
      </nav>

      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 32px 64px" }}>

        {/* ── AGGREGATE STAT CARDS ── */}
        {statsErr && (
          <div style={{ padding: "12px 16px", background: `${C.coral}0a`, border: `1px solid ${C.coral}33`, color: C.coral, fontFamily: C.M, fontSize: 11, marginBottom: 16, borderRadius: 2 }}>
            ⚠ Backend offline — start <code>node backend/server.js</code> to see live data. ({statsErr})
          </div>
        )}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 1, marginBottom: 1 }}>
          {[
            { label: "Total Uploads",  value: stats?.totalUploads ?? "—", color: C.blue },
            { label: "Total Rows",     value: stats?.totalRows    ?? "—", color: C.violet },
            { label: "Unique Users",   value: stats?.uniqueUsers  ?? "—", color: C.teal },
            { label: "High Risk",      value: stats?.high         ?? "—", color: C.coral },
            { label: "Medium Risk",    value: stats?.medium       ?? "—", color: C.amber },
            { label: "Safe",           value: stats?.safe         ?? "—", color: C.teal },
          ].map((c, i) => <StatCard key={i} {...c} index={i} />)}
        </div>

        {/* ── UPLOADS TABLE + DETAIL SPLIT ── */}
        <div style={{ display: "grid", gridTemplateColumns: detail ? "1fr 1.1fr" : "1fr", gap: 1, alignItems: "start" }}>

          {/* LEFT — upload list */}
          <TiltCard color={C.blue} style={{ background: C.card, border: `1px solid ${C.rule}`, borderRadius: 2, overflow: "hidden" }}>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 20px", borderBottom: `1px solid ${C.rule}`, background: C.panel, gap: 12, flexWrap: "wrap" }}>
              <div style={{ fontFamily: C.D, fontSize: 20, color: C.ink, fontWeight: 300, fontStyle: "italic" }}>
                All Uploads <span style={{ fontFamily: C.M, fontSize: 10, color: C.muted, fontStyle: "normal" }}>({total})</span>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input
                  value={search} placeholder="Filter by email…"
                  onChange={e => { setSearch(e.target.value); setPage(0); }}
                  style={{ fontFamily: C.M, fontSize: 11, background: C.bg, border: `1px solid ${C.rule}`, color: C.ink, padding: "7px 12px", outline: "none", width: 200, borderRadius: 2 }}
                  onFocus={e => e.target.style.borderColor = C.coral}
                  onBlur={e => e.target.style.borderColor = C.rule}
                />
                <button onClick={() => loadList(page, search)} style={{ fontFamily: C.M, fontSize: 10, padding: "7px 14px", background: C.coral, color: C.white, border: "none", borderRadius: 2, cursor: "none", letterSpacing: ".06em" }}>Refresh</button>
              </div>
            </div>

            {/* Table */}
            {listErr && <div style={{ padding: "12px 20px", fontFamily: C.M, fontSize: 11, color: C.coral }}>{listErr}</div>}
            {listLoading
              ? <div style={{ padding: 32, textAlign: "center", fontFamily: C.M, fontSize: 11, color: C.muted, letterSpacing: ".1em" }}>Loading…</div>
              : (
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        {["User", "Email", "File", "Rows", "High", "Med", "Safe", "Date", ""].map((h, i) => (
                          <th key={i} style={TH}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {uploads.length === 0
                        ? <tr><td colSpan={9} style={{ ...TD, textAlign: "center", padding: 40, color: C.muted, fontFamily: C.M, fontSize: 11 }}>No uploads found</td></tr>
                        : uploads.map((u, i) => {
                          const isActive = detail?._id === u._id;
                          return (
                            <tr key={u._id} className="row-tr" style={{ background: isActive ? `${C.blue}08` : i % 2 === 0 ? C.card : C.bg, cursor: "pointer" }}
                              onClick={() => openDetail(u._id)}>
                              <td style={{ ...TD, fontWeight: 500, color: C.ink, maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.userName || "—"}</td>
                              <td style={{ ...TD, fontFamily: C.M, fontSize: 10, color: C.blue, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.userEmail}</td>
                              <td style={{ ...TD, fontFamily: C.M, fontSize: 10, color: C.muted, maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.fileName}</td>
                              <td style={{ ...TD, fontFamily: C.M, fontSize: 11, color: C.violet, textAlign: "right" }}>{u.rowCount?.toLocaleString()}</td>
                              <td style={{ ...TD, fontFamily: C.M, fontSize: 10, color: C.coral, textAlign: "right" }}>{u.stats?.high ?? "—"}</td>
                              <td style={{ ...TD, fontFamily: C.M, fontSize: 10, color: C.amber, textAlign: "right" }}>{u.stats?.medium ?? "—"}</td>
                              <td style={{ ...TD, fontFamily: C.M, fontSize: 10, color: C.teal, textAlign: "right" }}>{u.stats?.safe ?? "—"}</td>
                              <td style={{ ...TD, fontFamily: C.M, fontSize: 9, color: C.muted, whiteSpace: "nowrap" }}>{fmtDate(u.uploadedAt)}</td>
                              <td style={{ ...TD, textAlign: "center" }}>
                                <button onClick={e => { e.stopPropagation(); deleteUpload(u._id); }}
                                  style={{ fontFamily: C.M, fontSize: 9, color: C.coral, background: `${C.coral}0d`, border: `1px solid ${C.coral}44`, padding: "3px 8px", borderRadius: 2, cursor: "none", letterSpacing: ".04em", transition: "all .15s" }}
                                  onMouseEnter={e => e.currentTarget.style.background = `${C.coral}22`}
                                  onMouseLeave={e => e.currentTarget.style.background = `${C.coral}0d`}>
                                  Del
                                </button>
                              </td>
                            </tr>
                          );
                        })
                      }
                    </tbody>
                  </table>
                </div>
              )
            }

            {/* Pagination */}
            {pages > 1 && (
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 20px", borderTop: `1px solid ${C.rule}`, background: C.panel }}>
                <span style={{ fontFamily: C.M, fontSize: 10, color: C.muted }}>{total} uploads · page {page + 1} of {pages}</span>
                <div style={{ display: "flex", gap: 3 }}>
                  <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
                    style={{ fontFamily: C.M, fontSize: 11, padding: "4px 10px", background: "transparent", border: `1px solid ${C.rule}`, color: page === 0 ? C.faint : C.body, cursor: page === 0 ? "not-allowed" : "none", borderRadius: 2 }}>‹</button>
                  {[...Array(Math.min(pages, 5))].map((_, i) => {
                    const t = Math.max(0, Math.min(pages - 5, page - 2)) + i;
                    return <button key={t} onClick={() => setPage(t)}
                      style={{ fontFamily: C.M, fontSize: 10, padding: "4px 9px", background: t === page ? C.blue : "transparent", border: `1px solid ${t === page ? C.blue : C.rule}`, color: t === page ? C.white : C.body, cursor: "none", borderRadius: 2 }}>{t + 1}</button>;
                  })}
                  <button onClick={() => setPage(p => Math.min(pages - 1, p + 1))} disabled={page === pages - 1}
                    style={{ fontFamily: C.M, fontSize: 11, padding: "4px 10px", background: "transparent", border: `1px solid ${C.rule}`, color: page === pages - 1 ? C.faint : C.body, cursor: page === pages - 1 ? "not-allowed" : "none", borderRadius: 2 }}>›</button>
                </div>
              </div>
            )}
          </TiltCard>

          {/* RIGHT — detail panel */}
          {detail && (
            <TiltCard color={C.coral} className="fu" style={{ background: C.card, border: `1px solid ${C.rule}`, borderRadius: 2, overflow: "hidden" }}>
              {/* Detail header */}
              <div style={{ padding: "16px 20px", borderBottom: `1px solid ${C.rule}`, background: C.panel, display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
                <div>
                  <div style={{ fontFamily: C.M, fontSize: 9, letterSpacing: ".14em", color: C.coral, marginBottom: 4 }}>UPLOAD DETAIL</div>
                  <div style={{ fontFamily: C.D, fontSize: 18, color: C.ink, fontStyle: "italic", fontWeight: 300, wordBreak: "break-all" }}>{detail.fileName}</div>
                  <div style={{ fontFamily: C.M, fontSize: 10, color: C.muted, marginTop: 4 }}>
                    {detail.userName} · <span style={{ color: C.blue }}>{detail.userEmail}</span>
                  </div>
                  <div style={{ fontFamily: C.M, fontSize: 9, color: C.muted, marginTop: 2 }}>{fmtDate(detail.uploadedAt)} · {detail.rowCount?.toLocaleString()} rows · {detail.headers?.length} columns</div>
                </div>
                <button onClick={() => setDetail(null)}
                  style={{ fontFamily: C.M, fontSize: 10, color: C.muted, background: "transparent", border: `1px solid ${C.rule}`, padding: "4px 10px", cursor: "none", borderRadius: 2, flexShrink: 0 }}>✕ Close</button>
              </div>

              {/* Risk summary pills */}
              {detail.stats && (
                <div style={{ display: "flex", gap: 1, borderBottom: `1px solid ${C.rule}` }}>
                  {[
                    { label: "HIGH",   value: detail.stats.high,   color: C.coral },
                    { label: "MEDIUM", value: detail.stats.medium, color: C.amber },
                    { label: "SAFE",   value: detail.stats.safe,   color: C.teal },
                    { label: "AVG CHURN", value: detail.stats.avgChurn ? detail.stats.avgChurn + "%" : "—", color: C.violet },
                  ].map(x => (
                    <div key={x.label} style={{ flex: 1, padding: "12px 16px", background: `${x.color}08`, borderRight: `1px solid ${C.rule}` }}>
                      <div style={{ fontFamily: C.M, fontSize: 8, letterSpacing: ".12em", color: x.color, marginBottom: 4 }}>{x.label}</div>
                      <div style={{ fontFamily: C.D, fontSize: 22, color: x.color, fontWeight: 700 }}>{x.value ?? "—"}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Column mapping */}
              {detail.mapping && Object.keys(detail.mapping).length > 0 && (
                <div style={{ padding: "12px 20px", borderBottom: `1px solid ${C.rule}`, display: "flex", flexWrap: "wrap", gap: 6 }}>
                  <div style={{ fontFamily: C.M, fontSize: 8, letterSpacing: ".12em", color: C.muted, width: "100%", marginBottom: 4 }}>COLUMN MAPPING</div>
                  {Object.entries(detail.mapping).map(([k, v]) => {
                    const cols = Array.isArray(v) ? v : [v];
                    return cols.length ? cols.map(col => (
                      <span key={k + col} style={{ fontFamily: C.M, fontSize: 9, padding: "3px 8px", border: `1px solid ${C.rule}`, borderRadius: 2, color: C.body }}>
                        <span style={{ color: C.muted }}>{k}:</span> {col}
                      </span>
                    )) : null;
                  })}
                </div>
              )}

              {/* Row preview table */}
              <div style={{ overflowX: "auto", maxHeight: 480 }}>
                {detailLoading
                  ? <div style={{ padding: 32, textAlign: "center", fontFamily: C.M, fontSize: 11, color: C.muted }}>Loading rows…</div>
                  : detail.rows?.length > 0 && (() => {
                    const previewCols = ["customerID", "Customer ID", "churn_score", "Model Churn Score", "risk_tier", "Risk Tier", "retention_strategy", "Retention Strategy"]
                      .filter(c => detail.headers?.includes(c) || detail.rows[0]?.[c] !== undefined)
                      .slice(0, 6);
                    const cols = previewCols.length > 0 ? previewCols : (detail.headers || Object.keys(detail.rows[0])).slice(0, 6);
                    return (
                      <>
                        <div style={{ fontFamily: C.M, fontSize: 8, letterSpacing: ".12em", color: C.muted, padding: "10px 20px 4px" }}>
                          ROW PREVIEW — first {Math.min(detail.rows.length, 50)} of {detail.rows.length}
                        </div>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                          <thead>
                            <tr>{cols.map(c => <th key={c} style={{ ...TH, fontSize: 8 }}>{c}</th>)}</tr>
                          </thead>
                          <tbody>
                            {detail.rows.slice(0, 50).map((r, i) => (
                              <tr key={i} className="row-tr" style={{ background: i % 2 === 0 ? C.card : C.bg }}>
                                {cols.map(c => {
                                  const v = r[c];
                                  const isScore = c === "churn_score" || c === "Model Churn Score";
                                  const isTier  = c === "risk_tier"   || c === "Risk Tier";
                                  const isStrat = c === "retention_strategy" || c === "Retention Strategy";
                                  let display = v ?? "—";
                                  let color = C.body;
                                  if (isScore && v !== undefined) { display = (parseFloat(v) * 100).toFixed(1) + "%"; color = parseFloat(v) >= 0.65 ? C.coral : parseFloat(v) >= 0.35 ? C.amber : C.teal; }
                                  if (isTier)  color = tierColor(String(v).toUpperCase());
                                  return (
                                    <td key={c} style={{ ...TD, color, fontFamily: isScore || isTier ? C.M : "inherit", fontWeight: isScore || isTier ? 500 : 400, maxWidth: isStrat ? 240 : 120, whiteSpace: isStrat ? "normal" : "nowrap", overflow: "hidden", textOverflow: isStrat ? "unset" : "ellipsis", lineHeight: 1.45, fontSize: isStrat ? 10 : 11 }}>
                                      {String(display)}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </>
                    );
                  })()
                }
              </div>
            </TiltCard>
          )}
        </div>
      </main>
    </div>
  );
}

/* ── SAVE TOAST ──────────────────────────────────────────────────────────────── */
function SaveToast({ status }) {
  // status: "" | "saving" | "saved" | "error"
  if (!status) return null;
  const cfg = {
    saving: { bg: C.amber,  icon: "⏳", msg: "Saving to MongoDB…" },
    saved:  { bg: C.teal,   icon: "✓",  msg: "Dataset saved to MongoDB" },
    error:  { bg: C.coral,  icon: "⚠",  msg: "MongoDB save failed — backend offline?" },
  }[status];
  if (!cfg) return null;
  return (
    <div style={{
      position: "fixed", bottom: 24, right: 24, zIndex: 9999,
      display: "flex", alignItems: "center", gap: 10,
      background: cfg.bg, color: C.white,
      padding: "11px 20px", borderRadius: 4,
      fontFamily: C.M, fontSize: 11, letterSpacing: ".06em",
      boxShadow: `0 4px 20px ${cfg.bg}55`,
      animation: "fadeUp .35s cubic-bezier(.22,1,.36,1)",
    }}>
      <span>{cfg.icon}</span>
      <span>{cfg.msg}</span>
    </div>
  );
}

export default function App() {
  // null = loading, false = logged out, object = logged in
  const [session, setSession] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [step, setStep] = useState("upload");
  const [csvFile, setCsvFile] = useState(null);          // raw File object for GridFS upload
  const [fileName, setFileName] = useState(null);
  const [headers, setHeaders] = useState([]);
  const [rows, setRows] = useState([]);
  const [mapping, setMapping] = useState({});
  const [nothing, setNothing] = useState(new Set());
  const [modelStatus, setModelStatus] = useState("");
  const [dbSaveStatus, setDbSaveStatus] = useState("");   // "" | "saving" | "saved" | "error"

  // Restore session automatically on page refresh via Firebase Auth state
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async fbUser => {
      if (fbUser) {
        try {
          const snap = await getDoc(doc(db, "users", fbUser.uid));
          const profile = snap.exists() ? snap.data() : {};
          setSession({ uid: fbUser.uid, user: { uid: fbUser.uid, email: fbUser.email, ...profile } });
        } catch {
          setSession({ uid: fbUser.uid, user: { uid: fbUser.uid, email: fbUser.email } });
        }
      } else {
        setSession(false);
      }
      setAuthReady(true);
    });
    return () => unsub();
  }, []);

  const handleFile = (file, text) => {
    const { headers: h, rows: r } = parseCSV(text);
    setCsvFile(file); setFileName(file.name); setHeaders(h); setRows(r); setMapping(autoDetect(h)); setNothing(new Set()); setStep("map");
  };
  const clear = () => { setStep("upload"); setCsvFile(null); setFileName(null); setHeaders([]); setRows([]); setMapping({}); setNothing(new Set()); };
  const login = data => { setSession(data); };
  const logout = async () => { await signOut(auth); setSession(false); clear(); };

  const arrMap = useMemo(() => Object.fromEntries(
    Object.entries(mapping)
      .filter(([k]) => !nothing.has(k))
      .map(([k, v]) => [k, v instanceof Set ? [...v] : Array.isArray(v) ? v : (v ? [v] : [])])
      .filter(([, v]) => v.length)
  ), [mapping, nothing]);
  const arrMapKey = useMemo(() => JSON.stringify(arrMap), [arrMap]);
  const stats = useMemo(() => step === "analytics" ? buildStats(rows, arrMap) : null, [step, rows, arrMapKey]);

  const exportCSV = () => {
    if (!rows.length) return;
    const headers = Object.keys(rows[0]);
    const h = headers.map(csvCell).join(",");
    const b = rows.map(r => headers.map(key => csvCell(r[key])).join(",")).join("\n");
    const blob = new Blob([h + "\n" + b], { type: "text/csv" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "churn_export.csv"; a.click();
  };

  const runAnalysis = async (m, n) => {
    const arr = Object.fromEntries(
      Object.entries(m)
        .filter(([k]) => !n.has(k))
        .map(([k, v]) => [k, v instanceof Set ? [...v] : Array.isArray(v) ? v : (v ? [v] : [])])
        .filter(([, v]) => v.length)
    );
    setModelStatus("Scoring customers...");
    // Local scoring — reads mapped churn output column directly (no backend required)
    const scored = localModelRows(rows, arr);
    const finalMapping = { ...m, churnOutput: new Set(["Model Churn Score"]) };
    setRows(scored); setMapping(finalMapping); setNothing(n);
    setModelStatus(""); setStep("analytics");

    // ── Persist to MongoDB (non-blocking) ───────────────────────────────────────
    const arrMapping = Object.fromEntries(
      Object.entries(finalMapping)
        .map(([k, v]) => [k, v instanceof Set ? [...v] : Array.isArray(v) ? v : (v ? [v] : [])])
    );
    const uploadStats = buildStats(scored, arrMapping);
    setDbSaveStatus("saving");
    const result = await saveToMongo({
      csvFile,                                             // raw File → stored in GridFS
      userName:  session?.user?.name  || session?.user?.email || "Unknown",
      userEmail: session?.user?.email || "",
      mapping: arrMapping,
      stats: {
        total:    uploadStats.total,
        high:     uploadStats.high,
        medium:   uploadStats.medium,
        safe:     uploadStats.safe,
        avgChurn: uploadStats.avgChurn,
      },
    });
    const toastStatus = result?.success ? "saved" : "error";
    setDbSaveStatus(toastStatus);
    setTimeout(() => setDbSaveStatus(""), toastStatus === "saved" ? 3500 : 5000);
  };

  // Show a neutral loading screen while Firebase resolves auth state
  if (!authReady) return (
    <div style={{ minHeight: "100vh", background: C.bg, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <Cursor />
      <div style={{ textAlign: "center" }}>
        <div style={{ fontFamily: C.D, fontSize: 36, color: C.coral, fontStyle: "italic", animation: "float 2s ease-in-out infinite" }}>Loading…</div>
        <div style={{ fontFamily: C.M, fontSize: 10, color: C.muted, marginTop: 10, letterSpacing: ".18em" }}>CHECKING SESSION</div>
      </div>
    </div>
  );
  if (!session) return <LoginScreen onLogin={login} />;
  if (session.user?.role === "admin") return <AdminDashboard session={session} onLogout={logout} />;

  return (
    <div style={{ minHeight: "100vh", background: C.bg, maxWidth: 1280, margin: "0 auto" }}>
      <Cursor />
      <SaveToast status={dbSaveStatus} />
      {step === "analytics" && <Marquee stats={stats} fileName={fileName} />}

      {/* NAV */}
      <nav style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "0 32px", height: 52, borderBottom: `1px solid ${C.rule}`,
        background: `${C.paper}ee`, backdropFilter: "blur(16px)",
        position: "sticky", top: step === "analytics" ? 28 : 0, zIndex: 100,
        boxShadow: "0 1px 0 rgba(0,0,0,.04)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ fontFamily: C.D, fontSize: 20, color: C.ink, letterSpacing: "-.01em", lineHeight: 1, fontWeight: 300 }}>
            Commu<span style={{ color: C.coral, fontStyle: "italic", fontWeight: 700 }}>no</span>
          </div>
          <div style={{ width: 1, height: 16, background: C.rule }} />
          <div style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".12em", textTransform: "uppercase" }}>Risk Analytics</div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
          {["upload", "map", "analytics"].map((s, i) => {
            const done = ["upload", "map", "analytics"].indexOf(step) > i, active = step === s;
            return (
              <div key={s} style={{ display: "flex", alignItems: "center" }}>
                <div style={{
                  display: "flex", alignItems: "center", gap: 6, padding: "5px 14px",
                  background: active ? C.coral : "transparent",
                  border: `1px solid ${active ? C.coral : done ? C.coral + "55" : C.rule}`,
                  borderRight: "none", transition: "all .3s",
                  boxShadow: active ? `0 2px 12px ${C.coral}33` : "none",
                }}>
                  <span style={{ fontFamily: C.M, fontSize: 9, fontWeight: 500, color: active ? C.white : done ? C.coral : C.muted, letterSpacing: ".06em" }}>{done ? "✓" : String(i + 1).padStart(2, "0")}</span>
                  <span style={{ fontFamily: C.S, fontSize: 11, color: active ? C.white : done ? C.body : C.muted, fontWeight: active ? 500 : 400 }}>{s === "upload" ? "Upload" : s === "map" ? "Map" : "Analyse"}</span>
                </div>
                {i < 2 && <div style={{ width: 8, height: 1, background: C.rule }} />}
              </div>
            );
          })}
          <div style={{ width: 1, height: 28, background: C.rule, marginLeft: 0 }} />
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div style={{ fontFamily: C.M, fontSize: 9, color: C.muted, letterSpacing: ".08em", marginRight: 6 }}>{session.user.name}</div>
          {step === "analytics" && <>
            <button className="nav-btn" onClick={() => setStep("map")} style={{ border: `1px solid ${C.rule}`, color: C.muted, borderRadius: 2 }}>Remap</button>
            <button className="nav-btn" onClick={clear} style={{ border: `1px solid ${C.rule}`, color: C.muted, borderRadius: 2 }}
              onMouseOver={e => { e.currentTarget.style.color = C.coral; e.currentTarget.style.borderColor = C.coral + "55"; }}
              onMouseOut={e => { e.currentTarget.style.color = C.muted; e.currentTarget.style.borderColor = C.rule; }}>Clear</button>
            <label className="nav-btn" style={{ border: `1px solid ${C.rule}`, color: C.muted, cursor: "none", borderRadius: 2 }}>
              New CSV
              <input type="file" accept=".csv" style={{ display: "none" }} onChange={e => { const f = e.target.files[0]; if (f) { const r = new FileReader(); r.onload = ev => handleFile(f, ev.target.result); r.readAsText(f); } }} />
            </label>
            <button className="glow-btn" onClick={exportCSV}
              style={{ fontSize: 10, background: C.coral, color: C.white, padding: "7px 16px", letterSpacing: ".08em", borderRadius: 2, boxShadow: `0 2px 12px ${C.coral}33` }}>
              Export ↓
            </button>
          </>}
          <button className="nav-btn" onClick={logout} style={{ border: `1px solid ${C.rule}`, color: C.muted, borderRadius: 2 }}>Logout</button>
        </div>
      </nav>

      <div style={{ padding: step === "analytics" ? "0 32px" : "0" }}>
        {modelStatus && step === "analytics" && <div style={{ fontFamily: C.M, fontSize: 10, color: C.muted, padding: "12px 0" }}>{modelStatus}</div>}
        {step === "upload" && <Upload onFile={handleFile} />}
        {step === "map" && <MapStep headers={headers} initialMapping={mapping} fileName={fileName} onConfirm={runAnalysis} />}
        {step === "analytics" && <Analytics rows={rows} mapping={arrMap} />}
      </div>
    </div>
  );
}
