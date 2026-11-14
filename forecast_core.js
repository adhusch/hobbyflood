/* Kettenvorhersage im Browser: Portierung von routing_core.c (Kaskade), lateral.response (Zwischengebiet) und chain_fc.forecast (ein Fall).
 * Gleiche Rechenvorschrift wie in Python; test_forecast_js.py vergleicht beide auf den Hochwasserfenstern der Karte.
 * Eingaben: Modellsatz (chain_models.json, ein Block), Hochwasserfenster (chain_floods.json, ein Eintrag), Zielpegel, Ausgabeindex, Variante. */
const FC = (() => {
  const DT = 0.25;
  const DN = { P2: "Prüm 2", SI: "Sinspelt", GI: "Giesdorf", EC: "Echtershausen", WI: "Wiersdorf", PZ: "Prümzurlay", SE: "Seffern", AL: "Alsdorf-Oberecken" };
  const UPS = { EC: ["P2"], WI: ["EC"], PZ: ["WI", "SI"], SE: ["GI"], AL: ["SE"] };
  const ORDER = ["P2", "SI", "GI", "EC", "WI", "PZ", "SE", "AL"];
  const HW = new Set(["P2", "SI", "GI"]);
  const NORAIN = new Set(["einzeln", "kette"]);
  const SWITCH = new Set(["kette_regen", "kette_ideal", "kette_vorhersage"]);    // Umschalten Winter / Starkregen (Einteilung je Zielpegel und Index in fl.storm)
  const FUTURE_RAIN = new Set(["kette_ideal", "kette_vorhersage"]);             // Regen nach der Ausgabezeit aus fl.p (gemessen = perfekte Vorhersage)
  let A = 192, H = 192, LEAD = 288, TAU = 168, TAU_S = 24, WET = {};

  function setConsts(c) { A = Math.round(c.A_H / DT); H = Math.round(c.H_H / DT); LEAD = Math.round(c.LEAD_H / DT); TAU = c.TAU_API_H; TAU_S = c.TAU_API_S_H; WET = c.WET || {}; }

  function chain(k) {
    const need = new Set(), todo = [k];
    while (todo.length) { const x = todo.pop(); need.add(x); (UPS[x] || []).forEach(u => todo.push(u)); }
    return ORDER.filter(x => need.has(x));
  }

  function celerity(q, lq, lc) {
    const x = Math.log(q > 1e-3 ? q : 1e-3), nn = lq.length;
    let v;
    if (nn === 1 || x <= lq[0]) v = lc[0];
    else if (x >= lq[nn - 1]) v = lc[nn - 1];
    else {
      let k = 0;
      while (k < nn - 2 && x > lq[k + 1]) k++;
      const f = (x - lq[k]) / (lq[k + 1] - lq[k]);
      v = lc[k] + f * (lc[k + 1] - lc[k]);
    }
    v = Math.exp(v);
    return v < 0.05 ? 0.05 : v > 12 ? 12 : v;
  }

  function cascade(x, n, length_km, lq, lc) {
    const T = x.length, dx = length_km / n, O = new Float64Array(n).fill(x[0]), y = new Float64Array(T);
    y[0] = x[0];
    for (let t = 1; t < T; t++) {
      let i0 = x[t - 1], i1 = x[t];
      for (let j = 0; j < n; j++) {
        const qref = 0.5 * (0.5 * (i0 + i1) + O[j]), c = celerity(qref, lq, lc), K = dx / (3.6 * c);
        const em = -Math.expm1(-DT / K), a = 1 - em, o0 = O[j];
        const o1 = o0 * a + i0 * em + (i1 - i0) * (1 - K * em / DT);
        i0 = o0; i1 = o1; O[j] = o1;
      }
      y[t] = O[n - 1];
    }
    return y;
  }

  /* Zwischengebiet: rain {p, api, aps, lead}, Rückgabe Länge p.length - lead in m³/s.
     Erweiterung fast: der Intensitätsanteil (1 - phi_a) * rho * P²/(P² + Pc²) läuft über zwei schnelle Speicher mit Kf (lateral.py) */
  /* nichtlineare Speicher (Erweiterung nlr, lateral._nl_stores): Inhalt je Speicher, K(S) = K / (1 + S / (K oref))^a, massentreu */
  function nlStores(peff, k1, k2, lead, f, out, add, oref, a) {
    const T = peff.length;
    let s1 = k1 * peff[0], s2 = k2 * peff[0];
    if (lead === 0) out[0] = (add ? out[0] : 0) + f * peff[0];
    for (let t = 1; t < T; t++) {
      const kt1 = k1 / Math.pow(1 + s1 / (k1 * oref), a), e1 = Math.exp(-DT / kt1);
      const s1n = s1 * e1 + peff[t] * kt1 * (1 - e1), i2 = peff[t] - (s1n - s1) / DT;
      s1 = s1n;
      const kt2 = k2 / Math.pow(1 + s2 / (k2 * oref), a), e2 = Math.exp(-DT / kt2);
      const s2n = s2 * e2 + i2 * kt2 * (1 - e2), o = i2 - (s2n - s2) / DT;
      s2 = s2n;
      if (t >= lead) out[t - lead] = (add ? out[t - lead] : 0) + f * o;
    }
  }
  function twoStores(peff, k1, k2, lead, f, out, add, nl) {
    if (nl) return nlStores(peff, k1, k2, lead, f, out, add, nl[0], nl[1]);
    const T = peff.length, a1 = Math.exp(-DT / k1), a2 = Math.exp(-DT / k2), w2 = 1 - k2 * (1 - a2) / DT;
    let o1 = peff[0], o2 = peff[0];
    if (lead === 0) out[0] = (add ? out[0] : 0) + f * o2;
    for (let t = 1; t < T; t++) {
      const o1n = a1 * o1 + (1 - a1) * peff[t];
      o2 = a2 * o2 + (1 - a2) * o1 + (o1n - o1) * w2;
      o1 = o1n;
      if (t >= lead) out[t - lead] = (add ? out[t - lead] : 0) + f * o2;
    }
  }
  function lateral(r, lt) {
    const T = r.p.length, terms = lt.terms || [], peff = new Float64Array(T), fast = terms.includes("fast"), sat = terms.includes("sat") || terms.includes("satx");
    const pf = fast || sat ? new Float64Array(T) : null;
    // Sättigungsüberschuss (sat): Rückhalt Sv = sev * exp(-g_ev * B), Anteil rs * SCS-Steigung bei Pe = api_s
    const sv = sat ? Math.exp(lt.log_sev) * Math.exp(-lt.g_ev * Math.max(r.bf ?? 0, 0)) : 0;
    const S = Math.exp(lt.log_s), pc2 = Math.exp(2 * (lt.log_pc ?? Math.log(10)));
    // Gebietsfeuchte (Erweiterung bf): B^g / (B^g + b50^g), B = Abfluss / MQ am Bezugspegel zu Beginn des Abflussfensters
    let wet = 1;
    if (terms.includes("bf")) { const g = Math.exp(lt.log_gb ?? 0), b = Math.pow(Math.max(r.bf ?? 0, 1e-6), g); wet = b / (b + Math.exp(g * lt.log_b50)); }
    for (let t = 0; t < T; t++) {
      let x = r.api[t];
      if (terms.includes("acc")) x += lt.w_acc * r.aps[t];
      if (terms.includes("bfa")) x += lt.w_bf * Math.max(r.bf ?? 0, 0);
      let phi = lt.phi_max * (1 - Math.exp(-x / S));
      if (terms.includes("int")) { const p2 = r.p[t] * r.p[t]; phi += (1 - phi) * lt.rho * p2 / (p2 + pc2); }
      if (terms.includes("bf")) phi *= wet;
      if (pf) {
        let qf = 0;
        if (fast) { const p2 = r.p[t] * r.p[t]; qf += (1 - phi) * lt.rho * p2 / (p2 + pc2); }
        if (sat) { const lam = lt.lam ?? 0.2, pe = r.aps[t], a = Math.max(pe - lam * sv, 0), den = pe + (1 - lam) * sv; qf += (1 - phi) * lt.rs * a * (pe + (2 - lam) * sv) / (den * den); }
        pf[t] = qf * r.p[t];
      }
      peff[t] = phi * r.p[t];
    }
    const out = new Float64Array(T - r.lead), f = lt.area_km2 / 3.6;
    const nl = terms.includes("nlr") || terms.includes("nlrx") ? [Math.exp(lt.log_oref ?? 0), lt.a_nl ?? 0] : null;
    twoStores(peff, Math.exp(lt.log_k1), Math.exp(lt.log_k2), r.lead, f, out, false, nl);
    if (pf) { const kf = Math.exp(lt.log_kf ?? 0); twoStores(pf, kf, kf, r.lead, f, out, true, nl); }
    return out;
  }

  function route(m, X, P) {
    const T = (Object.values(X)[0] || P[m.lat[0].name].p).length - (Object.keys(X).length ? 0 : P[m.lat[0].name].lead);
    const y = new Float64Array(T).fill(m.c0);
    m.links.forEach(lk => {
      const lq = lk.nodes_q.map(Math.log), c = cascade(X[lk.up], lk.n, lk.length_km, lq, lk.lc);
      for (let t = 0; t < T; t++) y[t] += lk.a * c[t];
    });
    if (P) (m.lat || []).forEach(lt => { const q = lateral(P[lt.name], lt); for (let t = 0; t < T; t++) y[t] += q[t]; });
    return y;
  }

  /* Vorregenindex über das ganze Regenfenster des Hochwassers (einmal je Hochwasser und Gebiet) */
  function apiSeries(p, a0, tau) {
    const k = Math.exp(-DT / tau), out = new Float64Array(p.length);
    let acc = a0;
    for (let i = 0; i < p.length; i++) { acc = acc * k + p[i] * DT; out[i] = acc; }
    return out;
  }
  function prepFlood(fl) {
    if (fl._prep) return fl._prep;
    const pr = { api: {}, aps: {} };
    if (fl.rain) Object.keys(fl.p).forEach(k => { pr.api[k] = apiSeries(fl.p[k], fl.api0[k][0], TAU); pr.aps[k] = apiSeries(fl.p[k], fl.api0[k][1], TAU_S); });
    fl._prep = pr;
    return pr;
  }
  /* Regenfenster für Ausgabeindex i (Index im Abflussfenster): Regenindex = Abflussindex + LEAD, Fenster ab i - A - LEAD */
  function rainWindow(fl, k, i, ideal) {
    const pr = prepFlood(fl), j0 = i - A, n = A + H + LEAD + 1;
    const w = WET[k], qb = w ? fl.q[w[0]][i - A] : null, bf = w && qb != null ? qb / w[1] : 0;
    const p = Float64Array.from(fl.p[k].slice(j0, j0 + n)), api = Float64Array.from(pr.api[k].slice(j0, j0 + n)), aps = Float64Array.from(pr.aps[k].slice(j0, j0 + n));
    if (!ideal) {
      const j = LEAD + A;
      for (let m = 1; m <= H; m++) { p[j + m] = 0; api[j + m] = api[j] * Math.exp(-m * DT / TAU); aps[j + m] = aps[j] * Math.exp(-m * DT / TAU_S); }
    }
    return { p, api, aps, lead: LEAD, bf };
  }

  /* Scheitel-Regel: kein Scheitel am Unterlieger, solange an seinen Oberliegern kein Scheitel gemessen ist (wie chain_fc.peak_passed / hold_after_max) */
  let HOLD = true;
  const PEAK_LAG = 2, PEAK_DROP = 0.01;   // Höchstwert >= 30 min zurück, jetzt >= 1 % darunter, über 30 min nicht gestiegen
  function peakPassed(q, i) {           // q: Messreihe des Oberliegers im Hochwasserfenster, i: Ausgabeindex
    let jm = 0, m = -Infinity;
    for (let t = 0; t <= A; t++) { const v = q[i - A + t]; if (v > m) { m = v; jm = t; } }
    let lo = Infinity;
    for (let t = 0; t <= jm; t++) lo = Math.min(lo, q[i - A + t]);
    return jm <= A - PEAK_LAG && q[i] <= (1 - PEAK_DROP) * m && q[i] <= q[i - PEAK_LAG] && m >= 1.2 * Math.max(lo, 1e-6);
  }
  function holdAfterMax(s, q0) {        // s: Reihe [A + H + 1], Vorhersage ab A + 1
    let jm = A + 1;
    for (let t = A + 2; t <= A + H; t++) if (s[t] > s[jm]) jm = t;
    if (s[jm] > q0) for (let t = jm + 1; t <= A + H; t++) s[t] = s[jm];
  }

  /* Starkregenhochwasser zur Ausgabezeit? (in Python vorberechnet: Monat Mai–September und größter 1-h-Gebietsregen der letzten 48 h >= 4 mm/h) */
  function isStorm(fl, target, i) { return !!(fl.storm && fl.storm[target] && fl.storm[target][i] === "1"); }

  /* Vorhersage aller Pegel der Kette; Rückgabe {Schlüssel: Float64Array(A + H + 1)} (bis Index A Messung, danach Vorhersage) oder null */
  function forecast(set, fl, target, i, variant) {
    if (i - A < 0 || i + H >= fl.q[DN[target]].length) return null;
    if (!NORAIN.has(variant) && !fl.rain) return null;
    const suffix = SWITCH.has(variant) ? (isStorm(fl, target, i) ? "_storm" : "_winter") : "";
    const ideal = FUTURE_RAIN.has(variant);
    const S = {};
    for (const k of chain(target)) {
      const q = fl.q[DN[k]], obs = q.slice(i - A, i + H + 1);
      if (obs.slice(0, A + 1).some(v => v == null)) return null;
      let sim, m;
      if (HW.has(k)) {
        if (NORAIN.has(variant)) { const s = new Float64Array(A + H + 1); for (let t = 0; t <= A + H; t++) s[t] = t <= A ? obs[t] : obs[A]; S[k] = s; continue; }
        m = set[k]["hw" + suffix];
        sim = route(m, {}, { [k]: rainWindow(fl, k, i, ideal) });
      } else {
        const X = {};
        UPS[k].forEach(u => {
          if (variant === "einzeln") { const v = fl.q[DN[u]].slice(i - A, i + H + 1), s = new Float64Array(A + H + 1); for (let t = 0; t <= A + H; t++) s[t] = t <= A ? v[t] : v[A]; X[DN[u]] = s; }
          else X[DN[u]] = S[u];
        });
        m = set[k][((NORAIN.has(variant) || k === "WI") ? "route_full" : "rain_full") + suffix];
        const P = m.lat && m.lat.length ? { [k]: rainWindow(fl, k, i, ideal) } : null;
        sim = route(m, X, P);
      }
      const err0 = obs[A] - sim[A], s = new Float64Array(A + H + 1);
      for (let t = 0; t <= A; t++) s[t] = obs[t];
      for (let t = 1; t <= H; t++) s[A + t] = Math.max(sim[A + t] + err0 * m.beta[t - 1], 0);
      if (HOLD && variant !== "kette_ideal" && UPS[k] && !UPS[k].every(u => peakPassed(fl.q[DN[u]], i))) holdAfterMax(s, obs[A]);
      S[k] = s;
    }
    return S;
  }

  /* Hält die Regel für Pegel k zur Ausgabezeit i die Vorhersage nach dem Höchstwert? (für die Anzeige) */
  function upstreamOpen(fl, k, i) { return !!(HOLD && UPS[k] && !UPS[k].every(u => peakPassed(fl.q[DN[u]], i))); }

  return { forecast, isStorm, upstreamOpen, setConsts, chain, DN, ORDER, lateral, get A() { return A; }, get H() { return H; } };
})();

/* Unsicherheit und Wasserstand: wie uncert.py (Fehlertabellen aus der Kreuzvalidierung) und rating.py (W-Q-Beziehung).
 * Tabellen U = data/uncert_tables.json; Klasse = Zustand * 6 + Abflussstufe * 2 + Feuchte (0: jetzt unter 0,3 MQ, sonst 1), in einer Starkregenlage
 * + 24 (eigene Tabelle aus allen Pegeln). Zustand und Starkregenlage vorberechnet je Hochwasser, Pegel und Index als Hexadezimalziffer: Zustand
 * + 4 * Starkregenlage aus dem Regen bis zur Ausgabezeit + 8 * einschließlich der 24 h danach (Variante kette_vorhersage, gemessener Regen als
 * perfekte Vorhersage). Abflussstufe aus dem größeren Wert von jetzt und Vorhersage-Höchstwert in MQ: unter 2, 2 bis 8, ab 8.
 * Fehler r = ln((Messung + d) / (Vorhersage + d)). */
const UQ = (() => {
  let U = null, LEADS = null;
  function setTables(u) { U = u; LEADS = u.leads_h.map(h => Math.round(h * 4)); }
  function tableSet(fl, variant) {
    if (variant === "kette_vorhersage") return (U.vorhersage || {})[fl.models] || U.vorhersage.alle;
    return fl.rain ? U.regen[fl.models] : U.kette.alle;
  }
  /* Quantile für Vorlauf L (Schritte 1..H), linear zwischen den gespeicherten Vorläufen */
  function qAt(T, L) {
    if (L <= LEADS[0]) return T[0];
    let j = 0;
    while (j < LEADS.length - 2 && LEADS[j + 1] < L) j++;
    const a = LEADS[j], b = LEADS[j + 1], f = Math.min(1, (L - a) / (b - a));
    return T[j].map((v, k) => v + f * (T[j + 1][k] - v));
  }
  /* P(r <= x) wie uncert.cdf_fast */
  function cdf(x, q) {
    const lev = U.lev, n = lev.length;
    if (x < q[0]) { const s = Math.max((q[1] - q[0]) / Math.LN2, 1e-3); return 0.05 * Math.exp((x - q[0]) / s); }
    if (x > q[n - 1]) { const s = Math.max((q[n - 1] - q[n - 2]) / Math.LN2, 1e-3); return 1 - 0.05 * Math.exp(-(x - q[n - 1]) / s); }
    let k = 0;
    for (let j = 0; j < n; j++) if (q[j] <= x) k++;
    const kk = Math.max(1, Math.min(n - 1, k)), qa = q[kk - 1], qb = q[kk];
    return lev[kk - 1] + (lev[kk] - lev[kk - 1]) * Math.max(0, Math.min(1, (x - qa) / Math.max(qb - qa, 1e-9)));
  }
  /* Klasse zur Ausgabezeit i; s: Vorhersagereihe [A + H + 1]; variant wie FC.forecast (Standard: mit Regen bis zur Ausgabezeit, falls vorhanden) */
  function klass(fl, key, i, s, A, H, variant) {
    const code = parseInt(U.floods[fl.id][key][i], 16), st = code & 3;
    const heavy = !fl.rain || variant === "kette" ? 0 : variant === "kette_vorhersage" ? (code >> 3) & 1 : (code >> 2) & 1;
    let mx = s[A];
    for (let t = A + 1; t <= A + H; t++) if (s[t] > mx) mx = s[t];
    const lv = mx / U.mq[key], e = U.level_edges || [2, 8], wet = s[A] >= (U.wet_edge ?? 0.3) * U.mq[key] ? 1 : 0;
    return st * 6 + (lv >= e[1] ? 2 : lv >= e[0] ? 1 : 0) * 2 + wet + 24 * heavy;
  }
  /* Band und Wahrscheinlichkeiten: {cls, state, q: [H][Stufen] in m³/s, exceed(thr) -> [H] kumulierte Wahrscheinlichkeit, peak|null} */
  function analyse(fl, key, i, s, A, H, variant) {
    variant = variant || (fl.rain ? "kette_regen" : "kette");
    const cls = klass(fl, key, i, s, A, H, variant), tab = tableSet(fl, variant)[key], T = tab.q[cls], d = U.offset[key];
    const qs = [], Ts = [];
    for (let L = 1; L <= H; L++) {
      const qq = qAt(T, L), f = Math.max(s[A + L], 0);
      Ts.push(qq);
      qs.push(qq.map(r => Math.max((f + d) * Math.exp(r) - d, 0)));
    }
    const exceed = thr => {
      const out = new Float64Array(H);
      let m = 0;
      for (let L = 1; L <= H; L++) {
        const p = 1 - cdf(Math.log((thr + d) / (Math.max(s[A + L], 0) + d)), Ts[L - 1]);
        if (p > m) m = p;
        out[L - 1] = m;
      }
      return out;
    };
    // Scheitel: Welle unterwegs, Vorhersage steigt (mehr als 2 %, Höchstwert mindestens 2 MQ) und hat ihren Scheitel vor dem Horizontende
    let peak = null;
    let ja = A + 1;
    for (let t = A + 2; t <= A + H; t++) if (s[t] > s[ja]) ja = t;
    const fcmax = s[ja], fcarg = ja - A;
    if (Math.floor((cls % 24) / 6) === 2 && tab.peak && fcmax > U.peak_rise * s[A] && fcmax >= U.peak_min_mq * U.mq[key] && fcarg < H - 4) {
      const e = U.peak_buckets_h, lead = fcarg * 0.25;
      let b = 0;
      while (b < e.length && e[b] < lead) b++;
      b = Math.max(0, Math.min(e.length - 2, b - 1));
      peak = { lead_h: lead, bucket: b, q: tab.peak.lnr[b].map(r => Math.max((fcmax + d) * Math.exp(r) - d, 0)), t_h: tab.peak.dt[b].map(x => lead + x) };
    }
    return { cls, state: Math.floor((cls % 24) / 6), level: Math.floor((cls % 6) / 2), dry: cls % 2 === 0, heavy: cls >= 24, q: qs, exceed, peak, fcmax, fcarg };
  }
  /* Wasserstand (cm) aus Abfluss: linear in log Q, an den Enden mit der Steigung des ersten bzw. letzten Abschnitts (wie rating.w_of_q) */
  function wOfQ(c, q) {
    const n = c.q.length, x = Math.log(Math.max(q, 1e-4)), lq = c.q.map(Math.log), w = c.w;
    let y;
    if (x > lq[n - 1]) y = w[n - 1] + (w[n - 1] - w[n - 2]) / (lq[n - 1] - lq[n - 2]) * (x - lq[n - 1]);
    else if (x < lq[0]) y = w[0] + (w[1] - w[0]) / (lq[1] - lq[0]) * (x - lq[0]);
    else { let k = 0; while (k < n - 2 && x > lq[k + 1]) k++; y = w[k] + (w[k + 1] - w[k]) * (x - lq[k]) / (lq[k + 1] - lq[k]); }
    return Math.max(y, 0);
  }
  function qOfW(c, wv) {
    const n = c.q.length, lq = c.q.map(Math.log), w = c.w;
    if (wv > w[n - 1]) return Math.exp(lq[n - 1] + (wv - w[n - 1]) / ((w[n - 1] - w[n - 2]) / (lq[n - 1] - lq[n - 2])));
    if (wv <= w[0]) return Math.exp(lq[0]);
    let k = 0; while (k < n - 2 && wv > w[k + 1]) k++;
    return Math.exp(lq[k] + (lq[k + 1] - lq[k]) * (wv - w[k]) / (w[k + 1] - w[k]));
  }
  return { setTables, analyse, klass, cdf, qAt, wOfQ, qOfW, get U() { return U; } };
})();
if (typeof module !== "undefined") module.exports = Object.assign(FC, { UQ });
