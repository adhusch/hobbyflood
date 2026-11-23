"""Routing-Modell für das Prüm-Netz: Kaskade nichtlinearer Speicher mit abflussabhängiger Wellengeschwindigkeit c(Q).

Je Unterlieger-Pegel:   Q_unten(t) = c0 + sum_k a_k * Kaskade_k[ Q_oben,k ](t)
  Kaskade_k   Strecke der Länge L_k in n_k Teilstrecken, jede ein Speicher S = K*O, K = dx / c(Q)   (routing_core.c)
  c(Q)        log c linear in log Q über drei Stützstellen (Wellengeschwindigkeit bei kleinem, mittlerem, großem Abfluss)
  a_k         Verstärkung: Zwischenzufluss, der mit dem Oberlieger mitläuft (Flächenverhältnis, gleiche Regenverteilung)
  c0          Sockel
Verzögerung der Strecke bei Abfluss Q: T(Q) = L / c(Q). Wäre die Vermutung richtig (große Hochwasser laufen schneller), steigt c(Q).

Der Rechenkern ist in C (routing_core.c, wird beim ersten Import mit "cc" übersetzt); ohne Compiler läuft eine langsamere numpy-Fassung.
"""
from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from lateral import BOUNDS as LAT_BOUNDS, Lateral, Rain, response as lateral_response

HERE = Path(__file__).parent
DT_H = 0.25
C_MIN, C_MAX = 0.15, 4.0           # m/s, Grenzen bei der Anpassung (Stausee: bis 12)
A_MAX = 8.0                         # Standardobergrenze der Verstärkung a je Zufluss
LAM1 = 0.1                         # Strafe auf Unterschiede von log c zwischen Nachbar-Stützstellen: c(Q) weicht nur ab, wo die Daten es verlangen
_lib = None


# ---------------------------------------------------------------- Rechenkern
def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    src, so = HERE / "routing_core.c", HERE / "_routing_core.so"
    try:
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(so), str(src), "-lm"], check=True, capture_output=True)
        lib = ctypes.CDLL(str(so))
        P = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        lib.cascade.argtypes = [P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int, P, P]
        lib.cascade.restype = None
        D, I = ctypes.c_double, ctypes.c_int
        lib.nl_stores.argtypes = [P, P, I, I, D, D, D, D, D]
        lib.nl_stores.restype = None
        lib.two_stores.argtypes = [P, P, I, I, D, D, D]
        lib.two_stores.restype = None
        lib.api_series.argtypes = [P, P, I, D, D]
        lib.api_series.restype = None
        _lib = lib
    except Exception:
        _lib = False
    return _lib


def celerity(q, lq, lc):
    """Wellengeschwindigkeit (m/s) bei Abfluss q (m³/s)."""
    x = np.log(np.maximum(np.asarray(q, float), 1e-3))
    v = np.exp(np.interp(x, lq, lc))
    return np.clip(v, 0.05, 12.0)


def _cascade_np(x, n, length_km, dt_h, lq, lc):
    """numpy-Fassung desselben Schemas (über alle Reihen gleichzeitig), nur als Ersatz ohne C-Compiler."""
    E, T = x.shape
    dx = length_km / n
    O = np.repeat(x[:, :1], n, axis=1)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, T):
        i0, i1 = x[:, t - 1].copy(), x[:, t].copy()
        for j in range(n):
            o0 = O[:, j].copy()
            c = celerity(0.5 * (0.5 * (i0 + i1) + o0), lq, lc)
            K = dx / (3.6 * c)
            em = -np.expm1(-dt_h / K)
            o1 = o0 * (1 - em) + i0 * em + (i1 - i0) * (1 - K * em / dt_h)
            i0, i1 = o0, o1
            O[:, j] = o1
        out[:, t] = O[:, n - 1]
    return out


def cascade(x, n: int, length_km: float, lq, lc, dt_h: float = DT_H) -> np.ndarray:
    x2 = np.atleast_2d(np.ascontiguousarray(x, dtype=np.float64))
    lq, lc = np.ascontiguousarray(lq, dtype=np.float64), np.ascontiguousarray(lc, dtype=np.float64)
    lib = _load_lib()
    if lib:
        out = np.empty_like(x2)
        lib.cascade(x2, out, x2.shape[0], x2.shape[1], int(n), float(length_km), float(dt_h), len(lq), lq, lc)
    else:
        out = _cascade_np(x2, int(n), length_km, dt_h, lq, lc)
    return out.reshape(np.shape(x))


# ---------------------------------------------------------------- Modell
@dataclass
class Link:
    up: str
    length_km: float
    nodes_q: tuple[float, ...]          # Stützstellen von c(Q) in m³/s (Abfluss am Oberlieger)
    n: int = 8
    lc: np.ndarray = None               # log c je Stützstelle
    a: float = 1.0

    @property
    def lq(self) -> np.ndarray:
        return np.log(np.asarray(self.nodes_q, float))

    def c(self, q):
        return celerity(q, self.lq, self.lc)

    def travel_h(self, q):
        return self.length_km / (3.6 * self.c(q))


@dataclass
class Model:
    dn: str
    links: list[Link]
    c0: float = 0.0
    tied: bool = False                   # True: c unabhängig von Q (Vergleichsmodell)
    cmax: float = C_MAX                  # Obergrenze von c bei der Anpassung (m/s)
    lat: list = field(default_factory=list)   # Regen-Abfluss-Anteile der Zwischengebiete (lateral.Lateral); leer = reines Routing
    amax: float = A_MAX                  # Obergrenze der Verstärkung a je Zufluss bei der Anpassung (Massenbilanz: etwa 1,3; sonst frei bis 8)

    def route(self, X: dict[str, np.ndarray], P: dict | None = None) -> np.ndarray:
        """P: Regenfenster je Zwischengebiet (lateral.Rain); ohne P oder ohne lat ist das Ergebnis das reine Routing."""
        y = np.full(np.shape(next(iter(X.values()))), self.c0, dtype=float)
        for lk in self.links:
            y = y + lk.a * cascade(X[lk.up], lk.n, lk.length_km, lk.lq, lk.lc)
        if self.lat and P is not None:
            for lt in self.lat:
                y = y + lateral_response(P[lt.name], lt)
        return y

    def to_dict(self) -> dict:
        return dict(dn=self.dn, c0=float(self.c0), tied=self.tied, cmax=self.cmax, **({"amax": self.amax} if self.amax != A_MAX else {}),
                    links=[dict(up=l.up, length_km=l.length_km, nodes_q=list(map(float, l.nodes_q)), n=int(l.n),
                                lc=list(map(float, l.lc)), a=float(l.a)) for l in self.links],
                    lat=[lt.to_dict() for lt in self.lat])

    @staticmethod
    def from_dict(d: dict) -> "Model":
        return Model(d["dn"], [Link(l["up"], l["length_km"], tuple(l["nodes_q"]), l["n"], np.array(l["lc"]), l["a"]) for l in d["links"]],
                     d["c0"], d.get("tied", False), d.get("cmax", C_MAX), [Lateral.from_dict(x) for x in d.get("lat", [])], d.get("amax", A_MAX))


# ---------------------------------------------------------------- Ereignisse
def find_events(Q: pd.DataFrame, dn: str, q_thr: float = 0.95, pre_h: float = 60, post_h: float = 24, dist_h: float = 72) -> np.ndarray:
    """Positionen der Scheitel von dn über dem Quantil q_thr (Glättung 1 h, mindestens dist_h Abstand), nur vollständige Fenster."""
    s = Q[dn]
    thr = float(s.quantile(q_thr))
    sm = s.rolling(4, center=True, min_periods=1).mean().to_numpy()
    pk, _ = find_peaks(sm, height=thr, distance=int(dist_h / DT_H), prominence=0.3 * thr)
    a, b = int(pre_h / DT_H), int(post_h / DT_H)
    return np.array([p for p in pk if p - a >= 0 and p + b < len(s)], dtype=int)


def stack(Q: pd.DataFrame, peaks: np.ndarray, names: list[str], pre_h: float = 60, post_h: float = 24) -> dict[str, np.ndarray]:
    a, b = int(pre_h / DT_H), int(post_h / DT_H)
    idx = peaks[:, None] + np.arange(-a, b + 1)[None, :]
    return {n: Q[n].to_numpy()[idx] for n in names}


@dataclass
class Data:
    """Fenster um die Scheitel des Unterliegers: X[up] und Y sind [Ereignis, Zeit]; mask = in die Anpassung eingehende Werte."""
    peaks: np.ndarray
    X: dict
    Y: np.ndarray
    mask: np.ndarray
    wts: np.ndarray
    times: np.ndarray                    # Zeit des Scheitels je Ereignis
    P: dict | None = None                # Regenfenster je Zwischengebiet (lateral.Rain), gleiche Ereignisse
    psel: np.ndarray | None = None       # Fenster, deren Scheitelfehler zählt (None = alle)

    def subset(self, keep: np.ndarray) -> "Data":
        return Data(self.peaks[keep], {k: v[keep] for k, v in self.X.items()}, self.Y[keep], self.mask[keep], self.wts[keep], self.times[keep],
                    None if self.P is None else {k: v.subset(keep) for k, v in self.P.items()},
                    None if self.psel is None else self.psel[keep])


def make_data(Q: pd.DataFrame, W: pd.DataFrame | None, dn: str, ups: list[str], peaks: np.ndarray | None = None, pre_h: float = 60,
              post_h: float = 24, spin_h: float = 12, drop_clipped: bool = True, rain: dict | None = None, lead_h: float = 72,
              wts_floor: float = 1.0) -> Data:
    """rain: je Zwischengebiet ein DataFrame mit Spalten p (mm/h) und api (mm) auf dem Raster von Q (rain.areal_frame), optional bf (Gebietsfeuchte,
    Abfluss / MQ am Bezugspegel; je Fenster der Wert zu Beginn des Abflussfensters). Ereignisse, deren Regenfenster (inklusive lead_h Vorlauf zum
    Einschwingen der Speicher) Lücken hat, werden ausgelassen. wts_floor: Fehler je Fenster durch max(Scheitel, wts_floor) geteilt (bei kleinen
    Regenereignissen ohne Hochwasser zählt dann ein absolut großer Fehler, etwa eine viel zu hohe Vorhersage, nicht ein relativ großer)."""
    if peaks is None:
        peaks = find_events(Q, dn, pre_h=pre_h, post_h=post_h)
    X = stack(Q, peaks, ups, pre_h, post_h)
    Y = stack(Q, peaks, [dn], pre_h, post_h)[dn]
    mask = np.ones_like(Y, dtype=bool)
    mask[:, : int(spin_h / DT_H)] = False
    keep = np.ones(len(peaks), dtype=bool)
    if drop_clipped and W is not None and "Wiersdorf" in set(ups + [dn]):
        Wk = stack(W, peaks, ["Wiersdorf"], pre_h, post_h)["Wiersdorf"]
        Qw = stack(Q, peaks, ["Wiersdorf"], pre_h, post_h)["Wiersdorf"]
        keep = ~clipped_window(Wk, Qw)
    wts = np.maximum(Y.max(axis=1), max(1.0, wts_floor))
    P = None
    if rain:
        lead = int(round(lead_h / DT_H))
        a, b = int(pre_h / DT_H), int(post_h / DT_H)
        idx = peaks[:, None] + np.arange(-a - lead, b + 1)[None, :]
        inside = idx.min(axis=1) >= 0
        idx = np.clip(idx, 0, len(Q) - 1)
        P = {}
        for k, df in rain.items():
            r, ap = df["p"].to_numpy()[idx], df["api"].to_numpy()[idx]
            aps = df["api_s"].to_numpy()[idx] if "api_s" in df else None
            bf = df["bf"].to_numpy()[idx[:, lead]] if "bf" in df else None
            sm = df["sm"].to_numpy()[idx[:, lead]] if "sm" in df else None
            xc = [c for c in ("xr", "x4", "x10", "x20") if c in df]
            ex = np.stack([df[c].to_numpy(dtype="float32")[idx] for c in xc], axis=-1) if xc else None
            keep &= inside & np.isfinite(r).all(axis=1) & np.isfinite(ap).all(axis=1)
            if aps is not None:
                keep &= np.isfinite(aps).all(axis=1)
            if bf is not None:
                keep &= np.isfinite(bf)
            if sm is not None:
                keep &= np.isfinite(sm)
            P[k] = Rain(np.nan_to_num(r), np.nan_to_num(ap), lead, None if aps is None else np.nan_to_num(aps), None if bf is None else np.nan_to_num(bf),
                        None if ex is None else np.nan_to_num(ex), None if sm is None else np.nan_to_num(sm))
    d = Data(peaks, X, Y, mask, wts, Q.index.to_numpy()[peaks], P)
    return d.subset(keep) if not keep.all() else d


def clipped_window(Ww: np.ndarray, Qw: np.ndarray) -> np.ndarray:
    """Wiersdorf bei 250 cm abgeschnitten: höchster Wasserstand im Fenster 249 bis 250 cm bei Q um 137,1 m³/s (vier ältere Hochwasser)."""
    wmax, qmax = Ww.max(axis=1), Qw.max(axis=1)
    return (wmax >= 249) & (wmax <= 250) & (qmax >= 136.0) & (qmax <= 137.3)


# ---------------------------------------------------------------- Anpassung
def _nodes_for(Q: pd.DataFrame, up: str) -> tuple[float, ...]:
    """Stützstellen von c(Q): 95-%-, 99,9-%-Quantil und 70 % des Maximums des Oberliegers (soweit aufsteigend)."""
    s = Q[up]
    v = [float(s.quantile(0.95)), float(s.quantile(0.999)), 0.7 * float(s.max())]
    v[1] = max(v[1], v[0] * 2.0)
    v[2] = max(v[2], v[1] * 1.8)
    return tuple(round(x, 2) for x in v)


def _pack(model: Model):
    th = []
    for lk in model.links:
        th += [lk.lc[0]] if model.tied else list(lk.lc)
        th.append(lk.a)
    th.append(model.c0)
    for lt in model.lat:
        th += lt.theta()
    return np.array(th, float)


def _unpack(model: Model, th: np.ndarray) -> Model:
    m = Model(model.dn, [Link(l.up, l.length_km, l.nodes_q, l.n, None, 1.0) for l in model.links], 0.0, model.tied, model.cmax, [], model.amax)
    i = 0
    for lk in m.links:
        nn = len(lk.nodes_q)
        if model.tied:
            lk.lc = np.repeat(th[i], nn)
            i += 1
        else:
            lk.lc = th[i:i + nn].copy()
            i += nn
        lk.a = th[i]
        i += 1
    m.c0 = th[i]
    i += 1
    m.lat = []
    for lt in model.lat:
        n = len(lt.theta())
        m.lat.append(lt.with_theta(th[i:i + n]))
        i += n
    return m


def _bounds(model: Model):
    lo, hi = [], []
    for lk in model.links:
        k = 1 if model.tied else len(lk.nodes_q)
        lo += [np.log(C_MIN)] * k + [0.02]
        hi += [np.log(model.cmax)] * k + [model.amax]
    lo.append(-3.0)
    hi.append(20.0)
    for lt in model.lat:
        l, h = lt.bounds()
        lo += l
        hi += h
    return np.array(lo), np.array(hi)


PEAK_W = 0.0      # Gewicht des Scheitelfehlers je Fenster in der Anpassung (chain_fc.PEAK_W_FIT setzt es), 0 = aus


def peak_resid(Y: np.ndarray, sim: np.ndarray, mask: np.ndarray, wts: np.ndarray, n_res: int, sel: np.ndarray | None = None) -> np.ndarray:
    """Fehler des Fensterhöchstwerts (wie ml_fit.peak_err), so skaliert, dass seine Quadratsumme PEAK_W-mal dem mittleren Fehler je Zeitschritt
    entspricht. Die Abweichung je Zeitschritt bestraft eine steile Welle doppelt, sobald sie eine Stunde daneben liegt, und zieht die Anpassung
    zu flachen, späten Wellen."""
    ym = np.max(np.where(mask, sim, -np.inf), axis=1)
    om = np.max(np.where(mask, Y, -np.inf), axis=1)
    ok = np.isfinite(ym) & np.isfinite(om)
    if sel is not None:
        ok &= sel
    return np.sqrt(PEAK_W * n_res / max(int(ok.sum()), 1)) * ((ym - om) / wts)[ok]


def objective(model: Model, d: Data, th: np.ndarray, lam: float = 0.03) -> np.ndarray:
    m = _unpack(model, th)
    sim = m.route(d.X, d.P)
    r = ((d.Y - sim) / d.wts[:, None])[d.mask]
    if PEAK_W > 0:
        r = np.concatenate([r, peak_resid(d.Y, sim, d.mask, d.wts, r.size, d.psel)])
    reg = []
    for lk in m.links:
        if len(lk.lc) >= 3 and not model.tied:
            sc = np.sqrt(r.size / 50)
            reg.append(lam * (lk.lc[:-2] - 2 * lk.lc[1:-1] + lk.lc[2:]) * sc)
            reg.append(LAM1 * (lk.lc[1:] - lk.lc[:-1]) * sc)
    return np.concatenate([r] + reg) if reg else r


def _with_dx(m: Model, dx0: float, tied: bool) -> Model:
    return Model(m.dn, [Link(l.up, l.length_km, l.nodes_q, max(1, round(l.length_km / dx0)), l.lc.copy(), l.a) for l in m.links], m.c0, tied, m.cmax, list(m.lat), m.amax)


def _run(m0: Model, d: Data, lam: float, max_nfev: int, prior: tuple | None = None) -> tuple[Model, float]:
    """prior = (Bezugsmodell, Gewicht): zusätzlich Strafterm für Abweichungen vom Bezugsmodell (je Parameter auf ein Viertel der zulässigen Spanne
    bezogen). Für Anpassungen an wenigen Ereignissen (z. B. Starkregenhochwasser), die sich an das Gesamtmodell anlehnen sollen."""
    lo, hi = _bounds(m0)
    th0 = np.clip(_pack(m0), lo + 1e-6, hi - 1e-6)
    if prior is None:
        f = lambda th: objective(m0, d, th, lam)
    else:
        ref = np.clip(_pack(prior[0]), lo, hi)
        scale = (hi - lo) / 4.0
        n_res = int(d.mask.sum())
        wq = np.sqrt(prior[1] * n_res / 100.0)
        f = lambda th: np.concatenate([objective(m0, d, th, lam), wq * (th - ref) / scale])
    res = least_squares(f, th0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=max_nfev)
    return _unpack(m0, res.x), float(res.cost)


def fit(dn: str, links: list[tuple[str, float]], d: Data, Q: pd.DataFrame, dx_grid=(100, 20, 12, 8, 5, 3, 2), tied: bool = False,
        start: Model | None = None, c_starts=(0.8, 1.6, 3.0), lam: float = 0.03, max_nfev: int = 40, cmax: float = C_MAX,
        lat: list | None = None, amax: float = A_MAX, prior: tuple | None = None) -> tuple[Model, float, float]:
    """Anpassung aller Parameter. Ohne start: Mehrfachstart der Wellengeschwindigkeit bei mittlerer Teilstreckenlänge, danach Raster der Teilstreckenlänge dx0
    (bestimmt die Abflachung der Welle) mit Warmstart. Mit start (z. B. Ergebnis aus der Gesamtanpassung): nur die Nachbarn von dx0 prüfen.
    Rückgabe Modell, Zielwert, dx0."""
    ref = float(np.median(d.Y.max(axis=1)))
    mid = dx_grid[len(dx_grid) // 2]
    cands: list[tuple[Model, float, float]] = []
    if start is None:
        for c in [min(c, 0.8 * cmax) for c in c_starts]:
            lks = [Link(u, L, _nodes_for(Q, u), max(1, round(L / mid)), np.full(3, np.log(c)), 1.0) for u, L in links]
            m0 = Model(dn, lks, 0.0, tied, cmax, list(lat or []), amax)
            if tied:
                for lk in m0.links:
                    lk.lc = np.full(3, np.log(c))
            for lk in m0.links:
                lk.a = max(0.2, min(4.0, amax * 0.9, float(np.median(d.Y.max(axis=1) / np.maximum(d.X[lk.up].max(axis=1), 0.1))) / len(m0.links)))
            try:
                m, cost = _run(m0, d, lam, max_nfev)
                cands.append((m, cost, mid))
            except Exception:
                pass
        base = min(cands, key=lambda z: z[1])
        grid = [g for g in dx_grid if g != mid]
    else:
        i0 = min(range(len(dx_grid)), key=lambda i: abs(max(1, round(links[0][1] / dx_grid[i])) - start.links[0].n))
        base = (start, np.inf, dx_grid[i0])
        grid = [dx_grid[j] for j in (i0 - 1, i0, i0 + 1) if 0 <= j < len(dx_grid)]
        cands = []
    for dx0 in grid:
        try:
            pr = None if prior is None else (_with_dx(prior[0], dx0, tied), prior[1])
            m, cost = _run(_with_dx(base[0], dx0, tied), d, lam, max_nfev, pr)
            cands.append((m, cost, dx0))
        except Exception:
            pass
    return min(cands, key=lambda z: z[1])


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    o, s = np.asarray(obs, float), np.asarray(sim, float)
    return float(1 - ((o - s) ** 2).sum() / max(((o - o.mean()) ** 2).sum(), 1e-12))


# ---------------------------------------------------------------- Vergleichsmodell: fester Lag
def _shift(x: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([np.repeat(x[..., :1], k, axis=-1), x[..., : x.shape[-1] - k]], axis=-1) if k else x


@dataclass
class LagModel:
    """Q_unten(t) = c0 + sum_k a_k * Q_oben,k(t - L_k): fester Lag, feste Verstärkung (Vergleichsmodell ohne Abflussabhängigkeit und ohne Abflachung)."""
    dn: str
    ups: list[str]
    lags_h: list[float]
    a: list[float]
    c0: float

    def route(self, X: dict[str, np.ndarray]) -> np.ndarray:
        y = np.full(np.shape(X[self.ups[0]]), self.c0, dtype=float)
        for u, L, a in zip(self.ups, self.lags_h, self.a):
            y = y + a * _shift(X[u], int(round(L / DT_H)))
        return y

    def to_dict(self) -> dict:
        return dict(dn=self.dn, ups=self.ups, lags_h=list(map(float, self.lags_h)), a=list(map(float, self.a)), c0=float(self.c0))


def fit_lag_model(dn: str, ups: list[str], d: Data, max_h: float = 24, max_h2: float = 12, step: int = 2) -> LagModel:
    """Lag je Oberlieger über ein Raster, Verstärkungen und Sockel durch gewichtete kleinste Quadrate (Gewicht 1 / Ereignisscheitel)."""
    import itertools
    w = (1.0 / d.wts[:, None]) * np.ones_like(d.Y)
    wm, y = w[d.mask], d.Y[d.mask] * w[d.mask]
    grids = [range(0, int((max_h if i == 0 else max_h2) / DT_H) + 1, step if len(ups) > 1 else 1) for i in range(len(ups))]
    best = None
    for ks in itertools.product(*grids):
        cols = [_shift(d.X[u], k)[d.mask] * wm for u, k in zip(ups, ks)] + [wm]
        A = np.stack(cols, axis=1)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(((A @ sol - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, ks, sol)
    _, ks, sol = best
    return LagModel(dn, list(ups), [k * DT_H for k in ks], [float(v) for v in sol[:-1]], float(sol[-1]))


# ---------------------------------------------------------------- Vorhersage
def forecast_batch(model: Model, Q: pd.DataFrame, t0: np.ndarray, horizon_h: float = 12, hist_h: float = 48, correct_h: float | None = 6.0) -> dict:
    """Vorhersage zum Zeitpunkt t0 (Positionen): Oberlieger bis t0 beobachtet, danach konstant (Persistenz). Rückgabe sim[E,H], corr[E,H] (mit Fehlerkorrektur
    Q_beob(t0) - Q_sim(t0), abklingend mit correct_h) für Vorlaufzeiten 1..H Schritte."""
    a, H = int(hist_h / DT_H), int(horizon_h / DT_H)
    idx = t0[:, None] + np.arange(-a, H + 1)[None, :]
    X = {}
    for lk in model.links:
        v = Q[lk.up].to_numpy()[idx].copy()
        v[:, a + 1:] = v[:, a:a + 1]
        X[lk.up] = v
    y = model.route(X)
    sim = y[:, a + 1:]
    err0 = Q[model.dn].to_numpy()[t0] - y[:, a]
    lead = (np.arange(1, H + 1) * DT_H)[None, :]
    corr = sim + (err0[:, None] * np.exp(-lead / correct_h) if correct_h else 0.0)
    return dict(sim=sim, corr=np.maximum(corr, 0), lead_h=np.arange(1, H + 1) * DT_H)
