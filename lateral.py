"""Zwischengebiet: Abfluss aus Gebietsniederschlag (Regen-Abfluss-Anteil des Routing-Modells).

Je Zwischengebiet (Fläche zwischen Oberlieger-Pegeln und Unterlieger) vier Grundparameter:
  phi_max   größter Abflussbeiwert (Anteil des Regens, der bei nassem Gebiet abflusswirksam wird)
  S         Anfeuchtungsmaß in mm: phi_a(t) = phi_max * (1 - exp(-x / S)), x = API7(t) = Vorregenindex mit 7 Tagen Abklingzeit (siehe rain.py)
  K1, K2    Zeitkonstanten (h) zweier Linearspeicher in Reihe (schneller und langsamer Anteil)
Q_lat [m³/s] = A [km²] * O2 [mm/h] / 3,6.

Optionale Erweiterungen des Abflussbeiwerts (`terms`), damit große Ereignisse mehr Abfluss geben als kleine:
  "int"  Intensität: phi = phi_a + (1 - phi_a) * rho * P^2 / (P^2 + Pc^2)   (P = Regenintensität des Zeitschritts in mm/h; Pc in mm/h)
         Bei starkem Regen läuft ein Teil des Regens unabhängig von der Vorfeuchte ab (Infiltrationsüberschuss, versiegelte und gesättigte Flächen).
  "acc"  Ereignisfeuchte: x = API7 + w * API1 (API1 = Vorregenindex mit 24 h Abklingzeit)
         Der Abflussbeiwert wächst im Ereignis mit der bisher gefallenen Regenmenge (wie beim SCS-Verfahren), nicht nur mit der Vorfeuchte der letzten Tage.
  "bfa"  Gebietsfeuchte aus dem Abfluss, additiv im Anfeuchtungsmaß: x = API7 + w_bf * B, B = Abfluss am Bezugspegel zu Beginn des Abflussfensters
         geteilt durch MQ (je Fenster ein Wert, Rain.bf). Nach Trockenphasen (B klein) fließt viel weniger ab, als der 7-Tage-Vorregen allein sagt:
         Im Oktober 2026 gaben 65-70 mm Regen nach dem trockenen Sommer nur 7-10 mm Abfluss (Prüm 2 vorher bei 6 % von MQ). Im Ereignis wächst
         x weiter mit dem gefallenen Regen (API7), der Boden kann also nass werden. Genutzt in chain_fc.py (LAT_TERMS).
  "fast" schneller Abflussweg bei Starkregen: der Intensitätsanteil wie bei "int", f = (1 - phi_a) * rho * P^2 / (P^2 + Pc^2), läuft nicht durch
         K1/K2, sondern über zwei schnelle Linearspeicher mit Kf (Oberflächenabfluss); der Rest phi_a * P wie im Grundmodell. Anlass: Mit
         gemessenem Regen blieb die Kette bei den Hochwassern über HQ10 (2016, 2018, 2021) bei 7-33 % des Scheitels, und der Scheitel kam 2-13 h zu
         spät (extreme_fit.py). Schließt "int" aus (gleiche Parameter rho, Pc).
  "sat"  Sättigungsüberschuss bei großen Regenmengen (SCS-Verfahren): zusätzlicher Anteil (1 - phi_a) * rs * g(Pe; Sv), g = Ableitung der
         SCS-Abflussfunktion (Pe - 0,2 Sv)^2 / (Pe + 0,8 Sv) nach Pe (0 unter 0,2 Sv, wächst mit Pe gegen 1); Pe = Ereignisregen (Vorregenindex
         mit 24 h Abklingzeit, api_s), Rückhalt Sv = sev * exp(-g_ev * B) (B = Gebietsfeuchte wie bei "bfa"). Läuft wie "fast" über zwei schnelle
         Speicher mit Kf. Anlass: Bei 100-136 mm Regen auf nassem Gebiet (Juli 2021) lag der gemessene Abflussbeiwert bei 0,23-0,66, im Modell bei
         0,03-0,34; nach Trockenheit (September 2013, 60-72 mm) bei 0,03-0,06 wie im Modell (extreme_fit.py).
  "nlr"  nichtlineare Speicher (alle Speicher des Zwischengebiets): Speicherzeit K(S) = K / (1 + (S / K) / oref)^a, also kürzer, je voller
         der Speicher (S / K = Abfluss eines gleich gefüllten linearen Speichers, mm/h); Speicherinhalt wird mitgeführt, die Wasserbilanz geht
         exakt auf. Anlass: Juli 2021 brachte an Prüm 2 und Giesdorf mit gemessenem Regen genug Volumen (Beiwert 0,76 gegen 0,58-0,65 gemessen),
         aber nur den halben Scheitel; die Welle lief durch Speicher mit 17-25 h (extreme_fit.py). "nlrx": dieselben, mit festen Parametern.
  "ml"   lernender Abflussbeiwert (ml_fit.py): ein kleines Netz (11 Eingänge, 2 x 16 Knoten, für alle Gebiete dasselbe) verschiebt je Zeitschritt
         den langsamen Anteil phi_a und den schnellen Anteil (aus "satx") auf der Logit-Skala: s = sig(logit(phi_a) + d1),
         q = (1 - s) * sig(logit(q0) + d2), q0 = schneller Anteil / (1 - phi_a). Mit d1 = d2 = 0 (Netz aus) ist es das Modell ohne "ml" (bis auf den Boden 1e-4 der Anteile, damit das Netz auch aus null etwas machen kann).
         Eingänge: Vorregen 7 Tage, 24 h und 3 h, Gebietsfeuchte, Regen (Stationen), Radar-Gebietsmittel, Starkregenanteile aus den Radarpixeln
         (Regen über 4, 10, 20 mm/h, Flächenmittel; Rain.ex), dazu phi_a und q0. Anlass: Das Radar-Gebietsmittel als Ersatz für den Stationsregen
         war schlechter, und ein einzelner Starkregen-Parameter wurde auf 0,03 angepasst; es kommt auf die Kombination mit der Feuchte an.
  "bf"   dasselbe als Faktor: phi = phi * B^g / (B^g + b50^g). Im Vergleich (Versuch, Oberläufe Prüm 2 und Sinspelt, Prüfung 2019-2022 und
         Oktober 2026) unterschätzt diese Form die Oktoberwelle (ein trockener Fensterbeginn bremst das ganze Ereignis), nicht übernommen.
Ohne `terms` ist das Ergebnis identisch mit dem Grundmodell.

Die Speichergleichungen sind exakt gelöst: Eingang des ersten Speichers je Schritt konstant (Regenintensität des 15-min-Intervalls),
Eingang des zweiten Speichers linear zwischen den Schrittwerten (wie in routing_core.c).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT_H = 0.25
BOUNDS = dict(phi_max=(0.0, 1.0), log_s=(np.log(5.0), np.log(1000.0)), log_k1=(np.log(0.5), np.log(24.0)), log_k2=(np.log(2.0), np.log(96.0)),
              rho=(0.0, 1.0), log_pc=(np.log(2.0), np.log(60.0)), w_acc=(0.0, 10.0), log_b50=(np.log(0.01), np.log(10.0)), log_gb=(np.log(0.2), np.log(3.0)),
              w_bf=(0.0, 1000.0), log_kf=(np.log(0.25), np.log(12.0)), rs=(0.0, 1.0), log_sev=(np.log(10.0), np.log(3000.0)), g_ev=(0.0, 10.0),
              lam=(0.02, 0.9), log_oref=(np.log(0.02), np.log(50.0)), a_nl=(0.0, 2.0))
BASE_KEYS = ("phi_max", "log_s", "log_k1", "log_k2")
TERM_KEYS = {"int": ("rho", "log_pc"), "acc": ("w_acc",), "bf": ("log_b50", "log_gb"), "bfa": ("w_bf",), "fast": ("rho", "log_pc", "log_kf"),
             "sat": ("rs", "log_sev", "g_ev", "log_kf", "lam"), "satx": (), "nlr": ("log_oref", "a_nl"), "nlrx": (), "ml": ()}
TERM_ORDER = ("int", "acc", "bf", "bfa", "fast", "sat", "satx", "nlr", "nlrx", "ml")   # satx: Sättigungsüberschuss mit festen (gemeinsamen) Parametern, nicht angepasst


@dataclass
class Lateral:
    name: str                       # Schlüssel der Regenreihe (z. B. "EC")
    area_km2: float
    phi_max: float = 0.3
    log_s: float = float(np.log(40.0))
    log_k1: float = float(np.log(3.0))
    log_k2: float = float(np.log(18.0))
    terms: tuple = ()               # Erweiterungen des Abflussbeiwerts: "int", "acc", "bf"
    rho: float = 0.2
    log_pc: float = float(np.log(10.0))
    w_acc: float = 1.0
    log_b50: float = float(np.log(0.3))
    log_gb: float = 0.0
    w_bf: float = 20.0
    log_kf: float = 0.0
    rs: float = 0.0
    log_sev: float = float(np.log(200.0))
    g_ev: float = 1.0
    lam: float = 0.2
    log_oref: float = 0.0
    a_nl: float = 0.0
    net: dict | list | None = None  # Gewichte des Netzes für "ml" (ml_fit.py): W1, b1, W2, b2, W3, b3 als Listen

    def keys(self) -> list[str]:
        if "int" in self.terms and "fast" in self.terms:
            raise ValueError("Erweiterungen 'int' und 'fast' schließen sich aus")
        out = list(BASE_KEYS)
        for t in TERM_ORDER:
            if t in self.terms:
                out += [k for k in TERM_KEYS[t] if k not in out]
        return out

    def theta(self) -> list[float]:
        return [float(getattr(self, k)) for k in self.keys()]

    def bounds(self) -> tuple[list[float], list[float]]:
        ks = self.keys()
        return [float(BOUNDS[k][0]) for k in ks], [float(BOUNDS[k][1]) for k in ks]

    def with_theta(self, th) -> "Lateral":
        d = {k: float(v) for k, v in zip(self.keys(), th)}
        return Lateral(self.name, self.area_km2, d["phi_max"], d["log_s"], d["log_k1"], d["log_k2"], self.terms,
                       d.get("rho", self.rho), d.get("log_pc", self.log_pc), d.get("w_acc", self.w_acc), d.get("log_b50", self.log_b50),
                       d.get("log_gb", self.log_gb), d.get("w_bf", self.w_bf), d.get("log_kf", self.log_kf), d.get("rs", self.rs),
                       d.get("log_sev", self.log_sev), d.get("g_ev", self.g_ev), d.get("lam", self.lam),
                       d.get("log_oref", self.log_oref), d.get("a_nl", self.a_nl), self.net)

    def with_terms(self, terms) -> "Lateral":
        """Gleiche Grundparameter, andere Erweiterungen (Zusatzparameter mit Standardstartwerten, falls noch nie gesetzt)."""
        return Lateral(self.name, self.area_km2, self.phi_max, self.log_s, self.log_k1, self.log_k2, tuple(terms), self.rho, self.log_pc, self.w_acc,
                       self.log_b50, self.log_gb, self.w_bf, self.log_kf, self.rs, self.log_sev, self.g_ev, self.lam, self.log_oref,
                       self.a_nl, self.net)

    def to_dict(self) -> dict:
        d = dict(name=self.name, area_km2=self.area_km2, phi_max=self.phi_max, log_s=self.log_s, log_k1=self.log_k1, log_k2=self.log_k2)
        if self.terms:
            d.update(terms=list(self.terms), rho=self.rho, log_pc=self.log_pc, w_acc=self.w_acc)
            if "bf" in self.terms:
                d.update(log_b50=self.log_b50, log_gb=self.log_gb)
            if "bfa" in self.terms:
                d.update(w_bf=self.w_bf)
            if "fast" in self.terms or "sat" in self.terms or "satx" in self.terms:
                d.update(log_kf=self.log_kf)
            if "ml" in self.terms:
                d.update(net=self.net)
            if "sat" in self.terms or "satx" in self.terms:
                d.update(rs=self.rs, log_sev=self.log_sev, g_ev=self.g_ev, lam=self.lam)
            if "nlr" in self.terms or "nlrx" in self.terms:
                d.update(log_oref=self.log_oref, a_nl=self.a_nl)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Lateral":
        return Lateral(d["name"], d["area_km2"], d["phi_max"], d["log_s"], d["log_k1"], d["log_k2"], tuple(d.get("terms", ())),
                       d.get("rho", 0.2), d.get("log_pc", float(np.log(10.0))), d.get("w_acc", 1.0), d.get("log_b50", float(np.log(0.3))),
                       d.get("log_gb", 0.0), d.get("w_bf", 20.0), d.get("log_kf", 0.0), d.get("rs", 0.0), d.get("log_sev", float(np.log(200.0))),
                       d.get("g_ev", 1.0), d.get("lam", 0.2), d.get("log_oref", 0.0), d.get("a_nl", 0.0), d.get("net"))


@dataclass
class Rain:
    """Regenfenster um die Ereignisse: rain [E, T_ext] in mm/h, api [E, T_ext] in mm (7 Tage), api_s [E, T_ext] in mm (24 h, nur für "acc");
    die ersten `lead` Schritte dienen nur dem Einschwingen."""
    rain: np.ndarray
    api: np.ndarray
    lead: int
    api_s: np.ndarray | None = None
    bf: np.ndarray | None = None          # Gebietsfeuchte je Fenster (Abfluss / MQ am Bezugspegel zu Beginn des Abflussfensters), nur für "bf"
    ex: np.ndarray | None = None          # Radargrößen [E, T_ext, 4] in mm/h: Gebietsmittel und Starkregenanteile über 4, 10, 20 mm/h (rain_fit.XCOLS), nur für "ml"

    def subset(self, keep: np.ndarray) -> "Rain":
        return Rain(self.rain[keep], self.api[keep], self.lead, None if self.api_s is None else self.api_s[keep],
                    None if self.bf is None else self.bf[keep], None if self.ex is None else self.ex[keep])


def _fast_part(r: Rain, lat: Lateral, phi_a: np.ndarray) -> np.ndarray:
    p2 = np.square(r.rain)
    return (1.0 - phi_a) * lat.rho * p2 / (p2 + np.exp(2.0 * lat.log_pc))


def scs_slope(pe: np.ndarray, sv: np.ndarray, lam: float = 0.2) -> np.ndarray:
    """Ableitung der SCS-Abflussfunktion Q(P) = (P - lam S)^2 / (P + (1 - lam) S) nach P: 0 bis lam S (Anfangsverlust), dann steigend gegen 1."""
    a = np.maximum(pe - lam * sv, 0.0)
    return a * (pe + (2.0 - lam) * sv) / np.square(pe + (1.0 - lam) * sv)


def _sat_part(r: Rain, lat: Lateral, phi_a: np.ndarray) -> np.ndarray:
    if r.api_s is None or r.bf is None:
        raise ValueError("Erweiterung 'sat' braucht api_s und bf im Regenfenster")
    sv = np.exp(lat.log_sev) * np.exp(-lat.g_ev * np.maximum(np.asarray(r.bf, float), 0.0))[:, None]
    return (1.0 - phi_a) * lat.rs * scs_slope(r.api_s, sv, lat.lam)


def _quick_part(r: Rain, lat: Lateral, phi_a: np.ndarray) -> np.ndarray:
    """Anteil über den schnellen Weg (fast und/oder sat), je Schritt als Abflussbeiwert."""
    q = 0.0
    if "fast" in lat.terms:
        q = q + _fast_part(r, lat, phi_a)
    if "sat" in lat.terms or "satx" in lat.terms:
        q = q + _sat_part(r, lat, phi_a)
    return q


ML_TAU3_H = 3.0
ML_EXCESS_F = (4.0, 10.0, 20.0)      # Schwellen mm/h der Starkregenanteile (wie rain_fit.EXCESS_F)
ML_NF = 11


def ml_features(r: Rain, phi_a: np.ndarray, quick: np.ndarray, dt_h: float = DT_H) -> np.ndarray:
    """Eingänge des Netzes [E, T_ext, 11] (Skalierung fest, ml_fit.py rechnet dieselben in JAX)."""
    if r.api_s is None or r.bf is None:
        raise ValueError("Erweiterung 'ml' braucht api_s und bf im Regenfenster")
    from scipy.signal import lfilter
    c = float(np.exp(-dt_h / ML_TAU3_H))
    a3 = lfilter([dt_h], [1.0, -c], r.rain, axis=1)
    bf = np.minimum(np.maximum(np.asarray(r.bf, float), 0.0), 10.0)[:, None] * np.ones_like(r.rain)
    q0 = quick / np.maximum(1.0 - phi_a, 1e-6)
    if r.ex is None:        # Betrieb ohne Radar (Standard): dieselben Größen aus dem Gebietsmittel der Stationen
        ex = np.stack([r.rain] + [np.maximum(r.rain - f, 0.0) for f in ML_EXCESS_F], axis=-1)
    else:
        ex = np.asarray(r.ex, float)
    return np.stack([np.log1p(r.api) / 4.0, np.log1p(r.api_s) / 4.0, np.log1p(a3) / 3.0, np.log1p(bf), np.log1p(r.rain) / 3.0,
                     np.log1p(ex[..., 0]) / 3.0, np.log1p(ex[..., 1]) / 2.0, np.log1p(ex[..., 2]) / 2.0, np.log1p(ex[..., 3]) / 2.0, phi_a, q0], axis=-1)


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x))


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ml_fractions(r: Rain, lat: Lateral, phi_a: np.ndarray, quick: np.ndarray, dt_h: float = DT_H) -> tuple[np.ndarray, np.ndarray]:
    """Langsamer und schneller Anteil des Regens nach dem Netz (je [E, T_ext])."""
    if not lat.net:
        raise ValueError("Erweiterung 'ml' ohne Netz")
    nets = lat.net if isinstance(lat.net, list) else [lat.net]      # mehrere Netze (Startwerte): Mittel der Verschiebungen
    x = ml_features(r, phi_a, quick, dt_h)
    d = 0.0
    for n in nets:
        h = np.tanh(x @ np.asarray(n["W1"]) + np.asarray(n["b1"]))
        h = np.tanh(h @ np.asarray(n["W2"]) + np.asarray(n["b2"]))
        d = d + (h @ np.asarray(n["W3"]) + np.asarray(n["b3"])) / len(nets)
    s = _sig(_logit(phi_a) + d[..., 0])
    q = (1.0 - s) * _sig(_logit(quick / np.maximum(1.0 - phi_a, 1e-6)) + d[..., 1])
    return s, q


def _has_quick(lat: Lateral) -> bool:
    return any(t in lat.terms for t in ("fast", "sat", "satx"))


def runoff_coefficient(r: Rain, lat: Lateral) -> np.ndarray:
    """Abflussbeiwert phi(t) [E, T_ext] (bei "fast"/"sat" einschließlich des schnellen Anteils)."""
    phi = _slow_coefficient(r, lat)
    return phi + _quick_part(r, lat, phi) if _has_quick(lat) else phi


def _slow_coefficient(r: Rain, lat: Lateral) -> np.ndarray:
    x = r.api
    if "acc" in lat.terms:
        if r.api_s is None:
            raise ValueError("Erweiterung 'acc' braucht api_s im Regenfenster")
        x = x + lat.w_acc * r.api_s
    if "bfa" in lat.terms:
        if r.bf is None:
            raise ValueError("Erweiterung 'bfa' braucht bf im Regenfenster")
        x = x + lat.w_bf * np.maximum(np.asarray(r.bf, float), 0.0)[:, None]
    phi = lat.phi_max * (1.0 - np.exp(-x / np.exp(lat.log_s)))
    if "int" in lat.terms:
        p2 = np.square(r.rain)
        phi = phi + (1.0 - phi) * lat.rho * p2 / (p2 + np.exp(2.0 * lat.log_pc))
    if "bf" in lat.terms:
        if r.bf is None:
            raise ValueError("Erweiterung 'bf' braucht bf im Regenfenster")
        g = np.exp(lat.log_gb)
        b = np.maximum(np.asarray(r.bf, float), 1e-6)[:, None] ** g
        phi = phi * b / (b + np.exp(g * lat.log_b50))
    return phi


def _nl_stores(peff: np.ndarray, k1: float, k2: float, dt_h: float, oref: float, a: float) -> np.ndarray:
    """Zwei nichtlineare Speicher in Reihe: Inhalt S je Speicher, Speicherzeit K(S) = K / (1 + S / (K oref))^a je Schritt, Zufluss im Schritt
    konstant; Abfluss je Schritt aus der Bilanz (Zufluss - Änderung des Inhalts), also massentreu. Ausgabe: Abfluss des zweiten Speichers am
    Schrittende (S / K)."""
    E, T = peff.shape
    out = np.empty((E, T))
    s1 = k1 * peff[:, 0]                                   # Start im Gleichgewicht des linearen Speichers
    s2 = k2 * peff[:, 0]
    out[:, 0] = peff[:, 0]
    for t in range(1, T):
        kt1 = k1 / np.power(1.0 + s1 / (k1 * oref), a)
        e1 = np.exp(-dt_h / kt1)
        s1n = s1 * e1 + peff[:, t] * kt1 * (1.0 - e1)
        i2 = peff[:, t] - (s1n - s1) / dt_h                  # mittlerer Abfluss des ersten Speichers im Schritt
        s1 = s1n
        kt2 = k2 / np.power(1.0 + s2 / (k2 * oref), a)
        e2 = np.exp(-dt_h / kt2)
        s2n = s2 * e2 + i2 * kt2 * (1.0 - e2)
        out[:, t] = i2 - (s2n - s2) / dt_h
        s2 = s2n
    return out


def _core():
    """Rechenkern in C (routing_core.c, über routing._load_lib); False ohne Compiler. Spät importiert, weil routing dieses Modul importiert."""
    import routing
    return routing._load_lib()


def _two_stores(peff: np.ndarray, k1: float, k2: float, dt_h: float, nl: tuple | None = None) -> np.ndarray:
    lib = _core()
    if lib:                                                # dieselbe Rechnung in C (Raspberry Pi: Python-Schleife war 2/3 der Laufzeit)
        x = np.ascontiguousarray(peff, dtype=np.float64)
        out = np.empty_like(x)
        if nl is not None:
            lib.nl_stores(x, out, x.shape[0], x.shape[1], float(k1), float(k2), float(dt_h), float(nl[0]), float(nl[1]))
        else:
            lib.two_stores(x, out, x.shape[0], x.shape[1], float(k1), float(k2), float(dt_h))
        return out
    if nl is not None:
        return _nl_stores(peff, k1, k2, dt_h, nl[0], nl[1])
    a1, a2 = np.exp(-dt_h / k1), np.exp(-dt_h / k2)
    w2 = 1.0 - k2 * (1.0 - a2) / dt_h
    E, T = peff.shape
    o1 = np.empty((E, T))
    o2 = np.empty((E, T))
    o1[:, 0] = o2[:, 0] = peff[:, 0]                       # Start im Gleichgewicht mit dem ersten Wert
    for t in range(1, T):
        o1[:, t] = a1 * o1[:, t - 1] + (1.0 - a1) * peff[:, t]
        o2[:, t] = a2 * o2[:, t - 1] + (1.0 - a2) * o1[:, t - 1] + (o1[:, t] - o1[:, t - 1]) * w2
    return o2


def response(r: Rain, lat: Lateral, dt_h: float = DT_H) -> np.ndarray:
    """Q_lat [E, T_ext - lead] in m³/s."""
    rain = r.rain
    nl = (float(np.exp(lat.log_oref)), float(lat.a_nl)) if ("nlr" in lat.terms or "nlrx" in lat.terms) else None
    if _has_quick(lat):
        phi_a = _slow_coefficient(r, lat)
        quick = _quick_part(r, lat, phi_a)
        if "ml" in lat.terms:
            phi_a, quick = ml_fractions(r, lat, phi_a, quick, dt_h)
        kf = np.exp(lat.log_kf)
        o = _two_stores(phi_a * rain, np.exp(lat.log_k1), np.exp(lat.log_k2), dt_h, nl) + _two_stores(quick * rain, kf, kf, dt_h, nl)
    else:
        o = _two_stores(runoff_coefficient(r, lat) * rain, np.exp(lat.log_k1), np.exp(lat.log_k2), dt_h, nl)
    return lat.area_km2 / 3.6 * o[:, r.lead:]
