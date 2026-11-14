"""Unit-Test (synthetische Daten, nur zur Prüfung der Rechnung): das Netz in JAX (ml_fit.py) und in numpy (lateral.py) liefern denselben
Zwischengebietsabfluss; mit Netz aus (letzte Schicht null) ist es (bis auf den Boden 1e-4 der Anteile) das Modell ohne "ml"."""
import numpy as np
import jax.numpy as jnp

import lateral as L
import ml_fit as M


def test_ml():
    rng = np.random.default_rng(1)
    E, T, lead = 5, 400, 96
    rain = np.where(rng.random((E, T)) < 0.08, rng.gamma(1.2, 4.0, (E, T)), 0.0)
    api = np.maximum.accumulate(rain.cumsum(axis=1) * 0.05, axis=1) + 5
    aps = rain.cumsum(axis=1) * 0.02
    bf = rng.uniform(0.1, 3.0, E)
    ex = np.stack([rain * 0.9] + [np.maximum(rain - f, 0) * 1.3 for f in (4, 10, 20)], axis=-1).astype("float32")
    r = L.Rain(rain, api, lead, aps, bf, ex)
    base = L.Lateral("X", 120.0, 0.35, float(np.log(60.0)), float(np.log(3.0)), float(np.log(20.0)), ("bfa", "satx", "nlrx"), w_bf=15.0, log_kf=float(np.log(2.5)),
                     rs=0.9, log_sev=float(np.log(150.0)), g_ev=0.2, log_oref=float(np.log(5.0)), a_nl=1.2)
    th = M.init(0, 16, dict(log_kf=base.log_kf, log_oref=base.log_oref, a_nl=base.a_nl))
    phi = L._slow_coefficient(r, base); quick = L._quick_part(r, base, phi)
    a = dict(X=jnp.asarray(L.ml_features(r, phi, quick)), phi=jnp.asarray(phi), q0=jnp.asarray(quick / np.maximum(1 - phi, 1e-6)), rain=jnp.asarray(rain),
             k1=float(np.exp(base.log_k1)), k2=float(np.exp(base.log_k2)), area=base.area_km2, lead=lead)
    y0 = L.response(r, base)
    assert np.abs(np.asarray(M.lat_flow(th, a)) - y0).max() < 2e-3 * y0.max(), "Netz aus: nicht gleich dem Grundmodell"   # bis auf den Boden 1e-4 der Anteile
    th["net"]["W3"] = jnp.asarray(rng.normal(0, 0.5, (16, 2))); th["net"]["b3"] = jnp.asarray([0.3, -0.2])
    net = {k: np.asarray(v).tolist() for k, v in th["net"].items()}
    lt = base.with_terms(base.terms + ("ml",)); lt.net = [net]
    y1 = L.response(r, lt)
    yj = np.asarray(M.lat_flow(th, a))
    assert np.abs(yj - y1).max() < 1e-8 * y1.max(), np.abs(yj - y1).max()
    assert np.abs(y1 - y0).max() > 0.01 * y0.max()
    lt2 = L.Lateral.from_dict(lt.to_dict())
    assert np.allclose(L.response(r, lt2), y1)
    print("ml ok (synthetische Daten)")


if __name__ == "__main__":
    test_ml()
