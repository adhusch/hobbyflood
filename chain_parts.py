"""chain_fc.run() in Einzelaufträgen (je höchstens 2 h, fortsetzbar):

    python chain_parts.py job g 0        Modelle aus allen Ereignissen 2004-2022, Prüfung 1989-2003 -> data/chain_parts/g0.pkl
    python chain_parts.py job b <0..4>   Block ohne den Prüfblock, Prüfung im Block -> data/chain_parts/b<i>.pkl
    python chain_parts.py combine        wie das Ende von chain_fc.run(): chain_skill.csv, chain_peak.csv, chain_models.json, chain_floods.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

import chain_fc

HERE = Path(__file__).parent
PARTS = HERE / "data" / "chain_parts"


def job(kind: str, b: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"{kind}{b}.pkl"
    if out.exists():
        print(out.name, "schon fertig")
        return
    t = time.time()
    chain_fc.setup_state()
    res = chain_fc.lambda_job((kind, b))
    pickle.dump(res, open(out, "wb"))
    print(f"{out.name} fertig in {time.time() - t:.0f} s", flush=True)


def combine() -> None:
    C = chain_fc
    C.setup_state()
    Q, W, frames, info = C._S["Q"], C._S["W"], C._S["frames"], C._S["info"]
    res = [pickle.load(open(f, "rb")) for f in sorted(PARTS.glob("*.pkl"))]
    blocks = sorted([r for r in res if r["block"] is not None], key=lambda r: r["block"])
    assert len(blocks) == len(C.BLOCKS) and any(r["block"] is None for r in res), "nicht alle Teile fertig"
    glob = next(r for r in res if r["block"] is None)
    s1, p1 = C.summarize(blocks, C.VARIANTS, "cv_2004_2022")
    s3, p3 = C.summarize(blocks, C.VARIANTS, "cv_starkregen", storm_only=True)
    s2, p2 = C.summarize([glob], C.NORAIN, "alt_1989_2003")
    pd.DataFrame(s1 + s3 + s2).to_csv(HERE / "data" / "chain_skill.csv", index=False, float_format="%.4g")
    pd.DataFrame(p1 + p3 + p2).to_csv(HERE / "data" / "chain_peak.csv", index=False, float_format="%.4g")
    ev = {r["block"] if r["block"] is not None else "alt": {t: x["events"] for t, x in r["eval"].items()} for r in res}
    sets = {str(r["block"]): r["models"] for r in blocks}
    sets["alle"] = glob["models"]
    sat = {str(r["block"]) if r["block"] is not None else "alle": r.get("sat") for r in res}
    mlinfo = {str(r["block"]) if r["block"] is not None else "alle": r.get("ml") for r in res}
    consts = dict(A_H=C.A_H, H_H=C.H_H, LEAD_H=C.LEAD_H, WET={k: [C.WET_REF[k], v] for k, v in C.wet_mq().items()}, BFMIN={k: [list(v[0]), list(v[1])] for k, v in C.BF_FLOOR.items()}, TAU_API_H=C.RN.TAU_API_H,
                  TAU_API_S_H=C.RN.TAU_API_S_H, P_STORM=C.P_STORM, STORM_MONTHS=list(C.STORM_MONTHS), STORM_WIN_H=C.STORM_WIN_H)
    hw = {k: dict(**v, area_km2=info[k]["area_km2"], weights=info[k]["weights"]) for k, v in C.HW.items()}
    (HERE / "data" / "chain_models.json").write_text(json.dumps(dict(sets=sets, events=ev, sat=sat, ml=mlinfo, hw=hw, consts=consts), ensure_ascii=False,
                                                                indent=1), encoding="utf-8")
    (HERE / "data" / "chain_floods.json").write_text(json.dumps(C.flood_windows(Q, W, frames, sets), ensure_ascii=False, separators=(",", ":")),
                                                     encoding="utf-8")
    print("zusammengesetzt:", len(blocks), "Blöcke + alle", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "job":
        job(sys.argv[2], int(sys.argv[3]))
    else:
        combine()
