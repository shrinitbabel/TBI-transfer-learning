"""
TBI length-of-stay and mortality API (Render backend for the Babels.AI /tbi-length-of-stay page).

Serves the pooled India + Jordan + California model exported by train_calculator.py
(model.json): a gradient-boosted classifier for in-hospital death and gradient-boosted
quantile trees for length of stay among survivors, with a split-conformal 80% range.
The trees are evaluated in pure Python, so scikit-learn is not needed at runtime.

Per-prediction attribution is single-input substitution: each input is swapped to the
value of a typical pooled patient and the change in the prediction is reported.

Endpoints:
  GET  /health
  GET  /healthz
  GET  /metrics
  POST /predict
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
MODEL = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
VALIDATION = json.loads((ROOT / "validation.json").read_text(encoding="utf-8"))
FEATURES: List[str] = MODEL["features"]
MOI_CATEGORIES: List[str] = MODEL["moi_categories"]
REF_PATIENT: Dict = MODEL["reference_patient"]
REF = MODEL["reference"]
ADJ = MODEL["conformal_adjust"]

LABELS = {
    "age": "Age",
    "sex": "Sex",
    "gcs": "Glasgow Coma Scale",
    "icu": "ICU admission",
    "moi": "Mechanism of injury",
    "hr": "Heart rate",
    "rr": "Respiratory rate",
    "sbp": "Systolic blood pressure",
}


# ------------------------------------------------------------------ model evaluation
def _raw(m: Dict, x: List[float]) -> float:
    s = m["init"]
    for t in m["trees"]:
        n = 0
        while t["l"][n] != -1:
            n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
        s += m["lr"] * t["v"][n]
    return s


def _tachycardia(age: float, hr: float) -> bool:
    return hr > (160 if age < 1 else 150 if age < 3 else 140 if age < 6 else 120 if age < 12 else 100)


def _tachypnea(age: float, rr: float) -> bool:
    return rr > (60 if age < 1 else 40 if age < 3 else 34 if age < 6 else 30 if age < 12 else 20)


def _hypotension(age: float, sbp: float) -> bool:
    return sbp < (70 if age < 1 else (70 + 2 * age) if age < 10 else 90)


def _features(p: Dict) -> List[float]:
    age = max(0.0, float(p["age"]))
    x = {
        "age": age,
        "gcs": float(min(15, max(3, round(p["gcs"])))),
        "male": float(p["male"]),
        "icu": float(p["icu"]),
        # vital signs enter the model as age-specific categories; unknown = within normal limits
        "tachycardia": float(p["hr"] is not None and _tachycardia(age, p["hr"])),
        "tachypnea": float(p["rr"] is not None and _tachypnea(age, p["rr"])),
        "hypotension": float(p["sbp"] is not None and _hypotension(age, p["sbp"])),
    }
    for m in MOI_CATEGORIES:
        x[f"moi={m}"] = float(p["moi"] == m)
    return [x[f] for f in FEATURES]


def _predict(p: Dict) -> Dict[str, float]:
    x = _features(p)
    M = MODEL["models"]
    return {
        "death": 1.0 / (1.0 + math.exp(-_raw(M["death"], x))),
        "los_median": math.exp(_raw(M["q50"], x)),
        "los_low": max(1.0, math.exp(_raw(M["q_lo"], x) - ADJ)),
        "los_high": math.exp(_raw(M["q_hi"], x) + ADJ),
    }


# ------------------------------------------------------------------ API schema
class Patient(BaseModel):
    age: float = Field(..., ge=0, le=110, description="Age in years")
    sex: Literal["male", "female"]
    gcs: int = Field(..., ge=3, le=15, description="First recorded total Glasgow Coma Scale")
    icu: bool = Field(..., description="Admitted to intensive care during the stay")
    moi: str = Field(..., description="Mechanism of injury (see /metrics for categories)")
    hr: Optional[float] = Field(None, ge=20, le=250, description="Heart rate, beats/min; null if unknown")
    rr: Optional[float] = Field(None, ge=4, le=80, description="Respiratory rate, breaths/min; null if unknown")
    sbp: Optional[float] = Field(None, ge=40, le=300, description="Systolic blood pressure, mmHg; null if unknown")


app = FastAPI(title="TBI Length-of-Stay and Mortality API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": "model.json", "n_features": len(FEATURES),
            "trained_on": MODEL["trained_on"]}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return {
        "model": "Gradient-boosted classifier (in-hospital death) + gradient-boosted quantile trees "
                 "on log length of stay among survivors, with split-conformal 80% range",
        "trained_on": MODEL["trained_on"],
        "mechanism_categories": MOI_CATEGORIES,
        "reference": REF,
        "validation": VALIDATION,
        "disclaimer": "Research demonstration only. Not for clinical decision-making.",
    }


def _risk_band(p: float) -> str:
    if p < 0.10:
        return "Low"
    if p < 0.25:
        return "Moderate"
    if p < 0.50:
        return "High"
    return "Very high"


def _value_label(key: str, p: Dict) -> str:
    v = p[key]
    if key == "male":
        return "Male" if v else "Female"
    if key == "icu":
        return "Yes" if v else "No"
    if key in ("hr", "rr", "sbp"):
        if v is None:
            return "Unknown (normal)"
        unit = {"hr": "/min", "rr": "/min", "sbp": " mmHg"}[key]
        return f"{v:g}{unit}"
    if key == "age":
        return f"{v:g} y"
    if key == "gcs":
        return str(int(round(v)))
    return str(v)


def _attribution(p: Dict, base: Dict[str, float]) -> List[Dict]:
    rows = []
    for key in ["age", "male", "gcs", "icu", "moi", "hr", "rr", "sbp"]:
        swapped = dict(p)
        swapped[key] = REF_PATIENT[key]
        alt = _predict(swapped)
        d_death = base["death"] - alt["death"]
        d_los = base["los_median"] - alt["los_median"]
        rows.append({
            "feature": key if key != "male" else "sex",
            "clinical_label": LABELS["sex" if key == "male" else key],
            "value_label": _value_label(key, p),
            "reference_label": _value_label(key, {**p, key: REF_PATIENT[key]}),
            "delta_death": d_death,
            "delta_los_days": d_los,
        })
    rows.sort(key=lambda r: abs(r["delta_death"]), reverse=True)
    return rows


@app.post("/predict")
def predict(patient: Patient):
    if patient.moi not in MOI_CATEGORIES:
        raise HTTPException(status_code=422, detail=f"moi must be one of {MOI_CATEGORIES}")
    try:
        p = patient.model_dump()
        p["male"] = int(p.pop("sex") == "male")
        p["icu"] = int(p["icu"])
        out = _predict(p)
        return {
            "death_probability": out["death"],
            "death_percent": round(out["death"] * 100, 1),
            "risk_band": _risk_band(out["death"]),
            "los_median_days": round(out["los_median"], 1),
            "los_range80_days": [round(out["los_low"], 1), round(out["los_high"], 1)],
            "reference": {
                "death_rate": REF["death_rate"],
                "los_median_days": round(REF["los_median"], 1),
                "los_range80_days": [round(REF["los_p10"], 1), round(REF["los_p90"], 1)],
                "patient": REF_PATIENT,
            },
            "attribution": _attribution(p, out),
            "disclaimer": "Research demonstration only. Not for clinical decision-making.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
