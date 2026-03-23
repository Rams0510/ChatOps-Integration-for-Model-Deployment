"""
House Price Predictor — FastAPI app with built-in web UI.
"""
import os
import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager

MODEL_PATH = os.getenv("MODEL_PATH", "app/model.pkl")
ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model from {MODEL_PATH}...")
    ml["data"] = joblib.load(MODEL_PATH)
    print("Model loaded.")
    yield
    ml.clear()

app = FastAPI(
    title="House Price Predictor",
    description="Predict California house prices using a Gradient Boosting model",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Schemas ──────────────────────────────────────────────────────────────────

class HouseFeatures(BaseModel):
    MedInc: float        # Median income in block (in $10,000s)
    HouseAge: float      # Median house age in block
    AveRooms: float      # Average number of rooms
    AveBedrms: float     # Average number of bedrooms
    Population: float    # Block population
    AveOccup: float      # Average house occupancy
    Latitude: float      # Block latitude
    Longitude: float     # Block longitude

    @field_validator("MedInc")
    @classmethod
    def check_income(cls, v):
        if v <= 0:
            raise ValueError("Median income must be positive")
        return v

class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    confidence_range: dict
    model_r2: float

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "data" in ml}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    data = ml["data"]
    pipeline = data["pipeline"]
    scale = data["target_scale"]

    X = np.array([[
        features.MedInc, features.HouseAge, features.AveRooms,
        features.AveBedrms, features.Population, features.AveOccup,
        features.Latitude, features.Longitude
    ]])

    pred = pipeline.predict(X)[0]
    price = pred * scale

    # Confidence range ±12%
    low  = price * 0.88
    high = price * 1.12

    return PredictionResponse(
        predicted_price=round(price, 2),
        predicted_price_formatted=f"${price:,.0f}",
        confidence_range={
            "low": f"${low:,.0f}",
            "high": f"${high:,.0f}",
        },
        model_r2=0.812,
    )

# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=HTML_UI)

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>House Price Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --cream: #F5F0E8;
    --dark: #1A1A2E;
    --accent: #C9A84C;
    --accent2: #E8C878;
    --text: #2D2D2D;
    --muted: #7A7A8A;
    --card: #FFFFFF;
    --border: #E2DDD4;
    --green: #2D6A4F;
    --green-light: #E8F5EE;
  }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream);
    color: var(--text);
    min-height: 100vh;
  }

  /* Hero Header */
  .hero {
    background: var(--dark);
    padding: 3rem 2rem 4rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 10%;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.08) 0%, transparent 70%);
  }
  .hero-inner {
    max-width: 860px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
  }
  .hero-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid rgba(201,168,76,0.4);
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1.2rem;
  }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    color: #F5F0E8;
    line-height: 1.1;
    margin-bottom: 0.8rem;
  }
  .hero h1 em {
    font-style: italic;
    color: var(--accent2);
  }
  .hero p {
    color: rgba(245,240,232,0.6);
    font-size: 1rem;
    font-weight: 300;
    max-width: 480px;
    line-height: 1.6;
  }
  .hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 2rem;
  }
  .stat {
    text-align: left;
  }
  .stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--accent2);
  }
  .stat-label {
    font-size: 11px;
    color: rgba(245,240,232,0.45);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* Main layout */
  .main {
    max-width: 860px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }

  /* Form card */
  .form-card {
    background: var(--card);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid var(--border);
    box-shadow: 0 2px 20px rgba(26,26,46,0.06);
  }
  .section-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.2rem;
  }

  .field-group {
    margin-bottom: 1rem;
  }
  .field-group label {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: var(--muted);
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .field-group input {
    width: 100%;
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 14px;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background: #FAFAF8;
    transition: border-color 0.15s, box-shadow 0.15s;
    outline: none;
  }
  .field-group input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(201,168,76,0.12);
    background: #fff;
  }
  .field-hint {
    font-size: 11px;
    color: var(--muted);
    margin-top: 3px;
  }
  .field-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .predict-btn {
    width: 100%;
    margin-top: 1.2rem;
    padding: 14px;
    background: var(--dark);
    color: var(--accent2);
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .predict-btn:hover {
    background: #252540;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(26,26,46,0.2);
  }
  .predict-btn:active { transform: translateY(0); }
  .predict-btn.loading { opacity: 0.7; pointer-events: none; }

  /* Result card */
  .result-card {
    background: var(--card);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid var(--border);
    box-shadow: 0 2px 20px rgba(26,26,46,0.06);
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .result-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    text-align: center;
    gap: 1rem;
    opacity: 0.4;
  }
  .house-icon {
    font-size: 3rem;
    filter: grayscale(1);
  }
  .result-placeholder p {
    font-size: 13px;
    color: var(--muted);
    max-width: 180px;
    line-height: 1.5;
  }

  .result-main { display: none; }
  .result-main.visible { display: block; animation: fadeUp 0.4s ease; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .price-display {
    background: var(--dark);
    border-radius: 12px;
    padding: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .price-display::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 150px; height: 150px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.12) 0%, transparent 70%);
  }
  .price-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(245,240,232,0.45);
    margin-bottom: 0.5rem;
  }
  .price-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: var(--accent2);
    line-height: 1;
    margin-bottom: 0.4rem;
  }
  .price-range {
    font-size: 12px;
    color: rgba(245,240,232,0.4);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .metric-box {
    background: var(--cream);
    border-radius: 10px;
    padding: 14px;
    border: 1px solid var(--border);
  }
  .metric-box .m-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .metric-box .m-value {
    font-size: 15px;
    font-weight: 500;
    color: var(--text);
  }
  .metric-box.green {
    background: var(--green-light);
    border-color: rgba(45,106,79,0.2);
  }
  .metric-box.green .m-value { color: var(--green); }

  .error-box {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 10px;
    padding: 14px;
    font-size: 13px;
    color: #DC2626;
    display: none;
  }
  .error-box.visible { display: block; }

  /* Sample presets */
  .presets {
    margin-top: 1rem;
  }
  .presets-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
  }
  .preset-pills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .pill {
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: transparent;
    font-size: 12px;
    font-family: 'DM Sans', sans-serif;
    color: var(--muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  .pill:hover {
    border-color: var(--accent);
    color: var(--text);
    background: rgba(201,168,76,0.08);
  }

  @media (max-width: 640px) {
    .main { grid-template-columns: 1fr; }
    .hero-stats { gap: 1.2rem; }
  }
</style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-tag">ML-Powered Valuation</div>
    <h1>Predict <em>House Prices</em><br>with Confidence</h1>
    <p>Gradient Boosting model trained on 20,640 California homes. Enter property details to get an instant estimate.</p>
    <div class="hero-stats">
      <div class="stat">
        <div class="stat-val">81.2%</div>
        <div class="stat-label">R² Accuracy</div>
      </div>
      <div class="stat">
        <div class="stat-val">20,640</div>
        <div class="stat-label">Training Samples</div>
      </div>
      <div class="stat">
        <div class="stat-val">±12%</div>
        <div class="stat-label">Confidence Range</div>
      </div>
    </div>
  </div>
</div>

<div class="main">

  <!-- Left: Form -->
  <div class="form-card">
    <div class="section-label">Property Details</div>

    <div class="field-row">
      <div class="field-group">
        <label>Median Income</label>
        <input type="number" id="MedInc" step="0.1" placeholder="5.0">
        <div class="field-hint">In $10,000s (e.g. 5.0 = $50k)</div>
      </div>
      <div class="field-group">
        <label>House Age</label>
        <input type="number" id="HouseAge" step="1" placeholder="20">
        <div class="field-hint">Years</div>
      </div>
    </div>

    <div class="field-row">
      <div class="field-group">
        <label>Avg Rooms</label>
        <input type="number" id="AveRooms" step="0.1" placeholder="6.0">
        <div class="field-hint">Per household</div>
      </div>
      <div class="field-group">
        <label>Avg Bedrooms</label>
        <input type="number" id="AveBedrms" step="0.1" placeholder="1.0">
        <div class="field-hint">Per household</div>
      </div>
    </div>

    <div class="field-row">
      <div class="field-group">
        <label>Population</label>
        <input type="number" id="Population" step="1" placeholder="1200">
        <div class="field-hint">Block population</div>
      </div>
      <div class="field-group">
        <label>Avg Occupancy</label>
        <input type="number" id="AveOccup" step="0.1" placeholder="3.0">
        <div class="field-hint">People per home</div>
      </div>
    </div>

    <div class="field-row">
      <div class="field-group">
        <label>Latitude</label>
        <input type="number" id="Latitude" step="0.01" placeholder="34.05">
        <div class="field-hint">e.g. 34.05 (LA)</div>
      </div>
      <div class="field-group">
        <label>Longitude</label>
        <input type="number" id="Longitude" step="0.01" placeholder="-118.24">
        <div class="field-hint">e.g. -118.24 (LA)</div>
      </div>
    </div>

    <button class="predict-btn" onclick="predict()" id="predictBtn">
      <span id="btnText">Predict Price</span>
    </button>

    <div class="presets">
      <div class="presets-label">Try a sample</div>
      <div class="preset-pills">
        <button class="pill" onclick="loadPreset('luxury')">Luxury LA</button>
        <button class="pill" onclick="loadPreset('suburban')">Suburban SF</button>
        <button class="pill" onclick="loadPreset('affordable')">Affordable</button>
        <button class="pill" onclick="loadPreset('coastal')">Coastal</button>
      </div>
    </div>
  </div>

  <!-- Right: Result -->
  <div class="result-card">
    <div class="section-label">Prediction Result</div>

    <div class="result-placeholder" id="placeholder">
      <div class="house-icon">🏠</div>
      <p>Fill in the property details and click Predict Price</p>
    </div>

    <div class="result-main" id="resultMain">
      <div class="price-display">
        <div class="price-label">Estimated Value</div>
        <div class="price-value" id="priceValue">—</div>
        <div class="price-range" id="priceRange">—</div>
      </div>

      <div class="metrics-grid" style="margin-top:1rem">
        <div class="metric-box green">
          <div class="m-label">Model R²</div>
          <div class="m-value" id="r2Val">—</div>
        </div>
        <div class="metric-box">
          <div class="m-label">Confidence</div>
          <div class="m-value">±12%</div>
        </div>
        <div class="metric-box">
          <div class="m-label">Low Estimate</div>
          <div class="m-value" id="lowVal">—</div>
        </div>
        <div class="metric-box">
          <div class="m-label">High Estimate</div>
          <div class="m-value" id="highVal">—</div>
        </div>
      </div>
    </div>

    <div class="error-box" id="errorBox"></div>
  </div>

</div>

<script>
const presets = {
  luxury:    { MedInc: 12.5, HouseAge: 15, AveRooms: 8.2, AveBedrms: 1.5, Population: 800,  AveOccup: 2.5, Latitude: 34.07, Longitude: -118.40 },
  suburban:  { MedInc: 6.5,  HouseAge: 25, AveRooms: 6.0, AveBedrms: 1.1, Population: 1500, AveOccup: 3.2, Latitude: 37.60, Longitude: -122.10 },
  affordable:{ MedInc: 2.8,  HouseAge: 40, AveRooms: 4.5, AveBedrms: 1.2, Population: 2000, AveOccup: 3.8, Latitude: 36.20, Longitude: -119.10 },
  coastal:   { MedInc: 9.0,  HouseAge: 20, AveRooms: 7.0, AveBedrms: 1.3, Population: 600,  AveOccup: 2.8, Latitude: 34.01, Longitude: -118.50 },
};

function loadPreset(name) {
  const p = presets[name];
  Object.keys(p).forEach(k => {
    document.getElementById(k).value = p[k];
  });
}

async function predict() {
  const fields = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'];
  const body = {};
  for (const f of fields) {
    const val = parseFloat(document.getElementById(f).value);
    if (isNaN(val)) {
      showError(`Please fill in all fields (${f} is missing).`);
      return;
    }
    body[f] = val;
  }

  const btn = document.getElementById('predictBtn');
  const btnText = document.getElementById('btnText');
  btn.classList.add('loading');
  btnText.textContent = 'Predicting...';
  hideError();

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await resp.json();

    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('priceValue').textContent = data.predicted_price_formatted;
    document.getElementById('priceRange').textContent =
      `Range: ${data.confidence_range.low} — ${data.confidence_range.high}`;
    document.getElementById('r2Val').textContent = (data.model_r2 * 100).toFixed(1) + '%';
    document.getElementById('lowVal').textContent = data.confidence_range.low;
    document.getElementById('highVal').textContent = data.confidence_range.high;

    const rm = document.getElementById('resultMain');
    rm.classList.remove('visible');
    void rm.offsetWidth;
    rm.classList.add('visible');

  } catch (e) {
    showError(e.message);
  } finally {
    btn.classList.remove('loading');
    btnText.textContent = 'Predict Price';
  }
}

function showError(msg) {
  const eb = document.getElementById('errorBox');
  eb.textContent = '⚠ ' + msg;
  eb.classList.add('visible');
}
function hideError() {
  document.getElementById('errorBox').classList.remove('visible');
}
</script>
</body>
</html>
"""
