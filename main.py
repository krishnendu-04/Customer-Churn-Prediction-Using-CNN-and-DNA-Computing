"""
main.py  —  Single entry point for the Churn Risk Analytics system.

What this does:
  1. Runs the full CNN-DNA pipeline (train, evaluate, predict)
  2. Prints a results summary to the terminal
  3. Starts a Flask server that serves the frontend + exposes the CSV
  4. Automatically opens the browser to the dashboard
"""

import os
import sys
import threading
import webbrowser
import time

# ── Make sure project root is on the path ────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── ANSI colours for terminal output ─────────────────────────────────────────
G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
M  = "\033[95m"   # magenta
C  = "\033[96m"   # cyan
W  = "\033[97m"   # white bold
DIM= "\033[2m"
RST= "\033[0m"

def banner(text, color=C):
    width = 56
    print(f"\n{color}{'═' * width}{RST}")
    print(f"{color}  {text}{RST}")
    print(f"{color}{'═' * width}{RST}\n")

def section(text, color=B):
    print(f"\n{color}── {text} {'─' * (50 - len(text))}{RST}")


# ════════════════════════════════════════════════════════
#  STEP 1 — RUN THE PIPELINE
# ════════════════════════════════════════════════════════

banner("COMMUNO  CHURN RISK ANALYTICS", color=G)
print(f"{DIM}  Initialising CNN-DNA pipeline...{RST}\n")

import pandas as pd

from preprocessing.feature_mapping import map_ott_features
from preprocessing.scaling import scale_features
from preprocessing.dna_encoding import dna_encode_features, reshape_for_cnn
from models.cnn_dna.train import train_model
from models.cnn_dna.evaluate import evaluate_model
from strategy_engine.churn_reason import compute_non_churn_baseline, identify_churn_reasons
from strategy_engine.strategy_mapper import generate_personalized_strategy
from strategy_engine.business_impact import estimate_business_impact
from sklearn.model_selection import train_test_split


def assign_risk_tier(prob):
    if prob >= 0.7:   return "High Risk"
    elif prob >= 0.4: return "Medium Risk"
    elif prob >= 0.2: return "Low Risk"
    else:             return "Safe"

def recommend_action(risk):
    if risk == "High Risk":    return "Immediate retention offer / discount"
    elif risk == "Medium Risk": return "Engagement nudges & content recommendations"
    elif risk == "Low Risk":   return "Monitor only"
    else:                      return "No action needed"


# 1. Load data
section("Loading data")
train_df = pd.read_csv("data/raw/ott/train.csv")
test_df  = pd.read_csv("data/raw/ott/test.csv")
print(f"  {G}✓{RST} Train set : {W}{len(train_df):,}{RST} rows")
print(f"  {G}✓{RST} Test set  : {W}{len(test_df):,}{RST} rows")

# 2. Feature mapping
section("Feature mapping")
train_mapped     = map_ott_features(train_df)
baseline_profile = compute_non_churn_baseline(train_mapped)
test_mapped      = map_ott_features(test_df)
print(f"  {G}✓{RST} Features mapped — baseline profile computed")

# 3. Labels & features
y_train  = train_mapped["Churn"].values
X_train  = train_mapped.drop(columns=["CustomerID", "Churn"])
X_test   = test_mapped.drop(columns=["CustomerID"])

# 4. Scaling
section("Scaling")
X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
print(f"  {G}✓{RST} Features scaled")

# 5. DNA encoding
section("DNA encoding")
feature_cols  = X_train_scaled.columns.tolist()
X_train_dna   = reshape_for_cnn(dna_encode_features(X_train_scaled, feature_cols))
X_test_dna    = reshape_for_cnn(dna_encode_features(X_test_scaled,  feature_cols))
print(f"  {G}✓{RST} DNA encoding complete — shape: {W}{X_train_dna.shape}{RST}")

# 6. Train / val split + train
section("Training CNN")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_dna, y_train, test_size=0.2, random_state=42, stratify=y_train
)
model, history = train_model(X_tr, y_tr)
print(f"  {G}✓{RST} Model trained")

# 7. Evaluation
section("Model evaluation")
metrics = evaluate_model(model, X_val, y_val, threshold=0.4)

print(f"""
  {W}{'Metric':<14}{'Value':>10}{RST}
  {'─' * 26}
  {'Accuracy':<14}{G}{metrics['accuracy']:.4f}{RST}
  {'Precision':<14}{B}{metrics['precision']:.4f}{RST}
  {'Recall':<14}{Y}{metrics['recall']:.4f}{RST}
  {'F1 Score':<14}{M}{metrics['f1_score']:.4f}{RST}
""")
print(f"  {W}Confusion Matrix:{RST}")
cm = metrics["confusion_matrix"]
print(f"  {DIM}             Pred 0   Pred 1{RST}")
print(f"  {DIM}Actual 0  {RST}{W}{str(cm[0][0]):>8}{RST}  {R}{str(cm[0][1]):>6}{RST}")
print(f"  {DIM}Actual 1  {RST}{R}{str(cm[1][0]):>8}{RST}  {G}{str(cm[1][1]):>6}{RST}")

# 8. Predict on test set
section("Running inference on test set")
y_pred_prob = model.predict(X_test_dna)

test_results = test_mapped[["CustomerID"]].copy()
test_results["ChurnProbability"] = y_pred_prob.flatten()
test_results["RiskTier"]          = test_results["ChurnProbability"].apply(assign_risk_tier)
test_results["RecommendedAction"] = test_results["RiskTier"].apply(recommend_action)

churn_reasons_list, strategy_list, impact_list = [], [], []

for idx, row in test_mapped.iterrows():
    risk       = test_results.loc[idx, "RiskTier"]
    churn_prob = test_results.loc[idx, "ChurnProbability"]

    if risk in ["High Risk", "Medium Risk"]:
        reasons  = identify_churn_reasons(row, baseline_profile)
        strategy = generate_personalized_strategy(reasons, churn_probability=churn_prob)
        impact   = estimate_business_impact(strategies=strategy, churn_probability=churn_prob)
    else:
        reasons  = []
        strategy = []
        impact   = {"ExpectedRetentionGain": 0.0, "StrategyCostLevels": []}

    churn_reasons_list.append(reasons)
    strategy_list.append(strategy)
    impact_list.append(impact)

test_results["ChurnReasons"]          = churn_reasons_list
test_results["PersonalizedStrategy"]  = strategy_list
test_results["ExpectedRetentionGain"] = [i["ExpectedRetentionGain"] for i in impact_list]
test_results["StrategyCostLevels"]    = [i["StrategyCostLevels"]    for i in impact_list]

# Sort by risk priority
risk_priority = {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2, "Safe": 3}
test_results["_p"] = test_results["RiskTier"].map(risk_priority)
test_results = test_results.sort_values(by=["_p", "ChurnProbability"], ascending=[True, False])
test_results = test_results.drop(columns=["_p"])

# 9. Save CSV
os.makedirs("reports", exist_ok=True)
OUTPUT_CSV = "reports/ott_churn_risk_report.csv"
final_cols = ["CustomerID","ChurnProbability","RiskTier","ChurnReasons",
              "PersonalizedStrategy","ExpectedRetentionGain","StrategyCostLevels"]
test_results[final_cols].to_csv(OUTPUT_CSV, index=False)
print(f"  {G}✓{RST} Predictions complete")


# ════════════════════════════════════════════════════════
#  STEP 2 — PRINT TERMINAL SUMMARY
# ════════════════════════════════════════════════════════

banner("RESULTS SUMMARY", color=W)

total  = len(test_results)
high   = (test_results["RiskTier"] == "High Risk").sum()
medium = (test_results["RiskTier"] == "Medium Risk").sum()
low    = (test_results["RiskTier"] == "Low Risk").sum()
safe   = (test_results["RiskTier"] == "Safe").sum()
avg_p  = test_results["ChurnProbability"].mean() * 100

print(f"  {W}Total customers analysed : {G}{total:>6,}{RST}")
print(f"  {W}High Risk                : {R}{high:>6,}  ({high/total*100:.1f}%){RST}")
print(f"  {W}Medium Risk              : {Y}{medium:>6,}  ({medium/total*100:.1f}%){RST}")
print(f"  {W}Low Risk                 : {B}{low:>6,}  ({low/total*100:.1f}%){RST}")
print(f"  {W}Safe                     : {G}{safe:>6,}  ({safe/total*100:.1f}%){RST}")
print(f"  {W}Avg Churn Probability    : {M}{avg_p:.1f}%{RST}")

section("Top 10 High-Risk Customers", color=R)
top10 = test_results[test_results["RiskTier"] == "High Risk"].head(10)
if len(top10):
    print(f"  {DIM}{'CustomerID':<20} {'ChurnProb':>10}  {'Action'}{RST}")
    print(f"  {'─' * 60}")
    for _, row in top10.iterrows():
        print(f"  {W}{str(row['CustomerID']):<20}{RST} {R}{row['ChurnProbability']:.4f}{RST}    {DIM}{row['RecommendedAction']}{RST}")
else:
    print(f"  {DIM}No high-risk customers found.{RST}")

print(f"\n  {G}✓{RST} Full report saved → {W}{OUTPUT_CSV}{RST}\n")


# ════════════════════════════════════════════════════════
#  STEP 3 — FLASK SERVER + OPEN BROWSER
# ════════════════════════════════════════════════════════

banner("LAUNCHING DASHBOARD", color=C)

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder=None)
CORS(app)

FRONTEND_DIST = os.path.join(ROOT, "dist")   # built Vite output
FRONTEND_DEV  = os.path.join(ROOT, "src")    # fallback hint

# ── API: serve the results CSV as JSON ───────────────────────────────────────
@app.route("/api/results", methods=["GET"])
def get_results():
    df = pd.read_csv(OUTPUT_CSV)
    return jsonify(df.to_dict(orient="records"))

# ── API: health check ────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "rows": len(test_results)})

# ── Serve built frontend (npm run build) ─────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if not os.path.isdir(FRONTEND_DIST):
        return (
            f"<pre style='font-family:monospace;padding:40px'>"
            f"<b>Backend is running ✓</b>\n\n"
            f"API ready at <a href='/api/results'>/api/results</a>\n\n"
            f"To see the full dashboard:\n"
            f"  1. Open a second terminal\n"
            f"  2. Run: npm run dev\n"
            f"  3. Visit: http://localhost:5173\n"
            f"</pre>",
            200,
        )
    target = os.path.join(FRONTEND_DIST, path)
    if path and os.path.exists(target):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")


PORT = 5000

def open_browser():
    """Wait for Flask to start, then open the browser."""
    time.sleep(1.4)
    url = f"http://localhost:{PORT}"
    print(f"  {G}✓{RST} Opening browser → {W}{url}{RST}\n")
    webbrowser.open(url)

# Launch browser in background thread
threading.Thread(target=open_browser, daemon=True).start()

print(f"  {C}Server starting on http://localhost:{PORT}{RST}")
print(f"  {DIM}Press Ctrl+C to stop{RST}\n")

app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)