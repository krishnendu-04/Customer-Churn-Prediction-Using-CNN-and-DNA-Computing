import pandas as pd

# preprocessing
from preprocessing.feature_mapping import map_ott_features
from preprocessing.scaling import scale_features
from preprocessing.dna_encoding import dna_encode_features, reshape_for_cnn

# model
from models.cnn_dna.train import train_model
from models.cnn_dna.evaluate import evaluate_model

from strategy_engine.churn_reason import (
    compute_non_churn_baseline,
    identify_churn_reasons
)

from strategy_engine.strategy_mapper import generate_personalized_strategy
from strategy_engine.business_impact import estimate_business_impact


def assign_risk_tier(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    elif prob >= 0.2:
        return "Low Risk"
    else:
        return "Safe"

def recommend_action(risk):
    if risk == "High Risk":
        return "Immediate retention offer / discount"
    elif risk == "Medium Risk":
        return "Engagement nudges & content recommendations"
    elif risk == "Low Risk":
        return "Monitor only"
    else:
        return "No action needed"


# 1. Load data
train_df = pd.read_csv("data/raw/ott/train.csv")
test_df = pd.read_csv("data/raw/ott/test.csv")

# 2. Feature mapping
train_mapped = map_ott_features(train_df)
baseline_profile = compute_non_churn_baseline(train_mapped)

test_mapped = map_ott_features(test_df)

# 3. Separate labels
y_train = train_mapped["Churn"].values
y_test = None

X_train = train_mapped.drop(columns=["CustomerID", "Churn"])
X_test = test_mapped.drop(columns=["CustomerID"])

# 4. Scaling
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# 5. DNA encoding
feature_cols = X_train_scaled.columns.tolist()

X_train_dna = dna_encode_features(X_train_scaled, feature_cols)
X_test_dna = dna_encode_features(X_test_scaled, feature_cols)

X_train_dna = reshape_for_cnn(X_train_dna)
X_test_dna = reshape_for_cnn(X_test_dna)

# 6. Train model
model, history = train_model(X_train_dna, y_train)

# Predict churn probabilities for test data
y_pred_prob = model.predict(X_test_dna)

test_results = test_mapped[["CustomerID"]].copy()
test_results["ChurnProbability"] = y_pred_prob.flatten()
test_results["RiskTier"] = test_results["ChurnProbability"].apply(assign_risk_tier)
test_results["RecommendedAction"] = test_results["RiskTier"].apply(recommend_action)
churn_reasons_list = []
personalized_strategy_list = []
business_impact_list = []

for idx, row in test_mapped.iterrows():

    risk = test_results.loc[idx, "RiskTier"]
    churn_prob = test_results.loc[idx, "ChurnProbability"]

    if risk in ["High Risk", "Medium Risk"]:
        reasons = identify_churn_reasons(row, baseline_profile)

        strategy = generate_personalized_strategy(
            reasons,
            churn_probability=churn_prob
        )

        impact = estimate_business_impact(
            strategies=strategy,
            churn_probability=churn_prob
        )

    else:
        reasons = []
        strategy = []
        impact = {
            "ExpectedRetentionGain": 0.0,
            "StrategyCostLevels": []
        }

    churn_reasons_list.append(reasons)
    personalized_strategy_list.append(strategy)
    business_impact_list.append(impact)

test_results["ChurnReasons"] = churn_reasons_list
test_results["PersonalizedStrategy"] = personalized_strategy_list

test_results["ExpectedRetentionGain"] = [
    i["ExpectedRetentionGain"] for i in business_impact_list
]

test_results["StrategyCostLevels"] = [
    i["StrategyCostLevels"] for i in business_impact_list
]



# ---------- SORT BY RISK PRIORITY ----------
# ---------- SORT BY RISK TIER + CHURN PROBABILITY ----------

risk_priority = {
    "High Risk": 0,
    "Medium Risk": 1,
    "Low Risk": 2,
    "Safe": 3
}

# Map risk tier to priority
test_results["RiskPriority"] = test_results["RiskTier"].map(risk_priority)

# Sort:
#RiskPriority (High → Safe)
#ChurnProbability (High → Low)
test_results = test_results.sort_values(
    by=["RiskPriority", "ChurnProbability"],
    ascending=[True, False]
)

# Remove helper column
test_results = test_results.drop(columns=["RiskPriority"])


# ---------- SAVE FINAL REPORT ----------

output_path = "reports/ott_churn_risk_report.csv"

import os
os.makedirs("reports", exist_ok=True)

test_results[
    [
        "CustomerID",
        "ChurnProbability",
        "RiskTier",
        "ChurnReasons",
        "PersonalizedStrategy",
        "ExpectedRetentionGain",
        "StrategyCostLevels"
    ]
].to_csv(output_path, index=False)

print(f"\nFinal churn report saved to: {output_path}")

