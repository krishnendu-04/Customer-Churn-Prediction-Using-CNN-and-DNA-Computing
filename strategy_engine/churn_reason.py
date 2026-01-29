# strategy_engine/churn_reason.py

import pandas as pd

# Mapping low-level features to high-level churn drivers
CHURN_DRIVERS = {
    "Low Engagement": [
        "Engagement"
    ],

    "High Inactivity": [
        "Inactivity"
    ],

    "Price Sensitivity": [
        "PriceSensitivity"
    ],

    "Poor Service Experience": [
        "SupportIssues"
    ],

    "Short Tenure": [
        "Tenure"
    ]
}


def compute_non_churn_baseline(train_df: pd.DataFrame) -> dict:
    """
    Computes baseline feature values from non-churn customers.
    """

    non_churn_df = train_df[train_df["Churn"] == 0]

    baseline = {
        "Engagement": non_churn_df["Engagement"].mean(),
        "Inactivity": non_churn_df["Inactivity"].mean(),
        "PriceSensitivity": non_churn_df["PriceSensitivity"].mean(),
        "SupportIssues": non_churn_df["SupportIssues"].mean(),
        "Tenure": non_churn_df["Tenure"].mean()
    }

    return baseline


def identify_churn_reasons(
    customer_row: pd.Series,
    baseline: dict,
    top_k: int = 2
) -> list:
    """
    Identifies top churn drivers for a single customer
    by measuring deviation from non-churn baseline.
    """

    driver_scores = {}

    for driver, features in CHURN_DRIVERS.items():
        score = 0

        for feature in features:
            if feature in customer_row and feature in baseline:
                deviation = abs(customer_row[feature] - baseline[feature])
                score += deviation

        driver_scores[driver] = score

    sorted_drivers = sorted(
        driver_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [driver for driver, _ in sorted_drivers[:top_k]]
