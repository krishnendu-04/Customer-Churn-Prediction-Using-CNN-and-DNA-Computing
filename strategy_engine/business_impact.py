# strategy_engine/business_impact.py

# Estimated cost and effectiveness of strategies
STRATEGY_IMPACT = {
    "Personalized content recommendations": {
        "cost": "Low",
        "retention_gain": 0.25
    },
    "Genre-based push notifications": {
        "cost": "Low",
        "retention_gain": 0.15
    },
    "Because-you-watched email campaign": {
        "cost": "Low",
        "retention_gain": 0.20
    },
    "Re-engagement notifications": {
        "cost": "Low",
        "retention_gain": 0.20
    },
    "Free premium days": {
        "cost": "Medium",
        "retention_gain": 0.45
    },
    "Resume-watching reminders": {
        "cost": "Low",
        "retention_gain": 0.10
    },
    "Temporary discount offer": {
        "cost": "High",
        "retention_gain": 0.60
    },
    "Plan downgrade suggestion": {
        "cost": "Medium",
        "retention_gain": 0.40
    },
    "Bundled subscription offers": {
        "cost": "High",
        "retention_gain": 0.55
    },
    "Priority customer support": {
        "cost": "High",
        "retention_gain": 0.50
    },
    "Apology credit": {
        "cost": "Medium",
        "retention_gain": 0.35
    },
    "Dedicated human callback": {
        "cost": "High",
        "retention_gain": 0.45
    },
    "Onboarding walkthrough": {
        "cost": "Low",
        "retention_gain": 0.30
    },
    "Starter content packs": {
        "cost": "Low",
        "retention_gain": 0.25
    },
    "Trial period extension": {
        "cost": "Medium",
        "retention_gain": 0.40
    }
}


def estimate_business_impact(strategies: list, churn_probability: float) -> dict:
    """
    Estimates expected churn reduction and cost profile
    for a given customer.
    """

    total_expected_retention = 0
    costs = set()

    for strategy in strategies:
        if strategy in STRATEGY_IMPACT:
            impact = STRATEGY_IMPACT[strategy]
            total_expected_retention += churn_probability * impact["retention_gain"]
            costs.add(impact["cost"])

    return {
        "ExpectedRetentionGain": round(total_expected_retention, 3),
        "StrategyCostLevels": list(costs)
    }
