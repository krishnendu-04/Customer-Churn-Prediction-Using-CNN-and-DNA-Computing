# strategy_engine/strategy_mapper.py

STRATEGY_MAP = {
    "Low Engagement": [
        "Personalized content recommendations",
        "Genre-based push notifications",
        "Because-you-watched email campaign"
    ],

    "High Inactivity": [
        "Re-engagement notifications",
        "Free premium days",
        "Resume-watching reminders"
    ],

    "Price Sensitivity": [
        "Temporary discount offer",
        "Plan downgrade suggestion",
        "Bundled subscription offers"
    ],

    "Poor Service Experience": [
        "Priority customer support",
        "Apology credit",
        "Dedicated human callback"
    ],

    "Short Tenure": [
        "Onboarding walkthrough",
        "Starter content packs",
        "Trial period extension"
    ]
}

def generate_personalized_strategy(
    churn_reasons: list,
    churn_probability: float
) -> list:
    """
    Generates confidence-weighted personalized strategies
    based on churn probability intensity.
    """

    strategies = []

    # Decide intensity based on probability
    if churn_probability >= 0.7:
        intensity = "high"
    elif churn_probability >= 0.4:
        intensity = "medium"
    else:
        intensity = "low"

    for reason in churn_reasons:
        if reason in STRATEGY_MAP:

            if intensity == "low":
                strategies.append(STRATEGY_MAP[reason][0])

            elif intensity == "medium":
                strategies.extend(STRATEGY_MAP[reason][:2])

            else:  # high intensity
                strategies.extend(STRATEGY_MAP[reason])

    return list(set(strategies))
