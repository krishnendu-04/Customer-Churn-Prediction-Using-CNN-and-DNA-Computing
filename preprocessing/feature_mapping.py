import pandas as pd

def map_ott_features(df: pd.DataFrame) -> pd.DataFrame:
    mapped_df = pd.DataFrame()

    mapped_df["CustomerID"] = df["CustomerID"]

    # Only add target if present (train data)
    if "Churn" in df.columns:
        mapped_df["Churn"] = df["Churn"]

    
    mapped_df["Tenure"] = df["AccountAge"]


    mapped_df["Engagement"] = (
    df["ViewingHoursPerWeek"] +
    df["AverageViewingDuration"] +
    df["WatchlistSize"] +
    df["ContentDownloadsPerMonth"]
) / 4


    # Encode SubscriptionType safely
    subscription_map = {
        "Basic": 1,
        "Standard": 2,
        "Premium": 3
    }
    subscription_encoded = df["SubscriptionType"].map(subscription_map)
    mapped_df["PriceSensitivity"] = (
        df["MonthlyCharges"] + subscription_encoded
    ) / 2



    # Inactivity = inverse of viewing activity
    inactivity_score = (
        (df["ViewingHoursPerWeek"].max() - df["ViewingHoursPerWeek"]) +
        (df["AverageViewingDuration"].max() - df["AverageViewingDuration"])
    ) / 2
    mapped_df["Inactivity"] = inactivity_score




    # Invert UserRating so lower rating = higher issue
    inverted_rating = 5 - df["UserRating"]
    mapped_df["SupportIssues"] = (
        df["SupportTicketsPerMonth"] + inverted_rating
    ) / 2

    return mapped_df
