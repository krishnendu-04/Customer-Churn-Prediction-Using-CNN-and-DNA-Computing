def map_ott_features(df: pd.DataFrame) -> pd.DataFrame:
    mapped_df = pd.DataFrame()

    mapped_df["CustomerID"] = df["CustomerID"]

    # Only add target if present (train data)
    if "Churn" in df.columns:
        mapped_df["Churn"] = df["Churn"]

    mapped_df["Tenure"] = None
    mapped_df["Engagement"] = None
    mapped_df["PriceSensitivity"] = None
    mapped_df["Inactivity"] = None
    mapped_df["SupportIssues"] = None

    return mapped_df
