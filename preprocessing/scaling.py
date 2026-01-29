
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None):

    """Scales unified churn features using Min-Max scaling,
        Fit on train data, apply to test data."""

    feature_cols = [
        "Tenure",
        "Engagement",
        "PriceSensitivity",
        "Inactivity",
        "SupportIssues"
    ]

    scaler = MinMaxScaler()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

    if test_df is not None:
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        return train_df, test_df, scaler

    return train_df, scaler

"""min max ---> scaled_value = (value - min) / (max - min)"""