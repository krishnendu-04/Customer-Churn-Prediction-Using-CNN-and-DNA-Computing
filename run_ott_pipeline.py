import pandas as pd

# preprocessing
from preprocessing.feature_mapping import map_ott_features
from preprocessing.scaling import scale_features
from preprocessing.dna_encoding import dna_encode_features, reshape_for_cnn

# model
from models.cnn_dna.train import train_model
from models.cnn_dna.evaluate import evaluate_model


# 1. Load data
train_df = pd.read_csv("data/raw/ott/train.csv")
test_df = pd.read_csv("data/raw/ott/test.csv")

# 2. Feature mapping
train_mapped = map_ott_features(train_df)
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

print(test_results.head())
