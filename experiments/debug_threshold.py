import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, confusion_matrix

# preprocessing
from preprocessing.feature_mapping import map_ott_features
from preprocessing.scaling import scale_features
from preprocessing.dna_encoding import dna_encode_features, reshape_for_cnn

# model
from models.cnn_dna.train import train_model

# 1. Load TRAIN data ONLY (has labels)
df = pd.read_csv("data/raw/ott/train.csv")

# 2. Feature mapping
mapped = map_ott_features(df)

# 3. Separate X and y
y = mapped["Churn"].values
X = mapped.drop(columns=["CustomerID", "Churn"])

# 4. Scale
X_scaled, scaler = scale_features(X)

# 5. DNA encoding
feature_cols = X_scaled.columns.tolist()
X_dna = dna_encode_features(X_scaled, feature_cols)
X_dna = reshape_for_cnn(X_dna)

# 6. Train model
model, history = train_model(X_dna, y)

# 7. Get probabilities
y_prob = model.predict(X_dna).flatten()

# 8. Threshold tuning (THIS IS THE POINT)
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    recall = recall_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    print("\n==============================")
    print(f"Threshold: {t}")
    print(f"Recall: {recall:.3f}")
    print(f"Precision: {precision:.3f}")
    print("Confusion Matrix:")
    print(cm)
