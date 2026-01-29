# models/cnn_dna/train.py

import numpy as np
from models.cnn_dna.model import build_cnn_model
from sklearn.utils.class_weight import compute_class_weight

def train_model(X_train, y_train):
    """
    Trains the CNN model with class imbalance handling.
    """

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_cnn_model(input_shape)

    # Compute class weights

    classes = np.array([0, 1])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight_dict = {
        0: class_weights[0],
        1: class_weights[1]
    }

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        verbose=1
    )

    return model, history

