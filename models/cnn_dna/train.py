# models/cnn_dna/train.py

import numpy as np
from models.cnn_dna.model import build_cnn_model

def train_model(X_train, y_train):
    """
    Trains the CNN model on DNA-encoded data.
    """

    input_shape = (X_train.shape[1], X_train.shape[2])  # (5, 4)

    model = build_cnn_model(input_shape)

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    return model, history
