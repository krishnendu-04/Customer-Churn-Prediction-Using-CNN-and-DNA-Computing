import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from models.cnn_dna.model import build_cnn_model  # ← ADD THIS

def train_model(X_train, y_train):

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)

    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight_dict = dict(zip(classes, class_weights))

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )

    return model, history