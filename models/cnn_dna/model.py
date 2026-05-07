from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Flatten,
    Dense,
    Input,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


def build_cnn_model(input_shape):
    """
    Optimized 1D CNN for DNA-encoded churn prediction
    input_shape: (num_features, num_channels)
    Example: (5, 4)
    """

    model = Sequential([
        # Proper Input Layer (removes warning)
        Input(shape=input_shape),

        # -------- Convolution Block 1 --------
        Conv1D(
            filters=32,
            kernel_size=2,
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),

        # -------- Convolution Block 2 --------
        Conv1D(
            filters=16,
            kernel_size=2,
            activation="relu",
            padding="same"
        ),
        BatchNormalization(),

        # Flatten extracted features
        Flatten(),

        # -------- Dense Block --------
        Dense(32, activation="relu"),
        Dropout(0.3),

        # Output Layer
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc")   # Important for churn problems
        ]
    )

    return model
