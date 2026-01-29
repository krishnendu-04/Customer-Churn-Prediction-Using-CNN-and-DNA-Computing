
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape):

    """lightweight 1D CNN for DNA-encoded
        input_shape: (num_features, num_channels) → (5, 4)"""

    model = Sequential()

    # 1D Convolution layer
    model.add(
        Conv1D(
            filters=16,
            kernel_size=2,
            activation="relu",
            input_shape=input_shape
        )
    )

    # Flatten learned patterns
    model.add(Flatten())

    # Output layer (binary classification)
    model.add(Dense(1, activation="sigmoid"))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


"""f(x) = max(0, x) relu activation function"""
"""sigmoid(x) = 1 / (1 + e^(-x)) sigmoid activation function"""
"""Loss = −[ y·log(p) + (1−y)·log(1−p) ] binary crossentropy loss function"""