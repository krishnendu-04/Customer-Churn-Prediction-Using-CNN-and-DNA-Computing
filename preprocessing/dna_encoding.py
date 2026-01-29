
import numpy as np
import pandas as pd

def map_value_to_base(value: float) -> str:
    
    """Maps a scaled value in [0,1] to a DNA base."""

    if value < 0.25:
        return "A"
    elif value < 0.50:
        return "T"
    elif value < 0.75:
        return "G"
    else:
        return "C"


def one_hot_encode_base(base: str):
    mapping = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "C": [0, 0, 0, 1]
    }
    return mapping[base]




from preprocessing.dna_encoding import map_value_to_base
def dna_encode_features(df: pd.DataFrame, feature_cols: list):

    """Converts scaled numerical features into DNA-based representations,
        Input values are in the range [0, 1]."""

    dna_encoded_data = []

    for _, row in df[feature_cols].iterrows():
        dna_row = []

        for value in row:
            base = map_value_to_base(value)
            dna_row.append(one_hot_encode_base(base))

        dna_encoded_data.append(dna_row)

    return np.array(dna_encoded_data)



def reshape_for_cnn(dna_encoded_data):

    """Ensures DNA encoded data is in CNN-compatible shape."""

    return dna_encoded_data.astype("float32")
