import sys
import numpy as np

from fusion_model import predict, TABULAR_FEATURES


if len(sys.argv) != 21:
    print(
        "Usage: python src/run_fusion.py IMAGE_PATH AUDIO_PATH "
        + " ".join(TABULAR_FEATURES)
    )
    sys.exit(1)


image_path = sys.argv[1]
audio_path = sys.argv[2]

try:
    tabular_values = [
        float(value)
        for value in sys.argv[3:]
    ]
except ValueError:
    print("All tabular values must be numeric.")
    sys.exit(1)


if len(tabular_values) != len(TABULAR_FEATURES):
    print(
        f"Expected {len(TABULAR_FEATURES)} tabular values, "
        f"received {len(tabular_values)}."
    )
    sys.exit(1)


print("\nInput features:")

for feature, value in zip(
    TABULAR_FEATURES,
    tabular_values
):
    print(
        f"{feature}: {value}"
    )


predict(
    image_path,
    audio_path,
    tabular_values
)