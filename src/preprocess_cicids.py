import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

root = Path(__file__).resolve().parent.parent

data_dir = root / "data" / "processed_cicids"
out_dir = root / "data" / "features_cicids"
unsw_feature_dir = root / "data" / "features"

out_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Use exactly the same feature columns and order
# as UNSW
# --------------------------------------------------

feature_names = joblib.load(
    unsw_feature_dir / "feature_names.joblib"
)

print("Expected number of features:", len(feature_names))
print("Feature names:")
print(feature_names)

# Verify CICIDS v3 contains every required feature
header = pd.read_csv(
    data_dir / "train.csv",
    nrows=0
)

missing = [
    col
    for col in feature_names
    if col not in header.columns
]

if missing:
    raise ValueError(
        f"CICIDS is missing required UNSW features: {missing}"
    )

print("\nCICIDS contains all required UNSW features.")

# Columns that receive log1p transformation
log_cols = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "DNS_TTL_ANSWER",
]


def prepare(df):
    # Select exactly the same 39 predictive features as UNSW
    x = df[feature_names].copy()

    x[log_cols] = np.log1p(
        x[log_cols]
    )

    return x


# --------------------------------------------------
# Fit scaler using training data only
# --------------------------------------------------

scaler = StandardScaler()

print("\nFitting scaler...")

for chunk in pd.read_csv(
    data_dir / "train.csv",
    chunksize=200000
):
    x = prepare(chunk)
    scaler.partial_fit(x)

print("Scaler fitted.")

joblib.dump(
    scaler,
    out_dir / "scaler.joblib"
)

joblib.dump(
    feature_names,
    out_dir / "feature_names.joblib"
)

# Current chronological v3 split sizes
sizes = {
    "train": 13640938,
    "val": 1948706,
    "test": 3897411,
}

# --------------------------------------------------
# Transform each split
# --------------------------------------------------

for name, size in sizes.items():

    print(f"\nProcessing {name}...")

    x_out = np.lib.format.open_memmap(
        out_dir / f"X_{name}.npy",
        mode="w+",
        dtype=np.float32,
        shape=(size, len(feature_names)),
    )

    y_out = np.lib.format.open_memmap(
        out_dir / f"y_{name}.npy",
        mode="w+",
        dtype=np.int64,
        shape=(size,),
    )

    start = 0

    for chunk in pd.read_csv(
        data_dir / f"{name}.csv",
        chunksize=200000
    ):

        x = prepare(chunk)

        x = scaler.transform(
            x
        ).astype(np.float32)

        y = chunk[
            "Label"
        ].to_numpy(dtype=np.int64)

        end = start + len(chunk)

        x_out[start:end] = x
        y_out[start:end] = y

        start = end

        print(
            f"{name}: "
            f"{start:,}/{size:,}"
        )

    x_out.flush()
    y_out.flush()

    if start != size:
        raise ValueError(
            f"{name} row count mismatch: "
            f"expected {size}, got {start}"
        )

print("\nDone.")