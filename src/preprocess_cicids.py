import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

root = Path(__file__).resolve().parent.parent
data_dir = root / "data" / "processed_cicids"
out_dir = root / "data" / "features_cicids"
out_dir.mkdir(parents=True, exist_ok=True)

drop_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"]

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
    "DNS_TTL_ANSWER"
]

def prepare(df):
    x = df.drop(columns=drop_cols).copy()
    x[log_cols] = np.log1p(x[log_cols])
    return x

scaler = StandardScaler()

print("Fitting scaler...")

for chunk in pd.read_csv(data_dir / "train.csv", chunksize=200000):
    x = prepare(chunk)
    scaler.partial_fit(x)

print("Scaler fitted.")

feature_names = list(prepare(pd.read_csv(data_dir / "train.csv", nrows=1)).columns)

joblib.dump(scaler, out_dir / "scaler.joblib")
joblib.dump(feature_names, out_dir / "feature_names.joblib")

sizes = {
    "train": 13225372,
    "val": 1889339,
    "test": 3778678
}

for name, size in sizes.items():
    print(f"\nProcessing {name}...")

    x_out = np.lib.format.open_memmap(
        out_dir / f"X_{name}.npy",
        mode="w+",
        dtype=np.float32,
        shape=(size, 41)
    )

    y_out = np.lib.format.open_memmap(
        out_dir / f"y_{name}.npy",
        mode="w+",
        dtype=np.int64,
        shape=(size,)
    )

    start = 0

    for chunk in pd.read_csv(data_dir / f"{name}.csv", chunksize=200000):
        x = prepare(chunk)
        x = scaler.transform(x).astype(np.float32)
        y = chunk["Label"].to_numpy(dtype=np.int64)

        end = start + len(chunk)

        x_out[start:end] = x
        y_out[start:end] = y

        start = end
        print(f"{name}: {start:,}/{size:,}")

    x_out.flush()
    y_out.flush()

print("\nDone.")