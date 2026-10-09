import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

root = Path(__file__).resolve().parent.parent
data_dir = root / "data" / "processed"
out_dir = root / "data" / "features"
out_dir.mkdir(parents=True, exist_ok=True)

drop_cols = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "Label",
    "Attack",
    "MIN_TTL",
    "MAX_TTL",
]

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

train = pd.read_csv(data_dir / "train.csv")
val = pd.read_csv(data_dir / "val.csv")
test = pd.read_csv(data_dir / "test.csv")

def prepare(df):
    x = df.drop(columns=drop_cols).copy()
    x[log_cols] = np.log1p(x[log_cols])
    return x

x_train = prepare(train)
x_val = prepare(val)
x_test = prepare(test)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train).astype(np.float32)
x_val = scaler.transform(x_val).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)

np.save(out_dir / "X_train.npy", x_train)
np.save(out_dir / "X_val.npy", x_val)
np.save(out_dir / "X_test.npy", x_test)

np.save(out_dir / "y_train.npy", train["Label"].to_numpy())
np.save(out_dir / "y_val.npy", val["Label"].to_numpy())
np.save(out_dir / "y_test.npy", test["Label"].to_numpy())

joblib.dump(scaler, out_dir / "scaler.joblib")
joblib.dump(list(train.drop(columns=drop_cols).columns), out_dir / "feature_names.joblib")

print("Train:", x_train.shape)
print("Validation:", x_val.shape)
print("Test:", x_test.shape)

print("\nTrain mean:", round(float(x_train.mean()), 4))
print("Train std:", round(float(x_train.std()), 4))

print("\nNaN:", np.isnan(x_train).sum())
print("Inf:", np.isinf(x_train).sum())