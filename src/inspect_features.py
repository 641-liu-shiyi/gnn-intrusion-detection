import pandas as pd
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "data" / "NetFlow_v2_Features.csv"

df = pd.read_csv(path)

print(df.to_string(index=False))