import duckdb
from pathlib import Path

root = Path(__file__).resolve().parent.parent

input_path = root / "data" / "NF-CICIDS2018-v3.csv"
output_dir = root / "data" / "processed_cicids"
output_dir.mkdir(parents=True, exist_ok=True)

db_path = root / "data" / "cicids_processing.duckdb"
temp_dir = root / "data" / "duckdb_temp"
temp_dir.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(db_path))

# Limit memory usage and allow DuckDB to spill to disk
con.execute("SET memory_limit='4GB'")
con.execute(f"SET temp_directory='{temp_dir}'")

print("Loading, deduplicating, and sorting CICIDS v3...")

con.execute(f"""
CREATE OR REPLACE TABLE cicids_sorted AS
SELECT
    *,
    ROW_NUMBER() OVER (
        ORDER BY FLOW_START_MILLISECONDS
    ) AS _row_id
FROM (
    SELECT DISTINCT *
    FROM read_csv_auto(
        '{input_path}',
        header=true
    )
)
""")

total = con.execute(
    "SELECT COUNT(*) FROM cicids_sorted"
).fetchone()[0]

train_end = int(total * 0.70)
val_end = int(total * 0.80)

print(f"Total unique rows: {total:,}")
print(f"Train: {train_end:,}")
print(f"Validation: {val_end - train_end:,}")
print(f"Test: {total - val_end:,}")

train_path = output_dir / "train.csv"
val_path = output_dir / "val.csv"
test_path = output_dir / "test.csv"

print("\nWriting train split...")

con.execute(f"""
COPY (
    SELECT * EXCLUDE (_row_id)
    FROM cicids_sorted
    WHERE _row_id <= {train_end}
    ORDER BY _row_id
)
TO '{train_path}'
(HEADER, DELIMITER ',')
""")

print("Writing validation split...")

con.execute(f"""
COPY (
    SELECT * EXCLUDE (_row_id)
    FROM cicids_sorted
    WHERE _row_id > {train_end}
      AND _row_id <= {val_end}
    ORDER BY _row_id
)
TO '{val_path}'
(HEADER, DELIMITER ',')
""")

print("Writing test split...")

con.execute(f"""
COPY (
    SELECT * EXCLUDE (_row_id)
    FROM cicids_sorted
    WHERE _row_id > {val_end}
    ORDER BY _row_id
)
TO '{test_path}'
(HEADER, DELIMITER ',')
""")

print("\nTimestamp ranges:")

for split_name, start, end in [
    ("Train", 1, train_end),
    ("Validation", train_end + 1, val_end),
    ("Test", val_end + 1, total),
]:
    result = con.execute(f"""
        SELECT
            MIN(FLOW_START_MILLISECONDS),
            MAX(FLOW_START_MILLISECONDS)
        FROM cicids_sorted
        WHERE _row_id BETWEEN {start} AND {end}
    """).fetchone()

    print(
        f"{split_name}: "
        f"{result[0]} -> {result[1]}"
    )

con.close()

print("\nDone.")