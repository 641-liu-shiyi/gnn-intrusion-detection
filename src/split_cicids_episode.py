
"""Episode-aware chronological split for CICIDS2018-v3."""

from pathlib import Path

import duckdb
import pandas as pd


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "NF-CICIDS2018-v3.csv"

OUTPUT = ROOT / "data" / "processed_cicids_episode"

WORK_DB = ROOT / "data" / "cicids_episode_processing.duckdb"

TEMP = ROOT / "data" / "duckdb_temp_episode"


if not INPUT.is_file():
    raise FileNotFoundError(INPUT)

OUTPUT.mkdir(parents=True, exist_ok=True)
TEMP.mkdir(parents=True, exist_ok=True)


con = duckdb.connect(str(WORK_DB))

con.execute("SET memory_limit='4GB'")
con.execute("SET threads=4")

quoted_temp = str(TEMP).replace("'", "''")
con.execute(f"SET temp_directory='{quoted_temp}'")


DATE_EXPR = (
    "CAST(epoch_ms("
    "CAST(FLOW_START_MILLISECONDS AS BIGINT)"
    ") AS DATE)"
)


# --------------------------------------------------
# 2. Load and deduplicate dataset
# --------------------------------------------------

print(
    "1/6 Reading and deduplicating input...",
    flush=True
)

con.execute("""
    CREATE OR REPLACE TABLE unique_flows AS
    SELECT DISTINCT *
    FROM read_csv_auto(?, header=true)
""", [str(INPUT)])


source_count = con.execute(
    "SELECT COUNT(*) FROM unique_flows"
).fetchone()[0]

print(f"Unique flows: {source_count:,}", flush=True)


# --------------------------------------------------
# 3. Identify attack episodes
# --------------------------------------------------

print(
    "2/6 Identifying attack-day episodes...",
    flush=True
)

episodes = con.execute(f"""
    SELECT
        {DATE_EXPR} AS attack_date,
        Attack,
        MIN(
            CAST(FLOW_START_MILLISECONDS AS BIGINT)
        ) AS first_ts,
        MAX(
            CAST(FLOW_START_MILLISECONDS AS BIGINT)
        ) AS last_ts,
        COUNT(*) AS n
    FROM unique_flows
    WHERE Attack IS NOT NULL
      AND Attack <> 'Benign'
    GROUP BY 1, 2
    ORDER BY 1, 3, 4, 2
""").fetchdf()


if episodes.empty:
    raise ValueError("No attack episodes found.")


# Merge overlapping attack intervals.
# Overlapping attacks share chronological boundaries.

interval_rows = []
member_rows = []
group_id = 0

for date, day in episodes.groupby(
    "attack_date", sort=True
):
    current = None

    day = day.sort_values(
        ["first_ts", "last_ts", "Attack"]
    )

    for row in day.itertuples(index=False):

        start = int(row.first_ts)
        end = int(row.last_ts)

        if current is None or start > current["end_ts"]:

            if current is not None:
                interval_rows.append(current)

            group_id += 1

            current = {
                "group_id": group_id,
                "attack_date": str(date)[:10],
                "start_ts": start,
                "end_ts": end,
            }

        else:
            current["end_ts"] = max(
                current["end_ts"], end
            )

        member_rows.append({
            "group_id": group_id,
            "attack_date": str(date)[:10],
            "Attack": row.Attack,
        })

    if current is not None:
        interval_rows.append(current)


con.register(
    "interval_input",
    pd.DataFrame(interval_rows)
)

con.register(
    "member_input",
    pd.DataFrame(member_rows)
)


con.execute("""
    CREATE OR REPLACE TABLE episode_intervals AS
    SELECT
        CAST(group_id AS INTEGER) AS group_id,
        CAST(attack_date AS DATE) AS attack_date,
        CAST(start_ts AS BIGINT) AS start_ts,
        CAST(end_ts AS BIGINT) AS end_ts
    FROM interval_input
""")


con.execute("""
    CREATE OR REPLACE TABLE episode_members AS
    SELECT
        CAST(group_id AS INTEGER) AS group_id,
        CAST(attack_date AS DATE) AS attack_date,
        Attack
    FROM member_input
""")


print(
    f"Attack-day groups: {len(member_rows)}",
    flush=True
)

print(
    f"Non-overlapping intervals: {len(interval_rows)}",
    flush=True
)


# --------------------------------------------------
# 4. Calculate 70/10/20 chronological boundaries
# --------------------------------------------------

print(
    "3/6 Computing chronological boundaries...",
    flush=True
)


con.execute(f"""
    CREATE OR REPLACE TABLE attack_cutoffs AS
    SELECT
        m.group_id,

        quantile_disc(
            CAST(r.FLOW_START_MILLISECONDS AS BIGINT),
            0.70
        ) AS train_cut,

        quantile_disc(
            CAST(r.FLOW_START_MILLISECONDS AS BIGINT),
            0.80
        ) AS val_cut

    FROM unique_flows r

    JOIN episode_members m
      ON r.Attack = m.Attack
     AND {DATE_EXPR} = m.attack_date

    GROUP BY m.group_id
""")


# Benign flows outside attack intervals
# are split chronologically within each day.

con.execute(f"""
    CREATE OR REPLACE TABLE idle_benign_cutoffs AS
    SELECT
        {DATE_EXPR} AS attack_date,

        quantile_disc(
            CAST(r.FLOW_START_MILLISECONDS AS BIGINT),
            0.70
        ) AS train_cut,

        quantile_disc(
            CAST(r.FLOW_START_MILLISECONDS AS BIGINT),
            0.80
        ) AS val_cut

    FROM unique_flows r

    LEFT JOIN episode_intervals i
      ON {DATE_EXPR} = i.attack_date
     AND CAST(
         r.FLOW_START_MILLISECONDS AS BIGINT
     ) BETWEEN i.start_ts AND i.end_ts

    WHERE r.Attack = 'Benign'
      AND i.group_id IS NULL

    GROUP BY 1
""")


# --------------------------------------------------
# 5. Assign flows to train / val / test
# --------------------------------------------------

print(
    "4/6 Assigning flows to splits...",
    flush=True
)


con.execute(f"""
    CREATE OR REPLACE TABLE assigned AS

    WITH located AS (

        SELECT
            r.*,

            m.group_id AS attack_group_id,
            i.group_id AS benign_group_id,

            a.train_cut AS attack_train_cut,
            a.val_cut AS attack_val_cut,

            b.train_cut AS benign_train_cut,
            b.val_cut AS benign_val_cut,

            d.train_cut AS idle_train_cut,
            d.val_cut AS idle_val_cut

        FROM unique_flows r

        LEFT JOIN episode_members m
          ON r.Attack = m.Attack
         AND {DATE_EXPR} = m.attack_date

        LEFT JOIN episode_intervals i
          ON r.Attack = 'Benign'
         AND {DATE_EXPR} = i.attack_date
         AND CAST(
             r.FLOW_START_MILLISECONDS AS BIGINT
         ) BETWEEN i.start_ts AND i.end_ts

        LEFT JOIN attack_cutoffs a
          ON a.group_id = m.group_id

        LEFT JOIN attack_cutoffs b
          ON b.group_id = i.group_id

        LEFT JOIN idle_benign_cutoffs d
          ON r.Attack = 'Benign'
         AND {DATE_EXPR} = d.attack_date
    ),

    bounded AS (

        SELECT * EXCLUDE (
            attack_group_id,
            benign_group_id,
            attack_train_cut,
            attack_val_cut,
            benign_train_cut,
            benign_val_cut,
            idle_train_cut,
            idle_val_cut
        ),

        COALESCE(
            attack_train_cut,
            benign_train_cut,
            idle_train_cut
        ) AS _train_cut,

        COALESCE(
            attack_val_cut,
            benign_val_cut,
            idle_val_cut
        ) AS _val_cut

        FROM located
    )

    SELECT * EXCLUDE (
        _train_cut,
        _val_cut
    ),

    CASE
        WHEN _train_cut IS NULL
          OR _val_cut IS NULL
        THEN NULL

        WHEN CAST(
            FLOW_START_MILLISECONDS AS BIGINT
        ) <= _train_cut
        THEN 'train'

        WHEN CAST(
            FLOW_START_MILLISECONDS AS BIGINT
        ) <= _val_cut
        THEN 'val'

        ELSE 'test'

    END AS _split

    FROM bounded
""")


# --------------------------------------------------
# 6. Validate assignments
# --------------------------------------------------

stats = con.execute("""
    SELECT
        _split,
        COUNT(*) AS n
    FROM assigned
    GROUP BY _split
    ORDER BY _split
""").fetchdf()


print("\nSplit sizes:")
print(stats.to_string(index=False), flush=True)


assigned_count = int(stats["n"].sum())

if (
    assigned_count != source_count
    or stats["_split"].isna().any()
):
    raise RuntimeError(
        "Unassigned or duplicated flows detected."
    )


attack_counts = con.execute("""
    SELECT
        Attack,

        COUNT(*) FILTER (
            WHERE _split = 'train'
        ) AS train_count,

        COUNT(*) FILTER (
            WHERE _split = 'val'
        ) AS val_count,

        COUNT(*) FILTER (
            WHERE _split = 'test'
        ) AS test_count

    FROM assigned
    GROUP BY Attack
    ORDER BY Attack
""").fetchdf()


print("\nAttack class distribution:")
print(
    attack_counts.to_string(index=False),
    flush=True
)


count_columns = [
    "train_count",
    "val_count",
    "test_count",
]

if attack_counts[count_columns].eq(0).any().any():
    raise RuntimeError(
        "Some attack classes are missing from a split."
    )


# --------------------------------------------------
# 7. Export CSV files
# --------------------------------------------------

print(
    "5/6 Exporting new CSV files...",
    flush=True
)


for split in ["train", "val", "test"]:

    destination = str(
        OUTPUT / f"{split}.csv"
    ).replace("'", "''")

    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (_split)
            FROM assigned
            WHERE _split = '{split}'
            ORDER BY
                CAST(
                    FLOW_START_MILLISECONDS AS BIGINT
                ),
                Attack
        )
        TO '{destination}'
        (HEADER, DELIMITER ',')
    """)

    print(
        f"Saved {split}.csv",
        flush=True
    )


# --------------------------------------------------
# 8. Save attack distribution report
# --------------------------------------------------

print(
    "6/6 Writing class-count report...",
    flush=True
)


attack_counts.to_csv(
    OUTPUT / "attack_split_counts.csv",
    index=False
)


con.close()

print("\nEpisode-aware split completed successfully.")
print(f"Output directory: {OUTPUT}")
