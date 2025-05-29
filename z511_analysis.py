import pyarrow.parquet as pq
import pandas as pd

NIS_FILE = "NIS_16_20.parquet"
OUTPUT_FOLDER = "z511_analysis_output"

def analyze_z511_c_diagnoses(parquet_path, output_dir, batch_size=100_000):
    pf = pq.ParquetFile(parquet_path)
    columns_of_interest = [f"I10_DX{i}" for i in range(1, 41)]
    analysis_columns = ["AGE", "FEMALE", "RACE", "ZIPINC_QRTL", "DIED", "LOS"]
    c_group_stats = {}
    iteration = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns_of_interest + analysis_columns):
        print(f"Processing batch {iteration}")
        iteration += 1
        df = batch.to_pandas()

        z511_mask = df["I10_DX1"].fillna("").str.startswith("Z511")
        z511_df = df[z511_mask].copy()

        c_dx_mask = z511_df[columns_of_interest].apply(lambda col: col.astype(str).str.startswith("C") & col.notna())
        c_dx_counts = c_dx_mask.sum(axis=1)
        single_c_mask = c_dx_counts == 1

        z511_df = z511_df[single_c_mask].copy()
        z511_df["C_CODE"] = c_dx_mask.idxmax(axis=1)
        z511_df["C_VALUE"] = z511_df.apply(lambda row: row[row["C_CODE"]], axis=1)

        for c_value, group in z511_df.groupby("C_VALUE"):
            stats = group[analysis_columns].describe(include='all')
            if c_value not in c_group_stats:
                c_group_stats[c_value] = []
            c_group_stats[c_value].append(stats)

    for c_value, stats_list in c_group_stats.items():
        combined_stats = pd.concat(stats_list).groupby(level=0).mean()
        combined_stats.to_csv(f"{output_dir}/stats_{c_value}.csv")

    print(f"Saved grouped statistics by C diagnosis to {output_dir}")

analyze_z511_c_diagnoses(NIS_FILE, OUTPUT_FOLDER)