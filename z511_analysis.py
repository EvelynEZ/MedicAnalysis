import pyarrow.parquet as pq
import pandas as pd

NIS_FILE = "NIS_16_20.parquet"
OUTPUT_FOLDER = "z511_analysis_output"



def count_z511_single_c_diagnoses(parquet_path, batch_size=100_000):
    pf = pq.ParquetFile(parquet_path)
    columns_of_interest = [f"I10_DX{i}" for i in range(1, 30)]
    c_code_counts = {}
    iteration = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns_of_interest):
        print(f"Counting C codes in batch {iteration}")
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

        for c_val in z511_df["C_VALUE"]:
            c_code_counts[c_val] = c_code_counts.get(c_val, 0) + 1

    # Convert to DataFrame and save
    count_df = pd.DataFrame(list(c_code_counts.items()), columns=["C_VALUE", "COUNT"])
    count_df.sort_values(by="COUNT", ascending=False, inplace=True)
    count_df.to_csv(f"z511_c_counts.csv", index=False)
    print(f"Saved C code counts to z511_c_counts.csv")

# count_z511_single_c_diagnoses(NIS_FILE)

def analyze_z511_c_diagnoses(parquet_path, output_dir, batch_size=100_000):
    pf = pq.ParquetFile(parquet_path)
    columns_of_interest = [f"I10_DX{i}" for i in range(1, 30)]
    analysis_columns = ["AGE", "FEMALE", "RACE", "ZIPINC_QRTL", "DIED", "LOS"]
    c_group_data = {}
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
            if c_value not in c_group_data:
                c_group_data[c_value] = []
            c_group_data[c_value].append(group)

    for c_value, group_list in c_group_data.items():
        group = pd.concat(group_list, ignore_index=True)
        summary = {}
        summary["Total Rows"] = [len(group)]

        for col in ["AGE", "LOS"]:
            summary[f"{col}_min"] = [group[col].min()]
            summary[f"{col}_25%"] = [group[col].quantile(0.25)]
            summary[f"{col}_50%"] = [group[col].median()]
            summary[f"{col}_75%"] = [group[col].quantile(0.75)]
            summary[f"{col}_max"] = [group[col].max()]
            summary[f"{col}_std"] = [group[col].std()]

        cat_dfs = []
        for col in ["RACE", "ZIPINC_QRTL", "DIED"]:
            value_counts = group[col].value_counts().sort_index()
            value_counts.name = col
            cat_dfs.append(value_counts)

        summary_df = pd.DataFrame(summary)
        cat_df = pd.concat(cat_dfs, axis=1).fillna(0).astype(int)
        final_df = pd.concat([summary_df.T, cat_df.T])
        final_df.to_csv(f"{output_dir}/stats_{c_value}.csv")

    print(f"Saved grouped statistics by C diagnosis to {output_dir}")

analyze_z511_c_diagnoses(NIS_FILE, OUTPUT_FOLDER)
