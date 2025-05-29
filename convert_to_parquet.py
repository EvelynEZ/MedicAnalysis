import pyreadstat
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def convert_dta_to_parquet(dta_path, parquet_path):
    try:
        row_offset = 28500000
        chunk_size = 10000
        parquet_writer = None
        all_columns = None

        while True:
            print(f"Reading row {row_offset}")
            df_chunk, meta = pyreadstat.read_dta(dta_path, row_offset=row_offset, row_limit=chunk_size)
            if df_chunk.empty:
                break

            if all_columns is None:
                all_columns = [col for col in df_chunk.columns if not col.startswith((
                    "PCLA", "PRCCSR", "DXCCSR", "PRDAY2", "CMR", "MDC", "DISPUNIFORM",
                    "PRDAY", "TRAN_OUT", "I10_BIRTH", "I10_DELIVERY", "I10_PR", "H_CONTRL", "HOSP",
                    "PRDAY", "TRAN_OUT", "I10_BIRTH", "I10_DELIVERY", "I10_PR", "H_CONTRL", "HOSP", "APRDRG", "I10_ECAUSE",
                    "HCUP_ED", "N_", "S_", "TOTCHG", "TRAN_IN", "PRVER", "AGE_NEONATE",
                    # "I10",
                ))]
                print(f"all_columns - {len(all_columns)}: {all_columns}")

            df_chunk = df_chunk[all_columns]
            # df_chunk = df_chunk.dropna()
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(parquet_path, table.schema)
            # import pytest
            # pytest.set_trace()
            parquet_writer.write_table(table)
            row_offset += chunk_size
    except BaseException as e:
        print(f"An error occurred at row={row_offset}: {e}")
    finally:
        if parquet_writer:
            parquet_writer.close()
            print(f"Finished writing first 5 rows to {parquet_path}")

def save_first_n_rows_as_csv(parquet_path, csv_path, n=1000, batch_size=1000):
    """
    Reads the first `n` rows from a Parquet file and writes them to a CSV.

    Args:
        parquet_path (str): Path to the input .parquet file.
        csv_path (str): Path to the output .csv file.
        n (int): Number of rows to read.
        batch_size (int): Number of rows to read per batch.

    Returns:
        None
    """
    pf = pq.ParquetFile(parquet_path)
    batches = []
    row_count = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        remaining = n - row_count
        if len(df) > remaining:
            df = df.iloc[:remaining]
        batches.append(df)
        row_count += len(df)
        if row_count >= n:
            break

    if batches:
        final_df = pd.concat(batches, ignore_index=True)
        final_df.to_csv(csv_path, index=False)
        print(f"Saved first {n} rows to {csv_path}")
    else:
        print("No data found in Parquet file.")


# dta_path = "~/Downloads/nis_combined_2016to2020.dta"
# parquet_path = "nis_28500000.parquet"
# convert_dta_to_parquet(dta_path, parquet_path)
# save_first_n_rows_as_csv(parquet_path, "nis_first_50000_rows.csv", n=50000, batch_size=10000)


def merge_parquet_files(input_files, output_file, drop_columns=None, batch_size=10000):
    writer = None
    base_schema = None

    for file in input_files:
        pf = pq.ParquetFile(file)
        print(f"Reading {file}")
        iteration = 0
        for batch in pf.iter_batches(batch_size=batch_size):
            print(f"Iteration {iteration}")
            iteration += 1
            table = pa.Table.from_batches([batch])

            # Drop unwanted columns
            if drop_columns:
                drop_columns_set = set(drop_columns)
                keep_columns = [col for col in table.column_names if col not in drop_columns_set]
                table = table.select(keep_columns)

            if writer is None:
                base_schema = table.schema
                writer = pq.ParquetWriter(output_file, base_schema)
            else:
                # Ensure schema matches
                table = table.cast(base_schema)

            writer.write_table(table)

    if writer:
        writer.close()
        print(f"Merged and saved to {output_file}")

# input_files = ["nis_1_7140000.parquet", "nis_7140000_14340000.parquet", "nis_14340000_21400000.parquet", "nis_21400000_28500000.parquet", "nis_28500000.parquet"]
# merge_parquet_files(input_files, "nis_16_20.parquet", ["I10_NECAUSE", "I10_INJURY", "I10_MULTINJURY", "I10_SERVICELINE"])

# pf_1 = pq.ParquetFile("nis_1_7140000.parquet")
# pf_2 = pq.ParquetFile("nis_7140000_14340000.parquet")
# print(f"pf_1 schema: {pf_1.schema}")
# print(f"pf_2 schema: {pf_2.schema}")
pf = pq.ParquetFile("nis_16_20.parquet")
print(f"pf metadata: {pf.metadata}")