import pyreadstat
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq

def convert_dta_to_parquet(dta_path, parquet_path, chunk_size=100000):
    start_offset = 0
    row_offset = start_offset
    parquet_writer = None

    first_chunk, _ = pyreadstat.read_dta(dta_path, row_offset=start_offset, row_limit=chunk_size)
    all_columns = first_chunk.columns.tolist()
    while True:
        print(f"Reading rows from row_offset {row_offset}")
        df_chunk, _ = pyreadstat.read_dta(dta_path, row_offset=row_offset, row_limit=chunk_size)
        if df_chunk.empty:
            break

        # Add missing columns as None to match schema
        for col in all_columns:
            if col not in df_chunk.columns:
                df_chunk[col] = None
        df_chunk = df_chunk[all_columns]

        if parquet_writer is None:
            table = pa.Table.from_pandas(df_chunk)
            parquet_writer = pq.ParquetWriter(parquet_path, table.schema)

        parquet_writer.write_table(table)
        row_offset += chunk_size

    if parquet_writer:
        parquet_writer.close()
        print(f"Saved full dataset to {parquet_path}")

dta_path = "~/Downloads/nis_combined_2016to2020.dta"
parquet_path = "nis_combined_complete.parquet"
convert_dta_to_parquet(dta_path, parquet_path)