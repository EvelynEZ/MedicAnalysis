import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyarrow.parquet as pq

from pyarrow.parquet import ParquetFile
import pyarrow as pa


def is_tls(row):
    return any(str(code).startswith('E88.3') or str(code).startswith('E883') for code in row)


lymphoma_prefixes = [
    'C81', 'C82', 'C83', 'C84', 'C85', 'C86', 'C8800', 'C88.40', 'C8840', 'C88.20', 'C8820', 'C91.10', 'C9110','C91.12', 'C9112', 'C91.40', 'C9140','C91.42', 'C9142'
]
lymphoma_remission_codes = [
    'C81.0A', 'C81.1A', 'C81.2A', 'C81.3A', 'C81.4A', 'C81.7A', 'C81.9A',
    'C82.0A', 'C82.1A', 'C82.2A', 'C82.3A', 'C82.4A', 'C82.5A', 'C82.6A', 'C82.8A', 'C82.9A',
    'C83.0A', 'C83.1A', 'C83.3A', 'C83.5A', 'C83.7A', 'C83.8A', 'C83.9A',
    'C84.0A', 'C84.1A', 'C84.4A', 'C84.6A', 'C84.7A', 'C84.AA', 'C84.ZA', 'C84.9A',
    'C85.1A', 'C85.2A', 'C85.8A', 'C85.9A',
    'C86.01', 'C86.11', 'C86.21', 'C86.31', 'C86.41', 'C86.51', 'C86.61',
    'C88.01', 'C88.41', 'C91.11', 'C91.41'
]


def collect_lymphoma_data():
    pf = ParquetFile('nis_combined_complete.parquet')
    batch_size = 100_000

    filtered_batches = []

    remission_set = set(lymphoma_remission_codes)
    prefix_tuple = tuple(lymphoma_prefixes)

    for batch_num, batch in enumerate(pf.iter_batches(batch_size=batch_size), 1):
        print(f"Processing batch {batch_num}")
        df_chunk = pa.Table.from_batches([batch]).to_pandas()

        dx_cols = [col for col in df_chunk.columns if col.startswith('I10_DX')]
        dx = df_chunk[dx_cols].astype(str)

        # Flatten to numpy for vectorized operations
        dx_np = dx.values

        # Vectorized: mask for codes starting with any lymphoma prefix
        prefix_regex = r'^(' + '|'.join(map(repr, prefix_tuple)).replace("'", "") + ')'
        prefix_mask = np.apply_along_axis(lambda row: any(pd.Series(row).str.match(prefix_regex)), 1, dx_np)

        # Vectorized: mask for codes matching any remission code
        remission_mask = np.isin(dx_np, list(remission_set)).any(axis=1)

        # Lymphoma: has prefix, not in remission
        lymphoma_mask = prefix_mask & ~remission_mask

        df_chunk['LYMPHOMA'] = lymphoma_mask

        df_lymph_chunk = df_chunk[df_chunk['LYMPHOMA']].copy()
        filtered_batches.append(df_lymph_chunk)

    df_lymph = pd.concat(filtered_batches, ignore_index=True)
    print(f"Total lymphoma records after batch processing: {df_lymph.shape}")
    df_lymph.to_parquet('filtered_lymphoma.parquet', index=False)
    return df_lymph


def count_rows_by_year(df):
    if 'YEAR' not in df.columns:
        raise ValueError("YEAR column not found in DataFrame")
    year_counts = df.groupby('YEAR').size().reset_index(name='Row_Count')
    print(year_counts)
    return year_counts


def count_rows_by_prefix(df, prefixes):
    dx_cols = [col for col in df.columns if col.startswith('I10_DX')]
    results = []
    for prefix in prefixes:
        mask = df[dx_cols].apply(lambda row: any(str(code).startswith(prefix) for code in row), axis=1)
        count = mask.sum()
        results.append({'Prefix': prefix, 'Row_Count': count})
    return pd.DataFrame(results, columns=['Prefix', 'Row_Count'])

def count_complications():
    # Define prefixes, exclusions, and descriptions
    prefixes = {
        'N17': (None, 'Acute renal failure'),
        'E875': (None, 'Hyperkalemia'),
        'E8339': (None, 'Hyperphosphatemia'),
        'E8351': (None, 'Hypocalcemia'),
        'Z49': (None, 'Renal dialysis'),
        'I47': (None, 'Paroxysmal tachycardia, SVT, VT'),
        'I48': (['I482'], 'Afib, Aflutter (exclude chronic Afib)'),
        'I49': (None, 'Other cardiac arrhythmias'),
        'R56.9': (None, 'Seizure'),
        'I50': (['I5022', 'I5032', 'I5033', 'I5042', 'I5043', 'I50812', 'I50813'], 'Heart failure (exclude chronic)'),
        'E877': (['E8771'], 'Fluid overload (exclude TACO)'),
        'J960': (None, 'Acute respiratory failure'),
        'J969': (None, 'Acute respiratory failure'),
        'J80': (None, 'ARDS'),
        'Z9911': (None, 'Ventilator dependence'),
        'I46': (None, 'Cardiac arrest'),
    }

    dx_cols = [col for col in df.columns if col.startswith('I10_DX')]
    results = []

    for prefix, (exclude, description) in prefixes.items():
        mask = df[dx_cols].apply(lambda row: any(str(code).startswith(prefix) for code in row), axis=1)
        if exclude:
            for ex in exclude:
                mask &= ~df[dx_cols].apply(lambda row: any(str(code).startswith(ex) for code in row), axis=1)
        count = mask.sum()
        results.append({'Prefix': prefix, 'Row_Count': count, 'Description': description})

    results_df = pd.DataFrame(results, columns=['Prefix', 'Row_Count', 'Description'])
    print(results_df)

def c83_code_summary(df):
    dx_cols = [col for col in df.columns if col.startswith('I10_DX')]
    # Filter rows with any code starting with C83
    c83_mask = df[dx_cols].apply(lambda row: any(str(code).startswith('C83') for code in row), axis=1)
    c83_df = df[c83_mask].copy()
    # Find all unique C83 code types in these rows
    c83_codes = set()
    for row in c83_df[dx_cols].values:
        c83_codes.update([str(code) for code in row if str(code).startswith('C83')])
    # Count rows for each C83 code type
    summary = []
    for code in sorted(c83_codes):
        code_mask = c83_df[dx_cols].apply(lambda row: any(str(c) == code for c in row), axis=1)
        count = code_mask.sum()
        summary.append({'C83_Code': code, 'Row_Count': count})
    summary_df = pd.DataFrame(summary, columns=['C83_Code', 'Row_Count'])
    print(summary_df)
    return summary_df


def count_lymph_type():
    dx_cols = [col for col in df.columns if col.startswith('I10_DX')]

    counts = []
    for prefix in lymphoma_prefixes:
        mask = df[dx_cols].apply(lambda row: any(str(code).startswith(prefix) for code in row), axis=1)
        total = mask.sum()
        tls = df.loc[mask, 'TLS'].sum()
        counts.append({'Prefix': prefix, 'Total': total, 'TLS': tls})

    counts_df = pd.DataFrame(counts, columns=['Prefix', 'Total', 'TLS'])
    print(counts_df)


def group_rows_by_c_code(df):
    dx_cols = [col for col in df.columns if col.startswith('I10_DX')]
    c_codes = set()
    # Collect all unique C* codes
    for row in df[dx_cols].values:
        c_codes.update([str(code) for code in row if str(code).startswith('C')])
    # Count rows for each C* code
    results = []
    for code in sorted(c_codes):
        mask = df[dx_cols].apply(lambda row: any(str(c) == code for c in row), axis=1)
        count = mask.sum()
        results.append({'C_Code': code, 'Row_Count': count})
    result_df = pd.DataFrame(results, columns=['C_Code', 'Row_Count'])
    print(result_df)
    return result_df



def group_rows_by_c_code_from_parquet(parquet_path, batch_size=100_000):
    pf = ParquetFile(parquet_path)
    c_code_counts = {}

    counter = 1
    for batch in pf.iter_batches(batch_size=batch_size):
        print(f"Processing batch {counter}")
        counter += 1
        df_chunk = pa.Table.from_batches([batch]).to_pandas()
        dx_cols = [col for col in df_chunk.columns if col.startswith('I10_DX')]
        dx_values = df_chunk[dx_cols].astype(str).values

        # For each row, count how many codes start with 'C'
        c_code_mask = np.char.startswith(dx_values.astype(str), 'C')
        c_code_counts_per_row = c_code_mask.sum(axis=1)

        # Only keep rows with exactly one 'C' code
        single_c_mask = c_code_counts_per_row == 1
        single_c_rows = dx_values[single_c_mask]
        c_codes = single_c_rows[c_code_mask[single_c_mask]]

        # Count occurrences using numpy
        unique, counts = np.unique(c_codes, return_counts=True)
        for code, count in zip(unique, counts):
            c_code_counts[code] = c_code_counts.get(code, 0) + count

    result_df = pd.DataFrame(
        [{'C_Code': code, 'Row_Count': count} for code, count in sorted(c_code_counts.items())],
        columns=['C_Code', 'Row_Count']
    )
    print(result_df)
    result_df.to_csv('cancer.csv', index=False)
    return result_df


def count_z511_dx1_rows(parquet_path, batch_size=100_000):
    pf = ParquetFile(parquet_path)
    value_counts = {}

    counter = 1
    for batch in pf.iter_batches(batch_size=batch_size):
        print(f"Processing batch {counter}")
        counter += 1
        df_chunk = pa.Table.from_batches([batch]).to_pandas()
        if 'I10_DX1' in df_chunk.columns:
            dx1 = df_chunk['I10_DX1'].astype(str).values
            mask = np.char.startswith(dx1.astype(str), ('Z51.1', 'Z511')).any(axis=-1) if dx1.ndim > 1 else \
                np.char.startswith(dx1.astype(str), 'Z51.1') | np.char.startswith(dx1.astype(str), 'Z511')
            filtered = dx1[mask]
            unique, counts = np.unique(filtered, return_counts=True)
            for val, cnt in zip(unique, counts):
                value_counts[val] = value_counts.get(val, 0) + cnt

    for val, cnt in value_counts.items():
        print(f"{val}: {cnt} rows")
    pd.DataFrame(list(value_counts.items()), columns=['Z51.1_Code', 'Row_Count']).to_csv('z511_dx1_counts.csv',
                                                                                         index=False)
    return value_counts


def group_c_codes_for_z511_dx1(parquet_path, output_csv='z511_combined_c_code_counts.csv', batch_size=100_000):
    pf = ParquetFile(parquet_path)
    z5111_counts = {}
    z5112_counts = {}

    for batch_num, batch in enumerate(pf.iter_batches(batch_size=batch_size), 1):
        print(f"Processing batch {batch_num}")
        df = pa.Table.from_batches([batch]).to_pandas()
        if 'I10_DX1' not in df.columns:
            continue
        mask = df['I10_DX1'].astype(str).isin(['Z5111', 'Z5112'])
        filtered = df[mask]
        if filtered.empty:
            continue
        dx_cols = [col for col in filtered.columns if col.startswith('I10_DX')]
        dx_values = filtered[dx_cols].astype(str).values

        # Count how many codes start with 'C' in each row
        c_mask = np.char.startswith(dx_values.astype(str), 'C')
        c_code_counts_per_row = c_mask.sum(axis=1)
        single_c_mask = c_code_counts_per_row == 1

        # Get the C code for each row with exactly one C code
        single_c_rows = dx_values[single_c_mask]
        single_c_mask_flat = c_mask[single_c_mask]
        c_codes = single_c_rows[single_c_mask_flat]

        # Get corresponding Z5111/Z5112 for these rows
        z511x = filtered['I10_DX1'].values[single_c_mask]

        # Count for each C code and Z5111/Z5112
        for code, z in zip(c_codes, z511x):
            if z == 'Z5111':
                z5111_counts[code] = z5111_counts.get(code, 0) + 1
            elif z == 'Z5112':
                z5112_counts[code] = z5112_counts.get(code, 0) + 1

    all_c_codes = sorted(set(z5111_counts) | set(z5112_counts))
    combined_df = pd.DataFrame({
        'Z5111': [z5111_counts.get(code, 0) for code in all_c_codes],
        'Z5112': [z5112_counts.get(code, 0) for code in all_c_codes],
    }, index=all_c_codes)
    combined_df.index.name = 'C_Code'
    combined_df.to_csv(output_csv)
    print(f"Saved combined Z5111 and Z5112 C code counts to {output_csv}")
# Usage:
group_c_codes_for_z511_dx1('nis_combined_complete.parquet')