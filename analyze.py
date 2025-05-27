import pandas as pd
import pyarrow.parquet as pq

def analyze_e883_correlation(parquet_file, chunk_size=500_000):
    diagnosis_prefix = "I10_DX"
    e883_counts = {}
    categorical_diff = {}
    numerical_corr = {}

    pq_file = pq.ParquetFile(parquet_file)
    for batch in pq_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()

        diagnosis_columns = [col for col in df.columns if col.startswith(diagnosis_prefix)]
        df['has_E883'] = df[diagnosis_columns].apply(lambda row: 'E883' in row.values, axis=1)

        positive_cases = df[df['has_E883']]
        negative_cases = df[~df['has_E883']]

        for col in df.columns:
            if col in diagnosis_columns or col == 'has_E883':
                continue
            if df[col].dtype == 'object' or df[col].dtype.name.startswith('int'):
                pos_counts = positive_cases[col].value_counts(normalize=True)
                neg_counts = negative_cases[col].value_counts(normalize=True)
                diff = (pos_counts - neg_counts).dropna().abs()
                max_diff = diff.max() if not diff.empty else 0
                if max_diff > 0.1:
                    if col not in categorical_diff:
                        categorical_diff[col] = diff
                    else:
                        categorical_diff[col] = categorical_diff[col].add(diff, fill_value=0)
            elif df[col].dtype.name.startswith('float'):
                correlation = df[col].corr(df['has_E883'].astype(float))
                if col not in numerical_corr:
                    numerical_corr[col] = [correlation]
                else:
                    numerical_corr[col].append(correlation)

    print("\n=== Top Categorical Differences with E883 ===")
    for col, diff in sorted(categorical_diff.items(), key=lambda x: x[1].max(), reverse=True)[:10]:
        print(f"{col}:")
        print(diff.sort_values(ascending=False).head(5))

    print("\n=== Top Numerical Correlations with E883 ===")
    averaged_corr = {col: sum(vals)/len(vals) for col, vals in numerical_corr.items()}
    for col, corr_val in sorted(averaged_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"{col}: correlation = {corr_val:.3f}")

analyze_e883_correlation("nis_combined_complete.parquet")
