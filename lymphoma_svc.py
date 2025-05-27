import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyarrow.parquet as pq

# Load NIS data
from pyarrow.parquet import ParquetFile
import pyarrow as pa

pf = ParquetFile('nis_combined_complete.parquet')
batch_size = 100_000

filtered_batches = []

lymphoma_prefixes = [
    'C81', 'C82', 'C83', 'C84', 'C85', 'C86'
]
lymphoma_remission_codes = [
    'C81.0A', 'C81.1A', 'C81.2A', 'C81.3A', 'C81.4A', 'C81.7A', 'C81.9A',
    'C82.0A', 'C82.1A', 'C82.2A', 'C82.3A', 'C82.4A', 'C82.5A', 'C82.6A', 'C82.8A', 'C82.9A',
    'C83.0A', 'C83.1A', 'C83.3A', 'C83.5A', 'C83.7A', 'C83.8A', 'C83.9A',
    'C84.0A', 'C84.1A', 'C84.4A', 'C84.6A', 'C84.7A', 'C84.AA', 'C84.ZA', 'C84.9A',
    'C85.1A', 'C85.2A', 'C85.8A', 'C85.9A',
    'C86.0A', 'C86.1A', 'C86.2A', 'C86.3A', 'C86.4A', 'C86.5A', 'C86.6A'
]

def is_lymphoma(row):
    for code in row:
        code_str = str(code)
        if any(code_str.startswith(prefix) for prefix in lymphoma_prefixes):
            if not any(code_str.startswith(rem) for rem in lymphoma_remission_codes):
                return True
    return False


def is_svc_thrombosis(row):
    return any(str(code).startswith('I82.21') for code in row)

for batch in pf.iter_batches(batch_size=batch_size):
    df_chunk = pa.Table.from_batches([batch]).to_pandas()

    # All diagnosis columns
    dx_cols = [col for col in df_chunk.columns if col.startswith('I10_DX')]

    df_chunk['LYMPHOMA'] = df_chunk[dx_cols].apply(is_lymphoma, axis=1)
    df_chunk['SVC'] = df_chunk[dx_cols].apply(is_svc_thrombosis, axis=1)

    df_lymph_chunk = df_chunk[df_chunk['LYMPHOMA'] == True].copy()
    filtered_batches.append(df_lymph_chunk)

# Concatenate filtered lymphoma patients
df_lymph = pd.concat(filtered_batches, ignore_index=True)
print(f"Total lymphoma records after batch processing: {df_lymph.shape}")


# Convert categorical variables to category dtype
df_lymph['RACE'] = df_lymph['RACE'].astype('category')
df_lymph['ZIPINC_QRTL'] = df_lymph['ZIPINC_QRTL'].astype('category')
df_lymph['FEMALE'] = df_lymph['FEMALE'].astype('category')

# One-hot encode with get_dummies
X = pd.get_dummies(df_lymph[['AGE', 'FEMALE', 'RACE', 'ZIPINC_QRTL', 'SVC', 'LOS', 'TOTCHG']], drop_first=True)

# Convert boolean to int, and force all to float
X = X.apply(lambda col: col.astype(float), axis=0)

# Drop rows with any NaNs
X = X.dropna()

# Align y and convert to float
y = df_lymph.loc[X.index, 'DIED'].astype(float)

# Add constant
X = sm.add_constant(X)

# Filter to only 0.0 and 1.0 values
valid_y_mask = y.isin([0.0, 1.0])
y = y[valid_y_mask]
X = X.loc[valid_y_mask]


# Drop one ZIPINC_QRTL dummy to avoid perfect multicollinearity
if 'ZIPINC_QRTL_4.0' in X.columns:
    X = X.drop(columns=['ZIPINC_QRTL_4.0'])

# Optional: also drop const if NaNs persist
if 'const' in X.columns and X['const'].isna().all():
    X = X.drop(columns=['const'])
# Drop columns with near-zero variance
X = X.loc[:, X.std() > 1e-3]
import pytest
# pytest.set_trace()
# Now fit model
model = sm.Logit(y, X)
result = model.fit()


# Print results
print(result.summary())

# Odds Ratios
params = result.params
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']
conf['OR'] = np.exp(params)
conf['OR_lower'] = np.exp(conf['2.5%'])
conf['OR_upper'] = np.exp(conf['97.5%'])

print("\nAdjusted Odds Ratios:\n", conf[['OR', 'OR_lower', 'OR_upper']])