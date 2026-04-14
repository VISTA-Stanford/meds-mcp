from google.cloud import bigquery
import pandas as pd

client = bigquery.Client(project="som-nero-plevriti-deidbdf")

query = """
    SELECT person_id, embed_time, task, label, question, label_description
    FROM `som-nero-plevriti-deidbdf.vista_bench_v1_3.progression_recurrence_survival_1yr_2yr_3yr_4yr_5yr`
"""

df = client.query(query).to_dataframe()
df.to_csv('/Users/Ayeeshi/Documents/Stanford/Research/meds-mcp/data/collections/vista_bench/progression_subset.csv', index=False)
print(f"Done! {len(df)} rows saved.")