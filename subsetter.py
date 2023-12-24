import pandas as pd

data = pd.read_csv("data_clean_nosalt_canon_a2d.csv")

data = data.sample(100, random_state=42)

data.to_csv("data_clean_nosalt_canon_a2d_100.csv", index=False)