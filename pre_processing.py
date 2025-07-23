from pathlib import Path
import pandas as pd, csv

BASE      = Path(__file__).resolve().parent
DATA_CSV  = BASE / "data" / "movies.csv"
OUT_CSV   = BASE / "data" / "movies_clean.csv"

df = pd.read_csv(DATA_CSV)

df["title"]  = df["title"].str.strip()

df["year"] = df["title"].str.extract(r'\((\d{4})\)\s*$')

df["title"] = df["title"].str.replace(r'\s+\(\d{4}\)\s*$', '', regex=True)

df["genres"] = (
    df["genres"].fillna("")
      .str.lower()
      .str.replace(r'\s*\|\s*', ', ', regex=True)  # replace '|' with ", " cleanly
)




df.to_csv(OUT_CSV, index=False)
print("saved →", OUT_CSV)
