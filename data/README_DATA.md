# Data instructions

* **Source:** Craigslist Cars and Trucks Dataset (Austin Reese), https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
* **Licence:** see the licence shown on the Kaggle dataset page.
* **Raw file:** `vehicles.csv` (download manually from Kaggle; it is NOT stored in this repository because of its size).
* **Cleaned file:** `data/processed/vehicles_cleaned.csv` (240,186 rows). Regenerate it by running the notebook
  from the first cell with the raw file. It is too large for GitHub (limit 100 MB), so a 5,000-row random sample is provided in `data/sample/`.
* **Cleaning decisions:** see `outputs/cleaning_audit_log.csv` and `outputs/before_after_quality_report.csv`.
