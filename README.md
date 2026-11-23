# Used Vehicle Price Prediction and Market Valuation Using Machine Learning

CDS4003 - Data Science Final Project

## Business problem
Buyers, sellers and dealerships find it hard to judge whether the asking price of a used vehicle is reasonable. This project builds a
regression model that estimates the typical asking price of a used car or truck from its attributes and deploys it as a web app.

## Team members and roles
| Name | GitHub username | Role / contribution |
|---|---|---|
| *add name* | *add username* | *add role* |
| *add name* | *add username* | *add role* |

## Dataset
* **Source:** Craigslist Cars and Trucks Dataset (Austin Reese) - https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
* **Licence:** *quote the licence shown on the Kaggle dataset page*
* **How to obtain:** download `vehicles.csv` from Kaggle and place it in `data/raw/` (it is not stored in this repository because of its size).
  A 5,000-row sample of the cleaned data is in `data/sample/`.

## Folder structure
```
data/        raw/ (not tracked), processed/ (full cleaned file not tracked), sample/
notebooks/   Colab notebook with Stages 1-7
src/         vehicle_features.py (feature engineering shared by training and app)
models/      price_model.joblib, model_metadata.json
app/         app.py (Streamlit app)
outputs/     CSV reports and figures
docs/        data dictionary, feature importance figure
requirements.txt
```

## How to run
1. Install the libraries: `pip install -r requirements.txt`
2. Run the notebook in `notebooks/` from top to bottom in Google Colab (upload `vehicles.csv` when asked).
3. Start the app locally: `streamlit run app/app.py`

## Summary of results
* Cleaned dataset: 240,186 listings after removing 0 exact duplicates, 44,904 re-posted adverts and 10,361 records with an invalid price.
* Final model: **Hist. gradient boosting**, trained on log(price).
* Test set: R-squared 0.676; median absolute percentage error 18.7%; 52.7% of predictions within 20% of the asking price;
  mean absolute error $4,610.
* Limitations: asking prices (not sale prices), one website, one time period, seller-entered data.

## Live app
*Add the Streamlit Community Cloud link here after deployment.*

## AI tools used
*List the AI tools used in this project and how they were used (required by the course).*
