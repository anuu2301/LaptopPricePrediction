
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "laptop_data.csv"
MODEL_PATH = BASE_DIR / "laptop_model.joblib"

# 1. Load the dataset
df = pd.read_csv( r"C:\Users\HP\Downloads\archive (3)\Laptop_price.csv")

# Clean column names
df.columns = df.columns.str.strip()

# 2. Find the target column
target_candidates = ["Price", "Price_euros", "price", "price_euros"]
target = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target is None:
    raise ValueError(
        f"No price column found. Available columns: {list(df.columns)}"
    )

# 3. Select useful features available in the dataset
candidate_features = [
    "Company", "TypeName", "Inches", "Ram",
    "Memory", "Gpu", "OpSys", "Weight"
]

features = [
    col for col in candidate_features
    if col in df.columns
]

if not features:
    raise ValueError(
        f"No expected feature columns found: {list(df.columns)}"
    )

# 4. Convert numeric-looking columns into numbers
for col in ["Inches", "Ram", "Weight"]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(
                r"([-+]?\d*\.?\d+)"
            )[0],
            errors="coerce"
        )

df[target] = pd.to_numeric(df[target], errors="coerce")

# Remove rows without a valid target price
df = df.dropna(subset=[target]).copy()

# Remove impossible or invalid prices
df = df[df[target] > 0].copy()

if len(df) < 10:
    raise ValueError("The dataset needs at least 10 valid rows.")

# Prepare input and output
X = df[features].copy()
y = df[target].copy()

# Detect numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in features
    if col not in numeric_features
]

# 5. Prepare the data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# 6. Build the Multiple Linear Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 7. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train the model
model.fit(X_train, y_train)

# 9. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel training completed!")
print(f"Dataset rows: {len(df)}")
print(f"Features used: {features}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared score: {r2:.3f}")

# 10. Save model and input information
metadata = {
    "features": features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": target,
    "mae": float(mae),
    "r2": float(r2),
    "categories": {},
    "ranges": {}
}

for col in categorical_features:
    values = df[col].dropna().astype(str).unique().tolist()
    metadata["categories"][col] = sorted(values)

for col in numeric_features:
    metadata["ranges"][col] = {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "median": float(df[col].median())
    }

joblib.dump(
    {"model": model, "metadata": metadata},
    MODEL_PATH
)

print(f"\nSaved trained model to: {MODEL_PATH}")
