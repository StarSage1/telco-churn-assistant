from pathlib import Path
import json
import joblib
import pandas as pd
from catboost import CatBoostClassifier


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# =========================================================
# Load saved models and configuration
# =========================================================

logistic_model = joblib.load(
    MODEL_DIR / "logistic_raw.joblib"
)

catboost_model = CatBoostClassifier()
catboost_model.load_model(
    MODEL_DIR / "catboost_feature_engineered.cbm"
)

with open(
    MODEL_DIR / "ensemble_config.json",
    "r"
) as f:
    ensemble_config = json.load(f)

with open(
    MODEL_DIR / "feature_config.json",
    "r"
) as f:
    feature_config = json.load(f)


LOGISTIC_WEIGHT = ensemble_config["logistic_weight"]
CATBOOST_WEIGHT = ensemble_config["catboost_weight"]
THRESHOLD = ensemble_config["threshold"]

RAW_FEATURES = feature_config["raw_features"]
CATEGORICAL_FEATURES = feature_config["categorical_features"]


# =========================================================
# Feature engineering
# =========================================================

def add_engineered_features(X):
    """
    Add the same engineered features that were used
    during model training.

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    X = X.copy()

    # -----------------------------------------------------
    # Number of optional services
    # -----------------------------------------------------

    service_cols = [
        "Online_Security",
        "Online_Backup",
        "Device_Protection",
        "Tech_Support",
        "Streaming_TV",
        "Streaming_Movies"
    ]

    X["Num_Services"] = sum(
        (X[col] == "Yes").astype(int)
        for col in service_cols
    )

    # -----------------------------------------------------
    # Number of support/protection services
    # -----------------------------------------------------

    support_cols = [
        "Online_Security",
        "Online_Backup",
        "Device_Protection",
        "Tech_Support"
    ]

    X["Num_Support_Services"] = sum(
        (X[col] == "Yes").astype(int)
        for col in support_cols
    )

    # -----------------------------------------------------
    # Number of streaming services
    # -----------------------------------------------------

    X["Num_Streaming_Services"] = (
        (X["Streaming_TV"] == "Yes").astype(int)
        +
        (X["Streaming_Movies"] == "Yes").astype(int)
    )

    # -----------------------------------------------------
    # New customer indicator
    # -----------------------------------------------------

    X["Is_New_Customer"] = (
        X["tenure"] <= 6
    ).astype(int)

    # -----------------------------------------------------
    # Automatic payment indicator
    # -----------------------------------------------------

    X["Auto_Payment"] = X["Payment_Method"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]).astype(int)

    # -----------------------------------------------------
    # Interaction:
    # customer has neither security nor tech support
    # -----------------------------------------------------

    X["No_Security_And_No_Support"] = (
        (X["Online_Security"] == "No")
        &
        (X["Tech_Support"] == "No")
    ).astype(int)

    return X


# =========================================================
# Input validation
# =========================================================

def validate_customer_data(customer_data):
    """
    Check that the customer input contains all features
    required by the trained models.
    """

    if not isinstance(customer_data, dict):
        raise TypeError(
            "customer_data must be a dictionary."
        )

    missing_features = [
        feature
        for feature in RAW_FEATURES
        if feature not in customer_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )


# =========================================================
# Prediction
# =========================================================

def predict_churn(customer_data):
    """
    Predict churn for one customer using the hybrid ensemble.

    Hybrid model:
        30% Logistic Regression using raw features
        70% CatBoost using feature-engineered features

    Parameters
    ----------
    customer_data : dict
        Dictionary containing one customer's raw features.

    Returns
    -------
    dict
        Prediction, churn probability, and model details.
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    validate_customer_data(customer_data)

    # -----------------------------------------------------
    # Convert dictionary to DataFrame
    # -----------------------------------------------------

    customer_df = pd.DataFrame(
        [customer_data]
    )

    # Keep the exact raw feature order used during training
    customer_raw = customer_df[
        RAW_FEATURES
    ].copy()

    # -----------------------------------------------------
    # Logistic Regression prediction
    # Uses RAW features
    # -----------------------------------------------------

    logistic_probability = (
        logistic_model
        .predict_proba(customer_raw)[:, 1][0]
    )

    # -----------------------------------------------------
    # Feature engineering for CatBoost
    # -----------------------------------------------------

    customer_fe = add_engineered_features(
        customer_raw
    )

    # CatBoost expects categorical features as strings
    for col in CATEGORICAL_FEATURES:
        customer_fe[col] = (
            customer_fe[col]
            .astype(str)
        )

    # -----------------------------------------------------
    # CatBoost prediction
    # Uses FEATURE-ENGINEERED features
    # -----------------------------------------------------

    catboost_probability = (
        catboost_model
        .predict_proba(customer_fe)[:, 1][0]
    )

    # -----------------------------------------------------
    # Hybrid ensemble probability
    # -----------------------------------------------------

    churn_probability = (
        LOGISTIC_WEIGHT
        * logistic_probability
        +
        CATBOOST_WEIGHT
        * catboost_probability
    )

    # -----------------------------------------------------
    # Apply selected threshold
    # -----------------------------------------------------

    prediction = int(
        churn_probability
        >= THRESHOLD
    )

    if prediction == 1:
        prediction_label = "Churn"
    else:
        prediction_label = "No Churn"

    # -----------------------------------------------------
    # Risk level
    # This is only for easier interpretation.
    # It does not change the model prediction.
    # -----------------------------------------------------

    if churn_probability >= 0.70:
        risk_level = "High"

    elif churn_probability >= THRESHOLD:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,

        "churn_probability": round(
            float(churn_probability),
            4
        ),

        "risk_level": risk_level,

        "logistic_probability": round(
            float(logistic_probability),
            4
        ),

        "catboost_probability": round(
            float(catboost_probability),
            4
        ),

        "threshold": THRESHOLD,

        "model_weights": {
            "logistic_regression":
                LOGISTIC_WEIGHT,

            "catboost":
                CATBOOST_WEIGHT
        }
    }


def predict_churn_batch(customer_rows):
    """Score multiple validated customer dictionaries in one vectorized pass."""
    if not customer_rows:
        return []
    for customer_data in customer_rows:
        validate_customer_data(customer_data)

    customer_raw = pd.DataFrame(customer_rows)[RAW_FEATURES].copy()
    logistic_probabilities = logistic_model.predict_proba(customer_raw)[:, 1]

    customer_fe = add_engineered_features(customer_raw)
    for col in CATEGORICAL_FEATURES:
        customer_fe[col] = customer_fe[col].astype(str)
    catboost_probabilities = catboost_model.predict_proba(customer_fe)[:, 1]

    churn_probabilities = (
        LOGISTIC_WEIGHT * logistic_probabilities
        + CATBOOST_WEIGHT * catboost_probabilities
    )

    results = []
    for logistic_probability, catboost_probability, churn_probability in zip(
        logistic_probabilities, catboost_probabilities, churn_probabilities
    ):
        prediction = int(churn_probability >= THRESHOLD)
        risk_level = (
            "High"
            if churn_probability >= 0.70
            else "Medium"
            if churn_probability >= THRESHOLD
            else "Low"
        )
        results.append(
            {
                "prediction": prediction,
                "prediction_label": "Churn" if prediction else "No Churn",
                "churn_probability": round(float(churn_probability), 4),
                "risk_level": risk_level,
                "logistic_probability": round(float(logistic_probability), 4),
                "catboost_probability": round(float(catboost_probability), 4),
                "threshold": THRESHOLD,
                "model_weights": {
                    "logistic_regression": LOGISTIC_WEIGHT,
                    "catboost": CATBOOST_WEIGHT,
                },
            }
        )
    return results


# =========================================================
# Simple manual test
# =========================================================

if __name__ == "__main__":

    customer = {
        "gender": "Female",
        "Senior_Citizen": 0,
        "Is_Married": "No",
        "Dependents": "No",
        "tenure": 3,
        "Phone_Service": "Yes",
        "Dual": "No",
        "Internet_Service": "Fiber optic",
        "Online_Security": "No",
        "Online_Backup": "No",
        "Device_Protection": "No",
        "Tech_Support": "No",
        "Streaming_TV": "Yes",
        "Streaming_Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless_Billing": "Yes",
        "Payment_Method": "Electronic check",
        "Monthly_Charges": 95.0,
        "Total_Charges": 285.0
    }

    result = predict_churn(
        customer
    )

    print("\nPrediction Result")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")
