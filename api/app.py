import pandas as pd
from fastapi import FastAPI, HTTPException

from api.model_loader import load_model
from api.schemas import PredictionRequest


app = FastAPI(
    title="Phishing URL Detection API",
    description="ML API for phishing webpage classification",
    version="1.0.0",
)


# Load the trained model once when the application starts
model = load_model()


# Exact feature order used by the trained model
FEATURE_NAMES = [
    "has_title",
    "has_input",
    "has_button",
    "has_image",
    "has_submit",
    "has_link",
    "has_password",
    "has_email_input",
    "has_hidden_element",
    "has_audio",
    "has_video",
    "number_of_inputs",
    "number_of_buttons",
    "number_of_images",
    "number_of_option",
    "number_of_list",
    "number_of_th",
    "number_of_tr",
    "number_of_href",
    "number_of_paragraph",
    "number_of_script",
    "length_of_title",
    "has_h1",
    "has_h2",
    "has_h3",
    "length_of_text",
    "number_of_clickable_button",
    "number_of_a",
    "number_of_img",
    "number_of_div",
    "number_of_figure",
    "has_footer",
    "has_form",
    "has_text_area",
    "has_iframe",
    "has_text_input",
    "number_of_meta",
    "has_nav",
    "has_object",
    "has_picture",
    "number_of_sources",
    "number_of_span",
    "number_of_table",
]


@app.get("/")
def root():
    return {
        "application": "Phishing URL Detection API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "RandomForest",
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Extract the 43 features in the exact training order
        feature_values = [
            getattr(request, feature)
            for feature in FEATURE_NAMES
        ]

        # Create a DataFrame with the same feature names
        # used when the RandomForest model was trained.
        feature_df = pd.DataFrame(
            [feature_values],
            columns=FEATURE_NAMES,
        )

        # Make prediction
        prediction = int(
            model.predict(feature_df)[0]
        )

        # Convert numeric prediction to readable result
        result = (
            "Phishing"
            if prediction == 1
            else "Legitimate"
        )

        response = {
            "prediction": prediction,
            "result": result,
            "model": "RandomForest",
        }

        # Calculate confidence
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                feature_df
            )[0]

            response["confidence"] = float(
                max(probabilities)
            )

        return response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )