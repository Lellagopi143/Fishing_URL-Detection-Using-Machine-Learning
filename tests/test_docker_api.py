import joblib
import pandas as pd
import requests


MODEL_PATH = "models/phishing_url_model.pkl"
DATA_PATH = "data/raw/structured_data_legitimate.csv"

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


def test_docker_prediction():
    """
    Verify that the Dockerized FastAPI service
    produces the same prediction as the trained model.
    """

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Use the first real dataset row
    row = df.iloc[0]

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Prepare model input
    features = pd.DataFrame(
        [row[FEATURE_NAMES].to_dict()],
        columns=FEATURE_NAMES,
    )

    # Direct model prediction
    expected_prediction = int(
        model.predict(features)[0]
    )

    # Create API payload
    payload = {
        feature: int(row[feature])
        for feature in FEATURE_NAMES
    }

    # Call Docker container
    response = requests.post(
        "http://localhost:8000/predict",
        json=payload,
        timeout=10,
    )

    # API should respond successfully
    assert response.status_code == 200

    response_data = response.json()

    # Docker API prediction should match direct model
    assert (
        response_data["prediction"]
        == expected_prediction
    )

    # Validate response
    assert response_data["result"] in [
        "Legitimate",
        "Phishing",
    ]

    assert response_data["model"] == "RandomForest"

    assert 0.0 <= response_data["confidence"] <= 1.0