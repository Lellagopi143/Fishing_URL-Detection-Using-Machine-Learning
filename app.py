from flask import Flask, render_template, request
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load trained model and scaler
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler_fishing.pkl")

# All numeric features (44)
numeric_features = [
    'has_title', 'has_input', 'has_button', 'has_image', 'has_submit', 'has_link',
    'has_password', 'has_email_input', 'has_hidden_element', 'has_audio', 'has_video',
    'number_of_inputs', 'number_of_buttons', 'number_of_images', 'number_of_option',
    'number_of_list', 'number_of_th', 'number_of_tr', 'number_of_href', 'number_of_paragraph',
    'number_of_script', 'length_of_title', 'has_h1', 'has_h2', 'has_h3', 'length_of_text',
    'number_of_clickable_button', 'number_of_a', 'number_of_img', 'number_of_div',
    'number_of_figure', 'has_footer', 'has_form', 'has_text_area', 'has_iframe',
    'has_text_input', 'number_of_meta', 'has_nav', 'has_object', 'has_picture',
    'number_of_sources', 'number_of_span', 'number_of_table'
]

# Function to extract features from URL
def extract_features(url):
    features = dict.fromkeys(numeric_features, 0)
    try:
        r = requests.get(url, timeout=5)
        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        # Boolean features
        features['has_title'] = 1 if soup.title else 0
        features['has_input'] = 1 if soup.find_all("input") else 0
        features['has_button'] = 1 if soup.find_all("button") else 0
        features['has_image'] = 1 if soup.find_all("img") else 0
        features['has_submit'] = 1 if soup.find_all("input", {"type":"submit"}) else 0
        features['has_link'] = 1 if soup.find_all("a") else 0
        features['has_password'] = 1 if soup.find_all("input", {"type":"password"}) else 0
        features['has_email_input'] = 1 if soup.find_all("input", {"type":"email"}) else 0
        features['has_hidden_element'] = 1 if soup.find_all("input", {"type":"hidden"}) else 0
        features['has_audio'] = 1 if soup.find_all("audio") else 0
        features['has_video'] = 1 if soup.find_all("video") else 0
        features['has_h1'] = 1 if soup.find_all("h1") else 0
        features['has_h2'] = 1 if soup.find_all("h2") else 0
        features['has_h3'] = 1 if soup.find_all("h3") else 0
        features['has_footer'] = 1 if soup.find_all("footer") else 0
        features['has_form'] = 1 if soup.find_all("form") else 0
        features['has_text_area'] = 1 if soup.find_all("textarea") else 0
        features['has_iframe'] = 1 if soup.find_all("iframe") else 0
        features['has_text_input'] = 1 if soup.find_all("textarea") else 0
        features['has_nav'] = 1 if soup.find_all("nav") else 0
        features['has_object'] = 1 if soup.find_all("object") else 0
        features['has_picture'] = 1 if soup.find_all("picture") else 0

        # Count features
        features['number_of_inputs'] = len(soup.find_all("input"))
        features['number_of_buttons'] = len(soup.find_all("button"))
        features['number_of_images'] = len(soup.find_all("img"))
        features['number_of_option'] = len(soup.find_all("option"))
        features['number_of_list'] = len(soup.find_all("ul")) + len(soup.find_all("ol"))
        features['number_of_th'] = len(soup.find_all("th"))
        features['number_of_tr'] = len(soup.find_all("tr"))
        features['number_of_href'] = len(soup.find_all("a", href=True))
        features['number_of_paragraph'] = len(soup.find_all("p"))
        features['number_of_script'] = len(soup.find_all("script"))
        features['length_of_title'] = len(soup.title.text) if soup.title else 0
        features['length_of_text'] = len(soup.get_text())
        features['number_of_clickable_button'] = len(soup.find_all("button")) + len(soup.find_all("a"))
        features['number_of_a'] = len(soup.find_all("a"))
        features['number_of_img'] = len(soup.find_all("img"))
        features['number_of_div'] = len(soup.find_all("div"))
        features['number_of_figure'] = len(soup.find_all("figure"))
        features['number_of_meta'] = len(soup.find_all("meta"))
        features['number_of_sources'] = len(soup.find_all("source"))
        features['number_of_span'] = len(soup.find_all("span"))
        features['number_of_table'] = len(soup.find_all("table"))

    except:
        pass

    return pd.DataFrame([features])

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    url = ""
    if request.method == "POST":
        url = request.form.get("url")
        df_features = extract_features(url)
        X_scaled = scaler.transform(df_features)
        prediction = svm_model.predict(X_scaled)[0]
        prob = svm_model.predict_proba(X_scaled)[0][1]

        result = {
            "prediction": "Phishing" if prediction == 1 else "Legitimate",
            "probability": f"{prob*100:.2f}%"
        }

    return render_template("index.html", result=result, url=url)

if __name__ == "__main__":
    app.run(debug=True)
