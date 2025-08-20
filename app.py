from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import pandas as pd
from dotenv import load_dotenv
import os
import jwt
import datetime
from functools import wraps

# Charger variables d’environnement
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "default_secret")
JWT_EXP_HOURS = int(os.getenv("JWT_EXP_HOURS", 4))
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "admin123")

app = Flask(__name__)
swagger = Swagger(app)

# Charger les objets ML
model = joblib.load('model_rf.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Décorateur pour sécuriser les routes avec JWT
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            try:
                token = request.headers["Authorization"].split(" ")[1]
            except:
                return jsonify({"error": "Format du header invalide"}), 401

        if not token:
            return jsonify({"error": "Token manquant"}), 401

        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except Exception:
            return jsonify({"error": "Token invalide ou expiré"}), 401

        return f(*args, **kwargs)
    return decorated


@app.route('/')
def home():
    return jsonify({"message": "API Churn Banque fonctionne !"})

# Endpoint pour obtenir un token
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if username == AUTH_USERNAME and password == AUTH_PASSWORD:
        exp = datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXP_HOURS)
        token = jwt.encode({"exp": exp}, JWT_SECRET, algorithm="HS256")
        return jsonify({"token": token})

    return jsonify({"error": "Identifiants invalides"}), 401


# Endpoint protégé pour prédiction
@app.route('/predict', methods=['POST'])
@token_required
def predict():
    """
    Prédit la probabilité de churn d'un client bancaire.
    ---
    tags:
      - Prédiction
    """
    try:
        data = request.get_json()

        expected_fields = [
            "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary",
            "Geography_Germany", "Geography_Spain", "Gender_Male"
        ]

        missing_fields = [field for field in expected_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Champs manquants : {missing_fields}"}), 400

        df = pd.DataFrame([[data[field] for field in expected_fields]], columns=expected_fields)
        df_imputed = pd.DataFrame(imputer.transform(df), columns=expected_fields)
        df_scaled = scaler.transform(df_imputed)

        proba = model.predict_proba(df_scaled)[0][1]

        return jsonify({"churn_probability": round(float(proba), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)


