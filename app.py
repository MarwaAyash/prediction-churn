from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import pandas as pd

app = Flask(__name__)
swagger = Swagger(app)

# Charger les objets sauvegardés
model = joblib.load('model_rf.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return jsonify({"message": "API Churn Banque fonctionne !"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédit la probabilité de churn d'un client bancaire.
    ---
    tags:
      - Prédiction
    parameters:
      - in: body
        name: client_data
        required: true
        schema:
          type: object
          required:
            - CreditScore
            - Age
            - Tenure
            - Balance
            - NumOfProducts
            - HasCrCard
            - IsActiveMember
            - EstimatedSalary
            - Geography_Germany
            - Geography_Spain
            - Gender_Male
          properties:
            CreditScore:
              type: integer
              example: 600
            Age:
              type: integer
              example: 40
            Tenure:
              type: integer
              example: 3
            Balance:
              type: number
              example: 60000.0
            NumOfProducts:
              type: integer
              example: 2
            HasCrCard:
              type: integer
              example: 1
            IsActiveMember:
              type: integer
              example: 1
            EstimatedSalary:
              type: number
              example: 50000.0
            Geography_Germany:
              type: integer
              example: 0
            Geography_Spain:
              type: integer
              example: 1
            Gender_Male:
              type: integer
              example: 1
    responses:
      200:
        description: Probabilité de churn prédite
        schema:
          type: object
          properties:
            churn_probability:
              type: number
              example: 0.31
      400:
        description: Erreur de validation des données
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Champs manquants : ['Age']"
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

        df = pd.DataFrame([data])

        df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
        df_scaled = scaler.transform(df_imputed)

        proba = model.predict_proba(df_scaled)[0][1]

        return jsonify({"churn_probability": round(float(proba), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
