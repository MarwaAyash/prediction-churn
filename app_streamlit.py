import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Prédiction de Churn")


@st.cache_resource
def load_models():
    try:
        model = joblib.load("model_rf.pkl")
        scaler = joblib.load("scaler.pkl")
        imputer = joblib.load("imputer.pkl")
        return model, scaler, imputer
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None, None, None


def main():
    st.title("Prédiction de Churn ")
    st.write("Interface simple pour démonstration")

    model, scaler, imputer = load_models()

    if model is None:
        st.error("Modèles non trouvés")
        st.stop()

    st.success("Modèles chargés")

    # Formulaire simple
    st.header("Informations Client")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 40)
        credit_score = st.number_input("Score Crédit", 300, 850, 650)
        balance = st.number_input("Solde", 0, 500000, 75000)
        tenure = st.number_input("Ancienneté", 0, 20, 5)

    with col2:
        num_products = st.selectbox("Produits", [1, 2, 3, 4])
        has_card = st.selectbox("Carte", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        is_active = st.selectbox("Actif", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        salary = st.number_input("Salaire", 0, 300000, 80000)

    country = st.selectbox("Pays", ["France", "Allemagne", "Espagne"])
    gender = st.selectbox("Genre", ["Femme", "Homme"])

    if st.button("Prédire le Churn"):
        # Préparer données
        data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [has_card],
            'IsActiveMember': [is_active],
            'EstimatedSalary': [salary],
            'Geography_Germany': [1 if country == "Allemagne" else 0],
            'Geography_Spain': [1 if country == "Espagne" else 0],
            'Gender_Male': [1 if gender == "Homme" else 0]
        })

        try:
            # Preprocessing
            X_imputed = pd.DataFrame(
                imputer.transform(data),
                columns=data.columns
            )
            X_scaled = scaler.transform(X_imputed)

            # Prédiction
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0, 1]

            # Résultats
            st.markdown("---")
            st.header("Résultats")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Probabilité", f"{probability:.0%}")

            with col2:
                if probability >= 0.7:
                    st.metric("Risque", "ÉLEVÉ", delta="Urgent")
                    st.write("🔴")
                elif probability >= 0.3:
                    st.metric("Risque", "MOYEN", delta="Surveiller")
                    st.write("🟡")
                else:
                    st.metric("Risque", "FAIBLE", delta="OK")
                    st.write("🟢")

            with col3:
                st.metric("Prédiction", "CHURN" if prediction == 1 else "FIDÈLE")

            # Barre visuelle
            st.subheader("Niveau de Risque")
            st.progress(probability)

            # Message
            if probability >= 0.7:
                st.error("Action immédiate recommandée")
            elif probability >= 0.3:
                st.warning("Surveillance recommandée")
            else:
                st.success("Client stable")

        except Exception as e:
            st.error(f"Erreur prédiction: {e}")


if __name__ == "__main__":
    main()
