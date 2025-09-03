import pandas as pd
import numpy as np
import os


def create_reference_data():

    print(" Création de reference_data.csv...")
    print(" Répertoire de travail:", os.getcwd())


    np.random.seed(42)
    n_samples = 3000

    print(f" Génération de {n_samples} échantillons...")


    reference_data = pd.DataFrame({

        'CreditScore': np.clip(
            np.random.normal(650, 96, n_samples), 350, 850
        ).astype(int),


        'Age': np.clip(
            np.random.normal(39, 10, n_samples), 18, 92
        ).astype(int),


        'Tenure': np.clip(
            np.random.poisson(5, n_samples), 0, 10
        ),


        'Balance': np.clip(
            np.random.exponential(76000, n_samples), 0, 250898
        ).round(2),


        'NumOfProducts': np.random.choice(
            [1, 2, 3, 4], n_samples,
            p=[0.51, 0.46, 0.027, 0.003]
        ),


        'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.29, 0.71]),


        'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),


        'EstimatedSalary': np.random.uniform(11, 199992, n_samples).round(2),


        'Geography_Germany': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),


        'Geography_Spain': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),


        'Gender_Male': np.random.choice([0, 1], n_samples, p=[0.45, 0.55])
    })


    conflicting_geo = (reference_data['Geography_Germany'] == 1) & (reference_data['Geography_Spain'] == 1)
    reference_data.loc[conflicting_geo, 'Geography_Spain'] = 0

    print(" Données générées avec succès!")

    # Afficher des statistiques
    print("\n Aperçu des données créées:")
    print(f"    Forme: {reference_data.shape}")
    print(f"    Colonnes: {list(reference_data.columns)}")

    print("\n📈 Statistiques descriptives:")
    print(reference_data.describe())

    print("\n  Distribution des variables catégorielles:")
    categorical_cols = ['NumOfProducts', 'HasCrCard', 'IsActiveMember',
                        'Geography_Germany', 'Geography_Spain', 'Gender_Male']

    for col in categorical_cols:
        print(f"    {col}: {dict(reference_data[col].value_counts().sort_index())}")

    # Sauvegarder
    filename = 'reference_data.csv'
    reference_data.to_csv(filename, index=False)

    print(f"\n Fichier sauvegardé: {filename}")
    print(f" Emplacement: {os.path.abspath(filename)}")

    # Validation
    print("\n Validation du fichier créé:")
    try:
        test_load = pd.read_csv(filename)
        print(f"    Fichier lisible: {test_load.shape}")
        print(f"    Colonnes correctes: {list(test_load.columns) == list(reference_data.columns)}")
        print(f"    Pas de valeurs manquantes: {test_load.isnull().sum().sum() == 0}")
    except Exception as e:
        print(f"   Erreur de validation: {e}")

    print("\n reference_data.csv créé avec succès!")

    return reference_data


def show_sample_data(df, n=5):

    print(f"\n Aperçu ({n} premières lignes):")
    print(df.head(n))

    print(f"\n Aperçu ({n} dernières lignes):")
    print(df.tail(n))


if __name__ == "__main__":
    print(" CRÉATION DE DONNÉES DE RÉFÉRENCE POUR MONITORING CHURN")

    # Créer les données
    data = create_reference_data()

    # Afficher un échantillon
    show_sample_data(data)
