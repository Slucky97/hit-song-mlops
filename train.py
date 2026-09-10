import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Chargement des données
dataset = pd.read_excel("Hit songs Prédiction/dataset_ms_ls.xlsx")

#On retire les colonnes non pertinentes
dataset = dataset.drop(["Artistes", "songs", "url", "id", "image_path"], axis=1)

#On sépare les features (X) et la cible (y)
X = dataset.drop(["Streamed/Non-Streamed"], axis=1)
y = dataset["Streamed/Non-Streamed"]

#On définit les colonnes continues et catégorielles
categorical_features = ["key", "mode", "time_signature"]
numerical_features = ["energy", "loudness", "tempo"]

#Etape du preprocessing (Normalisation + encodage)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder="passthrough",
)

#pipeline complet : preprocessing + modèle, ensemble
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
])

#Séparation entrainement / validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Entrainement
print("Entrainement en cours...")
pipeline.fit(X_train, y_train)

#Evaluation
y_pred = pipeline.predict(X_val)
print("\nRésultats sur les données de validation :")
print(classification_report(y_val, y_pred))

#sauvegarde du modele
joblib.dump(pipeline, "model.joblib")
print("\n Modèle sauvegardé dans model.joblib")