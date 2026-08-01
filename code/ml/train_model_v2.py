"""
Entraînement amélioré — XGBoost + Feature Engineering + Seuil optimisé (Correction MLOps)
Dataset réel : fraudTest.csv
"""
import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                              precision_recall_curve, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from mlflow.tracking import MlflowClient

# ── CONFIGURATION ──────────────────────────────────────────────────────────
MLFLOW_URI = "http://127.0.0.1:5001"
DATA_URL   = "https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv"
MODEL_NAME = "fraud_detector"
EXPERIMENT_NAME = "fraud_detection_mlops" # Nouveau nom pour la clarté

mlflow.set_tracking_uri(MLFLOW_URI)

# ── Fix chemin d'artefact Windows ──────────────────────────────────────
# On force un chemin d'artefact propre (file:///) pour éviter tout bug sous Windows.
client = MlflowClient()
try:
    existing = client.get_experiment_by_name(EXPERIMENT_NAME)
except Exception:
    existing = None

if existing is None:
    artifact_location = "file:///" + os.path.abspath("mlruns").replace("\\", "/")
    experiment_id = client.create_experiment(
        EXPERIMENT_NAME, artifact_location=artifact_location
    )
    print(f"Expérience créée avec artifact_location = {artifact_location}")
else:
    experiment_id = existing.experiment_id
    print(f"Expérience existante, artifact_location = {existing.artifact_location}")

mlflow.set_experiment(EXPERIMENT_NAME)

# ── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("Chargement du dataset...")
df = pd.read_csv(DATA_URL)
print(f"Shape : {df.shape} | Fraudes : {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

fraud_amts = df[df["is_fraud"] == 1]["amt"]
normal_amts = df[df["is_fraud"] == 0]["amt"]

print("=== FRAUDES ===")
print(f"Moyenne : {fraud_amts.mean():.2f}€")
print(f"Médiane : {fraud_amts.median():.2f}€")
print(f"Écart-type : {fraud_amts.std():.2f}€")
print(f"Q1 (25%) : {fraud_amts.quantile(0.25):.2f}€")
print(f"Q3 (75%) : {fraud_amts.quantile(0.75):.2f}€")

print("\n=== NORMALES ===")
print(f"Moyenne : {normal_amts.mean():.2f}€")
print(f"Médiane : {normal_amts.median():.2f}€")

# ── 2. FEATURE ENGINEERING AVANCÉ ────────────────────────────────────────
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])

df["hour"]        = df["trans_date_trans_time"].dt.hour
df["day"]         = df["trans_date_trans_time"].dt.dayofweek
df["is_weekend"]  = (df["day"] >= 5).astype(int)
df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
df["age"]         = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

# Distance entre client et marchand
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

# Montant relatif à la moyenne du client (anomalie de dépense)
df["amt_log"] = np.log1p(df["amt"])
client_avg = df.groupby("cc_num")["amt"].transform("mean")
df["amt_vs_avg"] = df["amt"] / (client_avg + 1)

# Fréquence de transactions par marchand
merchant_freq = df["merchant"].value_counts()
df["merchant_freq"] = df["merchant"].map(merchant_freq)

# Encodage catégoriel
le_cat = LabelEncoder()
le_gen = LabelEncoder()
df["category_enc"] = le_cat.fit_transform(df["category"].astype(str))
df["gender_enc"]   = le_gen.fit_transform(df["gender"].astype(str))

feature_cols = [
    "amt", "amt_log", "amt_vs_avg", "category_enc", "gender_enc",
    "hour", "day", "is_weekend", "is_night", "age",
    "lat", "long", "city_pop", "merch_lat", "merch_long",
    "distance_km", "merchant_freq", "unix_time"
]

X = df[feature_cols].fillna(0).values
y = df["is_fraud"].values

# ── 3. SPLIT & PREPROCESSING ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print(f"Scaler + {len(feature_cols)} features sauvegardés")

# ── 4. SMOTE ──────────────────────────────────────────────────────────────
# Équilibrage UNIQUEMENT pour l'optimisation des hyperparamètres et du seuil
print("SMOTE en cours (pour l'optimisation uniquement)...")
sm = SMOTE(random_state=42, sampling_strategy=0.3) 
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print(f"Après SMOTE : {y_train_res.sum()} fraudes / {len(y_train_res)} total")

# ── 5. TRAIN & OPTIMIZE XGBOOST (Sur données SMOTE) ─────────────────────
with mlflow.start_run(run_name="XGBoost_SMOTE_Optimization"):
    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "scale_pos_weight": 3, 
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "aucpr", # Adapté au déséquilibre
    }
    mlflow.log_params(params)
    mlflow.log_param("feature_engineering", "distance_km, amt_vs_avg, is_night, merchant_freq")
    mlflow.log_param("data_strategy", "SMOTE (0.3) for optimization")

    print("Entraînement XGBoost (sur données équilibrées SMOTE) pour optimisation...")
    model_opt = XGBClassifier(**params, n_jobs=-1)
    model_opt.fit(X_train_res, y_train_res)

    y_proba_opt = model_opt.predict_proba(X_test_scaled)[:, 1]

    # ── 6. OPTIMISATION DU SEUIL DE DÉCISION (Sur données SMOTE) ────────
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_opt)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    print(f"\nMeilleur seuil trouvé sur SMOTE : {best_threshold:.3f} (F1={f1_scores[best_idx]:.4f})")
    mlflow.log_metric("optimized_threshold", best_threshold)

    # Métriques sur données SMOTE (pour info)
    auc_opt = roc_auc_score(y_test, y_proba_opt)
    mlflow.log_metric("auc_roc_smote", auc_opt)
    print(f"AUC-ROC (SMOTE model) : {auc_opt:.4f}")

    # Sauvegarde temporaire du seuil
    with open("models/threshold.pkl", "wb") as f:
        pickle.dump(best_threshold, f)

# ── 7. RE-TRAIN MODÈLE FINAL (Sur DONNÉES RÉELLES - Correction MLOps) ──
print("\n--- CORRECTION MLOPS CRITIQUE ---")
print("Ré-entraînement du modèle FINAL sur les données RÉELLES (X_train_scaled)...")

with mlflow.start_run(run_name="XGBoost_FINAL_Production"):
    # On utilise les mêmes hyperparamètres
    mlflow.log_params(params)
    mlflow.log_param("data_strategy", "FINAL (Real Distribution)")
    # On log le seuil optimisé précédemment
    mlflow.log_param("applied_threshold", best_threshold)

    model_final = XGBClassifier(**params, n_jobs=-1)
    model_final.fit(X_train_scaled, y_train)

    y_proba_final = model_final.predict_proba(X_test_scaled)[:, 1]
    
    # Application du SEUIL OPTIMISÉ (calculé sur le modèle SMOTE)
    y_pred_opt = (y_proba_final >= best_threshold).astype(int)

    # ── 8. ÉVALUATION FINALE (Sur distribution réelle avec seuil optimisé)
    auc_final = roc_auc_score(y_test, y_proba_final)
    f1_final = f1_score(y_test, y_pred_opt)
    prec_final = precision_score(y_test, y_pred_opt)
    rec_final = recall_score(y_test, y_pred_opt)

    mlflow.log_metric("auc_roc_final", auc_final)
    mlflow.log_metric("f1_score_final", f1_final)
    mlflow.log_metric("precision_final", prec_final)
    mlflow.log_metric("recall_final", rec_final)

    print(f"\n=== RÉSULTATS FINALS (Réel, seuil optimisé {best_threshold:.3f}) ===")
    print(classification_report(y_test, y_pred_opt, target_names=["Normal", "Fraude"]))
    print(f"AUC-ROC Final : {auc_final:.4f}")
    print(f"F1-Score Final : {f1_final:.4f}")

    # ── 9. SAUVEGARDE & ENREGISTREMENT MLFLOW ────────────────────────────────
    # On sauvegarde le MODÈLE FINAL entraîné sur données réelles
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model_final, f)
    print("\nModèle FINAL sauvegardé localement OK")

    # Enregistrement dans le Model Registry MLflow
    mlflow.sklearn.log_model(model_final, "model", registered_model_name=MODEL_NAME)
    print(f"Modèle enregistré dans MLflow : '{MODEL_NAME}' OK")

print("\nEntraînement terminé. Copie les fichiers de 'models/' vers le dossier Airflow.")