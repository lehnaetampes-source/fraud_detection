"""
DAG — Détection fraude temps réel (Corrigé et Unifié avec le modèle ML)
"""
from datetime import datetime
import json
import random
import os
import sys
import pandas as pd
import numpy as np
import requests
import pickle
import sqlalchemy

try:
    import xgboost
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost

from airflow import DAG
from airflow.operators.python import PythonOperator

API_URL = "https://sdacelo-real-time-fraud-detection.hf.space/current-transactions"
GLOBAL_AMT_MEAN = 65.0  
SECURITY_ALERT_THRESHOLD = 170.0 

default_args = {
    "owner": "data_team",
    "retries": 0,
    "email_on_failure": False,
}

# Charger les artefacts globaux (Sauvegardés à la racine du dossier dags)
DAG_DIR = os.path.dirname(__file__)
def load_pkl_safely(filename):
    path = os.path.join(DAG_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def fetch_transactions(**context):
    """Récupère l'unique transaction de l'API ou simule un flux unitaire réaliste."""
    df = None
    is_simulated = False
    print(f"[EXTRACT] Appel de l'API : {API_URL}")
    
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        raw_data = response.json()

        # L'API renvoie parfois une chaîne JSON encodée deux fois
        # (ex: '"{\\"columns\\":[...],\\"data\\":[[...]]}"').
        # response.json() ne renvoie alors qu'une str -> il faut la re-décoder.
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        if isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, dict):
            if "data" in raw_data:
                columns = raw_data.get("columns")
                df = pd.DataFrame(raw_data["data"], columns=columns) if columns else pd.DataFrame(raw_data["data"])
            else:
                df = pd.DataFrame([raw_data])
                
        if df is not None:
            print(f"[EXTRACT] ✅ Succès API ! {len(df)} transaction collectée.")
        else:
            print(f"[EXTRACT] ⚠️ Réponse API dans un format inattendu : {type(raw_data)} -> {str(raw_data)[:200]}")
        
    except Exception as e:
        print(f"[EXTRACT] ⏳ API Saturation / Rate Limit ({e}). Transition sur le flux simulé.")
        is_simulated = True

    if df is None or df.empty:
        categories = ["shopping_net", "grocery_pos", "gas_transport", "entertainment", "shopping_pos"]
        is_fraudulent_flow = (random.random() < 0.75)
        amt = round(random.uniform(175, 900), 2) if is_fraudulent_flow else round(random.uniform(5, 95), 2)
        
        single_tx = {
            "cc_num": 1000000000000 + random.randint(1, 9999), 
            "merchant": f"merchant_{random.randint(1, 30)}",
            "category": random.choice(categories),
            "amt": amt, 
            "gender": random.choice(["M", "F"]),
            "lat": 40.730610, "long": -73.935242, "city_pop": 8000000,
            "dob": "1985-03-15", 
            "merch_lat": 40.740610 if is_fraudulent_flow else 40.732610, 
            "merch_long": -73.945242 if is_fraudulent_flow else -73.936242,
            "unix_time": int(datetime.utcnow().timestamp())
        }
        df = pd.DataFrame([single_tx])
        print(f"[EXTRACT] 🔀 1 transaction simulée injectée (Montant : {amt}€).")

    # L'API réelle renvoie le vrai label "is_fraud" du dataset Kaggle.
    # On le renomme tout de suite pour ne pas qu'il soit écrasé plus tard
    # par la colonne "is_fraud" (= prédiction du modèle).
    if "is_fraud" in df.columns:
        df = df.rename(columns={"is_fraud": "is_fraud_ground_truth"})
        print(f"[EXTRACT] 🏷️ Vérité terrain disponible : is_fraud_ground_truth = {int(df['is_fraud_ground_truth'].iloc[0])}")

    context["ti"].xcom_push(key="raw_df", value=df.to_json())


def preprocess_transactions(**context):
    """Calcul des features à la volée de manière identique à l'entraînement."""
    raw_json = context["ti"].xcom_pull(key="raw_df", task_ids="fetch_transactions")
    df = pd.read_json(raw_json)

    df["amt"] = pd.to_numeric(df["amt"], errors="coerce").fillna(0)
    
    # 1. Date & Time Features
    now = datetime.utcnow()
    df["hour"] = now.hour
    df["day"] = now.weekday()
    df["is_weekend"] = (df["day"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    
    # 2. Age Calcul
    if "dob" in df.columns:
        try:
            df["dob"] = pd.to_datetime(df["dob"])
            df["age"] = now.year - df["dob"].dt.year
        except:
            df["age"] = 40
    else:
        df["age"] = 40

    # 3. Haversine Distance
    def haversine_np(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    df["distance_km"] = haversine_np(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["distance_km"] = df["distance_km"].fillna(5.0)

    # 4. Advanced Metrics
    df["amt_log"] = np.log1p(df["amt"])
    df["amt_vs_avg"] = df["amt"] / (GLOBAL_AMT_MEAN + 1)
    df["merchant_freq"] = 10 

    # Encodages simplifiés alignés sur les types majeurs du dataset
    df["category_enc"] = df["category"].map({"shopping_net": 1, "grocery_pos": 2, "gas_transport": 3}).fillna(0).astype(int)
    df["gender_enc"] = df["gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)

    # 5. unix_time : requis par le modèle mais absent des vraies transactions API
    # (l'API renvoie "current_time" à la place, avec le même format d'epoch).
    if "unix_time" not in df.columns:
        if "current_time" in df.columns:
            df["unix_time"] = df["current_time"]
        else:
            df["unix_time"] = int(now.timestamp())
    df["unix_time"] = pd.to_numeric(df["unix_time"], errors="coerce").fillna(int(now.timestamp())).astype(np.int64)

    context["ti"].xcom_push(key="processed_df", value=df.to_json())


def predict_fraud_mlflow(**context):
    """Prédiction unifiée à l'aide du vrai modèle ML et du seuil optimal."""
    processed_json = context["ti"].xcom_pull(key="processed_df", task_ids="preprocess_transactions")
    df = pd.read_json(processed_json)

    # Charger les colonnes et les modèles
    feature_cols = load_pkl_safely("feature_cols.pkl")
    scaler = load_pkl_safely("scaler.pkl")
    model = load_pkl_safely("model.pkl")
    threshold = load_pkl_safely("threshold.pkl")

    if threshold is None:
        threshold = 0.5

    if model is not None and scaler is not None and feature_cols is not None:
        try:
            # S'assurer que toutes les colonnes requises existent
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # --- CORRECTION ICI : Sélection stricte et sécurisée des colonnes pour éviter le bug Timestamp ---
            X_df = pd.DataFrame(df, columns=feature_cols).apply(pd.to_numeric, errors="coerce").fillna(0)
            X_raw = X_df.values.astype(np.float64)
            print(f"[PREDICT] 🔬 Features envoyées au modèle : {X_df.iloc[0].to_dict()}")
            X_scaled = scaler.transform(X_raw)
            
            y_proba = model.predict_proba(X_scaled)[:, 1]
            df["fraud_score"] = y_proba
            df["is_fraud"] = (y_proba >= threshold).astype(int)
            print(f"[PREDICT] Modèle ML appliqué avec succès (Seuil optimal: {threshold:.3f})")
        except Exception as e:
            print(f"[PREDICT] ⚠️ Erreur lors du calcul ML ({e}). Règle de secours appliquée.")
            df["is_fraud"] = (df["amt"] > SECURITY_ALERT_THRESHOLD).astype(int)
            df["fraud_score"] = df["is_fraud"].map({1: 0.95, 0: 0.01})
    else:
        print("[PREDICT] ⚠️ Artefacts pkl manquants dans le conteneur. Mode dégradé (Seuil €).")
        df["is_fraud"] = (df["amt"] > SECURITY_ALERT_THRESHOLD).astype(int)
        df["fraud_score"] = df["is_fraud"].map({1: 0.95, 0: 0.01})

    nb_fraudes = int(df["is_fraud"].sum())
    print(f"[PREDICT] Analyse complétée. Résultat : {nb_fraudes} fraude(s) détectée(s).")
    
    for _, row in df[df["is_fraud"] == 1].iterrows():
        print(f"  🚨 ALERTE DIRECTE → {row.get('merchant','N/A')} | {float(row['amt']):.2f}€ | Score ML = {row['fraud_score']:.3f}")

    # Comparatif avec la vérité terrain quand elle est disponible (données API réelles)
    if "is_fraud_ground_truth" in df.columns:
        for _, row in df.iterrows():
            reel = int(row["is_fraud_ground_truth"])
            predit = int(row["is_fraud"])
            if reel == 1 and predit == 1:
                verdict = "✅ VRAI POSITIF — le modèle a bien détecté une fraude réelle"
            elif reel == 1 and predit == 0:
                verdict = "❌ FAUX NÉGATIF — fraude réelle non détectée par le modèle"
            elif reel == 0 and predit == 1:
                verdict = "⚠️ FAUX POSITIF — transaction légitime signalée à tort"
            else:
                verdict = "✅ VRAI NÉGATIF — transaction légitime, correctement laissée passer"
            print(f"  🔍 COMPARATIF → réel={reel} | prédit={predit} | score={row['fraud_score']:.3f} | {verdict}")

    context["ti"].xcom_push(key="predictions", value=df.to_json())
    context["ti"].xcom_push(key="nb_fraudes", value=nb_fraudes)


def store_results(**context):
    """Archive chaque transaction scorée dans Postgres (exigence 'store real-time data in a db')."""
    predictions_json = context["ti"].xcom_pull(key="predictions", task_ids="predict_fraud_mlflow")
    df = pd.read_json(predictions_json)

    conn_str = os.environ.get("POSTGRES_FRAUD_CONN") or os.environ.get("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    if not conn_str:
        print("[LOAD] ⚠️ Aucune connexion Postgres disponible (ni POSTGRES_FRAUD_CONN, ni AIRFLOW__DATABASE__SQL_ALCHEMY_CONN). Rien n'a été persisté.")
        return

    try:
        engine = sqlalchemy.create_engine(conn_str)
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text("""
                CREATE TABLE IF NOT EXISTS fraud_transactions (
                    id SERIAL PRIMARY KEY,
                    detected_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    merchant TEXT,
                    category TEXT,
                    amt DOUBLE PRECISION,
                    fraud_score DOUBLE PRECISION,
                    is_fraud_predicted INTEGER,
                    is_fraud_ground_truth INTEGER
                )
            """))

            has_ground_truth = "is_fraud_ground_truth" in df.columns

            for _, row in df.iterrows():
                connection.execute(
                    sqlalchemy.text("""
                        INSERT INTO fraud_transactions
                            (merchant, category, amt, fraud_score, is_fraud_predicted, is_fraud_ground_truth)
                        VALUES
                            (:merchant, :category, :amt, :fraud_score, :is_fraud_predicted, :is_fraud_ground_truth)
                    """),
                    {
                        "merchant": str(row.get("merchant", "N/A")),
                        "category": str(row.get("category", "N/A")),
                        "amt": float(row["amt"]),
                        "fraud_score": float(row["fraud_score"]),
                        "is_fraud_predicted": int(row["is_fraud"]),
                        "is_fraud_ground_truth": int(row["is_fraud_ground_truth"]) if has_ground_truth else None,
                    },
                )
        print(f"[LOAD] ✅ {len(df)} transaction(s) archivée(s) dans Postgres (table fraud_transactions).")
    except Exception as e:
        print(f"[LOAD] ❌ Erreur lors de l'écriture en base : {e}")


def send_fraud_alert(**context):
    nb_fraudes = context["ti"].xcom_pull(key="nb_fraudes", task_ids="predict_fraud_mlflow")
    if nb_fraudes > 0:
        print("[ALERT] 🚨 NOTIFICATION DIRECTE : Activité frauduleuse bloquée par le modèle !")
    else:
        print("[ALERT] ✅ Fin de l'inspection : Transaction validée.")


with DAG(
    dag_id="fraud_detection_realtime_mlflow",
    description="Pipeline Streaming Réel avec Fallback et Modèle Unifié",
    schedule_interval="*/2 * * * *", 
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["streaming", "mlops-validated"],
) as dag:

    t1 = PythonOperator(task_id="fetch_transactions",      python_callable=fetch_transactions)
    t2 = PythonOperator(task_id="preprocess_transactions", python_callable=preprocess_transactions)
    t3 = PythonOperator(task_id="predict_fraud_mlflow",    python_callable=predict_fraud_mlflow)
    t4 = PythonOperator(task_id="store_results",           python_callable=store_results)
    t5 = PythonOperator(task_id="send_fraud_alert",        python_callable=send_fraud_alert)

    t1 >> t2 >> t3 >> t4 >> t5