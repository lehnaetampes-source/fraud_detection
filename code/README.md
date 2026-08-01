# 🛡️ Automatic Fraud Detection — ETL with Airflow

## Structure du projet
```
code/
├── dags/
│   ├── dag_realtime_fraud.py   # DAG temps réel (toutes les minutes)
│   └── dag_batch_report.py     # DAG batch quotidien (8h chaque matin)
├── ml/
│   └── train_model.py          # Entraînement RandomForest + MLflow
├── config/
│   └── init_db.sql             # Création des tables PostgreSQL
└── docker-compose.yml          # Infrastructure complète
```

## 🚀 Lancement en 4 étapes

### 1. Prérequis
```bash
docker --version   # >= 20.x
docker-compose --version
```

### 2. Démarrer l'infrastructure
```bash
cd code/
docker-compose up -d
# Attendre ~2 minutes que tout démarre
```

### 3. Entraîner le modèle
```bash
pip install scikit-learn imbalanced-learn mlflow pandas
mkdir -p models
python ml/train_model.py
# Puis dans MLflow UI (http://localhost:5000) → passer le modèle en "Production"
```

### 4. Configurer les variables Airflow
Dans l'UI Airflow (http://localhost:8080) mot de passe : airflow identifiant: airflow
| POSTGRES_CONN | [postgresql://airflow:airflow@postgres:5432/fraud_db |](http://localhost:8085)
API : SLACK_WEBHOOK_URL | https://huggingface.co/spaces/sdacelo/real-time-fraud-detection |


### 5. Activer les DAGs
- `fraud_detection_realtime` → s'exécute toutes les minutes
- `fraud_daily_report` → s'exécute chaque matin à 8h

## 📊 Architecture
```
API HuggingFace ──► DAG Temps Réel ──► PostgreSQL ──► Alert Slack
                         │
fraudTest.csv ──► ML Training ──► MLflow Registry
                         │
PostgreSQL ──► DAG Batch (8h) ──► Rapport HTML ──► Email
```

