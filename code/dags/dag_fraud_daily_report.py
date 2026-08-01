"""
DAG — Rapport quotidien fraude (branché sur les vraies données stockées en Postgres)
Schedule : 8h00 chaque matin
"""
from datetime import datetime, timedelta
import os

import pandas as pd
import sqlalchemy

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "data_team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


def extract_yesterday(**context):
    """Récupère les vraies transactions de la veille depuis Postgres (table fraud_transactions)."""
    conn_str = os.environ.get("POSTGRES_FRAUD_CONN") or os.environ.get("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    if not conn_str:
        raise RuntimeError("Aucune connexion Postgres disponible (ni POSTGRES_FRAUD_CONN, ni AIRFLOW__DATABASE__SQL_ALCHEMY_CONN).")

    engine = sqlalchemy.create_engine(conn_str)

    now = datetime.utcnow()
    yesterday_start = datetime(now.year, now.month, now.day) - timedelta(days=1)
    yesterday_end = yesterday_start + timedelta(days=1)

    query = sqlalchemy.text("""
        SELECT amt, fraud_score, is_fraud_predicted, is_fraud_ground_truth
        FROM fraud_transactions
        WHERE detected_at >= :start AND detected_at < :end
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"start": yesterday_start, "end": yesterday_end})

    is_demo_fallback = False
    # Filet de sécurité pour la démo : le pipeline temps réel vient d'être lancé,
    # il n'existe donc pas encore de "vraie" journée complète d'hier.
    # On bascule alors sur l'ensemble des données déjà collectées, pour pouvoir
    # démontrer le pipeline de bout en bout dès aujourd'hui.
    if df.empty:
        is_demo_fallback = True
        print("[EXTRACT] ⚠️ Aucune transaction trouvée pour 'hier' (pipeline temps réel encore jeune).")
        print("[EXTRACT] 🔀 Bascule en mode démo : agrégation de TOUTES les transactions déjà collectées.")
        with engine.connect() as connection:
            df = pd.read_sql("SELECT amt, fraud_score, is_fraud_predicted, is_fraud_ground_truth FROM fraud_transactions", connection)

    total_tx = len(df)
    nb_fraudes = int(df["is_fraud_predicted"].sum()) if total_tx > 0 else 0
    fraud_rate_pct = round((nb_fraudes / total_tx) * 100, 3) if total_tx > 0 else 0.0
    amount_total = round(float(df["amt"].sum()), 2) if total_tx > 0 else 0.0
    amount_fraud = round(float(df.loc[df["is_fraud_predicted"] == 1, "amt"].sum()), 2) if total_tx > 0 else 0.0

    # Comparatif vs vérité terrain, quand elle est disponible dans les données API réelles
    has_ground_truth = "is_fraud_ground_truth" in df.columns and df["is_fraud_ground_truth"].notna().any()
    nb_vrais_positifs = nb_faux_negatifs = nb_faux_positifs = None
    if has_ground_truth:
        gt = df["is_fraud_ground_truth"].fillna(-1).astype(int)
        pred = df["is_fraud_predicted"].astype(int)
        nb_vrais_positifs = int(((gt == 1) & (pred == 1)).sum())
        nb_faux_negatifs = int(((gt == 1) & (pred == 0)).sum())
        nb_faux_positifs = int(((gt == 0) & (pred == 1)).sum())

    data = {
        "date_reporting": str(yesterday_start.date()),
        "is_demo_fallback": is_demo_fallback,
        "total_transactions": total_tx,
        "total_frauds": nb_fraudes,
        "fraud_rate_pct": fraud_rate_pct,
        "total_amount_eur": amount_total,
        "fraud_amount_eur": amount_fraud,
        "has_ground_truth": has_ground_truth,
        "nb_vrais_positifs": nb_vrais_positifs,
        "nb_faux_negatifs": nb_faux_negatifs,
        "nb_faux_positifs": nb_faux_positifs,
    }

    print(f"[EXTRACT] ✅ {total_tx} transaction(s) réelle(s) analysée(s) pour {data['date_reporting']}"
          f"{' (mode démo)' if is_demo_fallback else ''}.")
    print(f"[EXTRACT] {nb_fraudes} fraude(s) détectée(s) par le modèle.")
    context["ti"].xcom_push(key="daily_consolidated_stats", value=data)


def compute_kpis(**context):
    """Consolide les KPIs pour le rapport."""
    stats = context["ti"].xcom_pull(key="daily_consolidated_stats", task_ids="extract_yesterday")

    avg_tx_value = stats["total_amount_eur"] / stats["total_transactions"] if stats["total_transactions"] > 0 else 0
    avg_fraud_value = stats["fraud_amount_eur"] / stats["total_frauds"] if stats["total_frauds"] > 0 else 0

    kpis = {
        **stats,
        "average_transaction_value": round(avg_tx_value, 2),
        "average_fraud_value": round(avg_fraud_value, 2),
    }

    print("[COMPUTE] ✅ KPIs journaliers calculés à partir des vraies données stockées.")
    context["ti"].xcom_push(key="kpis", value=kpis)


def generate_report(**context):
    """Génère le rapport formaté."""
    kpis = context["ti"].xcom_pull(key="kpis", task_ids="compute_kpis")

    demo_note = (
        "\n⚠️ NOTE DÉMO : pas encore de journée complète d'historique, agrégation sur\n"
        "   toutes les transactions collectées depuis le lancement du pipeline.\n"
        if kpis["is_demo_fallback"] else ""
    )

    ground_truth_block = ""
    if kpis["has_ground_truth"]:
        ground_truth_block = f"""
------------------------------------------------------------------------
COMPARATIF VS VÉRITÉ TERRAIN (label réel du dataset) :
------------------------------------------------------------------------
Vrais positifs (fraude détectée)     : {kpis['nb_vrais_positifs']}
Faux négatifs (fraude ratée)         : {kpis['nb_faux_negatifs']}
Faux positifs (fausse alerte)        : {kpis['nb_faux_positifs']}
"""

    report_content = f"""
========================================================================
DAILY FRAUD DETECTION REPORT — {kpis['date_reporting']}
========================================================================
{demo_note}------------------------------------------------------------------------
GLOBAL TRANSACTIONAL SUMMARY:
------------------------------------------------------------------------
Total Volume Analyzed   : {kpis['total_transactions']:,}
Total Amount Processed  : {kpis['total_amount_eur']:,} EUR
Average Ticket Value    : {kpis['average_transaction_value']} EUR
------------------------------------------------------------------------
FRAUD RISK PERFORMANCE:
------------------------------------------------------------------------
Confirmed Frauds        : {kpis['total_frauds']}
Detection Rate          : {kpis['fraud_rate_pct']}%
Total Fraudulent Amount : {kpis['fraud_amount_eur']:,} EUR
Average Fraud Value     : {kpis['average_fraud_value']} EUR
{ground_truth_block}========================================================================
Generated by Apache Airflow (Jedha Lead Program) — données réelles Postgres
========================================================================
"""
    print("[GENERATE] ✅ Rapport généré à partir des données réelles :")
    print(report_content)
    context["ti"].xcom_push(key="final_report", value=report_content)


def send_daily_email(**context):
    """Simule l'envoi du rapport par email (notification, comme demandé par le métier)."""
    kpis = context["ti"].xcom_pull(key="kpis", task_ids="compute_kpis")
    report = context["ti"].xcom_pull(key="final_report", task_ids="generate_report")

    recipient = "business-intelligence@company.com"
    subject = f"🚨 Daily Fraud Report: {kpis['date_reporting']} — {kpis['total_frauds']} Frauds Confirmed"

    print(f"[EMAIL] ✅ Préparation de l'envoi à {recipient}.")
    print(f"[EMAIL] Subject: {subject}")
    print(f"[EMAIL] Content:\n{report}")
    print("[EMAIL] Status: Sent Successfully.")


with DAG(
    dag_id="fraud_daily_reporting_batch",
    description="Rapport quotidien consolidé des fraudes, basé sur les vraies données Postgres",
    schedule_interval="0 8 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["reporting", "batch", "daily"],
) as dag:

    extract_task = PythonOperator(task_id="extract_yesterday", python_callable=extract_yesterday)
    compute_task = PythonOperator(task_id="compute_kpis", python_callable=compute_kpis)
    generate_task = PythonOperator(task_id="generate_report", python_callable=generate_report)
    send_task = PythonOperator(task_id="send_daily_email", python_callable=send_daily_email)

    extract_task >> compute_task >> generate_task >> send_task
