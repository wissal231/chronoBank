
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
TEMPORAL_DIR = BASE_DIR / "data" / "processed" / "temporal"
RESULTS_DIR = BASE_DIR / "data" / "results"

events = pd.read_csv(
    TEMPORAL_DIR / "temporal_events.csv"
)

daily = pd.read_csv(
    RESULTS_DIR / "res_temporal.csv"
)

alerts = pd.read_csv(
    RESULTS_DIR / "temporal_alerts.csv"
)

metrics = pd.read_csv(
    RESULTS_DIR / "temporal_evaluation.csv"
)

scenarios = pd.read_csv(
    RESULTS_DIR / "temporal_evaluation_by_scenario.csv"
)

profiles = pd.read_csv(
    RAW_DIR / "client_profiles.csv"
)

# 1. Vérifier les utilisateurs
assert events["user_id"].nunique() == len(profiles)

# 2. Vérifier les identifiants
assert events["event_id"].is_unique

# 3. Vérifier les scores
assert daily["temporal_score"].between(0, 100).all()

# 4. Vérifier les niveaux
assert set(daily["temporal_level"].unique()).issubset(
    {"Low", "Medium", "High"}
)

# 5. Vérifier les valeurs binaires
assert daily["temporal_flag"].isin([0, 1]).all()

# 6. Vérifier l'absence de valeurs manquantes
assert daily["temporal_score"].notna().all()

# 7. Vérifier le nombre de lignes quotidiennes
expected_rows = profiles["user_id"].nunique() * 30

assert len(daily) == expected_rows, (
    f"Nombre de lignes incorrect : "
    f"{len(daily)} au lieu de {expected_rows}"
)

# 8. Vérifier les trois scénarios
expected_scenarios = {
    "account_takeover",
    "small_payment_fraud",
    "foreign_login"
}

assert expected_scenarios.issubset(
    set(scenarios["attack_scenario"])
)

# 9. Vérifier que les métriques existent
assert len(metrics) == 1

print("\n===== VALIDATION TEMPORAL =====")

print("Nombre d'événements :", len(events))
print("Nombre d'utilisateurs :", events["user_id"].nunique())
print("Nombre de résultats quotidiens :", len(daily))
print("Nombre d'alertes :", len(alerts))

print("\nDistribution des niveaux :")
print(daily["temporal_level"].value_counts())

print("\nEvaluation :")
print(metrics)

print("\nScénarios :")
print(scenarios)

print("\nToutes les vérifications sont réussies !")
