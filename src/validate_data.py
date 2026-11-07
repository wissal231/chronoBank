
import pandas as pd
import numpy as np

from config import (
    N_USERS,
    N_DAYS,
    N_ATTACKS_PER_SCENARIO,
    PROFILES_FILE,
    EVENTS_FILE,
    ATTACK_EVENTS_FILE,
    FEATURES_RAW_FILE,
    FEATURES_FILE,
    FEATURE_COLUMNS_FILE
)

# Chargement des fichiers
profiles = pd.read_csv(PROFILES_FILE)
events = pd.read_csv(EVENTS_FILE)
attacks = pd.read_csv(ATTACK_EVENTS_FILE)
features_raw = pd.read_csv(FEATURES_RAW_FILE)
features = pd.read_csv(FEATURES_FILE)

# Lecture de la liste des features
with open(FEATURE_COLUMNS_FILE, "r", encoding="utf-8") as file:
    feature_columns = file.read().splitlines()


def check(condition, message):

    if condition:
        print("[OK]", message)
    else:
        print("[ERREUR]", message)


print("\n========== 1. PROFILS ==========")

check(
    len(profiles) == N_USERS,
    f"Nombre de clients : {len(profiles)}"
)

check(
    profiles["user_id"].is_unique,
    "Identifiants clients uniques"
)

check(
    not profiles.isnull().any().any(),
    "Aucune valeur manquante dans les profils"
)


print("\n========== 2. ÉVÉNEMENTS ==========")

check(
    events["event_id"].is_unique,
    "Identifiants d'événements uniques"
)

check(
    events["user_id"].nunique() == N_USERS,
    "Les 1000 clients sont représentés"
)

check(
    events["is_attack"].isin([0, 1]).all(),
    "Le label is_attack contient uniquement 0 et 1"
)

check(
    (events["amount"] >= 0).all(),
    "Aucun montant négatif"
)

check(
    not events.isnull().any().any(),
    "Aucune valeur manquante dans les événements"
)

check(
    set(events["event_type"].unique()).issubset({
        "login",
        "transaction",
        "password_change"
    }),
    "Types d'événements valides"
)

print("\nRépartition des événements :")
print(events["event_type"].value_counts())

print("\nRépartition des labels :")
print(events["is_attack"].value_counts())


print("\n========== 3. SCÉNARIOS D'ATTAQUE ==========")

expected_scenarios = {
    "account_takeover",
    "small_payment_fraud",
    "foreign_login"
}

scenario_counts = (
    attacks.groupby("attack_scenario")["attack_id"]
    .nunique()
)

print(scenario_counts)

check(
    set(scenario_counts.index) == expected_scenarios,
    "Les trois scénarios sont présents"
)

check(
    all(
        scenario_counts.get(scenario, 0) == N_ATTACKS_PER_SCENARIO
        for scenario in expected_scenarios
    ),
    "100 cas d'attaque distincts par scénario"
)

check(
    attacks["event_id"].is_unique,
    "Chaque événement attaqué apparaît une seule fois dans le registre"
)

check(
    set(attacks["event_id"]).issubset(set(events["event_id"])),
    "Tous les événements attaqués existent dans events.csv"
)

check(
    (events["is_attack"].sum() == len(attacks)),
    "Tous les événements marqués comme attaques sont répertoriés"
)

# Vérifier les couples client/jour attaqués
events_check = events.copy()
events_check["timestamp"] = pd.to_datetime(events_check["timestamp"])
events_check["day"] = events_check["timestamp"].dt.strftime("%Y-%m-%d")

attack_user_days = events_check[
    events_check["is_attack"] == 1
][["user_id", "day"]].drop_duplicates()

check(
    len(attack_user_days) == 3 * N_ATTACKS_PER_SCENARIO,
    "300 couples client/jour attaqués"
)


print("\n========== 4. FEATURES BRUTES ==========")

check(
    len(features_raw) == N_USERS * N_DAYS,
    "30 000 lignes client/jour"
)

check(
    len(feature_columns) == 13,
    "13 features définies"
)

check(
    set(feature_columns).issubset(features_raw.columns),
    "Les 13 features sont présentes"
)

check(
    not features_raw.isnull().any().any(),
    "Aucune valeur manquante dans features_raw.csv"
)

check(
    features_raw["is_attack"].isin([0, 1]).all(),
    "Labels de features valides"
)

check(
    features_raw["is_attack"].sum()
    == 3 * N_ATTACKS_PER_SCENARIO,
    "300 lignes client/jour marquées comme attaquées"
)

check(
    features_raw["pct_night"].between(0, 1).all(),
    "pct_night est compris entre 0 et 1"
)

check(
    (features_raw["amount_ratio"] >= 0).all(),
    "amount_ratio est positif ou nul"
)


print("\n========== 5. FEATURES NORMALISÉES ==========")

check(
    features.shape == (N_USERS * N_DAYS, 16),
    f"Dimensions attendues : (30000, 16), obtenues : {features.shape}"
)

check(
    set(feature_columns).issubset(features.columns),
    "Toutes les features normalisées sont présentes"
)

check(
    not features.isnull().any().any(),
    "Aucune valeur manquante après normalisation"
)

check(
    features["is_attack"].isin([0, 1]).all(),
    "Le label est conservé sans normalisation"
)

check(
    features["user_id"].equals(features_raw["user_id"])
    and features["day"].equals(features_raw["day"]),
    "Les identifiants et les jours sont conservés"
)

print("\nMoyennes des features normalisées :")
print(features[feature_columns].mean().round(3))

print("\nÉcarts-types des features normalisées :")
print(features[feature_columns].std().round(3))

print("\n========== VALIDATION TERMINÉE ==========")


