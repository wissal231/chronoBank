
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ======================================================
# CHEMINS DU PROJET
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
TEMPORAL_DIR = BASE_DIR / "data" / "processed" / "temporal"
RESULTS_DIR = BASE_DIR / "data" / "results"

EVENTS_FILE = RAW_DIR / "events.csv"
PROFILES_FILE = RAW_DIR / "client_profiles.csv"
ATTACKS_FILE = RAW_DIR / "attack_events.csv"

TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FLAG_COLUMNS = [
    "flag_brute_force",
    "flag_account_takeover",
    "flag_foreign_login",
    "flag_foreign_transaction",
    "flag_small_payment_fraud",
    "flag_high_transaction"
]


# ======================================================
# 1. CHARGEMENT DES DONNEES
# ======================================================
def load_data():
    """Charge les données brutes du Jour 1."""

    events = pd.read_csv(EVENTS_FILE)
    profiles = pd.read_csv(PROFILES_FILE)

    events["timestamp"] = pd.to_datetime(
        events["timestamp"],
        errors="coerce"
    )

    events["success"] = (
        pd.to_numeric(events["success"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    events["amount"] = (
        pd.to_numeric(events["amount"], errors="coerce")
        .fillna(0)
    )

    events["is_attack"] = (
        pd.to_numeric(events["is_attack"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    events = events.dropna(
        subset=["user_id", "timestamp", "event_type"]
    )

    events = events.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    return events, profiles


# ======================================================
# 2. TIMELINE
# ======================================================
def build_timeline(events):
    """Construit une timeline pour chaque utilisateur."""

    df = events.copy()

    # Date de l'événement
    df["day"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # Heure sous forme numérique
    df["hour"] = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60
        + df["timestamp"].dt.second / 3600
    )

    # Temps écoulé depuis l'événement précédent
    df["gap_seconds"] = (
        df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    df["gap_minutes"] = df["gap_seconds"] / 60

    # Événement précédent
    df["previous_event"] = (
        df.groupby("user_id")["event_type"].shift(1)
    )

    # Indique si l'événement est réalisé pendant la nuit
    df["is_night"] = (
        (df["hour"] >= 22) | (df["hour"] < 6)
    ).astype(int)

    return df


# ======================================================
# 3. FENETRES GLISSANTES
# ======================================================
def calculate_rolling_features(df):
    """
    Calcule les statistiques temporelles
    avec des fenêtres glissantes.
    """

    result = df.copy()

    result["failed_login"] = (
        (result["event_type"] == "login")
        & (result["success"] == 0)
    ).astype(int)

    result["transaction"] = (
        result["event_type"] == "transaction"
    ).astype(int)

    result["transaction_amount"] = np.where(
        result["event_type"] == "transaction",
        result["amount"],
        0
    )

    # Tri nécessaire : le timestamp devient l'index temporel
    result = result.sort_values(
        ["user_id", "timestamp"]
    )

    def calculate_user_windows(group):

        group = group.copy()
        group = group.set_index("timestamp")

        # Connexions échouées en 15 minutes (brute force)
        group["failed_login_15m"] = (
            group["failed_login"]
            .rolling("15min", closed="both")
            .sum()
        )

        # Connexions échouées en 30 minutes (account takeover)
        # CORRECTION : cette colonne manquait, d'où le KeyError
        group["failed_login_30m"] = (
            group["failed_login"]
            .rolling("30min", closed="both")
            .sum()
        )

        # Nombre de transactions en 60 minutes
        group["transactions_60m"] = (
            group["transaction"]
            .rolling("60min", closed="both")
            .sum()
        )

        # Montant total des transactions en 60 minutes
        group["amount_60m"] = (
            group["transaction_amount"]
            .rolling("60min", closed="both")
            .sum()
        )

        # Nombre de transactions en 5 minutes
        group["transactions_5m"] = (
            group["transaction"]
            .rolling("5min", closed="both")
            .sum()
        )

        return group.reset_index()

    # Boucle explicite : garde la colonne user_id
    # quelle que soit la version de pandas
    pieces = [
        calculate_user_windows(group)
        for _, group in result.groupby("user_id")
    ]

    result = pd.concat(pieces, ignore_index=True)

    rolling_outputs = [
        "failed_login_15m",
        "failed_login_30m",
        "transactions_60m",
        "amount_60m",
        "transactions_5m"
    ]

    result[rolling_outputs] = result[rolling_outputs].fillna(0)

    return result


# ======================================================
# 4. CONTEXTE UTILISATEUR
# ======================================================
def add_user_context(events, profiles):
    """
    Ajoute les informations habituelles des utilisateurs
    à chaque événement.
    """

    context = profiles[
        [
            "user_id",
            "home_country",
            "device_1",
            "device_2",
            "usual_amount"
        ]
    ].copy()

    df = events.merge(
        context,
        on="user_id",
        how="left"
    )

    # Détection d'un nouvel appareil
    df["is_new_device"] = (
        (df["device_id"] != df["device_1"])
        & (df["device_id"] != df["device_2"])
    ).astype(int)

    # Détection d'un pays inhabituel
    df["is_foreign"] = (
        df["country"] != df["home_country"]
    ).astype(int)

    # Rapport entre le montant et le montant habituel
    df["amount_ratio"] = np.where(
        df["usual_amount"] > 0,
        df["amount"] / df["usual_amount"],
        0
    )

    return df


# ======================================================
# 5. DETECTION DES SEQUENCES SUSPECTES
# ======================================================
def detect_suspicious_sequences(df):
    """
    Détecte les trois scénarios d'attaque :
    1. Account Takeover
    2. Foreign Login
    3. Small Payment Fraud

    La détection n'utilise jamais is_attack.
    """

    result = df.copy()

    # Initialisation des indicateurs
    for column in FLAG_COLUMNS:
        result[column] = 0

    result = result.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    # Analyse séparée de chaque utilisateur
    for user_id, group in result.groupby("user_id"):

        indexes = group.index.tolist()

        # Mémorise les dernières connexions suspectes
        suspicious_login_time = None
        foreign_login_time = None

        for idx in indexes:

            row = result.loc[idx]

            event_type = row["event_type"]
            timestamp = row["timestamp"]

            # 1. BRUTE FORCE
            if row["failed_login_15m"] >= 5:
                result.at[idx, "flag_brute_force"] = 1

            # 2. CONNEXION SUSPECTE APRES PLUSIEURS ECHECS
            if (
                event_type == "login"
                and row["success"] == 1
                and row["failed_login_30m"] >= 5
                and (
                    row["is_new_device"] == 1
                    or row["is_foreign"] == 1
                )
            ):
                suspicious_login_time = timestamp
                result.at[idx, "flag_account_takeover"] = 1

            # 3. CHANGEMENT DE MOT DE PASSE OU GROS MONTANT
            #    APRES LA CONNEXION SUSPECTE
            if suspicious_login_time is not None:

                elapsed = (
                    timestamp - suspicious_login_time
                ).total_seconds() / 60

                if 0 <= elapsed <= 30:

                    if event_type == "password_change":
                        result.at[idx, "flag_account_takeover"] = 1

                    if (
                        event_type == "transaction"
                        and row["amount_ratio"] >= 10
                    ):
                        result.at[idx, "flag_account_takeover"] = 1

                else:
                    suspicious_login_time = None

            # 4. FOREIGN LOGIN
            if (
                event_type == "login"
                and row["success"] == 1
                and (
                    row["is_foreign"] == 1
                    or row["is_new_device"] == 1
                )
            ):
                foreign_login_time = timestamp
                result.at[idx, "flag_foreign_login"] = 1

            # 5. TRANSACTION APRES FOREIGN LOGIN
            if (
                event_type == "transaction"
                and foreign_login_time is not None
            ):

                elapsed = (
                    timestamp - foreign_login_time
                ).total_seconds() / 60

                if 0 <= elapsed <= 10:
                    if row["amount_ratio"] >= 6:
                        result.at[
                            idx,
                            "flag_foreign_transaction"
                        ] = 1

                if elapsed > 10:
                    foreign_login_time = None

            # 6. SMALL PAYMENT FRAUD
            if (
                event_type == "transaction"
                and row["transactions_60m"] >= 20
            ):
                result.at[idx, "flag_small_payment_fraud"] = 1

            # 7. TRANSACTION TRES ELEVEE
            if (
                event_type == "transaction"
                and row["amount_ratio"] >= 10
            ):
                result.at[idx, "flag_high_transaction"] = 1

    return result


# ======================================================
# 6. SCORE TEMPOREL
# ======================================================
def calculate_temporal_score(df):
    """
    Calcule un score temporel pour chaque événement.
    Score compris entre 0 et 100.
    """

    result = df.copy()

    result["temporal_score_event"] = (
        result["flag_brute_force"] * 15
        + result["flag_account_takeover"] * 35
        + result["flag_foreign_login"] * 10
        + result["flag_foreign_transaction"] * 35
        + result["flag_small_payment_fraud"] * 35
        + result["flag_high_transaction"] * 15
    )

    result["temporal_score_event"] = (
        result["temporal_score_event"].clip(0, 100)
    )

    return result


# ======================================================
# 7. AGREGATION QUOTIDIENNE
# ======================================================
def aggregate_daily_results(df):
    """Agrège les scores temporels par utilisateur et par jour."""

    daily = df.groupby(
        ["user_id", "day"]
    ).agg(
        temporal_score=("temporal_score_event", "max"),
        n_temporal_alerts=(
            "temporal_score_event",
            lambda x: (x > 0).sum()
        ),
        max_failed_login_15m=("failed_login_15m", "max"),
        max_transactions_60m=("transactions_60m", "max"),
        max_amount_60m=("amount_60m", "max"),
        **{
            column: (column, "max")
            for column in FLAG_COLUMNS
        }
    ).reset_index()

    # Ajouter les couples utilisateur/jour sans événement
    # pour conserver tous les jours de chaque utilisateur.
    profiles = pd.read_csv(PROFILES_FILE)

    days = pd.date_range(
        start=pd.to_datetime(df["day"]).min(),
        end=pd.to_datetime(df["day"]).max(),
        freq="D"
    ).strftime("%Y-%m-%d")

    all_days = pd.MultiIndex.from_product(
        [profiles["user_id"].unique(), days],
        names=["user_id", "day"]
    ).to_frame(index=False)

    daily = all_days.merge(
        daily,
        on=["user_id", "day"],
        how="left"
    )

    numeric_columns = [
        "temporal_score",
        "n_temporal_alerts",
        "max_failed_login_15m",
        "max_transactions_60m",
        "max_amount_60m",
        *FLAG_COLUMNS
    ]

    daily[numeric_columns] = daily[numeric_columns].fillna(0)

    daily["temporal_score"] = daily["temporal_score"].clip(0, 100)

    daily["temporal_flag"] = (
        daily["temporal_score"] >= 30
    ).astype(int)

    daily["temporal_level"] = np.select(
        [
            daily["temporal_score"] >= 60,
            daily["temporal_score"] >= 30
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    return daily


# ======================================================
# 8. ALERTES DETAILLEES
# ======================================================
def create_alerts(df):
    """Construit une table détaillée des événements suspects."""

    alert_definitions = {
        "flag_brute_force": "Brute Force",
        "flag_account_takeover": "Account Takeover",
        "flag_foreign_login": "Foreign Login",
        "flag_foreign_transaction": "Foreign Transaction",
        "flag_small_payment_fraud": "Small Payment Fraud",
        "flag_high_transaction": "High Transaction"
    }

    columns = [
        "event_id",
        "user_id",
        "timestamp",
        "day",
        "event_type",
        "alert_type",
        "reason",
        "temporal_score_event"
    ]

    alerts = []

    for flag_column, alert_name in alert_definitions.items():

        detected = df[df[flag_column] == 1].copy()

        if detected.empty:
            continue

        detected["alert_type"] = alert_name

        detected["reason"] = detected["user_id"].apply(
            lambda user: f"{alert_name} détecté pour {user}"
        )

        alerts.append(detected[columns])

    if not alerts:
        return pd.DataFrame(columns=columns)

    return pd.concat(
        alerts,
        ignore_index=True
    ).sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)


# ======================================================
# 9. EVALUATION
# ======================================================
def evaluate_temporal_detection(daily_results, events):
    """
    Évalue les prédictions temporelles
    en utilisant les labels uniquement pour l'évaluation.
    """

    # Vérité terrain : un jour est attaqué si au moins
    # un événement de cette journée est marqué comme attaque.
    truth = (
        events.groupby(["user_id", "day"])["is_attack"]
        .max()
        .reset_index(name="y_true")
    )

    evaluation = daily_results.merge(
        truth,
        on=["user_id", "day"],
        how="left"
    )

    evaluation["y_true"] = (
        evaluation["y_true"].fillna(0).astype(int)
    )

    y_true = evaluation["y_true"]
    y_pred = evaluation["temporal_flag"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    })

    metrics_df = pd.DataFrame([metrics])

    print("\n===== EVALUATION TEMPORELLE =====")

    for metric, value in metrics.items():
        if metric in [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive"
        ]:
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0
        )
    )

    return metrics_df, evaluation


def evaluate_by_scenario(daily_results, events):
    """Mesure le taux de détection pour chaque scénario simulé."""

    attacks = pd.read_csv(ATTACKS_FILE)

    # Évite les colonnes en double (user_id_x / user_id_y)
    # si attack_events.csv contient déjà user_id ou day
    attacks = attacks.drop(
        columns=[c for c in ["user_id", "day"] if c in attacks.columns]
    )

    # Récupérer le user_id et le jour de chaque événement attaqué
    attack_details = attacks.merge(
        events[["event_id", "user_id", "day"]],
        on="event_id",
        how="left"
    )

    # Chaque attaque est identifiée par attack_id
    cases = (
        attack_details.groupby(
            ["attack_id", "attack_scenario", "user_id", "day"]
        )
        .size()
        .reset_index(name="n_attack_events")
    )

    cases = cases.merge(
        daily_results[
            ["user_id", "day", "temporal_flag", "temporal_score"]
        ],
        on=["user_id", "day"],
        how="left"
    )

    cases["detected"] = (
        cases["temporal_flag"].fillna(0).astype(int)
    )

    scenario_results = (
        cases.groupby("attack_scenario")
        .agg(
            total_cases=("attack_id", "nunique"),
            detected_cases=("detected", "sum")
        )
        .reset_index()
    )

    scenario_results["detection_rate"] = (
        scenario_results["detected_cases"]
        / scenario_results["total_cases"]
    )

    return scenario_results


# ======================================================
# 10. PROGRAMME PRINCIPAL
# ======================================================
def main():

    print("\n===================================")
    print(" CHRONOBANK AI - TEMPORAL ANALYSIS")
    print("===================================\n")

    # 1. Chargement des données
    print("[1/8] Chargement des données...")

    events, profiles = load_data()

    print("Nombre d'événements :", len(events))
    print("Nombre d'utilisateurs :", profiles["user_id"].nunique())

    # 2. Ajout du contexte utilisateur
    print("\n[2/8] Ajout du contexte utilisateur...")

    events = add_user_context(events, profiles)

    # 3. Construction de la timeline
    print("\n[3/8] Construction de la timeline...")

    events = build_timeline(events)

    # 4. Calcul des fenêtres glissantes
    print("\n[4/8] Calcul des fenêtres temporelles...")

    events = calculate_rolling_features(events)

    # 5. Détection des séquences suspectes
    print("\n[5/8] Détection des attaques...")

    events = detect_suspicious_sequences(events)

    # 6. Calcul des scores
    print("\n[6/8] Calcul des scores temporels...")

    events = calculate_temporal_score(events)

    # Sauvegarde des événements enrichis
    events.to_csv(
        TEMPORAL_DIR / "temporal_events.csv",
        index=False
    )

    # 7. Agrégation quotidienne
    print("\n[7/8] Agrégation quotidienne...")

    daily_results = aggregate_daily_results(events)

    daily_results.to_csv(
        RESULTS_DIR / "res_temporal.csv",
        index=False
    )

    # Création des alertes
    alerts = create_alerts(events)

    alerts.to_csv(
        RESULTS_DIR / "temporal_alerts.csv",
        index=False
    )

    # 8. Évaluation
    print("\n[8/8] Évaluation du détecteur...")

    metrics, evaluation = evaluate_temporal_detection(
        daily_results,
        events
    )

    metrics.to_csv(
        RESULTS_DIR / "temporal_evaluation.csv",
        index=False
    )

    scenario_results = evaluate_by_scenario(
        daily_results,
        events
    )

    scenario_results.to_csv(
        RESULTS_DIR / "temporal_evaluation_by_scenario.csv",
        index=False
    )

    print("\n===== RESULTATS PAR SCENARIO =====")
    print(scenario_results)

    print("\n===== FICHIERS GENERES =====")

    print("1. temporal_events.csv")
    print("2. res_temporal.csv")
    print("3. temporal_alerts.csv")
    print("4. temporal_evaluation.csv")
    print("5. temporal_evaluation_by_scenario.csv")

    print("\nAnalyse temporelle terminée avec succès !")


if __name__ == "__main__":
    main()