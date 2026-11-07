
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from config import (
    N_USERS,
    N_DAYS,
    START_DATE,
    PROFILES_FILE,
    EVENTS_FILE,
    FEATURES_RAW_FILE,
    FEATURES_FILE,
    SCALER_FILE,
    FEATURE_COLUMNS_FILE
)


def load_data():

    profiles = pd.read_csv(PROFILES_FILE)

    events = pd.read_csv(EVENTS_FILE)

    events["timestamp"] = pd.to_datetime(
        events["timestamp"]
    )

    return profiles, events


def prepare_events(events, profiles):

    events = events.copy()

    # Jour de l'événement
    events["day"] = (
        events["timestamp"].dt.strftime("%Y-%m-%d")
    )

    # Heure de l'événement
    events["hour"] = (
        events["timestamp"].dt.hour
        + events["timestamp"].dt.minute / 60
    )

    # Récupérer les appareils habituels et le pays
    events = events.merge(
        profiles[
            ["user_id", "home_country", "device_1", "device_2"]
        ],
        on="user_id",
        how="left"
    )

    # Nouvel appareil
    events["is_new_device"] = (
        (events["device_id"] != events["device_1"])
        & (events["device_id"] != events["device_2"])
    ).astype(int)

    # Pays inhabituel
    events["is_foreign"] = (
        events["country"] != events["home_country"]
    ).astype(int)

    # Activité nocturne : de 22 h à 6 h
    events["is_night"] = (
        (events["hour"] >= 22)
        | (events["hour"] < 6)
    ).astype(int)

    return events


def calculate_daily_features(events):

    daily = events.groupby(
        ["user_id", "day"]
    ).agg(
        n_events=("event_id", "count"),
        n_new_device=("is_new_device", "sum"),
        n_foreign=("is_foreign", "sum"),
        pct_night=("is_night", "mean"),
        hour_mean=("hour", "mean"),
        is_attack=("is_attack", "max")
    ).reset_index()

    return daily



def calculate_login_features(events):

    login_events = events[
        events["event_type"] == "login"
    ].copy()

    login_events["failed_login"] = (
        login_events["success"] == 0
    ).astype(int)

    login_features = login_events.groupby(
        ["user_id", "day"]
    ).agg(
        n_login=("event_id", "count"),
        n_failed=("failed_login", "sum")
    ).reset_index()

    return login_features


def calculate_transaction_features(events):

    transactions = events[
        events["event_type"] == "transaction"
    ].copy()

    transaction_features = transactions.groupby(
        ["user_id", "day"]
    ).agg(
        n_tx=("event_id", "count"),
        amount_mean=("amount", "mean"),
        amount_max=("amount", "max")
    ).reset_index()

    return transaction_features


def calculate_time_gaps(events):

    sorted_events = events.sort_values(
        ["user_id", "day", "timestamp"]
    ).copy()

    sorted_events["gap_seconds"] = (
        sorted_events.groupby(
            ["user_id", "day"]
        )["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    gap_features = sorted_events.groupby(
        ["user_id", "day"]
    ).agg(
        min_gap=("gap_seconds", "min"),
        mean_gap=("gap_seconds", "mean")
    ).reset_index()

    return gap_features


def merge_features(
    daily,
    login_features,
    transaction_features,
    gap_features,
    profiles
):

    features = daily.merge(
        login_features,
        on=["user_id", "day"],
        how="left"
    )

    features = features.merge(
        transaction_features,
        on=["user_id", "day"],
        how="left"
    )

    features = features.merge(
        gap_features,
        on=["user_id", "day"],
        how="left"
    )

    # Construire toutes les combinaisons client/jour
    start = pd.Timestamp(START_DATE)

    all_days = [
        (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(N_DAYS)
    ]

    all_user_days = pd.MultiIndex.from_product(
        [profiles["user_id"].tolist(), all_days],
        names=["user_id", "day"]
    ).to_frame(index=False)

    # Garder toutes les journées, même sans événement
    features = all_user_days.merge(
        features,
        on=["user_id", "day"],
        how="left"
    )

    # Remplir les variables d'activité absentes
    zero_columns = [
        "n_events",
        "n_new_device",
        "n_foreign",
        "pct_night",
        "n_login",
        "n_failed",
        "n_tx",
        "amount_mean",
        "amount_max",
        "min_gap",
        "mean_gap",
        "is_attack"
    ]

    features[zero_columns] = (
        features[zero_columns].fillna(0)
    )

    # Pour un jour sans activité, utiliser l'heure habituelle
    usual_hours = profiles.set_index(
        "user_id"
    )["usual_hour"]

    features["hour_mean"] = (
        features["hour_mean"].fillna(
            features["user_id"].map(usual_hours)
        )
    )

    # Calcul du ratio montant maximal / montant habituel
    usual_amounts = profiles.set_index(
        "user_id"
    )["usual_amount"]

    features["amount_ratio"] = (
        features["amount_max"]
        / features["user_id"].map(usual_amounts)
    )

    features["amount_ratio"] = (
        features["amount_ratio"].replace(
            [np.inf, -np.inf], 0
        ).fillna(0)
    )

    return features



FEATURE_COLUMNS = [
    "n_events",
    "n_tx",
    "n_login",
    "n_failed",
    "n_new_device",
    "n_foreign",
    "pct_night",
    "hour_mean",
    "amount_mean",
    "amount_max",
    "amount_ratio",
    "min_gap",
    "mean_gap"
]


def save_raw_features(features):

    columns = [
        "user_id",
        "day",
        "is_attack"
    ] + FEATURE_COLUMNS

    features = features[columns].copy()

    features.to_csv(
        FEATURES_RAW_FILE,
        index=False
    )

    print("Features brutes sauvegardées :", FEATURES_RAW_FILE)

    return features

#normaulisation des features

LOG_COLUMNS = [
    "n_events",
    "n_tx",
    "n_login",
    "n_failed",
    "n_new_device",
    "n_foreign",
    "amount_mean",
    "amount_max",
    "amount_ratio",
    "min_gap",
    "mean_gap"
]


def transform_features(features):

    transformed = features.copy()

    for column in LOG_COLUMNS:

        transformed[column] = np.log1p(
            transformed[column].clip(lower=0)
        )

    return transformed


def normalize_features(features):

    X = features[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    scaled_features = pd.DataFrame(
        X_scaled,
        columns=FEATURE_COLUMNS
    )

    # Réintégrer les colonnes d'identification
    result = pd.concat(
        [
            features[["user_id", "day", "is_attack"]].reset_index(drop=True),
            scaled_features.reset_index(drop=True)
        ],
        axis=1
    )

    # Sauvegarde du scaler
    joblib.dump(
        scaler,
        SCALER_FILE
    )

    # Sauvegarde de l'ordre des features
    with open(FEATURE_COLUMNS_FILE, "w", encoding="utf-8") as file:

        file.write("\n".join(FEATURE_COLUMNS))

    result.to_csv(
        FEATURES_FILE,
        index=False
    )

    print("Données normalisées :", FEATURES_FILE)
    print("Scaler sauvegardé :", SCALER_FILE)

    return result




def main():

    print("\n===== CONSTRUCTION DES FEATURES =====")

    # 1. Charger les données
    profiles, events = load_data()

    print("Profils :", profiles.shape)
    print("Événements :", events.shape)

    # 2. Préparer les événements
    prepared_events = prepare_events(
        events,
        profiles
    )

    # 3. Calculer les features
    daily = calculate_daily_features(
        prepared_events
    )

    login_features = calculate_login_features(
        prepared_events
    )

    transaction_features = calculate_transaction_features(
        prepared_events
    )

    gap_features = calculate_time_gaps(
        prepared_events
    )

    # 4. Fusionner les features
    features = merge_features(
        daily,
        login_features,
        transaction_features,
        gap_features,
        profiles
    )

    # 5. Sauvegarder les données non normalisées
    features_raw = save_raw_features(
        features
    )

    # 6. Transformation logarithmique
    transformed = transform_features(
        features_raw
    )

    # 7. Normalisation
    normalized = normalize_features(
        transformed
    )

    print("\n===== FIN DE LA PRÉPARATION =====")
    print("Dimensions :", normalized.shape)
    print("Nombre de features :", len(FEATURE_COLUMNS))

    print("\nAperçu :")
    print(normalized.head())


if __name__ == "__main__":
    main()
