
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
# CORRECTION : temporal_events.csv est dans processed/temporal
TEMPORAL_DIR = BASE_DIR / "data" / "processed" / "temporal"
RESULTS_DIR = BASE_DIR / "data" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

daily = pd.read_csv(
    RESULTS_DIR / "res_temporal.csv"
)

scenario = pd.read_csv(
    RESULTS_DIR / "temporal_evaluation_by_scenario.csv"
)

# CORRECTION : lecture depuis TEMPORAL_DIR et non RESULTS_DIR
events = pd.read_csv(
    TEMPORAL_DIR / "temporal_events.csv"
)

attacks = pd.read_csv(
    RAW_DIR / "attack_events.csv"
)

events["timestamp"] = pd.to_datetime(
    events["timestamp"]
)

# ----------------------------------
# 1. DISTRIBUTION DES SCORES
# ----------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=daily,
    x="temporal_score",
    bins=20,
    kde=True
)

plt.title("Distribution des scores temporels")
plt.xlabel("Temporal Risk Score")
plt.ylabel("Nombre d'utilisateurs-jours")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "temporal_scores.png",
    dpi=300
)

plt.close()


# ----------------------------------
# 2. TAUX DE DETECTION PAR SCENARIO
# ----------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=scenario,
    x="attack_scenario",
    y="detection_rate"
)

plt.title("Taux de détection par scénario")
plt.xlabel("Scénario")
plt.ylabel("Taux de détection")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "temporal_detection.png",
    dpi=300
)

plt.close()


# ----------------------------------
# 3. ATTACK TIMELINE
# ----------------------------------

# Choisir un cas Account Takeover s'il existe
account_cases = attacks[
    attacks["attack_scenario"] == "account_takeover"
]

if not account_cases.empty:

    selected_attack = account_cases["attack_id"].iloc[0]

    selected_events = attacks[
        attacks["attack_id"] == selected_attack
    ]

    attack_event_ids = set(
        selected_events["event_id"]
    )

    user_id = events[
        events["event_id"].isin(attack_event_ids)
    ]["user_id"].iloc[0]

    attack_times = events[
        events["event_id"].isin(attack_event_ids)
    ]["timestamp"]

    start = attack_times.min() - pd.Timedelta(minutes=10)
    end = attack_times.max() + pd.Timedelta(minutes=10)

    timeline = events[
        (events["user_id"] == user_id)
        & (events["timestamp"] >= start)
        & (events["timestamp"] <= end)
    ].copy()

    timeline["injected_attack"] = (
        timeline["event_id"].isin(attack_event_ids)
    )

    plt.figure(figsize=(13, 6))

    sns.scatterplot(
        data=timeline,
        x="timestamp",
        y="event_type",
        hue="injected_attack",
        style="injected_attack",
        s=100
    )

    plt.title(
        f"Attack Timeline - {user_id} - {selected_attack}"
    )

    plt.xlabel("Date et heure")
    plt.ylabel("Type d'événement")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "attack_timeline.png",
        dpi=300
    )

    plt.close()

else:
    print("Aucun cas Account Takeover trouvé.")

print("Visualisations temporelles générées.")