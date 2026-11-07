
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import EVENTS_FILE, FEATURES_RAW_FILE

events = pd.read_csv(EVENTS_FILE)
features = pd.read_csv(FEATURES_RAW_FILE)

# 1. Distribution des types d'événements
plt.figure(figsize=(8, 5))

sns.countplot(
    data=events,
    x="event_type",
    order=events["event_type"].value_counts().index
)

plt.title("Distribution des types d'événements")
plt.xlabel("Type d'événement")
plt.ylabel("Nombre d'événements")
plt.tight_layout()
plt.savefig("data/event_types.png")
plt.show()


# 2. Répartition normal / attaque
plt.figure(figsize=(6, 5))

sns.countplot(
    data=events,
    x="is_attack"
)

plt.title("Événements normaux et événements attaqués")
plt.xlabel("is_attack")
plt.ylabel("Nombre d'événements")
plt.tight_layout()
plt.savefig("data/attack_distribution.png")
plt.show()


# 3. Distribution des montants des transactions
transactions = events[
    events["event_type"] == "transaction"
]

plt.figure(figsize=(8, 5))

sns.histplot(
    data=transactions,
    x="amount",
    hue="is_attack",
    bins=50,
    element="step"
)

plt.title("Distribution des montants des transactions")
plt.xlabel("Montant (MAD)")
plt.ylabel("Nombre de transactions")
plt.tight_layout()
plt.savefig("data/transaction_amounts.png")
plt.show()


# 4. Nombre d'événements par client et par jour
plt.figure(figsize=(8, 5))

sns.histplot(
    data=features,
    x="n_events",
    bins=30
)

plt.title("Distribution du nombre d'événements par client/jour")
plt.xlabel("Nombre d'événements")
plt.ylabel("Nombre de clients/jours")
plt.tight_layout()
plt.savefig("data/daily_events.png")
plt.show()
