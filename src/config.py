
from pathlib import Path

# Chemin principal du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossiers
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Création automatique des dossiers s'ils n'existent pas
for folder in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Paramètres de simulation
RANDOM_STATE = 42

N_USERS = 1000
N_DAYS = 30

# Date de départ de la simulation
START_DATE = "2026-01-01"

# Nombre de scénarios et d'attaques
N_ATTACKS_PER_SCENARIO = 100

# Fréquences quotidiennes moyennes
LOGIN_RATE_MIN = 0.7
LOGIN_RATE_MAX = 0.9

TX_RATE_MIN = 0.4
TX_RATE_MAX = 0.6

# Pays possibles
COUNTRIES = ["MA", "FR", "ES", "US", "UK", "DE"]

# Types d'événements
EVENT_TYPES = [
    "login",
    "transaction",
    "password_change"
]

# Fichiers de sortie
PROFILES_FILE = RAW_DIR / "client_profiles.csv"
EVENTS_FILE = RAW_DIR / "events.csv"
ATTACK_EVENTS_FILE = RAW_DIR / "attack_events.csv"

FEATURES_RAW_FILE = PROCESSED_DIR / "features_raw.csv"
FEATURES_FILE = PROCESSED_DIR / "features.csv"

SCALER_FILE = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.txt"
