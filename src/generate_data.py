
import numpy as np
import pandas as pd
import uuid

from datetime import timedelta

from config import (
    RANDOM_STATE,
    N_USERS,
    N_DAYS,
    START_DATE,
    N_ATTACKS_PER_SCENARIO,
    LOGIN_RATE_MIN,
    LOGIN_RATE_MAX,
    TX_RATE_MIN,
    TX_RATE_MAX,
    COUNTRIES,
    PROFILES_FILE,
    EVENTS_FILE,
    ATTACK_EVENTS_FILE
)

rng = np.random.default_rng(RANDOM_STATE)  #random number generator

start_date = pd.Timestamp(START_DATE)

events = []
attack_records = []



def generate_client_profiles():

    profiles = []

    for i in range(N_USERS):

        user_id = f"U{i + 1:04d}"

        home_country = rng.choice(
            COUNTRIES,
            p=[0.55, 0.12, 0.08, 0.10, 0.08, 0.07]
        )

        device_1 = f"DEV_{i+1}_A"
        device_2 = f"DEV_{i+1}_B"

        usual_hour = int(rng.integers(7, 22))

        usual_amount = float(
            rng.lognormal(mean=5.5, sigma=0.6)   # pour avoir des valeurs positif only
        )

        login_frequency = float(
            rng.uniform(LOGIN_RATE_MIN, LOGIN_RATE_MAX)
        )

        tx_frequency = float(
            rng.uniform(TX_RATE_MIN, TX_RATE_MAX)    #fraquenc quotidienne de transaction
        )

        profiles.append({
            "user_id": user_id,
            "home_country": home_country,
            "device_1": device_1,
            "device_2": device_2,
            "usual_hour": usual_hour,
            "usual_amount": round(usual_amount, 2),
            "login_frequency": login_frequency,
            "tx_frequency": tx_frequency
        })

    profiles_df = pd.DataFrame(profiles)

    profiles_df.to_csv(PROFILES_FILE, index=False)

    print("Profils clients créés :", len(profiles_df))
    print("Fichier sauvegardé :", PROFILES_FILE)

    return profiles_df



def add_event(
    user_id,
    timestamp,
    event_type,
    amount,
    device_id,
    country,
    success,
    is_attack=0
):

    event = {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "timestamp": pd.Timestamp(timestamp),
        "event_type": event_type,
        "amount": float(amount),
        "device_id": device_id,
        "country": country,
        "success": int(success),
        "is_attack": int(is_attack)
    }

    events.append(event)

    return event["event_id"]




def generate_normal_events(profiles_df):

    for _, profile in profiles_df.iterrows():

        user_id = profile["user_id"]
        home_country = profile["home_country"]

        device_1 = profile["device_1"]
        device_2 = profile["device_2"]

        usual_hour = int(profile["usual_hour"])
        usual_amount = float(profile["usual_amount"])

        login_frequency = float(profile["login_frequency"])
        tx_frequency = float(profile["tx_frequency"])

        for day in range(N_DAYS):

            current_day = start_date + pd.Timedelta(days=day)

            # Nombre de connexions
            n_logins = rng.poisson(login_frequency)

            # Nombre de transactions
            n_transactions = rng.poisson(tx_frequency)

            # Connexions normales
            for _ in range(n_logins):

                hour = int(np.clip(
                    rng.normal(usual_hour, 2.5), 0, 23
                ))

                minute = int(rng.integers(0, 60))

                timestamp = (
                    current_day
                    + pd.Timedelta(hours=hour, minutes=minute)
                )

                device = rng.choice([device_1, device_2])

                success = int(rng.random() > 0.04)

                add_event(
                    user_id=user_id,
                    timestamp=timestamp,
                    event_type="login",
                    amount=0,
                    device_id=device,
                    country=home_country,
                    success=success,
                    is_attack=0
                )

            # Transactions normales
            for _ in range(n_transactions):

                hour = int(np.clip(
                    rng.normal(usual_hour, 3), 0, 23
                ))

                minute = int(rng.integers(0, 60))

                timestamp = (
                    current_day
                    + pd.Timedelta(hours=hour, minutes=minute)
                )

                device = rng.choice([device_1, device_2])

                amount = float(
                    rng.lognormal(
                        mean=np.log(max(usual_amount, 1)),
                        sigma=0.35
                    )
                )

                add_event(
                    user_id=user_id,
                    timestamp=timestamp,
                    event_type="transaction",
                    amount=amount,
                    device_id=device,
                    country=home_country,
                    success=1,
                    is_attack=0
                )

    print("Génération des événements normaux terminée.")


def select_attack_targets(profiles_df):

    possible_targets = [
        (user_id, day)
        for user_id in profiles_df["user_id"]
        for day in range(N_DAYS)
    ]

    selected_indices = rng.choice(
        len(possible_targets),
        size=3 * N_ATTACKS_PER_SCENARIO,
        replace=False
    )

    selected_targets = [
        possible_targets[i] for i in selected_indices
    ]

    targets = {
        "account_takeover": selected_targets[:100],
        "small_payment_fraud": selected_targets[100:200],
        "foreign_login": selected_targets[200:300]
    }

    return targets


#scenario 1  account takeover

def inject_account_takeover(profiles_df, targets):

    profiles = profiles_df.set_index("user_id")

    for user_id, day in targets:

        profile = profiles.loc[user_id]

        current_day = start_date + pd.Timedelta(days=day)

        foreign_country = rng.choice([
            country for country in COUNTRIES
            if country != profile["home_country"]
        ])

        new_device = f"UNKNOWN_{user_id}_{day}"

        base_time = current_day + pd.Timedelta(hours=2)

        attack_id = f"ATO_{user_id}_{day}"

        # 1. Tentatives de connexion échouées
        for i in range(6):

            event_id = add_event(
                user_id=user_id,
                timestamp=base_time + pd.Timedelta(minutes=i * 3),
                event_type="login",
                amount=0,
                device_id=new_device,
                country=foreign_country,
                success=0,
                is_attack=1
            )

            attack_records.append({
                "event_id": event_id,
                "attack_id": attack_id,
                "attack_scenario": "account_takeover"
            })

        # 2. Connexion réussie
        event_id = add_event(
            user_id=user_id,
            timestamp=base_time + pd.Timedelta(minutes=20),
            event_type="login",
            amount=0,
            device_id=new_device,
            country=foreign_country,
            success=1,
            is_attack=1
        )

        attack_records.append({
            "event_id": event_id,
            "attack_id": attack_id,
            "attack_scenario": "account_takeover"
        })

        # 3. Changement du mot de passe
        event_id = add_event(
            user_id=user_id,
            timestamp=base_time + pd.Timedelta(minutes=25),
            event_type="password_change",
            amount=0,
            device_id=new_device,
            country=foreign_country,
            success=1,
            is_attack=1
        )

        attack_records.append({
            "event_id": event_id,
            "attack_id": attack_id,
            "attack_scenario": "account_takeover"
        })

        # 4. Transaction frauduleuse
        amount = float(
            profile["usual_amount"] * rng.uniform(10, 20)
        )

        event_id = add_event(
            user_id=user_id,
            timestamp=base_time + pd.Timedelta(minutes=35),
            event_type="transaction",
            amount=amount,
            device_id=new_device,
            country=foreign_country,
            success=1,
            is_attack=1
        )

        attack_records.append({
            "event_id": event_id,
            "attack_id": attack_id,
            "attack_scenario": "account_takeover"
        })

    print("Scénario Account Takeover terminé.")


    
def inject_small_payment_fraud(profiles_df, targets):

    profiles = profiles_df.set_index("user_id")

    for user_id, day in targets:

        profile = profiles.loc[user_id]

        current_day = start_date + pd.Timedelta(days=day)

        base_time = current_day + pd.Timedelta(hours=14)

        device = profile["device_1"]
        country = profile["home_country"]

        attack_id = f"SPF_{user_id}_{day}"

        n_payments = int(rng.integers(25, 41))

        for i in range(n_payments):

            timestamp = (
                base_time
                + pd.Timedelta(minutes=int(i * 60 / n_payments))
            )

            amount = float(rng.uniform(5, 40))

            event_id = add_event(
                user_id=user_id,
                timestamp=timestamp,
                event_type="transaction",
                amount=amount,
                device_id=device,
                country=country,
                success=1,
                is_attack=1
            )

            attack_records.append({
                "event_id": event_id,
                "attack_id": attack_id,
                "attack_scenario": "small_payment_fraud"
            })

    print("Scénario Small Payment Fraud terminé.")


def inject_foreign_login(profiles_df, targets):

    profiles = profiles_df.set_index("user_id")

    for user_id, day in targets:

        profile = profiles.loc[user_id]

        current_day = start_date + pd.Timedelta(days=day)

        foreign_country = rng.choice([
            country for country in COUNTRIES
            if country != profile["home_country"]
        ])

        new_device = f"FOREIGN_{user_id}_{day}"

        base_time = current_day + pd.Timedelta(hours=3)

        attack_id = f"FL_{user_id}_{day}"

        # Connexion depuis un pays inhabituel
        event_id = add_event(
            user_id=user_id,
            timestamp=base_time,
            event_type="login",
            amount=0,
            device_id=new_device,
            country=foreign_country,
            success=1,
            is_attack=1
        )

        attack_records.append({
            "event_id": event_id,
            "attack_id": attack_id,
            "attack_scenario": "foreign_login"
        })

        # Transaction importante quelques minutes après
        amount = float(
            profile["usual_amount"] * rng.uniform(6, 12)
        )

        event_id = add_event(
            user_id=user_id,
            timestamp=base_time + pd.Timedelta(minutes=5),
            event_type="transaction",
            amount=amount,
            device_id=new_device,
            country=foreign_country,
            success=1,
            is_attack=1
        )

        attack_records.append({
            "event_id": event_id,
            "attack_id": attack_id,
            "attack_scenario": "foreign_login"
        })

    print("Scénario Foreign Login terminé.")





    
def main():

    print("\n===== CHRONOBANK AI =====")
    print("Génération des données bancaires")
    print("==========================\n")

    # Création des profils
    profiles_df = generate_client_profiles()

    # Génération des activités normales
    generate_normal_events(profiles_df)

    # Sélection des cas d'attaque
    targets = select_attack_targets(profiles_df)

    # Injection des attaques
    inject_account_takeover(
        profiles_df,
        targets["account_takeover"]
    )

    inject_small_payment_fraud(
        profiles_df,
        targets["small_payment_fraud"]
    )

    inject_foreign_login(
        profiles_df,
        targets["foreign_login"]
    )

    # Construction du DataFrame des événements
    events_df = pd.DataFrame(events)

    events_df["timestamp"] = pd.to_datetime(
        events_df["timestamp"]
    )

    events_df = events_df.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    # Sauvegarde des événements
    events_df.to_csv(EVENTS_FILE, index=False)

    # Sauvegarde du détail des attaques
    attack_df = pd.DataFrame(attack_records)

    attack_df.to_csv(
        ATTACK_EVENTS_FILE,
        index=False
    )

    print("\n===== RÉSULTATS =====")
    print("Nombre de clients :", len(profiles_df))
    print("Nombre d'événements :", len(events_df))
    print("Événements normaux :", (events_df["is_attack"] == 0).sum())
    print("Événements d'attaque :", (events_df["is_attack"] == 1).sum())

    print("\nTypes d'événements :")
    print(events_df["event_type"].value_counts())

    print("\nFichiers créés :")
    print(PROFILES_FILE)
    print(EVENTS_FILE)
    print(ATTACK_EVENTS_FILE)


if __name__ == "__main__":
    main()


