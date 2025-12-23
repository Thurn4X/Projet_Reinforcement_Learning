import gymnasium as gym
import gym_trading_env
from gym_trading_env.wrapper import DiscreteActionsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import os
import shutil
import wandb
import glob
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
PROJECT_NAME = "Projet_Reinforcement_Learning"
RUN_NAME = "H4_Hard_Mode_NoCrypto" 

# ON REPART DU MODELE EXISTANT (Mets ici le chemin de ton meilleur modèle actuel)
# Si tu as fait le 'continue', mets 'agent_h4_v2'. Sinon 'agent_h4_probe'
OLD_MODEL_PATH = "models_h4/agent_h4_probe" 
OLD_STATS_PATH = "models_h4/stats_h4_probe.pkl"

# LISTE DES MOTS INTERDITS (Pour exclure les cryptos)
EXCLUDE_KEYWORDS = ["DOGE", "BTC", "ETH", "SOL", "BNB"]

# --- PREPROCESSOR (Toujours le même) ---
def preprocess_smart(df):
    df = df.sort_index().drop_duplicates()
    mid = (df['high'] + df['low']) / 2
    lowest = mid.rolling(10).min()
    highest = mid.rolling(10).max()
    raw = 2 * ((mid - lowest) / (highest - lowest + 0.00001)) - 1
    smooth = raw.ewm(alpha=0.33).mean().clip(-0.99, 0.99)
    df['feature_fisher'] = 0.5 * np.log((1 + smooth) / (1 - smooth))
    window = 24 
    sma = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    df['feature_elasticity'] = (df['close'] - sma) / (std + 0.00001)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['feature_rsi'] = (rsi - 50) / 20.0 
    sma200 = df['close'].rolling(200).mean()
    df['feature_trend'] = (df['close'] - sma200) / sma200 * 10
    df['feature_hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['feature_hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['feature_log_ret'] = np.log(df['close'] / df['close'].shift(1)) * 100
    return df.dropna()

class WandbFinanceCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_count += 1
            info = self.locals["infos"][0]
            val = info.get("portfolio_valuation", 1000)
            wandb.log({
                "global_step": self.num_timesteps,
                "Finance/Portfolio_Value": val,
                "Episodes": self.episode_count
            })
        return True

def train_hard_mode():
    print(f"--- TRAINING HARD MODE (No Crypto) : {RUN_NAME} ---")
    wandb.init(project=PROJECT_NAME, name=RUN_NAME, config={
        "Type": "Fine-Tuning",
        "Data": "Low Volatility Only",
        "Goal": "Force Precision"
    })

    # 1. TRI SÉLECTIF DES DONNÉES
    temp_dir = "data_h4_hard"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    all_files = glob.glob("data_h4/*.pkl")
    kept_files = []
    
    print("🧐 Filtrage des données...")
    for f in all_files:
        filename = os.path.basename(f).upper()
        # Si le fichier contient un mot interdit (DOGE, BTC...), on le saute
        if any(bad_word in filename for bad_word in EXCLUDE_KEYWORDS):
            print(f"   🚫 Exclusion : {filename}")
        else:
            shutil.copy(f, temp_dir)
            kept_files.append(filename)
            print(f"   ✅ Ajout : {filename}")

    if not kept_files:
        print("❌ Erreur : Aucun fichier retenu. Vérifie tes noms de fichiers.")
        return

    # 2. ENVIRONNEMENT
    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir=f"{temp_dir}/*.pkl",
        preprocess=preprocess_smart,
        portfolio_initial_value=1000,
        trading_fees=0.001, # 0.1% Réel
        borrow_interest_rate=0.0002/6,
    )
    env = DiscreteActionsWrapper(env, positions=[-1, 0, 1])
    env = DummyVecEnv([lambda: env])
    
    # 3. CHARGEMENT
    try:
        env = VecNormalize.load(OLD_STATS_PATH, env)
        env.training = True 
        model = RecurrentPPO.load(OLD_MODEL_PATH, env=env)
    except:
        print("❌ Erreur chargement modèle/stats. Vérifie les chemins.")
        return

    # 4. RÉGLAGES DE PRÉCISION
    model.learning_rate = 0.0001 # Plus lent pour affiner
    model.ent_coef = 0.01        # Moins de chaos, plus de concentration
    
    print("🚀 Lancement Fine-Tuning Hard Mode (300k steps)...")
    
    model.learn(
        total_timesteps=300_000, 
        callback=WandbFinanceCallback(verbose=1), 
        progress_bar=False, 
        reset_num_timesteps=False
    )
    
    model.save("models_h4/agent_h4_hard_mode")
    env.save("models_h4/stats_h4_hard_mode.pkl")
    
    wandb.finish()
    print("✅ Terminé. Lance l'analyse sur GOLD et S&P500 !")

if __name__ == "__main__":
    train_hard_mode()