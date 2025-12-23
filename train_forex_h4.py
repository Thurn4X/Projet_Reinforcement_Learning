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
RUN_NAME = "H4_Probe_200k" # On indique clairement que c'est un test

# --- PREPROCESSOR (Identique, adapté H4) ---
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

def train_probe():
    print(f"--- TRAINING PROBE H4 (200k) : {RUN_NAME} ---")
    
    wandb.init(project=PROJECT_NAME, name=RUN_NAME, config={"Timeframe": "H4", "Type": "Probe"})

    # Préparation Data
    temp_dir = "data_h4_temp"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # On prend les fichiers H4
    source_files = glob.glob("data_h4/*.pkl")
    if not source_files:
        print("❌ ERREUR : Dossier data_h4 vide ! Lance convert_to_h4.py d'abord.")
        return
    for f in source_files: shutil.copy(f, temp_dir)

    # Environnement
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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.95,
        ent_coef=0.02, # On garde une curiosité saine
        batch_size=256,
        n_steps=1024,
        tensorboard_log=f"runs/{RUN_NAME}",
        device="cpu"
    )

    print("🚀 Lancement Sonde (200k steps)...")
    # ICI : Seulement 200,000 steps
    model.learn(total_timesteps=200_000, callback=WandbFinanceCallback(verbose=1), progress_bar=False)
    
    os.makedirs("models_h4", exist_ok=True)
    # On l'appelle "probe" pour ne pas confondre
    model.save("models_h4/agent_h4_probe")
    env.save("models_h4/stats_h4_probe.pkl")
    
    wandb.finish()
    print("✅ Terminé. Analyse ce modèle avec 'analyze_trades.py' (en pointant vers agent_h4_probe).")

if __name__ == "__main__":
    train_probe()