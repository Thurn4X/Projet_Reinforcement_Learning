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
RUN_NAME = "H4_Binance_Crypto_LongOnly" 
OLD_MODEL_PATH = "models_h4/agent_h4_probe" 
OLD_STATS_PATH = "models_h4/stats_h4_probe.pkl"

# --- WRAPPER POUR CENSURER LE SHORT ---
class LongOnlyWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
    def action(self, action):
        # Positions via DiscreteActionsWrapper([-1, 0, 1]) sont mappées :
        # 0 -> -1 (Short), 1 -> 0 (Flat), 2 -> 1 (Long)
        # On force l'action 0 (Short) à devenir 1 (Flat/Cash)
        if action == 0:
            return 1 
        return action

# --- 1. FONCTION DE CONVERSION 1H -> 4H ---
def resample_to_h4(input_path, output_dir):
    df = pd.read_pickle(input_path)
    df = df.sort_index().drop_duplicates()
    df_h4 = df.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    df_h4.to_pickle(output_path)
    return output_path

# --- 2. PREPROCESSOR SMART ---
def preprocess_smart(df):
    df = df.sort_index().drop_duplicates()
    mid = (df['high'] + df['low']) / 2
    lowest = mid.rolling(10).min()
    highest = mid.rolling(10).max()
    raw = 2 * ((mid - lowest) / (highest - lowest + 1e-5)) - 1
    smooth = raw.ewm(alpha=0.33).mean().clip(-0.99, 0.99)
    df['feature_fisher'] = 0.5 * np.log((1 + smooth) / (1 - smooth))
    
    sma = df['close'].rolling(24).mean()
    std = df['close'].rolling(24).std()
    df['feature_elasticity'] = (df['close'] - sma) / (std + 1e-5)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['feature_rsi'] = ( (100 - (100 / (1 + rs))) - 50 ) / 20.0 
    
    sma200 = df['close'].rolling(200).mean()
    df['feature_trend'] = (df['close'] - sma200) / (sma200 + 1e-9) * 10
    
    df['feature_hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['feature_hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['feature_log_ret'] = np.log(df['close'] / df['close'].shift(1)) * 100
    return df.dropna()

class WandbFinanceCallback(BaseCallback):
    def __init__(self, verbose=0): super().__init__(verbose)
    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            info = self.locals["infos"][0]
            wandb.log({"Finance/Portfolio_Value": info.get("portfolio_valuation", 1000)})
        return True

def train_binance_specialist():
    # --- ÉTAPE A : PRÉPARATION DES DONNÉES ---
    source_dir = "data"
    target_dir = "data_h4_binance_only"
    if os.path.exists(target_dir): shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    binance_files = [f for f in glob.glob(f"{source_dir}/*.pkl") if os.path.basename(f).lower().startswith("binance")]
    if not binance_files: print("❌ Aucun fichier Binance trouvé !"); return

    for f in binance_files:
        resample_to_h4(f, target_dir)

    # --- ÉTAPE B : ENTRAÎNEMENT ---
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir=f"{target_dir}/*.pkl",
        preprocess=preprocess_smart,
        portfolio_initial_value=1000,
        trading_fees=0.001,
        borrow_interest_rate=0, # Pas d'intérêt car pas de levier/short
    )
    # 1. On garde l'Action Space à 3 pour la compatibilité du chargement
    env = DiscreteActionsWrapper(env, positions=[-1, 0, 1])
    # 2. On applique le filtre Long-Only
    env = LongOnlyWrapper(env)
    env = DummyVecEnv([lambda: env])
    
    # Chargement Stats & Modèle
    env = VecNormalize.load(OLD_STATS_PATH, env)
    env.training = True 
    model = RecurrentPPO.load(OLD_MODEL_PATH, env=env)

    # Tuning agressif pour capter les hausses Crypto
    model.learning_rate = 0.0002
    model.ent_coef = 0.05 # On augmente l'entropie pour forcer l'exploration Long
    
    print(f"🚀 Fine-tuning Spécialiste Crypto (Long-Only) sur {len(binance_files)} fichiers...")
    model.learn(total_timesteps=300_000, callback=WandbFinanceCallback(), reset_num_timesteps=False)
    
    model.save("models_h4/agent_h4_crypto_specialist")
    env.save("models_h4/stats_h4_crypto_specialist.pkl")
    wandb.finish()
    print("✅ Terminé !")

if __name__ == "__main__":
    train_binance_specialist()