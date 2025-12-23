import gymnasium as gym
import gym_trading_env
from gym_trading_env.renderer import Renderer
from gym_trading_env.wrapper import DiscreteActionsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
import numpy as np
import os
import glob
import shutil
import sys

# --- CONFIGURATION DES AGENTS ---
AGENTS = {
    "HARD_MODE": {
        "model": "models_h4/agent_h4_hard_mode",
        "stats": "models_h4/stats_h4_hard_mode.pkl",
    },
    "CRYPTO_SPECIALIST": {
        "model": "models_h4/agent_h4_crypto_specialist",
        "stats": "models_h4/stats_h4_crypto_specialist.pkl",
    }
}
RENDER_DIR = "render_logs"

def preprocess_smart_final(df):
    """ Conversion 1H -> 4H + Features """
    df = df.sort_index().drop_duplicates()
    df = df.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()

    mid = (df['high'] + df['low']) / 2
    lowest, highest = mid.rolling(10).min(), mid.rolling(10).max()
    raw = 2 * ((mid - lowest) / (highest - lowest + 0.00001)) - 1
    smooth = raw.ewm(alpha=0.33).mean().clip(-0.99, 0.99)
    df['feature_fisher'] = 0.5 * np.log((1 + smooth) / (1 - smooth))
    
    sma = df['close'].rolling(24).mean()
    std = df['close'].rolling(24).std()
    df['feature_elasticity'] = (df['close'] - sma) / (std + 0.00001)
    
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

def main(input_pkl):
    if os.path.exists(RENDER_DIR): shutil.rmtree(RENDER_DIR)
    os.makedirs(RENDER_DIR, exist_ok=True)

    # 1. Détermination de l'agent
    filename = os.path.basename(input_pkl).lower()
    agent_key = "CRYPTO_SPECIALIST" if "binance" in filename else "HARD_MODE"
    config = AGENTS[agent_key]
    print(f"Utilisation de l'agent : {agent_key}")

    # 2. Préparation des données
    df_raw = pd.read_pickle(input_pkl)
    df_final = preprocess_smart_final(df_raw)

    # 3. Environnement
    raw_env = gym.make(
        "TradingEnv",
        df=df_final,
        portfolio_initial_value=1000,
        trading_fees=0.001,
        borrow_interest_rate=0.02/100/24,
        verbose=0
    )
    raw_env = DiscreteActionsWrapper(raw_env, positions=[-1, 0, 1])

    # 4. Chargement IA & Stats
    dummy_env = DummyVecEnv([lambda: raw_env])
    norm_env = VecNormalize.load(config['stats'], dummy_env)
    norm_env.training = False
    norm_env.norm_reward = False
    model = RecurrentPPO.load(config['model'], env=norm_env)

    # 5. Boucle de prédiction
    print(f"Simulation sur {filename}...")
    obs, _ = raw_env.reset()
    done = False
    lstm_states = None
    starts = np.ones((1,), dtype=bool)

    while not done:
        norm_obs = norm_env.normalize_obs(obs)
        action, lstm_states = model.predict(norm_obs, state=lstm_states, episode_start=starts, deterministic=True)
        
        action_val = action.item()
        # Sécurité Long Only pour Crypto
        if agent_key == "CRYPTO_SPECIALIST" and action_val == 0:
            action_val = 1 # Force Flat au lieu de Short

        obs, reward, terminated, truncated, info = raw_env.step(action_val)
        done = terminated or truncated
        starts = np.array([done])

    # 6. Rendu final
    raw_env.unwrapped.save_for_render(dir=RENDER_DIR)
    renderer = Renderer(render_logs_dir=RENDER_DIR)
    
    # Ajout d'indicateurs sur le graphique
    renderer.add_line(name="RSI (Signal)", function=lambda df: df["feature_rsi"])
    renderer.add_line(name="Elasticity", function=lambda df: df["feature_elasticity"])
    
    print("Serveur prêt")
    renderer.run()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rendu_visuel_hybride.py data/binance-DOGEEUR-1h.pkl par exemple")
    else:
        main(sys.argv[1])