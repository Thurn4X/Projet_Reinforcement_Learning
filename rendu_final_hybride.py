import gymnasium as gym
import gym_trading_env
from gym_trading_env.wrapper import DiscreteActionsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION DES AGENTS ---
AGENTS = {
    "HARD_MODE": {
        "model": "models_h4/agent_h4_hard_mode",
        "stats": "models_h4/stats_h4_hard_mode.pkl",
        "desc": "Expert Finance Traditionnelle (Shorts autorisés)"
    },
    "CRYPTO_SPECIALIST": {
        "model": "models_h4/agent_h4_crypto_specialist",
        "stats": "models_h4/stats_h4_crypto_specialist.pkl",
        "desc": "Spécialiste Binance (LONG ONLY forcé)"
    }
}

def preprocess_smart_final(df):
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

def run_prediction(input_pkl):
    if not os.path.exists(input_pkl):
        print(f"❌ Erreur : Fichier {input_pkl} introuvable."); return

    filename = os.path.basename(input_pkl).lower()
    agent_key = "CRYPTO_SPECIALIST" if "binance" in filename else "HARD_MODE"
    config = AGENTS[agent_key]

    print(f"🤖 AGENT : {agent_key} | {config['desc']}")

    df_raw = pd.read_pickle(input_pkl)
    df_final = preprocess_smart_final(df_raw)

    env = gym.make("TradingEnv", df=df_final, portfolio_initial_value=1000, trading_fees=0.001, verbose=0)
    env = DiscreteActionsWrapper(env, positions=[-1, 0, 1])
    env = DummyVecEnv([lambda: env])
    
    norm_env = VecNormalize.load(config['stats'], env)
    norm_env.training = False
    norm_env.norm_reward = False
    model = RecurrentPPO.load(config['model'], env=norm_env)

    obs = env.reset()
    done, lstm_states = False, None
    starts = np.ones((1,), dtype=bool)
    history = []
    raw_env = env.envs[0].unwrapped
    
    while not done:
        norm_obs = norm_env.normalize_obs(obs)
        action, lstm_states = model.predict(norm_obs, state=lstm_states, episode_start=starts, deterministic=True)
        
        action_to_apply = action[0]
        
        # --- LOGIQUE FORCÉE LONG ONLY POUR LA CRYPTO ---
        if agent_key == "CRYPTO_SPECIALIST":
            if action_to_apply == 0: # Si l'IA veut Short (-1)
                action_to_apply = 1  # On force le Flat (0)
        
        idx = raw_env._idx
        if idx >= len(raw_env.df): break
        
        date, price = raw_env.df.index[idx], raw_env.df.iloc[idx]['close']
        obs, rewards, dones, infos = env.step([action_to_apply])
        
        history.append({
            "Date": date,
            "Prix": round(price, 4),
            "Agent": agent_key,
            "Position": [-1, 0, 1][action_to_apply],
            "Valeur_Portefeuille": round(infos[0]['portfolio_valuation'], 2)
        })
        done, starts = dones[0], np.array([dones[0]])

    output_name = f"RAPPORT_FINAL_{agent_key}_{os.path.basename(input_pkl).replace('.pkl', '.csv')}"
    pd.DataFrame(history).to_csv(output_name, index=False)
    print(f"✅ Terminé ! Valeur finale : {history[-1]['Valeur_Portefeuille']}€")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rendu_final.py data/binance-BTCUSD-1h.pkl")
    else:
        run_prediction(sys.argv[1])