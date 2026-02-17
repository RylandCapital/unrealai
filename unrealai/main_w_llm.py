import os
import io
import random
import time
import gc
import re
import csv
import json
import base64
import multiprocessing
import threading
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

import gym
from gym import spaces

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

import psutil  # RAM logging

from pathlib import Path
import importlib.util

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI (UPDATED): use the modern SDK + gpt-5-mini + reasoning high
from openai import OpenAI

# Use your existing env var name if you want.
# Recommended is OPENAI_API_KEY, but we'll keep SECRET to match your setup.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("SECRET")

def _get_openai_client():
    # Create client lazily per process (important for multiprocessing spawn)
    if not hasattr(_get_openai_client, "_client"):
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing API key. Set OPENAI_API_KEY (recommended) or SECRET.")
        _get_openai_client._client = OpenAI(api_key=OPENAI_API_KEY)
    return _get_openai_client._client

def overwatch_decision_gpt5mini(raw_action: int, img_b64_str: str) -> str:
    """
    Calls GPT-5-mini with reasoning=high and the chart image as an image input.
    Returns model output text.
    """
    client = _get_openai_client()

    instructions = (
        "You are lead equity analyst on the tradedesk. You are overseeing a reinforcement trading agent "
        "that decides whether to do nothing/hold(0), long(1), short(2), close(3), or stay flat each day. "
        "Each day you get a chart with a 10Y monthly chart, 10yr weekly chart, and a 3yr daily chart. "
        "Each chart has 3 anchored VWAPS: most recent high (blue), most recent low (pink), highest volume day (yellow). "
        "You are given the current price and YTD return in the daily chart title. The chart also informs you if we are "
        "currently long/short/flat as well as the current trade's PnL. "
        "Your job is to use this chart combined with the recommendation from the AI trading agent (0,1,2,3) to choose "
        "whether to accept the current trading_action or return a new one. "
        "Reply ONLY with: <number choice> - <two-sentence reason>."
    )

    # Send image as a proper image input (not inside JSON)
    image_data_url = f"data:image/png;base64,{img_b64_str}"

    payload = {"trading_action": int(raw_action)}

    resp = client.responses.create(
        model="gpt-5-mini",
        reasoning={"effort": "high"},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instructions},
                    {"type": "input_text", "text": json.dumps(payload)},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    )

    return resp.output_text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Load your charting module
FILE = Path(r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\report_vwapv1.py")
spec   = importlib.util.spec_from_file_location("report_vwapv1", FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# ─────────────────────────────────────────────────────────────────────────────
# Action constants
ACTION_HOLD  = 0
ACTION_LONG  = 1
ACTION_SHORT = 2
ACTION_CLOSE = 3

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def load_csv(csv_path):
    """Load only the feature columns required by the model."""
    cols = [
        'Date',
        'above_21_sma', 'above_55_sma', 'above_200_sma', '21day_rsi',
        'above_21_sma_spy', 'above_55_sma_spy', 'above_200_sma_spy', '21day_rsi_spy',
        "#SPXADR_close", "#NDXADR_close", "#SPXMCOSC_close", "#NDXMCOSC_close",
        "#NDXZWBT_close", "#SPXZWBT_close", "#OEX%MA50_close", "#OEX%MA200_close",
        "#M2FED3_close", "#M2FED2_close", '$VIX_close', 'adjusted_close'
    ]
    return pd.read_csv(csv_path, usecols=cols, parse_dates=['Date'], dtype={c: 'float32' for c in cols})

def logging_listener(q):
    while True:
        msg = q.get()
        if msg == "END":
            break
        print(msg, flush=True)

def child_log(q, msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    q.put(f"[{ts}] {msg}")

def get_ram_usage():
    return psutil.virtual_memory().percent

# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment
class TradingEnv(gym.Env):
    """Custom trading environment"""

    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 df,
                 window_size=20,
                 initial_cash=100_000.0,
                 slippage_rate=0.0002,
                 stop_loss=0.05,
                 take_profit=1_000,   # intentionally huge to disable TP auto-close
                 cooldown_days=2,
                 episode_length=252):
        super().__init__()

        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        self.full_df = df.reset_index(drop=True)
        self.window_size    = window_size
        self.initial_cash   = float(initial_cash)
        self.slippage_rate  = slippage_rate
        self.stop_loss      = stop_loss
        self.take_profit    = take_profit
        self.cooldown_days  = cooldown_days
        self.episode_length = episode_length

        self.feature_cols = [
            'above_21_sma', 'above_55_sma', 'above_200_sma', '21day_rsi',
            'above_21_sma_spy', 'above_55_sma_spy', 'above_200_sma_spy', '21day_rsi_spy',
            "#SPXADR_close", "#NDXADR_close", "#SPXMCOSC_close", "#NDXMCOSC_close",
            "#NDXZWBT_close", "#SPXZWBT_close", "#OEX%MA50_close", "#OEX%MA200_close",
            "#M2FED3_close", "#M2FED2_close", '$VIX_close'
        ]

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.window_size, len(self.feature_cols)),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self._reset_episode_slice()
        self.reset()

    # ---------------------------------------------------------------------
    # Episode helpers
    def _reset_episode_slice(self):
        if self.episode_length is None:
            self.df = self.full_df  # view only
        else:
            max_start = len(self.full_df) - self.episode_length
            if max_start <= 0:
                raise ValueError("Dataframe too short for desired episode length.")
            start = random.randint(0, max_start)
            end   = start + self.episode_length
            self.df = self.full_df.iloc[start:end]

    def reset(self):
        self._reset_episode_slice()
        self.current_step       = self.window_size
        self.position           = 0   # 0 = flat, 1 = long, -1 = short
        self.shares             = 0.0
        self.entry_price        = 0.0
        self.entry_equity       = None
        self.position_open_step = None
        self.cash               = self.initial_cash
        self.equity             = self.initial_cash
        self.cooldown_remaining = 0
        self.trades             = []
        self.close_hits         = 0   # count of explicit agent closes in this episode
        self.done               = False
        return self._get_obs()

    @property
    def current_datetime(self):
        """Returns the datetime of the current step from the 'Date' column."""
        idx = min(max(self.current_step, 0), len(self.df) - 1)
        return self.df['Date'].iat[idx]

    # ---------------------------------------------------------------------
    # Internal helpers
    def _get_obs(self):
        start = self.current_step - self.window_size
        return self.df[self.feature_cols].iloc[start:self.current_step].values.astype(np.float32)

    def _apply_slippage(self, value):
        return value * (1 - self.slippage_rate)

    def _do_close(self, price):
        """Close current position, return percentage profit."""
        gross_value = self.shares * price
        slip_cost   = gross_value * self.slippage_rate

        if self.position == 1:  # long → sell
            proceeds    = gross_value - slip_cost
            pnl_dollars = proceeds - (self.shares * self.entry_price)
            self.cash  += proceeds
            trade_label = 'close_long'
        else:  # short → buy back
            buy_back_cost = gross_value + slip_cost
            pnl_dollars   = (self.shares * self.entry_price) - buy_back_cost
            self.cash    -= buy_back_cost
            trade_label   = 'close_short'

        pct = pnl_dollars / (self.shares * self.entry_price) if self.entry_price else 0.0
        ttm = (self.current_step - self.position_open_step) or 1
        self.trades.append((trade_label, self.current_step, price, pct, ttm))

        # reset position
        self.position           = 0
        self.shares             = 0.0
        self.entry_price        = 0.0
        self.position_open_step = None
        self.equity             = self.cash
        self.entry_equity       = None
        return pct

    # ---------------------------------------------------------------------
    # Core step method
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode already done")

        # Forced close at end of data
        if self.current_step >= len(self.df) - 1:
            if self.position != 0:
                self._do_close(self.df['adjusted_close'].iat[self.current_step])
            self.done = True
            return self._get_obs(), 0.0, True, {}

        price        = self.df['adjusted_close'].iat[self.current_step]
        reward       = 0.0

        # 1) Handle agent-initiated CLOSE
        if self.position != 0 and action == ACTION_CLOSE:
            self.close_hits += 1
            reward          += self._do_close(price)
            self.cooldown_remaining = self.cooldown_days

        # 2) Auto stop-loss / take-profit
        if self.position != 0:
            pct_move = ((price - self.entry_price) / self.entry_price
                        if self.position == 1 else
                        (self.entry_price - price) / self.entry_price)
            if pct_move <= -self.stop_loss or pct_move >= self.take_profit:
                reward += self._do_close(price)
                self.cooldown_remaining = self.cooldown_days

        # 3) Opening logic (only if flat and not in cooldown)
        if self.position == 0 and self.cooldown_remaining == 0:
            if action in (ACTION_LONG, ACTION_SHORT):
                trade_value = self.cash            # all-in sizing
                slip_cost   = trade_value * self.slippage_rate
                shares      = (trade_value - slip_cost) / price if price else 0.0

                if action == ACTION_LONG:
                    self.cash  -= trade_value
                    self.position = 1
                    self.trades.append(('open_long', self.current_step, price, None, None))
                else:  # SHORT
                    self.cash  += trade_value
                    self.position = -1
                    self.trades.append(('open_short', self.current_step, price, None, None))

                self.shares             = shares
                self.entry_price        = price
                self.position_open_step = self.current_step
                self.entry_equity = self.cash + (self.shares * price if self.position == 1
                                     else -self.shares * price)

        elif self.position == 0 and self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # 4) Update equity mark-to-market
        if   self.position == 1:
            self.equity = self.cash + self.shares * price
        elif self.position == -1:
            self.equity = self.cash + self.shares * price
        else:
            self.equity = self.cash

        if self.position != 0:
            # Unrealised P&L in %, relative to entry equity
            step_reward = (self.equity - self.entry_equity) / self.entry_equity
        else:
            step_reward = 0.0

        reward += step_reward

        # Advance
        self.current_step += 1
        if self.current_step >= len(self.df):
            self.done = True

        return self._get_obs(), reward, self.done, {}

    # ---------------------------------------------------------------------
    def render(self, mode='human'):
        print(f"Step={self.current_step}, Pos={self.position}, Shares={self.shares:.2f}, "
              f"Cash={self.cash:.2f}, Eq={self.equity:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DQN Agent (unchanged architecture; only variable names)
class DQNAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.999,
                 epsilon_min=0.05, batch_size=32,
                 memory_size=50_000):
        self.state_size  = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.batch_size    = batch_size
        self.memory        = deque(maxlen=memory_size)
        self.train_step_counter = 0
        self.target_update_freq = 50
        self.overwatch_counter = 0
        self.override_count   = 0

        self.overwatch_enabled = True

        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        m = Sequential()
        m.add(Flatten(input_shape=self.state_size))
        m.add(Dense(64, activation='relu'))
        m.add(Dense(32, activation='relu'))
        m.add(Dense(self.action_size, activation='linear'))
        m.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                  loss='mse')
        return m

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, d))

    def act(self, state, symbol=None, env=None, use_overwatch=True):
        self.overwatch_counter += 1

        allow_overwatch = (
            use_overwatch
            and self.overwatch_enabled
            and symbol is not None
        )

        # 1) raw DQN action
        if np.random.rand() <= self.epsilon:
            raw_action = random.randrange(self.action_size)
        else:
            q_vals     = self.model.predict(state[np.newaxis, :], verbose=0)[0]
            raw_action = int(np.argmax(q_vals))

        # 2) optionally call your manager
        if allow_overwatch and self.overwatch_counter % 10 == 0:
            fig = module.build_fig(symbol, end=env.current_datetime.strftime("%Y-%m-%d"))

            pos_map = {0: "Flat", 1: "Long", -1: "Short"}
            pos_str = pos_map.get(env.position, "Unknown")
            if env.position != 0 and env.entry_equity:
                pnl_pct = (env.equity - env.entry_equity) / env.entry_equity * 100
                days    = env.current_step - env.position_open_step + 1
            else:
                pnl_pct = 0.0
                days    = 0

            textstr = (
                f"Position: {pos_str}\n"
                f"PnL: {pnl_pct:.2f}%\n"
                f"Days: {days}"
            )

            fig.text(
                0.01, 0.99, textstr,
                va='top', ha='left',
                color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.6)
            )

            img_bytes  = fig_to_bytes(fig)
            img_b64_str = base64.b64encode(img_bytes).decode("utf-8")

            sdir = r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\reports"
            fn = 'test{0}.png'.format(self.overwatch_counter)
            full_path = os.path.join(sdir, fn)
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # 3) call GPT-5-mini (UPDATED)
            try:
                resp_txt = overwatch_decision_gpt5mini(raw_action, img_b64_str)
            except Exception as e:
                # If OpenAI call fails, fall back to raw_action
                # (You can add logging here if you want.)
                return raw_action

            m = re.search(r'\b([0-3])\b', resp_txt)
            if m:
                override = int(m.group(1))
                if override != raw_action:
                    self.override_count += 1
                return override

        return raw_action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        S  = np.array([m[0] for m in minibatch])
        A  = np.array([m[1] for m in minibatch])
        R  = np.array([m[2] for m in minibatch])
        S2 = np.array([m[3] for m in minibatch])
        D  = np.array([m[4] for m in minibatch])

        target      = self.model.predict(S,  verbose=0)
        target_next = self.target_model.predict(S2, verbose=0)

        for i in range(self.batch_size):
            if D[i]:
                target[i][A[i]] = R[i]
            else:
                target[i][A[i]] = R[i] + self.gamma * np.max(target_next[i])

        self.model.fit(S, target, epochs=1, verbose=0)

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Stats helpers

def summarize_trades(trades, title="Trades Summary", log_queue=None):
    close_trades = [t for t in trades if 'close' in t[0]]
    pnls         = [pct for *_, pct, _ in close_trades if pct is not None]
    wins   = sum(p > 0 for p in pnls)
    losses = sum(p < 0 for p in pnls)
    total_pct = np.nansum(pnls)
    avg_pct   = np.nanmean(pnls) if pnls else np.nan

    def _log(msg):
        child_log(log_queue, msg) if log_queue else print(msg)

    _log("\n" + "=" * 40)
    _log(title)
    _log("=" * 40)
    _log(f" Total Realized PnL (%): {total_pct * 100:.2f}%")
    _log(f" Avg   Realized PnL (%): {avg_pct * 100:.2f}%")
    _log("=" * 40 + "\n")

    return {'total_pnl_pct': total_pct, 'avg_pnl_pct': avg_pct}

def compute_portfolio_stats(equity_curve, invested_curve, trades,
                            initial_value, buy_hold_curve=None):
    if not equity_curve:
        return {}
    final_v   = equity_curve[-1]
    tot_ret   = (final_v - initial_value) / initial_value
    ann_ret   = ((final_v / initial_value) ** (252 / len(equity_curve)) - 1
                 if len(equity_curve) > 1 else np.nan)
    run_max   = np.maximum.accumulate(equity_curve)
    dd        = (run_max - equity_curve) / run_max
    max_dd    = np.max(dd)

    trade_profits = [t[3] for t in trades if t[3] is not None]
    wins  = sum(p > 0 for p in trade_profits)
    total = len(trade_profits)
    win_pct = wins / total * 100 if total else 0

    longs  = [t for t in trades if 'close_long'  == t[0]]
    shorts = [t for t in trades if 'close_short' == t[0]]
    win_long  = sum(t[3] > 0 for t in longs)
    win_short = sum(t[3] > 0 for t in shorts)

    if len(equity_curve) > 1:
        dr = np.diff(equity_curve) / equity_curve[:-1]
        sr = np.mean(dr) / (np.std(dr) + 1e-9) * np.sqrt(252)
    else:
        sr = np.nan

    stats = {
        'final_value': final_v,
        'total_return_pct':      tot_ret  * 100,
        'annualized_return_pct': ann_ret  * 100,
        'max_drawdown_pct':      max_dd   * 100,
        'sharpe_ratio':          sr,
        'total_trades':          total,
        'win_pct_overall':       win_pct,
        'num_long_trades':       len(longs),
        'num_short_trades':      len(shorts),
        'win_pct_long':  win_long  / len(longs)  * 100 if longs  else 0,
        'win_pct_short': win_short / len(shorts) * 100 if shorts else 0,
    }

    if buy_hold_curve and len(buy_hold_curve) > 1:
        fbv = buy_hold_curve[-1]
        bhr = (fbv - initial_value) / initial_value
        stats.update({
            'buy_hold_total_return_pct': bhr * 100,
            'alpha_over_buy_hold':       (tot_ret - bhr) * 100,
        })

    return stats

def plot_aggregated_results(agent_eq, bh_eq, invested, stats, save_path=None):
    days = list(range(len(agent_eq)))
    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("Equity Curve", "Daily Invested"),
                         vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=days, y=agent_eq, mode='lines', name='Agent'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=bh_eq, mode='lines', name='BuyHold'),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=days, y=invested, name='Invested'),
                  row=2, col=1)
    fig.update_layout(title="Aggregated Performance", hovermode='x unified')
    if save_path:
        fig.write_html(save_path)
    fig.show()

    print("\n--- Aggregated Stats ---")
    for k, v in stats.items():
        suffix = '%' if 'pct' in k else ''
        print(f"{k}: {v:.2f}{suffix}")
    print("------------------------\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Training & Testing helpers

def train_agent_on_df(agent, df, *, episodes=10, window_size=20,
                      initial_cash=100_000, episode_length=252,
                      report_csv_path=None, log_queue=None, symbol=""):
    reward_hist = []
    all_trades  = []

    csv_file   = open(report_csv_path, 'w', newline='') if report_csv_path else None
    csv_writer = None
    if csv_file:
        csv_writer = csv.DictWriter(csv_file,
                                    fieldnames=["episode", "step", "epsilon",
                                                "action", "equity", "reward"])
        csv_writer.writeheader()

    env = TradingEnv(df, window_size=window_size,
                     initial_cash=initial_cash,
                     slippage_rate=0.0002,
                     stop_loss=0.05,
                     take_profit=1_000,
                     cooldown_days=2,
                     episode_length=episode_length)

    for e in range(episodes):
        agent.override_count = 0
        state = env.reset()
        tot_r = 0.0
        while True:
            action               = agent.act(state, env=env, symbol=symbol)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            if csv_writer:
                csv_writer.writerow({
                    "episode": e,
                    "step":    env.current_step,
                    "epsilon": agent.epsilon,
                    "action":  action,
                    "equity":  env.equity,
                    "reward":  reward,
                })
                csv_file.flush()

            state = next_state
            tot_r += reward
            if done:
                break

        reward_hist.append(tot_r)
        all_trades.extend(env.trades)

        child_log(log_queue, f"[{symbol}] Episode {e + 1}/{episodes} "
                     f"epsilon={agent.epsilon:.4f} "
                     f"reward={tot_r:.4f} "
                     f"closes={env.close_hits} "
                     f"overrides={agent.override_count}")

    if csv_file:
        csv_file.close()

    summarize_trades(all_trades, title=f"{symbol} Train Trades",
                     log_queue=log_queue)
    gc.collect()
    return reward_hist, all_trades

def test_agent_on_df(agent, df, *, window_size=20, initial_cash=100_000,
                     log_queue=None, symbol=""):
    env = TradingEnv(df, window_size=window_size,
                     initial_cash=initial_cash,
                     episode_length=None)
    agent.override_count = 0
    state  = env.reset()
    steps, portfolio, invested = [], [], []
    while True:
        action               = agent.act(state, env=env, symbol=symbol)
        next_state, _, done, _ = env.step(action)
        state = next_state

        steps.append(env.current_step)
        portfolio.append(env.equity)
        invested.append(initial_cash if env.position != 0 else 0)
        if done:
            break

    summarize_trades(env.trades, title=f"{symbol} Test Trades",
                     log_queue=log_queue)

    start_price = df['adjusted_close'].iat[window_size]
    bh_shares   = initial_cash / start_price if start_price else 0
    bh_curve    = [bh_shares * df['adjusted_close'].iat[i] for i in steps]
    override_ct  = agent.override_count
    override_pct = override_ct / len(steps) * 100 if steps else 0
    return {
        'steps':            steps,
        'portfolio_values': portfolio,
        'invested_history': invested,
        'buy_hold_values':  bh_curve,
        'trades':           env.trades,
        'override_count':   override_ct,
        'override_pct':     override_pct
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. Worker function for multiprocessing

def process_stock(args):
    train_csv, test_csv, agent_params, win, ep, ini, ep_len, out_dir, log_q = args
    sym = os.path.splitext(os.path.basename(train_csv))[0]
    child_log(log_q, f"=== {sym} start ===")

    agent = DQNAgent(**agent_params)

    # Train
    agent.overwatch_enabled = False
    df_tr   = load_csv(train_csv)
    rewards, _ = train_agent_on_df(agent, df_tr, episodes=ep, window_size=win,
                                   initial_cash=ini, episode_length=ep_len,
                                   report_csv_path=os.path.join(out_dir,
                                                                 f"{sym}_train.csv"),
                                   log_queue=log_q, symbol=sym)

    plt.figure(figsize=(8, 3))
    plt.plot(rewards)
    plt.title(f"{sym} Reward")
    plt.savefig(os.path.join(out_dir, f"{sym}_reward.png"))
    plt.close()

    # Test
    agent.overwatch_enabled = True
    df_te = load_csv(test_csv)
    agent.epsilon = 0.0  # greedy
    res = test_agent_on_df(agent, df_te, window_size=win,
                           initial_cash=ini, log_queue=log_q, symbol=sym)

    fig = make_subplots()
    fig.add_trace(go.Scatter(x=res['steps'], y=res['portfolio_values'],
                             mode='lines', name='Agent'))
    fig.add_trace(go.Scatter(x=res['steps'], y=res['buy_hold_values'],
                             mode='lines', name='BuyHold'))
    fig.update_layout(title=f"{sym} Test", xaxis_title="Step")
    fig.write_html(os.path.join(out_dir, f"{sym}_test.html"))

    pd.DataFrame(res['trades'],
                 columns=['type', 'step', 'price', 'pnl_pct', 'time']).to_csv(
        os.path.join(out_dir, f"{sym}_trades.csv"), index=False)

    stats = compute_portfolio_stats(res['portfolio_values'],
                                    res['invested_history'], res['trades'],
                                    ini, res['buy_hold_values'])
    stats['override_count'] = res['override_count']
    stats['override_pct']   = res['override_pct']
    pd.DataFrame([stats]).to_csv(os.path.join(out_dir, f"{sym}_stats.csv"),
                                 index=False)

    child_log(log_q, f"=== {sym} done ===")

    tf.keras.backend.clear_session()
    gc.collect()
    return res

# ─────────────────────────────────────────────────────────────────────────────
# 6. Orchestrator with aggregated curve

def main_loop(agent_params, train_files, test_files, output_dir, *,
              window_size=20, episodes=10, initial_cash=100_000,
              episode_length=252, log_queue=None):
    os.makedirs(output_dir, exist_ok=True)

    args = [(tr, te, agent_params, window_size, episodes, initial_cash,
             episode_length, output_dir, log_queue)
            for tr, te in zip(train_files, test_files)]

    child_log(log_queue, "Starting multiprocessing pool")
    with multiprocessing.Pool(min(len(args), 2), maxtasksperchild=1) as pool:
        results = pool.map(process_stock, args)
    child_log(log_queue, "Pool finished")

    if not results:
        return

    min_len   = min(len(r['portfolio_values']) for r in results)
    norm_curves, bh_curves, inv_curves = [], [], []
    for r in results:
        norm_curves.append(np.array(r['portfolio_values'][:min_len]) / initial_cash)
        bh_curves.append(np.array(r['buy_hold_values'][:min_len])  / initial_cash)
        inv_curves.append(np.array(r['invested_history'][:min_len]) / initial_cash)

    agg_equity = np.mean(norm_curves, axis=0) * initial_cash
    agg_bh     = np.mean(bh_curves,   axis=0) * initial_cash
    agg_inv    = np.mean(inv_curves,  axis=0) * initial_cash

    total_overrides = sum(r['override_count'] for r in results)
    total_steps     = sum(len(r['steps'])      for r in results)
    agg_override_pct = total_overrides / total_steps * 100 if total_steps else 0

    agg_stats = compute_portfolio_stats(list(agg_equity), list(agg_inv), [],
                                        initial_cash, list(agg_bh))
    agg_stats['override_count'] = total_overrides
    agg_stats['override_pct']   = agg_override_pct
    pd.DataFrame([agg_stats]).to_csv(os.path.join(output_dir,
                                                  "aggregated_stats.csv"),
                                     index=False)

    plot_aggregated_results(agg_equity, agg_bh, agg_inv,
                            agg_stats, save_path=os.path.join(output_dir,
                                                              "aggregated_curve.html"))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry point
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    mgr   = multiprocessing.Manager()
    log_q = mgr.Queue()
    threading.Thread(target=logging_listener, args=(log_q,), daemon=True).start()

    train_files = [
        r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\traindata\CSCO.csv",
    ]
    test_files = [
        r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\testdata\CSCO.csv",
    ]

    output_dir     = r"P:\10_CWP Trade Department\Ryland\unrealai\unrealai\reports"
    window_size    = 20
    episodes       = 100
    initial_cash   = 100_000
    episode_length = 252

    dummy_df   = load_csv(train_files[0])
    dummy_df['Date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(dummy_df))
    state_size = (window_size, len(TradingEnv(dummy_df).feature_cols))
    action_size = 4

    agent_parameters = {
        'state_size':     state_size,
        'action_size':    action_size,
        'learning_rate':  0.0005,
        'gamma':          0.95,
        'epsilon':        1.0,
        'epsilon_decay':  0.9995,
        'epsilon_min':    0.02,
        'batch_size':     32,
        'memory_size':    50_000,
    }

    start_time = time.time()
    main_loop(agent_parameters, train_files, test_files, output_dir,
              window_size=window_size, episodes=episodes,
              initial_cash=initial_cash, episode_length=episode_length,
              log_queue=log_q)

    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, _   = divmod(rem, 60)
    print(f"Total runtime: {int(h)}h {int(m)}m")

    log_q.put("END")
    gc.collect()
