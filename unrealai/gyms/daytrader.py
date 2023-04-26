import numpy as np
import pandas as pd
import sys 
import time
import tensorflow as tf
import random
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


from utils.finance_data import get_intra_equity, momentum_score

tf.keras.utils.disable_interactive_logging()




'''creating custom trading env mimicking openAI gymnasium
and eod historical data (need api key) and learns on 
intraday data only. no daily data.
'''

#number of features/states in the env
# class observation_space:
#     def __init__(self, n):
#         self.shape = (n,) #number of features in the statespace

#number of possible actions
#long, short, flat
class action_space:
    def __init__(self, n):
        self.n = n
    def seed(self, seed):
        pass
    def sample(self):
        return random.randint(0,2)

class Trader:
    def __init__(self, symbol, start, stop, interval, pct_train):
        self.symbol = symbol
        self.start = start
        self.stop = stop
        self.interval = interval
        self.pct_train = pct_train
        self.bar = 0
        self.treward = 0
        self.performance = 1
        self.trade_penalty = .001
        self.min_performance = .90
        self.current_position = 0
        # self.observation_space = observation_space(
        #     len(self._prepare_data().columns)
        #     )
        self.action_space = action_space(3)
        self._prepare_data()

    
    def _prepare_data(self):
        df = get_intra_equity(
            self.symbol,
            self.start,
            self.stop,
            self.interval
            )
        df.set_index('timestamp', inplace=True)
        timedelta = pd.to_datetime(
            df['datetime']
            ).diff().dt.components['minutes'].iloc[1]
        
        #1 Day MA Distance and Slope
        df['d_1d_ma'] = ((df['close']-df['close'].rolling(int(390/timedelta)).mean())/df['close']).round(4)
        df['s_1d_ma'] = df['close'].rolling(int(390/timedelta)).mean().rolling(int(390/timedelta)).apply(
        lambda x: momentum_score(x)  
        )
        
        #5 Day MA Distance and Slope
        df['d_5d_ma'] = ((df['close']-df['close'].rolling(int(1950/timedelta)).mean())/df['close']).round(4)
        df['s_5d_ma'] = df['close'].rolling(int(1950/timedelta)).mean().rolling(int(1950/timedelta)).apply(
        lambda x: momentum_score(x)  
        )

        #20 Day MA Distance and Slope
        df['d_20m_ma'] = ((df['close']-df['close'].rolling(int(7800/timedelta)).mean())/df['close']).round(4)
        df['s_20m_ma'] = df['close'].rolling(int(7800/timedelta)).mean().rolling(int(7800/timedelta)).apply(
        lambda x: momentum_score(x)  
        )


        #1 Day Price Return 
        df['r_1d'] = df['close'].rolling(int(390/timedelta)).apply(lambda x: (x.iloc[-1]/x.iloc[0])-1)

        #5 Day Price Return 
        df['r_5d'] = df['close'].rolling(int(1950/timedelta)).apply(lambda x: (x.iloc[-1]/x.iloc[0])-1)

        #20 Day Price Return 
        df['r_20d'] = df['close'].rolling(int(7800/timedelta)).apply(lambda x: (x.iloc[-1]/x.iloc[0])-1)


        #1 Day Rolling VWAP
        df['d_1d_vwp'] = (df['close']-((df['close']*df['volume']).rolling(int(390/timedelta), min_periods=1).sum() / \
            df['volume'].rolling(int(390/timedelta), min_periods=1).sum())) / df['close']
        df['s_1d_vwp'] = ((df['close']*df['volume']).rolling(int(390/timedelta), min_periods=1).sum() / \
            df['volume'].rolling(int(390/timedelta), min_periods=1).sum()).rolling(int(390/timedelta)).apply(
        lambda x: momentum_score(x)  
        )
        
        #5 Day Rolling VWAP
        df['d_5d_vwp'] = (df['close']-((df['close']*df['volume']).rolling(int(1950/timedelta), min_periods=1).sum() / \
            df['volume'].rolling(int(1950/timedelta), min_periods=1).sum())) / df['close']
        df['s_5d_vwp'] = ((df['close']*df['volume']).rolling(int(1950/timedelta), min_periods=1).sum() / \
            df['volume'].rolling(int(1950/timedelta), min_periods=1).sum()).rolling(int(1950/timedelta)).apply(
        lambda x: momentum_score(x)  
        )
        
        #20 Day Rolling VWAP
        df['d_20d_vwp'] = (df['close']-((df['close']*df['volume']).rolling(int(7800/timedelta), min_periods=1).sum() / \
            df['volume'].rolling(int(7800/timedelta), min_periods=1).sum())) / df['close']
        df['s_20d_vwp'] = ((df['close']*df['volume']).rolling(int(7800/timedelta), min_periods=1).sum() / \
            df['volume'].rolling(int(7800/timedelta), min_periods=1).sum()).rolling(int(7800/timedelta)).apply(
        lambda x: momentum_score(x)  
        )

        #Step PnL Reward
        df['pnl'] = (df['close'].diff()/df['close'].shift(1)).shift(-1)

        df=df.round(4)
        self.original_data = df.dropna()
        df=df.drop([
            'gmtoffset',
            'datetime',
            'open',
            'high',
            'low',
            'close',
            'volume'
        ], axis=1)
        self._data = df.dropna()
        
    
    def _get_state(self):
        df = self._data.drop('pnl',axis=1).iloc[self.bar]
        df.loc['positioning'] = self.current_position
        return df
    
    def reset(self):
        self.treward = 0
        self.performance = 1
        self.bar = 0
        self.current_position = 0
        state = self._data.drop('pnl',axis=1).iloc[self.bar]
        state.loc['positioning'] = self.current_position
        return state.values
    
    #new action vs current_position vs current_pnl
    def step(self, action):

        pnl = self._data.iloc[self.bar]['pnl']
        
        #0
        if (self.current_position==0) & (action==0):
            reward = 0
        if (self.current_position==0) & (action!=0):
            reward = pnl - self.trade_penalty if action==1 else (-1*pnl) - self.trade_penalty
        #1
        if (self.current_position==1) & (action==1):
            reward = pnl
        if (self.current_position==1) & (action!=1):
            reward = self.trade_penalty if action==0 else (-1*pnl) - self.trade_penalty
        #2
        if (self.current_position==2) & (action==2):
            reward = -pnl
        if (self.current_position==2) & (action!=2):
            reward = self.trade_penalty if action==0 else pnl - self.trade_penalty

        
        self.current_position=action
        self.treward += reward
        self.performance *= (1+reward)
        self.bar += 1
        if self.bar>=len(self._data):
            done=True
        elif self.performance<self.min_performance:
            done=True
        else:
            done=False
        
        state = self._get_state()
        info = {}
        
        return state.values, self.performance, done, info
    


class DQNAgent:
    def __init__(self, state_size, action_size, maxmem):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=maxmem)
        self.history = pd.DataFrame([], columns=['score', 'barsin'])
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.005
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def analyze_training(self, plot='score_avg', average=10):
        self.history['score_avg'] = self.history['score'].rolling(average).mean()
        self.history['barsin_avg'] = self.history['barsin'].rolling(average).mean()
        self.history[plot].plot()        



if __name__ == "__main__":
    env = Trader('AAPL', '2023-02-02','2023-04-01', '5m', .5)
    state_size = 16
    action_size = env.action_space.n
    max_mem = 100
    agent = DQNAgent(state_size, action_size, max_mem)
    episodes = 125
    batch_size = 16

    for e in np.arange(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in np.arange(len(env._data)-2):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}, position: {}, epsilon: {}, num_candles: {}, len_mem: {}".format(
                    e,
                    episodes,
                    env.performance, 
                    env.current_position, 
                    agent.epsilon,
                    time, 
                    len(agent.memory)))
                
                agent.history.loc[e,'score'] = env.performance
                agent.history.loc[e,'barsin'] = time

                break
            
            if time==np.arange(len(env._data)-2)[-1]:
                
                print("WE FINISHED -> episode: {}/{}, score: {}, position: {}, epsilon: {}, num_candles: {}, len_mem: {}".format(
                    e,
                    episodes,
                    env.performance, 
                    env.current_position, 
                    agent.epsilon,
                    time, 
                    len(agent.memory)))
                
                agent.history.loc[e,'score'] = env.performance
                agent.history.loc[e,'barsin'] = time

            if (len(agent.memory) > batch_size):                
                agent.replay(batch_size)


