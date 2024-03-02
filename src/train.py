from gymnasium.wrappers import TimeLimit
from env_hivssh import HIVPatient
from matplotlib import pyplot as plt
from evaluate import evaluate_agent
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from evaluate import evaluate_HIV, evaluate_HIV_population
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
# import lightgbm as lgb
import xgboost as xgb
import _pickle as cPickle
import random


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

config_DQN = {'learning_rate': 0.01,
          'gamma': 0.9,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 0.2,
          'batch_size': 512,
          'gradient_steps': 1,
          'update_target_strategy': 'ema', # or 'ema'
          'update_target_freq': 20,
          'update_target_tau': 0.05,
          'criterion': torch.nn.SmoothL1Loss()}



config_regressor = {'gamma': 0.99,'horizon': 10000, 'num_episodes': 400, 'max_depth':10, 'n_estimators':100}




class ProjectAgent:
        def __init__(self,config_DQN = config_DQN, config_regressor = config_regressor, deep = True, mean = False, vote = False, n_dqn = 1):
            self.action_idx = [0, 1, 2, 3]
            self.size_observation_space = 6
            self.size_action_space = 4
            self.nb_neurons = 64
            # self.path = "myagent.pt"
            self.deep = deep
            self.mean = mean
            self.vote = vote
            self.n_dqn = n_dqn
            if not self.deep:
                self.DQN = [torch.nn.Sequential(nn.Linear(self.size_observation_space, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.size_action_space)) for _ in range(n_dqn)]
            else: 
                self.DQN = [torch.nn.Sequential(nn.Linear(self.size_observation_space, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.size_action_space)) for _ in range(n_dqn)]
            self.path = "src/DQN"
            self.config_DQN = config_DQN
            self.config_regressor = config_regressor
            self.regr_extratrees = ExtraTreesRegressor(n_estimators=self.config_regressor['n_estimators'])
            self.regr_xgb = xgb.XGBRegressor(max_depth=self.config_regressor['max_depth'], n_estimators=self.config_regressor['n_estimators'])
        
        def act(self, observation, use_random=False):
            if use_random:
                return np.random.choice(self.action_idx)
            else:
                # Qsa_extra, Qsa_xgb = self.get_Q_regr(observation)
                Q_list = self.get_Q_DQN(observation)
 
                
                if self.mean:
                    Q_DQN = np.vstack(Q_list).transpose()
                    # print('Q_DQN:',Q_DQN.shape)
                    Q = np.mean(Q_DQN, axis=1)
                    # print('Q:',Q.shape)
                    return np.argmax(Q)
                
                elif self.vote:
                    a = np.zeros(4)
                    for q in Q_list:
                        a[np.argmax(q)] += 1
                    # a[np.argmax(Qsa_extra)] += 1
                    # a[np.argmax(Qsa_xgb)] += 1
                    return np.argmax(a)
                else :
                    Q_DQN = np.hstack(Q_list).transpose()
                    # Q = np.concatenate((Qsa_extra,Qsa_xgb, Q_DQN), axis=1)
                    # print('Shape concatenated Q:',Q_DQN.shape)
                    # print('Argmax:',np.argmax(Q_DQN))
                    return np.argmax(Q_DQN)%4
                   

            
        def get_Q_regr(self, state):
            Qsa_extra = []
            for action in range(self.num_actions):
                sa = np.append(state, action).reshape(1, -1)
                Qsa_extra.append(self.regr_extratrees.predict(sa))
            Qsa_xgb = []
            for action in range(self.num_actions):
                sa = np.append(state, action).reshape(1, -1)
                Qsa_xgb.append(self.regr_xgb.predict(sa))
            return Qsa_extra, Qsa_xgb
        
        def get_Q_DQN(self, state):
            Q = []
            for i in range(self.n_dqn):
                Q.append(self.DQN[i](torch.Tensor(state).unsqueeze(0)).detach().numpy())
            return Q

        def save(self, path):
            pass

        def load(self):
            device = torch.device('cpu')
            list_dqns = ['DQN_deep_pop_bs1024_0.pt']
            for i, name in enumerate(list_dqns):
                state_dict = torch.load(self.path+'/'+name, map_location=device)
                self.DQN[i].load_state_dict(state_dict)
                self.DQN[i].eval()
        
            # with open('src/extra_trees_model.pkl', 'rb') as f:
            #     self.regr_extratrees = cPickle.load(f) 

            # self.regr_xgb = xgb.XGBRegressor(max_depth=self.config_regressor['max_depth'], n_estimators=self.config_regressor['n_estimators'])
            # self.regr_xgb.load_model('src/xgb_model.json')    
            print('Models loaded')
            pass



"""Trained agent utils
"""
        
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
class DQNAgent:
    def __init__(self,config = config_DQN, deep = False, double = False, path = ''):
        
        self.action_idx = [0, 1, 2, 3]
        self.size_observation_space = 6
        self.size_action_space = 4
        self.nb_neurons = 64
        self.double = double
    
        # self.path = "myagent.pt"
        self.deep = deep
        if not self.deep:
            self.DQN = torch.nn.Sequential(nn.Linear(self.size_observation_space, self.nb_neurons),
                            nn.ReLU(),
                            nn.Linear(self.nb_neurons, self.nb_neurons),
                            nn.ReLU(), 
                            nn.Linear(self.nb_neurons, self.size_action_space))
            self.path = path
        else: 
            self.DQN = torch.nn.Sequential(nn.Linear(self.size_observation_space, self.nb_neurons),
                            nn.ReLU(),
                            nn.Linear(self.nb_neurons, self.nb_neurons),
                            nn.ReLU(), 
                            nn.Linear(self.nb_neurons, self.nb_neurons),
                            nn.ReLU(), 
                            nn.Linear(self.nb_neurons, self.size_action_space))
            self.path = path
        self.target_model = deepcopy(self.DQN)
        self.config = config
        
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(self.action_idx)
        else:
            return self.greedy_action(observation)
        
    def greedy_action(self, state):
        device = "cuda" if next(self.DQN.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.DQN(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.DQN.state_dict(), path)
        pass

    def load(self):
        device = torch.device('cpu')
        state_dict = torch.load(self.path, map_location=device)
        self.DQN.load_state_dict(state_dict)
        self.DQN.eval()
        pass
    

    def gradient_step(self, memory, optimizer, criterion):
        if len(memory) > self.config['batch_size']:
            if self.double:
                X, A, R, Y, D = memory.sample(self.config['batch_size'])
                with torch.no_grad():  
                    nexta = self.DQN(Y).max(1)[1].unsqueeze(1)
                    
                QYmax = self.target_model(Y).gather(1, nexta).squeeze()
                update = torch.addcmul(R, 1-D, QYmax, value=self.config['gamma']) # do if non-terminal(s'): update = r + gamma * max(Q')   else:  update = r
                QXA = self.DQN(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = criterion(QXA, update.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            else:
                X, A, R, Y, D = memory.sample(self.config['batch_size'])
                QYmax = self.target_model(Y).max(1)[0].detach()
                update = torch.addcmul(R, 1-D, QYmax, value=self.config['gamma']) # do if non-terminal(s'): update = r + gamma * max(Q')   else:  update = r
                QXA = self.DQN(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = criterion(QXA, update.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            
        """Training pipeline for the agent
            """
    def train(self, nb_episodes, env):
        """
        Train the agent in the HIV environment.

        Args:
            nb_episodes (int): The number of episodes to train the agent.

        Returns:
            Agent: The trained agent.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DQN.to(device)
        self.target_model.to(device)
 

        optimizer = torch.optim.Adam(self.DQN.parameters(), lr=self.config['learning_rate'])
        criterion = self.config['criterion']
        
        memory = ReplayBuffer(self.config['buffer_size'], device)
        cum_reward = 0
        step = 0
        episode_return = [0]
        episode = 0
        done = False
        
        obs, info = env.reset()
        epsilon =  self.config['epsilon_max']
        best_cum_reward = 0
        while episode < nb_episodes:
            
            # update epsilon
            if step > self.config['epsilon_delay_decay']*nb_episodes*200:
                epsilon = max(self.config["epsilon_min"], epsilon-(self.config['epsilon_max']-self.config['epsilon_min'])/self.config['epsilon_decay_period'])
                
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = self.act(obs, use_random=True)
            else:
                action = self.act(obs)
            
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            memory.append(obs, action, reward, next_state, done)
            cum_reward += reward

            # train
            for _ in range(self.config['gradient_steps']):
                self.gradient_step(memory, optimizer, criterion)
            
            # next transition
            step += 1
            # update target network if needed
            if self.config['update_target_strategy'] == 'replace':
                if step % self.config['update_target_freq'] == 0: 
                    self.target_model.load_state_dict(self.DQN.state_dict())
            if self.config['update_target_strategy'] == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.DQN.state_dict()
                tau = self.config['update_target_tau']
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
                
            if trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:5d}'.format(len(memory)), 
                        ", episode return ", '{:4.1f}'.format(cum_reward/200),
                        sep='')
                obs, _ = env.reset()

                if cum_reward > best_cum_reward:
                    best_cum_reward = cum_reward
                    print('Best model so far')
                    self.save(self.path)
                    
                episode_return.append((cum_reward+episode_return[-1]*(episode-1))/episode)
                cum_reward = 0
            else:
                obs = next_state
                

"""Regressor Agent
"""
    
class RegressorAgent:
    def __init__(self, num_features = 6, num_actions = 4, regressor='XGBoost', config = config_regressor):
        self.num_features = num_features
        self.num_actions = num_actions
        self.regr_model = None
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.next_state_history = []
        self.regressor = regressor
        self.config = config

    def select_action(self, state):
        Qsa = []
        for action in range(self.num_actions):
            sa = np.append(state, action).reshape(1, -1)
            Qsa.append(self.regr_model.predict(sa))
        return np.argmax(Qsa)

    def train(self, env):
        state = self.reset_state(env)  # Reset the environment at the beginning of each episode
        
        for _ in range(self.config['horizon']): # collect samples
            action = self.act(state, use_random=True)
            next_state, reward, done, trunc = self.transition(action, env)
            self.state_history.append(state)
            self.action_history.append(action)
            self.reward_history.append(reward)
            self.next_state_history.append(next_state)
            if done or trunc:
                state = self.reset_state(env)
            else:
                state = next_state
        
        Q_func = []
        SA = np.append(np.array(self.state_history), np.array(self.action_history).reshape((-1,1)),axis=1)
        for iter in range(self.config['num_episodes']):
            if iter % 50 == 0:
                print("Iteration ", '{:3d}'.format(iter), sep='')
            if iter == 0:
                value = np.array(self.reward_history).copy()
            else:
                Q2 = np.zeros((len(self.state_history), self.num_actions))
                for next_action in range(self.num_actions):
                    A2 = next_action*np.ones((len(self.next_state_history), 1))
                    SA_next = np.append(np.array(self.next_state_history), A2,axis=1)
                    Q2[:, next_action] = Q_func[-1].predict(SA_next)
                max_Q2 = np.max(Q2, axis=1)
                value = np.array(self.reward_history).copy() + self.config['gamma']*max_Q2
            if self.regressor == 'RandomForest':
                Q = RandomForestRegressor(n_estimators=self.config['n_estimators'], max_depth=self.config['max_depth'])
            # elif self.regressor == 'LightGBM':
            #     Q = lgb.LGBMRegressor()
            elif self.regressor == 'XGBoost':
                Q = xgb.XGBRegressor(max_depth=self.config['max_depth'], n_estimators=self.config['n_estimators'])
            elif self.regressor == 'ExtraTrees':
                Q = ExtraTreesRegressor(n_estimators=self.config['n_estimators'])
            Q.fit(SA, value)
            Q_func.append(Q)
        self.regr_model = Q_func[-1]

    def reset_state(self,env):
        observation, _ = env.reset()
        return observation # Initialize state randomly

    def transition(self, action, env):
        next_state, reward, done, trunc, _ = env.step(action)
        return next_state, reward, done, trunc
    
        
    def act(self, state, use_random=False):
        if use_random:
            return np.random.choice([0,1,2,3])
        else: 
            return self.select_action(state)
        
    def save(self, path):
        if self.regressor == 'RandomForest':
            with open('src/random_forest_model.pkl', 'wb') as f:
                cPickle.dump(self.regr_model, f)
                
        # elif self.regressor == 'LightGBM':
        #     self.regr_model.booster_.save_model('lgbm_model.txt')
        elif self.regressor == 'ExtraTrees':
            with open('src/extra_trees_model.pkl', 'wb') as f:
                cPickle.dump(self.regr_model, f)
            
        elif self.regressor == 'XGBoost':
            self.regr_model.save_model('src/xgb_model.json')
        pass

    def load(self):
        if self.regressor == 'RandomForest':
            with open('src/random_forest_model.pkl', 'rb') as f:
                self.regr_model = cPickle.load(f)
                
        elif self.regressor == 'ExtraTrees':
            with open('src/extra_trees_model.pkl', 'rb') as f:
                self.regr_model = cPickle.load(f) 
                
        # elif self.regressor == 'LightGBM':
        #     self.regr_model = lgb.Booster(model_file='src/lgbm_model.txt')
            
        elif self.regressor == 'XGBoost':
            self.regr_model = xgb.XGBRegressor(max_depth=self.config['max_depth'], n_estimators=self.config['n_estimators'])
            self.regr_model.load_model('src/xgb_model.json')    
            print('Model loaded')
        pass

"""Random Agent to compare with the trained agent.
"""

class RandomAgent:

    def act(self, observation, use_random=False):
        return np.random.choice([0, 1, 2, 3])

    def save(self, path):
        pass

    def load(self):

        pass
    
"""Launch training
"""

if __name__ == "__main__":

    agent = ProjectAgent()
    agent.load()
    score_agent = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)
    print('Final Model performance Argmax')
    print("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")
    
    agent = ProjectAgent(mean=True)
    agent.load()
    score_agent = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)
    print('Final Model performance Mean')
    print("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")
    
    agent = ProjectAgent(vote=True)
    agent.load()
    score_agent = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)
    print('Final Model performance Vote')
    print("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")
    

    
    
    
    
