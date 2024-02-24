from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from matplotlib import pyplot as plt
from evaluate import evaluate_agent
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb
import xgboost as xgb
import _pickle as cPickle
import random

config = {'learning_rate': 0.0001,
          'gamma': 0.98,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 40,
          'batch_size': 40,
          'gradient_steps': 1,
          'update_target_strategy': 'ema', # or 'ema'
          'update_target_freq': 50,
          'update_target_tau': 0.05,
          'horizon': 200,
          'criterion': torch.nn.SmoothL1Loss()}

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


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
    
def gradient_step(agent_model, target_model, memory, optimizer, criterion):
    if len(memory) > config['batch_size']:
        X, A, R, Y, D = memory.sample(config['batch_size'])
        QYmax = target_model(Y).max(1)[0].detach()
        update = torch.addcmul(R, 1-D, QYmax, value=config['gamma']) # do if non-terminal(s'): update = r + gamma * max(Q')   else:  update = r
        QXA = agent_model(X).gather(1, A.to(torch.long).unsqueeze(1))
        loss = criterion(QXA, update.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
            
        """Training pipeline for the agent
        """
def train(agent, nb_episodes, env):
    """
    Train the agent in the HIV environment.

    Args:
        nb_episodes (int): The number of episodes to train the agent.

    Returns:
        Agent: The trained agent.
    """
    
    if torch.cuda.is_available():
        agent.DQN.to("cuda")
    device = "cuda" if next(agent.DQN.parameters()).is_cuda else "cpu"
    target_model = deepcopy(agent.DQN).to(device)
    optimizer = torch.optim.Adam(agent.DQN.parameters(), lr=config['learning_rate'])
    criterion = config['criterion']
    
    memory = ReplayBuffer(config['buffer_size'], device)
    cum_reward = 0
    step = 0
    episode_return = [0]
    episode = 0
    done = False
    
    obs, info = env.reset()
    epsilon =  config['epsilon_max']
    count = 0
    while episode < nb_episodes:
        
         # update epsilon
        if step > config['epsilon_delay_decay']:
            epsilon = max(config["epsilon_min"], epsilon-(config['epsilon_max']-config['epsilon_min'])/config['epsilon_decay_period'])
            
        # select epsilon-greedy action
        if np.random.rand() < epsilon:
            action = agent.act(obs, use_random=True)
        else:
            action = agent.act(obs)
        
        # step
        next_state, reward, done, trunc, _ = env.step(action)
        memory.append(obs, action, reward, next_state, done)
        cum_reward += reward

        # train
        for _ in range(config['gradient_steps']):
            gradient_step(agent.DQN,target_model, memory, optimizer, criterion)
        
        # next transition
        step += 1
        # update target network if needed
        if config['update_target_strategy'] == 'replace':
            if step % config['update_target_freq'] == 0: 
                target_model.load_state_dict(agent.DQN.state_dict())
        if config['update_target_strategy'] == 'ema':
            target_state_dict = target_model.state_dict()
            model_state_dict = agent.DQN.state_dict()
            tau = config['update_target_tau']
            for key in model_state_dict:
                target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
            target_model.load_state_dict(target_state_dict)
            
        if trunc:
            episode += 1
            print("Episode ", '{:3d}'.format(episode), 
                    ", epsilon ", '{:6.2f}'.format(epsilon), 
                    ", batch size ", '{:5d}'.format(len(memory)), 
                    ", episode return ", '{:4.1f}'.format(cum_reward/step),
                    sep='')
            obs, _ = env.reset()
            
            if cum_reward/step > 100000:
                count += 1
            else:
                count = 0
            if count == 3:
                break
                
            episode_return.append((cum_reward+episode_return[-1]*(episode-1))/episode)
            step = 0
            cum_reward = 0
        else:
            obs = next_state

    fig = plt.figure() 
    plt.title("Strategy loss over time")
    plt.xlabel("Episodes")
    plt.ylabel("Strategy loss")
    plt.plot(episode_return)
    plt.show()
    fig.savefig("strategy_loss.png")
            
    return agent

class DQNAgent:
    def __init__(self, device = "cpu"):
        self.action_idx = [0, 1, 2, 3]
        self.size_observation_space = 6
        self.size_action_space = 4
        self.nb_neurons = 64
        self.path = "src/DQN_ag.pt"
        # self.path = "myagent.pt"
        self.DQN = torch.nn.Sequential(nn.Linear(self.size_observation_space, self.nb_neurons),
                          nn.ReLU(),
                          nn.Linear(self.nb_neurons, self.nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(self.nb_neurons, self.size_action_space)).to(device)
        
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
        """
        Saves the agent's current state to a file specified by the path.

        This method should serialize the agent's state (e.g., model weights, configuration settings)
        and save it to a file, allowing the agent to be later restored to this state using the `load` method.

        Args:
            path (str): The file path where the agent's state should be saved.

        """
        torch.save(self.DQN.state_dict(), path)
        pass

    def load(self):
        state_dict = torch.load(self.path, map_location='cpu')
        self.DQN.load_state_dict(state_dict)
        self.DQN.eval()
        pass

"""Regressor Agent
"""
    

class ProjectAgent:
    def __init__(self, num_features = 6, num_actions = 4, regressor='XGBBoost'):
        self.num_features = num_features
        self.num_actions = num_actions
        self.regr_model = None
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.next_state_history = []
        self.regressor = regressor

    def select_action(self, state):
        Qsa = []
        for action in range(self.num_actions):
            sa = np.append(state, action).reshape(1, -1)
            Qsa.append(self.regr_model.predict(sa))
        return np.argmax(Qsa)

    def train(self, env, horizon = 200, num_episodes=200):
        state = self.reset_state(env)  # Reset the environment at the beginning of each episode
        
        for _ in range(horizon): # collect samples
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
        for iter in range(num_episodes):
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
                value = np.array(self.reward_history).copy() + config['gamma']*max_Q2
            if self.regressor == 'RandomForest':
                Q = RandomForestRegressor()
            # elif self.regressor == 'LightGBM':
            #     Q = lgb.LGBMRegressor()
            elif self.regressor == 'XGBoost':
                Q = xgb.XGBRegressor()
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
        
    def save(self):
        if self.regressor == 'RandomForest':
            with open('src/random_forest_model.pkl', 'wb') as f:
                cPickle.dump(self.regr_model, f)
                
        # elif self.regressor == 'LightGBM':
        #     self.regr_model.booster_.save_model('lgbm_model.txt')
            
        elif self.regressor == 'XGBoost':
            self.regr_model.save_model('src/xgb_model.json')
        pass

    def load(self, path = ''):
        if self.regressor == 'RandomForest':
            with open('src/random_forest_model.pkl', 'rb') as f:
                self.regr_model = cPickle.load(f)
                
        # elif self.regressor == 'LightGBM':
        #     self.regr_model = lgb.Booster(model_file='src/lgbm_model.txt')
            
        elif self.regressor == 'XGBoost':
            self.regr_model = xgb.XGBRegressor()
            self.regr_model.load_model('src/xgb_model.json')    
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

    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
    DQN_ag = DQNAgent()
    # agent = train( DQN_ag,nb_episodes=1000, env=env)
    # agent.save("src/DQN_ag.pt")
    # print('Agent saved')
    DQN_ag.load()
    
    randforest_ag = ProjectAgent(regressor='RandomForest')
    randforest_ag.train(env = env, horizon = int(5*1e4), num_episodes=20)
    randforest_ag.save()
    # randforest_ag = RegressorAgent(regressor='RandomForest')
    # randforest_ag.load()
    
    xgb_ag = ProjectAgent(regressor='XGBoost')
    xgb_ag.train(env = env, horizon = int(5*1e4), num_episodes=20)
    xgb_ag.save()
    # xgb_ag = RegressorAgent(regressor='XGBoost')
    # xgb_ag.load()
    
    # lgbm_ag = RegressorAgent(regressor='LightGBM')
    # lgbm_ag.train(env = env, horizon = 100, num_episodes=200)
    # lgbm_ag.save()
    # lgbm_ag = RegressorAgent(regressor='LightGBM')
    # lgbm_ag.load()
    
    rand_ag = RandomAgent()    
    
    env_pop = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)
    env_part = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
    
    print("Evaluation on particular")
    result_DQN_part = evaluate_agent(DQN_ag, env_part, nb_episode = 10) 
    result_random_part = evaluate_agent(rand_ag, env_part, nb_episode = 10)
    result_randomforest_part = evaluate_agent(randforest_ag, env_part, nb_episode = 10)
    result_xgb_part = evaluate_agent(xgb_ag, env_part, nb_episode = 10)
    # result_lgbm_part = evaluate_agent(lgbm_ag, env_part, nb_episode = 10)
    
    print("Evaluation on population")
    result_DQN_pop = evaluate_agent(DQN_ag, env_pop, nb_episode = 10)
    result_random_pop = evaluate_agent(rand_ag, env_pop, nb_episode = 10)
    result_randomforest_pop = evaluate_agent(randforest_ag, env_pop, nb_episode = 10)  
    result_xgb_pop = evaluate_agent(xgb_ag, env_pop, nb_episode = 10)
    # result_lgbm_pop = evaluate_agent(lgbm_ag, env_pop, nb_episode = 10)
     
    print("DQN partiel", result_DQN_part)
    print("Random agent partiel", result_random_part)
    print("Random forest agent partiel", result_randomforest_part)    
    print("XGBoost agent partiel", result_xgb_part)
    # print("LightGBM agent partiel", result_lgbm_part)
    
    print("DQN population", result_DQN_pop)
    print("Random agent population", result_random_pop)
    print("Random forest agent population", result_randomforest_pop)
    print("XGBoost agent population", result_xgb_pop)
    # print("LightGBM agent population", result_lgbm_pop)

    
    
    
    
