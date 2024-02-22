from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
# from matplotlib import pyplot as plt
from evaluate import evaluate_agent
import numpy as np
import torch
import torch.nn as nn
import random

config = {'learning_rate': 0.0001,
          'gamma': 0.98,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 40,
          'batch_size': 40}

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, device = "cpu"):
        self.action_idx = [0, 1, 2, 3]
        self.size_observation_space = 6
        self.size_action_space = 4
        self.nb_neurons = 64
        self.path = "src/myagent.pt"
        self.DQN = torch.nn.Sequential(nn.Linear(self.size_observation_space, self.nb_neurons),
                          nn.ReLU(),
                          nn.Linear(self.nb_neurons, self.nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(self.nb_neurons, self.size_action_space)).to(device)
        
    def act(self, observation, use_random=False):
        # must return action in self.action_idx (0 pas de medoc, 1 medoc 1, 2 medoc 2, 3 les deux)
        # use random define if agent should act randomly (for exploration)
        # Il faut state 5 E elevé et state 4 V faible 
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
        """
        Loads the agent's state from a file specified by the path (HARDCODED). This not a good practice,
        but it will simplify the grading process.

        This method should deserialize the saved state (e.g., model weights, configuration settings)
        from the file and restore the agent to this state. Implementations must ensure that the
        agent's state is compatible with the `act` method's expectations.

        Note:
            It's important to ensure that neural network models (if used) are loaded in a way that is
            compatible with the execution device (e.g., CPU, GPU). This may require specific handling
            depending on the libraries used for model implementation. WARNING: THE GITHUB CLASSROOM
        HANDLES ONLY CPU EXECUTION. IF YOU USE A NEURAL NETWORK MODEL, MAKE SURE TO LOAD IT IN A WAY THAT
        DOES NOT REQUIRE A GPU.
        """
        state_dict = torch.load(self.path, map_location='cpu')
        self.DQN.load_state_dict(state_dict)
        self.DQN.eval()
        pass
    

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
    

    
def gradient_step(agent_model, memory, optimizer, criterion):
    if len(memory) > config['batch_size']:
        X, A, R, Y, D = memory.sample(config['batch_size'])
        QYmax = agent_model(Y).max(1)[0].detach()
        update = torch.addcmul(R, 1-D, QYmax, value=config['gamma']) # do if non-terminal(s'): update = r + gamma * max(Q')   else:  update = r
        QXA = agent_model(X).gather(1, A.to(torch.long).unsqueeze(1))
        loss = criterion(QXA, update.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
class RandomAgent:

    def act(self, observation, use_random=False):
        return np.random.choice([0, 1, 2, 3])

    def save(self, path):
        pass

    def load(self):

        pass
    
    
def train(nb_episodes):
    """
    Train the agent in the HIV environment.

    Args:
        nb_episodes (int): The number of episodes to train the agent.

    Returns:
        Agent: The trained agent.
    """

    agent = ProjectAgent()
    
    if torch.cuda.is_available():
        agent.DQN.to("cuda")
    device = "cuda" if next(agent.DQN.parameters()).is_cuda else "cpu"
    
    optimizer = torch.optim.Adam(agent.DQN.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    memory = ReplayBuffer(config['buffer_size'], device)
    cum_reward = 0
    step = 0
    episode_return = [0]
    episode = 0
    done = False
    
    obs, info = env.reset()
    epsilon =  config['epsilon_max']
    
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
        gradient_step(agent.DQN, memory, optimizer, criterion)
        
        # next transition
        step += 1
        # print('Step', step, 'Reward', reward, 'Cumulative reward', cum_reward, 'Epsilon',
        # epsilon, 'Action', action, 'Next state', next_state, 'Done', done, 'Truncated', trunc)
        if trunc:
            episode += 1
            print("Episode ", '{:3d}'.format(episode), 
                    ", epsilon ", '{:6.2f}'.format(epsilon), 
                    ", batch size ", '{:5d}'.format(len(memory)), 
                    ", episode return ", '{:4.1f}'.format(cum_reward/step),
                    sep='')
            obs, _ = env.reset()
            episode_return.append((cum_reward+episode_return[-1]*(episode-1))/episode)
            step = 0
            cum_reward = 0
        else:
            obs = next_state

    # fig = plt.figure() 
    # plt.title("Strategy loss over time")
    # plt.xlabel("Episodes")
    # plt.ylabel("Strategy loss")
    # plt.plot(np.arange(nb_episodes),episode_return)
    # plt.show()
    # fig.savefig("strategy_loss.png")
            
    return agent


if __name__ == "__main__":

    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.

    agent = train(nb_episodes=1000)
    agent.save("myagent.pt")
    print('Agent saved')
    agent.load()
    rand_ag = RandomAgent()
    
    env_pop =TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)
    env_part = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
    result_myagent_part = evaluate_agent(agent, env_part, nb_episode = 10) 
    result_random_part = evaluate_agent(rand_ag, env_part, nb_episode = 10)
    result_myagent_pop = evaluate_agent(agent, env_pop, nb_episode = 10)
    result_random_pop = evaluate_agent(rand_ag, env_pop, nb_episode = 10)   
    print("My agent partiel", result_myagent_part)
    print("Random agent partiel", result_random_part)
    print("My agent population", result_myagent_pop)
    print("Random agent population", result_random_pop)

    
    
    
    
