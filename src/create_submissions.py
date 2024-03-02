from gymnasium.wrappers import TimeLimit
from env_hivssh import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
from copy import deepcopy
import torch
from train import DQNAgent, RegressorAgent
import itertools

config_DQN = {'learning_rate': 0.01,
          'gamma': 0.9,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 0.2,
          'batch_size': 40,
          'gradient_steps': 1,
          'update_target_strategy': 'ema', # or 'ema'
          'update_target_freq': 20,
          'update_target_tau': 0.05,
          'criterion': torch.nn.SmoothL1Loss()}



config_regressor = {'gamma': 0.99,'horizon': 200, 'num_episodes': 400, 'max_depth':10, 'n_estimators':20}

if __name__ == "__main__":
    # Define the hyperparameter values to search


    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
    env2 = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)
    

    config_DQN['batch_size'] = 2048
    for i in range(2):
        agent = DQNAgent(config = config_DQN, deep=True, double=True, path=f"src/DQN/DQN_deep_largebs_double_bs2048_{i}.pt")
        agent.train(env = env, nb_episodes = 1000)  
        agent.load()      
        score_agent = evaluate_HIV(agent=agent, nb_episode=1)
        score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)

        with open(file="train_score_tosubmit.txt", mode="a") as f:
            f.write(f"DQN_deep_largebs_double_bs2048_{i}.pt\n")
            f.write("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")
        
    # for i in range(2):
    #     agent = DQNAgent(config = config_DQN, deep=True, double=False, path=f"src/DQN/DQN_deep_bs1024_{i}.pt")
    #     agent.train(env = env, nb_episodes = 1000)
    #     agent.load()
    #     score_agent = evaluate_HIV(agent=agent, nb_episode=1)
    #     score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)
    #     with open(file="train_score_tosubmit.txt", mode="a") as f:
    #         f.write(f"DQN_deep_bs1024_{i}.pt\n")
    #         f.write("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")
    
    for i in range(2):
        agent = DQNAgent(config = config_DQN, deep=True, double=True, path=f"src/DQN/DQN_deep_largebs_double_pop_bs2048_{i}.pt")
        agent.train(env = env2, nb_episodes = 1000)
        agent.load()
        score_agent = evaluate_HIV(agent=agent, nb_episode=1)
        score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)

        with open(file="train_score_tosubmit.txt", mode="a") as f:
            f.write(f"DQN_deep_largebs_pop_bs2048_{i}.pt\n")
            f.write("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")
        
    for i in range(2):
        agent = DQNAgent(config = config_DQN, deep=False, double=False, path=f"src/DQN/DQN_deep_largebs_pop_bs2048_{i}.pt")
        agent.train(env = env2, nb_episodes = 1000)
        agent.load()
        score_agent = evaluate_HIV(agent=agent, nb_episode=1)
        score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)

        with open(file="train_score_tosubmit.txt", mode="a") as f:
            f.write(f"DQN_deep_largebs_bs2048_{i}.pt\n")
            f.write("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")

    # # High episodes low horizon    
    # config_regressor['n_estimators'] = 100
    # config_regressor['num_episodes'] = 800
    # config_regressor['horizon'] = 5000
    
    # agent = RegressorAgent(config = config_regressor, regressor = 'ExtraTrees')
    # agent.train(env = env)
    # agent.save(f"src/ExtraTrees_800_5000_100.pt")

    # score_agent = evaluate_HIV(agent=agent, nb_episode=1)
    # score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)

    # with open(file="train_score_tosubmit.txt", mode="a") as f:
    #     f.write(f" ExtraTrees regressor 800 episodes: horizon={5000}, n_estimators={100}\n")
    #     f.write("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")
        
        
    # config_regressor['horizon'] = 10000
    # config_regressor['max_depth'] = 15
    # config_regressor['n_estimators'] = 200
    
    # agent = RegressorAgent(config = config_regressor, regressor = 'XGBoost')
    # agent.train(env = env)

    # score_agent = evaluate_HIV(agent=agent, nb_episode=1)
    # score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)

    # with open(file="train_score_tosubmit.txt", mode="a") as f:
    #     f.write(f" XGBoost regressor 400 episodes: horizon={10000}, n_estimators = {200}, max_depth={15}\n")
    #     f.write("{:e}".format(score_agent)+"\n"+"{:e}".format(score_agent_dr)+"\n")