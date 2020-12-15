"""
RL trials for cooperative-competitive games!
"""
import numpy as np
import sys
import os
import gym
from stable_baselines import TRPO, DQN, ACKTR, ACER, A2C, PPO1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import datetime
import time
from termcolor import colored

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-I', '--total_iters', default=100, type=float, help='Iterations for training')
parser.add_argument('-t', '--train', default=0, type=int, help='1 to train; 0 to test')
parser.add_argument('-m', '--model', default=None, type=str, help='Trained model')
parser.add_argument('-n', '--num_test', default=1000, type=float, help='Number of games to test.')

args = parser.parse_args()

sys.path.append('/Users/cusgadmin/Documents/UCB/Academics/SSastry/\
    Multi_agent_competition')
os.chdir('/Users/cusgadmin/Documents/UCB/Academics/SSastry/Multi_agent_competition/')

algo_list = ['TRPO','DQN','ACKTR','ACER','A2C','PPO1']
alg = int(input(colored('Enter code for algorithm to use:\n\t0: TRPO\n\t'\
        +'1: DQN\n\t2: ACKTR\n\t3: ACER\n\t4: A2C\n\t5: PPO\n')))

if args.train:
    now = datetime.datetime.now()
    print('Training policy using {} for {} iterations!'.format(algo_list[alg],int(args.total_iters)))
    start_time = time.time()
    if alg == 0:
        model = TRPO('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', verbose=1)
    elif alg == 1:
        model = DQN('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', verbose=1)
    elif alg == 2:
        model = ACKTR('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', verbose=1)
    elif alg == 3:
        model = ACER('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', verbose=1)
    elif alg == 4:
        model = A2C('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', verbose=1)
    elif alg == 5:
        model = PPO1('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', verbose=1)
    # Note: in practice, you need to train for 1M steps to have a working policy
    model.learn(total_timesteps=int(args.total_iters))
    model.save('{}_iters{}_{}_pursuitevasion_small'.format(algo_list[alg],int(args.total_iters),str(now.strftime('%Y%m%d'))))
    end_time = time.time()
    print('Training time for algorithm {}: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(algo_list[alg],\
        end_time-start_time,(end_time-start_time)/60,(end_time-start_time)/3600))
    print('Trained using RL')
else: #test
    print('Testing {} learnt policy from model file {} for {} games!'.format(algo_list[alg],\
        args.model,int(args.num_test)))
    start_time = time.time()
    if alg == 0:
        model = TRPO.load(args.model)
    elif alg == 1:
        model = DQN.load(args.model)
    elif alg == 2:
        model = ACKTR.load(args.model)
    elif alg == 3:
        model = ACER.load(args.model)
    elif alg == 4:
        model = A2C.load(args.model)
    elif alg == 5:
        model = PPO1.load(args.model)
    env = gym.make('gym_pursuitevasion_small:pursuitevasion_small-v0')
    g = 1
    obs = env.reset(ep=g)
    e_win_games = int(0)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, e_win = env.step(action)
        if done:
            g += 1
            obs = env.reset(ep=g)
            if g % 100 == 0:
                print('Playing game {}'.format(g))
            if e_win:
                e_win_games += 1
            if g > int(args.num_test):
                break
    end_time = time.time()
    print('{} Evader won {}/{} games = {:.2f}%!'.format(algo_list[alg],e_win_games,\
        int(args.num_test),e_win_games*100/args.num_test))
    print('Testing time for algorithm {}: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(algo_list[alg],\
        end_time-start_time,(end_time-start_time)/60,(end_time-start_time)/3600))
    print('Tested RL policy')
