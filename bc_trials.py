import numpy as np
import sys
import os
import gym
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import datetime
import time
from termcolor import colored

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-f', '--exp_file', type=str, help='Expert game plays')
parser.add_argument('-I', '--total_iters', default=100, type=float, help='Iterations for training')
parser.add_argument('-t', '--train', default=0, type=int, help='1 to train; 0 to test')
parser.add_argument('-m', '--model', default=None, type=str, help='Trained model')
parser.add_argument('-n', '--num_test', default=1e3, type=float, help='Number of games to test.')

args = parser.parse_args()

sys.path.append('/Users/cusgadmin/Documents/UCB/Academics/SSastry/\
	Multi_agent_competition')
os.chdir('/Users/cusgadmin/Documents/UCB/Academics/SSastry/Multi_agent_competition/')

if args.train:
	now = datetime.datetime.now()

	print(colored('Loading expert data from {}!'.format(args.exp_file),'red'))
	exp_data = np.load(args.exp_file)
	print(colored('Expert evader has won {} games!'\
		.format(len(exp_data['episode_returns'])),'red'))
	dataset = ExpertDataset(expert_path=args.exp_file, verbose=1)

	start_time = time.time()
	model = GAIL('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', dataset, verbose=1)

	print(colored('Training a behaviour cloning agent for {} iterations!'.format(int(args.total_iters)),'red'))
	model.pretrain(dataset=dataset,n_epochs=int(args.total_iters))
	model.save('games{}_iters{}_{}_bc_pursuitevasion_small'.format(len(exp_data['episode_returns']),\
			int(args.total_iters),str(now.strftime('%Y%m%d'))))
	end_time = time.time()
	print(colored('Training time: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(end_time-start_time,\
		(end_time-start_time)/60,(end_time-start_time)/3600),'red'))
	print(colored('Trained BC policy','red'))
	
else: #test
	print(colored('Trained on expert data from {}!'.format(args.exp_file),'red'))
	# exp_data = np.load(args.exp_file)s
	print(colored('Testing learnt policy from model file {} for {} games!'.\
		format(args.model,int(args.num_test)),'red'))
	start_time = time.time()
	model = GAIL.load(args.model)
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
			if g > args.num_test:
				break
	end_time = time.time()
	# print(colored('Expert evader had won {} games!'\
	# 	.format(len(exp_data['episode_returns'])),'red'))
	print(colored('BC Evader won {}/{} games = {:.2f}%!'.format(e_win_games,int(args.num_test),\
		e_win_games*100/args.num_test),'red'))
	print(colored('Testing time: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(end_time-start_time,\
		(end_time-start_time)/60,(end_time-start_time)/3600),'red'))
	print(colored('Tested BC policy','red'))