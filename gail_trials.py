import numpy as np
import sys
import os
import gym
from stable_baselines import GAIL, TRPO
from stable_baselines.gail import ExpertDataset, generate_expert_traj
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
parser.add_argument('-pI', '--pretrain_iters', default=10, type=float, help='Iterations for pretraining with BC')
parser.add_argument('-t', '--train', default=0, type=int, help='1 to train; 0 to test')
parser.add_argument('-pt', '--pretrain', default=0, type=int, help='1 to pretrain; 0 to not')
parser.add_argument('-ptf', '--pretrained_model', type=str, help='Pass pretrained file if not pretraining')
parser.add_argument('-m', '--model', default=None, type=str, help='Trained model')
parser.add_argument('-n', '--num_test', default=1000, type=int, help='Number of games to test.')
parser.add_argument('-g', '--generate_expert', default=0, type=int,\
 help='1 to use trpo to generate expert data; 0 to not')
parser.add_argument('-gn', '--generate_num', default=1e3, type=float,\
 help='1 to use trpo to generate expert data; 0 to not')

args = parser.parse_args()

sys.path.append('/Users/cusgadmin/Documents/UCB/Academics/SSastry/\
	Multi_agent_competition')
os.chdir('/Users/cusgadmin/Documents/UCB/Academics/SSastry/Multi_agent_competition/')

if args.train:
	now = datetime.datetime.now()

	if args.generate_expert:
		start_time = time.time()
		print(colored('Generating {} games of expert data using TRPO!'.format(int(args.generate_num)),'red'))
		exp_model = TRPO('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', verbose=1)
		generate_expert_traj(exp_model,'{}_expert_trpo_pe_small'.format(int(args.generate_num)), \
			n_timesteps=int(1e1*args.generate_num), n_episodes=int(args.generate_num))
		dataset = ExpertDataset(expert_path='{}_expert_trpo_pe_small.npz'.format(int(args.generate_num)), verbose=1)
		end_time = time.time()
		print(colored('Time to generate expert data: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(end_time-start_time,\
		(end_time-start_time)/60,(end_time-start_time)/3600),'red'))

	else:
		print(colored('Loading expert data from {}!'.format(args.exp_file),'red'))
		exp_data = np.load(args.exp_file)
		print(colored('Expert evader has won {} games!'\
			.format(len(exp_data['episode_returns'])),'red'))
		dataset = ExpertDataset(expert_path=args.exp_file, verbose=1)

	start_time1 = time.time()
	# if args.pretrained_model:
	# 	print(colored('Loading pretrained model from {}!'.format(args.pretrained_model),'red'))
	# 	env = gym.make('gym_pursuitevasion_small:pursuitevasion_small-v0')
	# 	model = GAIL.load(args.pretrained_model,env=env)
	# else:
	model = GAIL('MlpPolicy', 'gym_pursuitevasion_small:pursuitevasion_small-v0', dataset, verbose=1)

	# if args.pretrain:
	print(colored('Pretraining a behaviour cloning agent!','red'))
	model.pretrain(dataset=dataset,n_epochs=int(args.pretrain_iters))
		# model.save('games{}_pretrained_{}_bc{}_trpo_pursuitevasion_small'.format(int(args.generate_num),\
		# 		str(now.strftime('%Y%m%d')),int(args.pretrain_iters)))

	end_time1 = time.time()
	print(colored('Pretraining time: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(end_time1-start_time1,\
		(end_time1-start_time1)/60,(end_time1-start_time1)/3600),'red'))

	start_time2 = time.time()
	print(colored('Training a GAIL agent!','red'))
	model.learn(total_timesteps=int(args.total_iters))

	if args.generate_expert:
		model.save('games{}_iters{}_{}_bc{}_gail_trpo_pursuitevasion_small'.format(int(args.generate_num),\
			int(args.total_iters),str(now.strftime('%Y%m%d')),int(args.pretrain_iters)))
	else:
		model.save('games{}_iters{}_{}_bc{}_gail_trpo_pursuitevasion_small'.format(len(exp_data['episode_returns']),\
			int(args.total_iters),str(now.strftime('%Y%m%d')),int(args.pretrain_iters)))
	end_time2 = time.time()
	print(colored('Total Training time: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(end_time2-start_time1,\
		(end_time2-start_time1)/60,(end_time2-start_time1)/3600),'red'))
	print(colored('Trained TRPO+GAIL policy','red'))
else: #test
	if not args.generate_expert:
		print(colored('Trained on expert data from {}!'.format(args.exp_file),'red'))
		# exp_data = np.load(args.exp_file)
	print(colored('Testing learnt policy from model file {} for {} games!'.\
		format(args.model,args.num_test),'red'))
	start_time = time.time()
	model = GAIL.load(args.model)
	env = gym.make('gym_pursuitevasion_small:pursuitevasion_small-v0')
	g = 1
	obs = env.reset(ep=g)
	# print('Playing game {}'.format(g))
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
	# if not args.generate_expert:
	# 	print(colored('Expert evader won {}/{} games = {:.2f}%!'.format(int(sum(exp_data['evader_wins'])),\
	# 		exp_data['number_games'],np.sum(exp_data['evader_wins'])*100/exp_data['number_games']),'red'))
	print(colored('GAIL Evader won {}/{} games = {:.2f}%!'.format(e_win_games,args.num_test,\
		e_win_games*100/args.num_test),'red'))
	print(colored('Testing time: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(end_time-start_time,\
		(end_time-start_time)/60,(end_time-start_time)/3600),'red'))
	print(colored('Tested TRPO+GAIL policy','red'))