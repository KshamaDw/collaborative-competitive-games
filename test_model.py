import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import argparse
from termcolor import colored
from stable_baselines import GAIL
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-m', '--model', default=None, type=str, help='Model to be tested')
parser.add_argument('-n', '--num_test', default=1, type=int, help='Number of games to test.')
parser.add_argument('-s', '--save', default=True, type=bool)

args = parser.parse_args()

sys.path.append('/Users/cusgadmin/Documents/UCB/Academics/SSastry/\
    Multi_agent_competition')
os.chdir('/Users/cusgadmin/Documents/UCB/Academics/SSastry/Multi_agent_competition/')

print(colored('Testing learnt policy from model file {} for {} games!'.\
		format(args.model,args.num_test),'red'))
start_time = time.time()
model = GAIL.load(args.model)
env = gym.make('gym_pursuitevasion_small:pursuitevasion_small-v0')
g = 1
obs = env.reset(ep=g)
e_win_games = int(0)
env.render(mode='human',highlight=True,ep=g)
if args.save:
	metadata = dict(title='Game')
	writer = FFMpegWriter(fps=5,metadata=metadata)
	writer.setup(env.window.fig, "test_game.mp4", 300)
	writer.grab_frame()
while True:
	action, _states = model.predict(obs)
	obs, rewards, done, e_win = env.step(action)
	env.render(mode='human', highlight=True, ep=g)
	if args.save:
		writer.grab_frame()
	if done:
		g += 1
		obs = env.reset(ep=g)
		if g % 100 == 0:
			print('Playing game {}'.format(g))
		if e_win:
			e_win_games += 1
		if g > args.num_test:
			break
if args.save:
	writer.finish()
end_time = time.time()
print(colored('Trained Evader won {}/{} games = {:.2f}%!'.format(e_win_games,args.num_test,\
		e_win_games*100/args.num_test),'red'))
print(colored('Testing time: {:.2f}s = {:.2f}min = {:.4f}hrs'.format(end_time-start_time,\
		(end_time-start_time)/60,(end_time-start_time)/3600),'red'))
