import numpy as np
import argparse
import datetime
import time
from termcolor import colored
import sys
import os

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-f', '--exp_file', type=str, help='Expert game plays')
parser.add_argument('-fs', '--mod_exp_file', type=str, help='File to save expert won game plays')

args = parser.parse_args()

sys.path.append('/Users/cusgadmin/Documents/UCB/Academics/SSastry/\
	Multi_agent_competition')
os.chdir('/Users/cusgadmin/Documents/UCB/Academics/SSastry/Multi_agent_competition/')

print(colored('Loading expert data from {}!'.format(args.exp_file),'red'))
exp_data = np.load(args.exp_file) #'actions', 'obs', 'rewards', 'episode_returns', 'episode_starts'
max_steps = 50
score_factors = np.asarray([1,1,0,1e-2])
scores = max_steps*score_factors

num_games = len(exp_data['episode_returns'])
print(colored('Total number of expert games: {}'.format(num_games),'red'))

print(colored('Picking only successful trajectories!','red'))
start_time = time.time()
a = exp_data['actions']
obs = exp_data['obs']
r = exp_data['rewards']
er = exp_data['episode_returns']
es = exp_data['episode_starts']
dist_goal = obs[:,2] #distance to goal of evader
# print(dist_goal)

#Evaders win only if at least one of them is at the goal
es_ind = np.where(es)[0] #indices of episode start
actions = []
observations = []
rewards = []
episode_returns = []
episode_starts = []

ep = 0
for i in es_ind:
	if ep == num_games - 1:
		i_ =  len(r) - 1
	if ep < num_games - 1:
		i_ = es_ind[ep+1] #next start of episode
	if dist_goal[i_-1]<=1:
		for j in range(i,i_):
			observations.append(obs[j,:])
			actions.append(a[j,:])
			rewards.append(r[j])
			episode_starts.append(es[j])
		episode_returns.append(er[ep])
	ep += 1

if len(episode_starts) > np.shape(observations)[0]:
	print('len(obs): {}; len(episode_starts): {}'.format(np.shape(observations)[0],len(episode_starts)))
	print('Removing last entry of episode_starts!')
	episode_starts = episode_starts[:-1]
end_time = time.time()
print('Modification time: {:.2f}s = {:.2f}min'.format(end_time-start_time,(end_time-start_time)/60))
actions = np.asarray(actions,dtype=int)
observations = np.asarray(observations,dtype=int)
rewards = np.asarray(rewards)
episode_returns = np.asarray(episode_returns)
episode_starts = np.asarray(episode_starts,dtype=bool)
exp_dict = {
	'actions': actions,
	'obs': observations,
	'rewards': rewards,
	'episode_returns': episode_returns,
	'episode_starts': episode_starts,
	}  # type: Dict[str, np.ndarray]
if not args.mod_exp_file:
	args.mod_exp_file = 'mod_'+args.exp_file
np.savez(args.mod_exp_file, **exp_dict)
print('Expert evader won {}/{} = {:.2f}%  games played!'.\
	format(np.sum(episode_starts),num_games,np.sum(episode_starts)*100/num_games))
