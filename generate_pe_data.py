import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import argparse
import sys,tty,termios
from termcolor import colored,cprint

class _Getch:
	def __call__(self):
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(3)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch

def get():
	a = 0
	inkey = _Getch()
	k = inkey()
	if k=='\x1b[A':
		a = 3 #go forward on forward arrow
	elif k=='\x1b[B':
		a = 0 #stay on downward arrow
	elif k=='\x1b[C':
		a = 2 #turn right on right arrow
	elif k=='\x1b[D':
		a = 1 #turn left on left arrow
	else:
		a = 100
	return a


parser = argparse.ArgumentParser(description=None)
parser.add_argument('-s', '--save', default=False, type=bool)
parser.add_argument('-f', '--log_file', default='pe_expertdata_small.npz', type=str)

args = parser.parse_args()

def main():

	if args.save:
		metadata = dict(title='Game')
		writer = FFMpegWriter(fps=5,metadata=metadata)

	seed = np.random.randint(2**10)
	env = gym.make('gym_pursuitevasion_small:pursuitevasion_small-v0',ep=1)

	nb_agents = len(env.agents)
	stop = False #if you want to stop playing
	num_ep = 100 #number of games
	max_steps = 100
	start_time = time.time()
	print(colored('You are player 0','red'))
	print(colored('Your input code:\n\tLeft arrow: Turn left\n\t'\
		+'Right arrow: Turn right\n\tUp arrow: Go forward\n\tDown arrow: Still'\
		+'\n\tAnything else: Stop Game','red'))
	actions = []
	observations = []
	rewards = []
	episode_returns = np.zeros((num_ep,))
	episode_starts = []
	e_win = np.zeros((num_ep,),dtype=bool)
	ng = num_ep
	for ep in range(1,num_ep+1):
		print(colored('Game number {}'.format(ep),'blue'))
		obs = env.reset(ep=ep)
		reward_sum = 0.0
		env.render(mode='human',highlight=True,ep=ep)
		if args.save:
			writer.setup(env.window.fig, "smallest_game{}.mp4".format(ep), 300)
		new_game = True
		while True:
			a = get()
			if a == 100: #stop playing
				episode_returns[ep-1] = reward_sum
				env.messages = ['Sorry to see you go! You only played {} games!'.format(ep)]
				env.render(mode='human', highlight=True, ep=ep)
				if args.save:
					writer.grab_frame()
				print(colored('\nSorry to see you go! You only played {} games!\n'.format(ep),'red'))
				stop = True
				ng = ep
				time.sleep(1.0)
				env.window.close()
			else:
				if new_game:
					episode_starts.append(True)
					new_game = False
				observations.append(obs)
				a = np.array([int(a)])
				obs, rew, done, ewi = env.step(ev_action=a)
				actions.append(a)
				rewards.append(rew)
				reward_sum += rew
				env.render(mode='human', highlight=True, ep=ep)
				if args.save:
					writer.grab_frame()
				if done:
					time.sleep(0.5)
					env.window.close()
					episode_returns[ep-1] = reward_sum
					e_win[ep-1] = ewi
					if ep == num_ep:
						stop = True
					break
				else:
					episode_starts.append(done)
			if stop:
				break
		if stop:
			break
	if args.save:
		writer.finish()
	if len(episode_starts) > np.shape(observations)[0]:
		print('len(obs): {}; len(episode_starts): {}'.format(np.shape(observations)[0],len(episode_starts)))
		print('Removing last entry of episode_starts!')
		episode_starts = episode_starts[:-1]
	end_time = time.time()
	print('Playing time: {:.2f}s = {:.2f}min'.format(end_time-start_time,(end_time-start_time)/60))
	actions = np.asarray(actions,dtype=int)
	observations = np.asarray(observations,dtype=int)
	rewards = np.asarray(rewards)
	exp_dict = {
			'actions': actions,
			'obs': observations,
			'rewards': rewards,
			'episode_returns': episode_returns,
			'episode_starts': episode_starts,
			'evader_wins': e_win,
			'number_games': ng
			}  # type: Dict[str, np.ndarray]
	np.savez(args.log_file, **exp_dict)
	print('Expert evader won {}/{} games played!'.format(np.sum(e_win),ng))

if __name__ == "__main__":
	main()