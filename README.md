# Learning for Collaborative-Competitive games

In this project, we formalize a collaborative-competitive game with competition between two teams and collaboration between members of each team. The players from the first team seek to reach their individual goals while avoiding capture by the second team. And, the second team seeks to capture all players in the first team. The competition between the two teams arises from the fact that the second team seeks to capture the first team's players, while the latter seek to reach their individual goals while avoiding capture. The players within each team can collaborate with each other in order to achieve their individual and team goals. The ground rules for game play are cast in the form of a Markov Decision Process with the goal of learning optimal game play strategies for members of the first team. We collect expert trajectories from human experts that played the game, and use this data to learn similar game play strategies designed to ensure that the first team wins the game. A recent approach for imitation learning called Generative Adversarial Imitation Learning (GAIL) is examined in the context of these collaborative-competitive games. The results of running GAIL on expert data are contrasted against those got from state of the art algorithms from the domain of imitation learning as well as (forward) reinforcement learning. We see that the learnt policies resemble in logic to those used by human experts in playing the game, while being successful in about 70\% of new games played. This success rate is very close to that of the human experts playing the game. 

# Reference paper
Generative Adversarial Imitation Learning https://arxiv.org/abs/1606.03476

# Package requirements


# Report of findings 


To cite this repository:
```python
@misc{collaborative-competitive-games,
  author = {Dwarakanath, Kshama},
  title = {Learning for Collaborative-Competitive games},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{}},
}
```
