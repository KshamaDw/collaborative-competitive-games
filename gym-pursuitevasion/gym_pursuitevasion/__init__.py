from gym.envs.registration import register

register(
    id='pursuitevasion-v0',
    entry_point='gym_pursuitevasion.envs:PursuitEvasionEnv',
)