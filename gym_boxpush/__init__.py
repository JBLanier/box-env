from gym.envs.registration import register

register(
    id='boxpush-v0',
    entry_point='gym_boxpush.envs:BoxPush',
)
