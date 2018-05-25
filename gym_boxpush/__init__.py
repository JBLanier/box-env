from gym.envs.registration import register

register(
    id='boxpush-v0',
    entry_point='gym_boxpush.envs:BoxPush',
)

register(
    id='boxpushsimple-v0',
    entry_point='gym_boxpush.envs:BoxPushSimple',
)

register(
    id='boxpushmaze-v0',
    entry_point='gym_boxpush.envs:BoxPushMaze',
)
