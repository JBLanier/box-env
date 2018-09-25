from gym.envs.registration import register

register(
    id='BoxPush-v0',
    entry_point='gym_boxpush.envs:BoxPush',
)

register(
    id='BoxPushRdmTeleporter-v0',
    entry_point='gym_boxpush.envs:BoxPushRdmTeleporter',
)
