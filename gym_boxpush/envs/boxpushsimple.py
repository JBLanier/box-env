import numpy as np
from math import pi
from gym_boxpush.envs.boxpush import *

class BoxPushSimple(BoxPush):

    def reset_state(self):
        self.force_applied = np.asarray([0.0, 0.0])

        self.boxes = []
        self.teleporter_pairs = []

        self.player = Box(
            x=50,
            y=50,
            width=20,
            height=98,
            mass=100,
            color=(0, 1, 0),
            friction=0.1,
            is_controlled=True,
            bounciness=0
        )

        self.player.vel = self.player.vel + [0, 0]

        self.boxes.append(self.player)

        # Add walls
        self.boxes.append(Box(
            x=0,
            y=50,
            width=2,
            height=100,
            movable=False,
            friction=0.01,
        ))
        self.boxes.append(Box(
            x=100,
            y=50,
            width=2,
            height=100,
            movable=False,
            friction=0.01,
        ))
        self.boxes.append(Box(
            x=50,
            y=0,
            width=100,
            height=2,
            movable=False,
            friction=0.01,
        ))
        self.boxes.append(Box(
            x=50,
            y=100,
            width=100,
            height=2,
            movable=False,
            friction=0.01,
        ))

    def step(self, action):
        # print("ACTION: {}".format(action))
        assert self.action_space.contains(action)

        self.force_applied = pol2cart(*action)

        self.log_location()


        self._handle_physics(PHYSICS_DELTA_TIME * 6)
        self._handle_physics(PHYSICS_DELTA_TIME * 6)

        state = self.render("state_pixels")
        reward = 0

        done = False

        return state, reward, done, {}

