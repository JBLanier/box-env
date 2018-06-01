import numpy as np
from math import pi
from gym_boxpush.envs.boxpush import *

class BoxPushSimpleColorChange(BoxPush):

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
        if action[0] == 1:
            self.player.color = (1, 0, 0)
        else:
            self.player.color = (0, 1, 0)

        return super(BoxPushSimpleColorChange, self).step(action)

    def render(self, mode='human'):
        if self.viewer is not None:
            for box in self.boxes:
                box.geom.set_color(*box.color)
        return super(BoxPushSimpleColorChange, self).render(mode=mode)
