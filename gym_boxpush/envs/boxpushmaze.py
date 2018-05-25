import numpy as np
from math import pi
from gym_boxpush.envs.boxpush import *

class BoxPushMaze(BoxPush):

    def reset_state(self):
        self.force_applied = np.asarray([0.0, 0.0])

        self.boxes = []
        self.teleporter_pairs = []

        self.player = Box(
            x=85,
            y=15,
            width=20,
            height=20,
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
            width=10,
            height=100,
            movable=False,
            friction=0.01,
        ))
        self.boxes.append(Box(
            x=100,
            y=50,
            width=10,
            height=100,
            movable=False,
            friction=0.01,
        ))
        self.boxes.append(Box(
            x=50,
            y=0,
            width=100,
            height=10,
            movable=False,
            friction=0.01,
        ))
        self.boxes.append(Box(
            x=50,
            y=100,
            width=100,
            height=10,
            movable=False,
            friction=0.01,
        ))

        self.boxes.append(Box(
            x=37.5,
            y=50,
            width=20,
            height=50,
            movable=False,
            friction=0.01,
        ))

        self.boxes.append(Box(
            x=5,
            y=62.5,
            width=4,
            height=75,
            movable=False,
            friction=0.01,
        ))

        self.boxes.append(Box(
            x=60,
            y=65,
            width=27.5,
            height=20,
            movable=False,
            friction=0.01,
        ))

        self.boxes.append(Box(
            x=62.5,
            y=30,
            width=68,
            height=10,
            movable=False,
            friction=0.01,
        ))

    def step(self, action):
        # print("ACTION: {}".format(action))
        assert self.action_space.contains(action)

        self.force_applied = pol2cart(*action)

        self.log_location()


        self._handle_physics(PHYSICS_DELTA_TIME * 1.5)

        state = self.render("state_pixels")
        reward = 0

        done = False

        return state, reward, done, {}

    def debug_show_player_at_location(self, location_x):
        """
        Returns rendering of player at specified location, does not affect actual game state.
        :param location_x: (float -1 to 1) show player at location_x
        :return: "state_pixels" rendering with player at location_x
        """
        old_center = np.copy(self.player.center)
        self.player.center[0] = (location_x + 1) * 50
        frame = self.render("state_pixels")
        self.player.center = old_center
        return frame

    def debug_get_player_location(self):
        return (self.player.center[0] - 50) / 50