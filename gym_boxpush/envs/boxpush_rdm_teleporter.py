from gym_boxpush.envs.boxpush import *
import numpy as np


class BoxPushRdmTeleporter(BoxPush):

    def reset_state(self):

        self.force_applied = np.asarray([0.0, 0.0])
        self.boxes = []

        self.player = Box(
            x=20,
            y=30,
            width=10,
            height=10,
            mass=100,
            color=(0, 1, 0),
            friction=0.1,
            is_controlled=True,
        )

        # box1 = Box(
        #     x=45,
        #     y=15,
        #     width=10,
        #     height=10,
        #     mass=100,
        #     color=(.55, .2, .2),
        #     friction=0.1,
        # )
        #
        # box2 = Box(
        #     x=55,
        #     y=15,
        #     width=10,
        #     height=10,
        #     mass=100,
        #     color=(.55, .2, .2),
        #     friction=0.1,
        # )
        #
        # bouncy_box = Box(
        #     x=75,
        #     y=15,
        #     width=10,
        #     height=10,
        #     mass=100,
        #     color=(.4, .7, 1),
        #     friction=0.1,
        #     bounciness=0.8,
        # )

        self.player.vel = self.player.vel + [0, 0]
        # box1.vel = box1.vel + [0.0, 0.01]
        # box2.vel = box2.vel + [0.0, 0.01]
        # bouncy_box.vel = bouncy_box.vel + [0.03, 0.01]

        # self.boxes.append(box1)
        # self.boxes.append(box2)
        # self.boxes.append(bouncy_box)
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
        # self.boxes.append(Box(
        #     x=20,
        #     y=60,
        #     width=49,
        #     height=10,
        #     movable=False,
        #     friction=0.01,
        # ))
        #
        # self.boxes.append(Box(
        #     x=80,
        #     y=60,
        #     width=48,
        #     height=10,
        #     movable=False,
        #     friction=0.01,
        # ))
        #
        # self.boxes.append(Box(
        #     x=61,
        #     y=74,
        #     width=10,
        #     height=30,
        #     movable=False,
        #     friction=0.01,
        # ))
        #
        # self.boxes.append(Box(
        #     x=20,
        #     y=90,
        #     width=10,
        #     height=30,
        #     movable=False,
        #     friction=0.01,
        # ))
        #
        # self.boxes.append(Box(
        #     x=35,
        #     y=20,
        #     width=10,
        #     height=40,
        #     movable=False,
        #     friction=0.01,
        # ))
        #
        # self.boxes.append(Box(
        #     x=60,
        #     y=35,
        #     width=40,
        #     height=10,
        #     movable=False,
        #     friction=0.01,
        # ))

        self.teleporter_pairs = []
        # self.teleporter_pairs.append(TeleporterPair(
        #     7, 93, 15, 15,
        #     70, 93, 15, 15,
        # ))

        self.random_teleporters = []
        self.random_teleporters.append(RandomTeleporter(
            50,50,10,10
        ))

        is_valid_goal = False

        while not is_valid_goal:

            self.goal = self._sample_goal()

            self.goal_rect = BackgroundRect(
                x=self.goal[0],
                y=self.goal[1],
                width=self.goal_distance_threshold,
                height=self.goal_distance_threshold,
                color=(1.0, .5, .5)
            )

            is_goal_colliding_with_level = False
            for box in self.boxes:
                if self._check_rect_collision(
                        self.goal_rect.center, self.goal_rect.length/2, True,
                        box.center, box.length/2, True
                ):
                    is_goal_colliding_with_level = True

            if not is_goal_colliding_with_level:
                is_valid_goal = True

        self.background_rects = []
        self.background_rects.append(self.goal_rect)
