import sys
import pygame

from ple.games import base
import numpy as np

CONTINUOUS_ACTION = pygame.USEREVENT + 1


def percent_round_int(percent, x):
    return np.round(percent * x * 0.01).astype(int)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    # print("r:{}, phi: {}, x: {}, Y: {}".format(rho, phi, x,y))
    return np.asarray([x, y])


class TeleporterPair:
    
    class Teleporter(pygame.sprite.Sprite):

        def __init__(self, x, y, width, height, SCREEN_WIDTH, SCREEN_HEIGHT):
            self.occupied_after_transport = False
            self.empty = True
            self.movable = False
            self.SCREEN_WIDTH = SCREEN_WIDTH
            self.SCREEN_HEIGHT = SCREEN_HEIGHT
            self.length = np.asarray([width, height])

            pygame.sprite.Sprite.__init__(self)

            self.image = pygame.Surface((
                percent_round_int(SCREEN_WIDTH, width),
                percent_round_int(SCREEN_HEIGHT, height)))
            self.image.fill((240, 150, 150, 0))
            self.image.set_colorkey((255, 195, 195))

            self.center = np.asarray([x, y], dtype=np.float)

            self.rect = self.image.get_rect()
            self.rect.center = (x, y)

            pygame.draw.rect(
                self.image,
                (255, 180, 180),
                (percent_round_int(SCREEN_WIDTH, 1), percent_round_int(SCREEN_HEIGHT, 1),
                 percent_round_int(SCREEN_WIDTH, width - 2),
                 percent_round_int(SCREEN_HEIGHT, height - 2)),
                0
            )

        def draw(self, screen):
            screen.blit(self.image,
                        (percent_round_int(self.SCREEN_WIDTH, self.rect.center[0] - self.length[0] / 2),
                         percent_round_int(self.SCREEN_HEIGHT, self.rect.center[1] - self.length[1] / 2)))

        def is_box_inside(self, box):
            if np.all((box.center + box.length / 2 < self.center + self.length / 2) &
                      (box.center - box.length / 2 > self.center - self.length / 2)):
                return True
            return False

    def __init__(self, x1, y1, width1, height1, x2, y2, width2, height2, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.ports = [TeleporterPair.Teleporter(x1, y1, width1, height1, SCREEN_WIDTH, SCREEN_HEIGHT),
                      TeleporterPair.Teleporter(x2, y2, width2, height2, SCREEN_WIDTH, SCREEN_HEIGHT)]

    def draw(self, screen):
        for port in self.ports:
            port.draw(screen)


class Box(pygame.sprite.Sprite):

    def __init__(self, x, y, width, height, SCREEN_WIDTH, SCREEN_HEIGHT,
                 mass=100, color=(0, 0, 0), bounciness=0.01, friction=0.1, is_controlled=False, movable=True):

        self.length = np.asarray([width, height])

        self.movable = movable

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.mass = mass
        self.bounciness = bounciness
        self.friction = friction

        self.is_controlled = is_controlled
        pygame.sprite.Sprite.__init__(self)

        image = pygame.Surface((
            percent_round_int(SCREEN_WIDTH, width),
            percent_round_int(SCREEN_HEIGHT, height)))
        image.fill(color)
        image.set_colorkey((255, 255, 255))

        # pygame.draw.rect(
        #     image,
        #     color,
        #     (0, 0,
        #         percent_round_int(SCREEN_WIDTH, width),
        #         percent_round_int(SCREEN_HEIGHT, height)),
        #     0
        # )

        self.image = image
        self.rect = self.image.get_rect()
        self.center = np.asarray([x, y], dtype=np.float)
        self.vel = np.asarray([0.0, 0.0], dtype=np.float)

        self.rect = self.image.get_rect()
        self.rect.center = self.center

    def get_update(self, dt, axis, force_applied=None, gravitational_force=None):

        if self.movable:
            # print("force: {}, dt: {}".format(force_applied, dt))
            new_vel = self.vel
            force = 0

            if self.is_controlled and force_applied is not None:
                force += force_applied[axis]
            if gravitational_force is not None:
                force += gravitational_force[axis]

            accel = force / self.mass
            accel = accel * 0.01

            # print("accel: {}".format(accel))
            new_vel[axis] = new_vel[axis] + (accel * dt)

            new_center = np.copy(self.center)
            new_center[axis] = new_center[axis] + (self.vel[axis] * dt)

            # print("update: new_center: {}, old_center: {}, change: {}, new_vel: {}, change: {}".format(new_center, self.center, new_center-self.center, new_vel, new_vel - self.vel))
            return new_vel, new_center
        else:
            return [0, 0], self.center

        # print("vel: {}, center: {}".format(self.vel, self.center))

    def apply_update(self, update):
        if self.movable:
            self.vel = update[0]
            self.center = update[1]
            self.rect.center = self.center

    def draw(self, screen):
        screen.blit(self.image,
                    (percent_round_int(self.SCREEN_WIDTH, self.center[0] - self.length[0] / 2),
                     percent_round_int(self.SCREEN_HEIGHT, self.center[1] - self.length[1] / 2)))


class BoxPush(base.PyGameWrapper):
    """
    Based on `Eder Santana`_'s game idea.

    .. _`Eder Santana`: https://github.com/EderSantana

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    init_lives : int (default: 3)
        The number lives the agent has.

    """

    def __init__(self, display_width=500, display_height=500):
        actions = {
            "apply_force": (0, [(-1, 1), (0, 1)])
        }
        base.PyGameWrapper.__init__(self, display_width, display_height, actions=actions)

        self.force_applied = np.asarray([0.0, 0.0])

    def quit(self):
        pygame.quit()

    def _setAction(self, action, last_action=None):
        """
        Pushes the action to the pygame event queue.
        """
        if action is None:
            action = self.NOOP

        if isinstance(action, tuple):
            # action is continous
            action_event = pygame.event.Event(CONTINUOUS_ACTION, {"value": action})
            pygame.event.post(action_event)
        else:
            print("expecting a tuple of the form (action, [(value,value..)] for continous action space")

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == CONTINUOUS_ACTION:
                magnitude, degree = event.value[1]
                self.force_applied = pol2cart(magnitude, degree)

    @staticmethod
    def _check_rect_collision(center1, half_length1, movable1, center2, half_length2, movable2):

        if not movable1 and not movable2:
            # don't consider intersecting immobile objects as collisions
            return False

        return np.all(
            (center1 + half_length1 >= center2 - half_length2 + 0.0001) &
            (center1 - half_length1 + 0.0001 <= center2 + half_length2)
        )

    def _handle_collision(self, update_a, update_b, box_a, box_b, axis):
        # returns true is there was a collision

        if self._check_rect_collision(update_a[1], box_a.length / 2, box_a.movable,
                                      update_b[1], box_b.length / 2, box_b.movable):
            # print("COLLISION, index: {}".format(axis))

            d = update_a[1][axis] - update_b[1][axis]
            vertical_overlap = ((box_a.length[axis] + box_b.length[axis]) / 2 - abs(d)) * d / abs(d)

            # print("overlap: {}".format(vertical_overlap))

            im_a = 1 / box_a.mass if box_a.movable else 0
            im_b = 1 / box_b.mass if box_b.movable else 0

            update_i_adjust = update_a[1][axis] + vertical_overlap * im_a / (im_a + im_b)
            update_a[1][axis] = update_i_adjust
            # print("location adjust for i: {}".format(update_i_adjust))

            update_j_adjust = update_b[1][axis] - vertical_overlap * im_b / (im_a + im_b)
            update_b[1][axis] = update_j_adjust
            # print("location adjust for j: {}".format(update_j_adjust))

            velocity_diff = (update_a[0][axis] - update_b[0][axis]) * d / abs(d)

            if velocity_diff <= 0.0:
                impulse = -d / abs(d) * velocity_diff / (im_a + im_b)

                # print("impulse: {}".format(impulse))
                #
                # print("old update vel: {}, new update vel: {}".format(update_a[0][axis],
                #                                                       update_a[0][axis] + impulse * im_p))
                update_a[0][axis] = update_a[0][axis] + impulse * (1 + box_a.bounciness) * im_a
                update_b[0][axis] = update_b[0][axis] - impulse * (1 + box_b.bounciness) * im_b

                relative_tangential_velocity = update_a[0][(axis + 1) % 2] - update_b[0][(axis + 1) % 2]
                # print("relative_tangential_velocity: {}".format(relative_tangential_velocity))
                update_a[0][(axis + 1) % 2] = update_a[0][(axis + 1) % 2] - abs(impulse) * (
                            box_a.friction + box_b.friction) * relative_tangential_velocity
                update_b[0][(axis + 1) % 2] = update_b[0][(axis + 1) % 2] + abs(impulse) * (
                            box_a.friction + box_b.friction) * relative_tangential_velocity

            return True

        return False

    def _handle_physics(self, dt):

        for k in range(0, 2):

            updates = list(
                map(lambda x: x.get_update(dt, axis=k, force_applied=self.force_applied, gravitational_force=None),
                    self.boxes))

            done = False
            while not done:
                done = True
                for i in range(0, len(updates)):
                    for j in range(i + 1, len(updates)):

                        if self._handle_collision(updates[i], updates[j], self.boxes[i], self.boxes[j], axis=k):
                            done = False

            for i in range(len(updates)):
                self.boxes[i].apply_update(updates[i])

            # print("update: {}".format(updates[0]))

        for teleporter in self.teleporter_pairs:
            for p in range(2):
                if not teleporter.ports[p].empty:
                    empty = True
                    for box in self.boxes:
                        if self._check_rect_collision(box.center, box.length / 2, box.movable,
                                                      teleporter.ports[p].center, teleporter.ports[p].length / 2,
                                                      teleporter.ports[p].movable):
                            empty = False
                            break

                    teleporter.ports[p].empty = empty
                if teleporter.ports[p].occupied_after_transport:
                    teleporter.ports[p].occupied_after_transport = not teleporter.ports[p].empty

        for teleporter in self.teleporter_pairs:
            for p in range(2):
                if not teleporter.ports[p].occupied_after_transport and teleporter.ports[(p + 1) % 2].empty:
                    for box in self.boxes:
                        if self._check_rect_collision(box.center, box.length / 2, box.movable,
                                                      teleporter.ports[p].center, teleporter.ports[p].length / 2,
                                                      teleporter.ports[p].movable):
                            update = (box.vel, teleporter.ports[(p + 1) % 2].center)
                            # print("Teleporter update: {}".format(update))
                            box.apply_update(update)
                            teleporter.ports[(p + 1) % 2].occupied_after_transport = True
                            teleporter.ports[(p + 1) % 2].empty = False
                            break

    def init(self):
        self.boxes = []

        player = Box(
            x=50,
            y=15,
            width=10,
            height=10,
            mass=100,
            color=(0, 255, 0),
            friction=0.1,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height,
            is_controlled=True
        )

        box1 = Box(
            x=45,
            y=83,
            width=10,
            height=10,
            mass=100,
            color=(140, 50, 50),
            friction=0.1,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height
        )

        box2 = Box(
            x=55,
            y=85,
            width=10,
            height=10,
            mass=100,
            color=(140, 50, 50),
            friction=0.1,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height
        )

        bouncy_box = Box(
            x=75,
            y=85,
            width=10,
            height=10,
            mass=100,
            color=(100, 180, 250),
            friction=0.1,
            bounciness=0.8,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height
        )

        player.vel = player.vel + [0, 0.01]
        box1.vel = box1.vel + [0.0, -0.01]
        box2.vel = box2.vel + [0.0, -0.01]
        bouncy_box.vel = bouncy_box.vel + [0.03, -0.01]

        self.boxes.append(box1)
        self.boxes.append(box2)
        self.boxes.append(bouncy_box)
        self.boxes.append(player)

        # Add walls
        self.boxes.append(Box(
            x=0,
            y=50,
            width=2,
            height=100,
            movable=False,
            friction=0.01,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height
        ))
        self.boxes.append(Box(
            x=100,
            y=50,
            width=2,
            height=100,
            movable=False,
            friction=0.01,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height
        ))
        self.boxes.append(Box(
            x=50,
            y=0,
            width=100,
            height=2,
            movable=False,
            friction=0.01,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height
        ))
        self.boxes.append(Box(
            x=50,
            y=100,
            width=100,
            height=2,
            movable=False,
            friction=0.01,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height
        ))
        self.boxes.append(Box(
            x=30,
            y=60,
            width=30,
            height=30,
            movable=False,
            friction=0.01,
            SCREEN_WIDTH=self.width,
            SCREEN_HEIGHT=self.height
        ))

        self.teleporter_pairs = []
        self.teleporter_pairs.append(TeleporterPair(
            7, 7, 15, 15,
            70, 7, 15, 15,
            self.width, self.height
        ))

    def getGameState(self):
        raise NotImplementedError

    def getScore(self):
        return 0

    def game_over(self):
        return False

    def step(self, dt):
        self.screen.fill((255, 255, 255))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        self._handle_physics(dt)

        for teleporters in self.teleporter_pairs:
            teleporters.draw(self.screen)

        for box in self.boxes:
            box.draw(self.screen)

        if self.lives == 0:
            self.score += self.rewards["loss"]

#
# if __name__ == "__main__":
#
#     pygame.init()
#     game = BoxPush(width=256, height=256)
#     game.rng = np.random.RandomState(24)
#     game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
#     game.clock = pygame.time.Clock()
#     game.init()
#
#     while True:
#         dt = game.clock.tick_busy_loop(30)
#         if game.game_over():
#             game.reset()
#
#         game.step(dt)
#         pygame.display.update()
