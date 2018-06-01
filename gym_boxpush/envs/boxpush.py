import numpy as np
from math import pi
import gym
from PIL import Image
import os
import random

# import sys
#
# import pyglet
# from pyglet import gl

# pyglet.options['shadow_window'] = False

FPS = 300

PHYSICS_DELTA_TIME = 300
ACTIONS = [[0, 0], [1, 0], [1, 0.5], [1, -0.5], [1, 1]]

STATE_W = 64
STATE_H = 64
WINDOW_W = 64
WINDOW_H = 64


def percent_round_int(percent, x):
    return np.round(percent * x * 0.01).astype(int)


def pol2cart(rho, phi):
    x = rho * np.cos(phi * pi)
    y = rho * np.sin(phi * pi)

    # print("r:{}, phi: {}, x: {}, Y: {}".format(rho, phi, x,y))
    return np.asarray([x, y])


class TeleporterPair:
    
    class Teleporter():

        def __init__(self, x, y, width, height):
            self.geom = None
            self.transform = None

            self.occupied_after_transport = False
            self.empty = True
            self.movable = False
            self.length = np.asarray([width, height])

            self.center = np.asarray([x, y], dtype=np.float)

        def is_box_inside(self, box):
            if np.all((box.center + box.length / 2 < self.center + self.length / 2) &
                      (box.center - box.length / 2 > self.center - self.length / 2)):
                return True
            return False

    def __init__(self, x1, y1, width1, height1, x2, y2, width2, height2):
        self.ports = [TeleporterPair.Teleporter(x1, y1, width1, height1),
                      TeleporterPair.Teleporter(x2, y2, width2, height2)]


class Box():

    def __init__(self, x, y, width, height, mass=100, color=(0, 0, 0), bounciness=0.01,
                 friction=0.1, is_controlled=False, movable=True):

        self.geom = None
        self.transform = None

        self.length = np.asarray([width, height])

        self.movable = movable
        self.color = color
        self.mass = mass
        self.bounciness = bounciness
        self.friction = friction

        self.is_controlled = is_controlled

        self.center = np.asarray([x, y], dtype=np.float)
        self.vel = np.asarray([0.0, 0.0], dtype=np.float)

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
            accel = accel * 0.0004

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


class BoxPush(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self,  max_episode_length=2000):
        # self.action_space = gym.spaces.Box( np.array([0, -1]), np.array([+1, +1]), dtype=np.float32)  # force magnitude and direction
        self.action_space = gym.spaces.Discrete(5)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

        self.max_episode_length = max_episode_length

        self.pyglet = None
        self.gl = None

        self.viewer = None
        self.state_pixels_context = None
        self.human_render = False

        self.force_applied = None
        self.boxes = None
        self.teleporter_pairs = None
        self.player = None

        self.location_record = None
        self.record_write_dir = None
        self.record_write_prefix = None
        self.record_write_file_number = 0
        self.record_write_steps_recorded = 0
        self.record_write_max_steps = 2000

        self.reset_state()

    def set_record_write(self, write_dir, prefix):
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)

        self.flush_record_write()
        if not self.record_write_dir:
            self.save_heatmap_picture(os.path.join(write_dir,'level.png'))
        self.record_write_dir = write_dir
        self.record_write_prefix = prefix
        self.record_write_file_number = 0

        return True


    def flush_record_write(self, create_new_record=True):
        if self.location_record is not None and self.record_write_steps_recorded > 0:
            write_file = os.path.join(self.record_write_dir,"{}_{}".format(self.record_write_prefix,
                                                                           self.record_write_file_number))
            np.save(write_file, self.location_record[:self.record_write_steps_recorded])
            self.record_write_file_number += 1
            self.record_write_steps_recorded = 0

        if create_new_record:
            self.location_record = np.empty(shape=(self.record_write_max_steps, 2), dtype=np.float32)

    def log_location(self):
        if self.location_record is not None:
            self.location_record[self.record_write_steps_recorded] = self.player.center
            self.record_write_steps_recorded += 1

            if self.record_write_steps_recorded >= self.record_write_max_steps:
                self.flush_record_write()

    def save_heatmap_picture(self, filename):
        background_picture_np = self.debug_show_player_at_location(location_x=10000)
        im = Image.fromarray(background_picture_np)
        im.save(filename)

    def reset_state(self):
        self.force_applied = np.asarray([0.0, 0.0])
        self.boxes = []

        self.player = Box(
            x=50,
            y=85,
            width=10,
            height=10,
            mass=100,
            color=(0, 1, 0),
            friction=0.1,
            is_controlled=True
        )

        box1 = Box(
            x=45,
            y=15,
            width=10,
            height=10,
            mass=100,
            color=(.55, .2, .2),
            friction=0.1,
        )

        box2 = Box(
            x=55,
            y=15,
            width=10,
            height=10,
            mass=100,
            color=(.55, .2, .2),
            friction=0.1,
        )

        bouncy_box = Box(
            x=75,
            y=15,
            width=10,
            height=10,
            mass=100,
            color=(.4, .7, 1),
            friction=0.1,
            bounciness=0.8,
        )

        self.player.vel = player.vel + [0, -0.01]
        box1.vel = box1.vel + [0.0, 0.01]
        box2.vel = box2.vel + [0.0, 0.01]
        bouncy_box.vel = bouncy_box.vel + [0.03, 0.01]

        self.boxes.append(box1)
        self.boxes.append(box2)
        self.boxes.append(bouncy_box)
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
        self.boxes.append(Box(
            x=30,
            y=60,
            width=30,
            height=30,
            movable=False,
            friction=0.01,
        ))

        self.teleporter_pairs = []
        self.teleporter_pairs.append(TeleporterPair(
            7, 93, 15, 15,
            70, 93, 15, 15,
        ))

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

    def step(self, action):
        # print(action)
        assert self.action_space.contains(action)

        self.force_applied = pol2cart(*ACTIONS[action])

        self.log_location()


        self._handle_physics(PHYSICS_DELTA_TIME)

        state = self.render("state_pixels")

        reward = 0

        self.current_step +=1
        if self.current_step >= 2 and (self.current_step >= self.max_episode_length or
                                       random.random() < 1/self.max_episode_length):
            self.done = True

        return state, reward, self.done, {}

    def render(self, mode='human'):

        if self.viewer is None:
            import pyglet
            from pyglet import gl
            from gym_boxpush.envs import rendering

            self.pyglet = pyglet
            self.gl = gl

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)

            for tp in self.teleporter_pairs:
                for teleporter in tp.ports:
                    r = teleporter.length[0] / 2
                    t = teleporter.length[1] / 2
                    l, b = -r, -t
                    teleporter.geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    teleporter.transform = rendering.Transform()
                    teleporter.transform.set_translation(percent_round_int(teleporter.center[0], WINDOW_W),
                                                         percent_round_int(teleporter.center[1], WINDOW_H))
                    teleporter.transform.set_scale(WINDOW_W / 100, WINDOW_H / 100)
                    teleporter.geom.add_attr(teleporter.transform)
                    teleporter.geom.set_color(1, .8, .8)
                    self.viewer.add_geom(teleporter.geom)
            for box in self.boxes:
                r = box.length[0] / 2
                t = box.length[1] / 2
                l, b = -r, -t
                box.geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                box.transform = rendering.Transform()
                box.transform.set_scale(WINDOW_W / 100, WINDOW_H / 100)
                box.geom.add_attr(box.transform)
                box.geom.set_color(*box.color)
                self.viewer.add_geom(box.geom)

            # self.transform = rendering.Transform()

            # platform = pyglet.window.get_platform()
            # display = platform.get_default_display()
            # screen = display.get_default_screen()
            #
            # config = None
            # for template_config in [
            #     gl.Config(double_buffer=True, depth_size=24),
            #     gl.Config(double_buffer=True, depth_size=16),
            #     None]:
            #     try:
            #         config = screen.get_best_config(template_config)
            #         break
            #     except NoSuchConfigException:
            #         pass
            # if not config:
            #     raise NoSuchConfigException('No standard config is available.')
            #
            # if not config.is_complete():
            #     config = screen.get_best_config(config)
            #
            # self.state_pixels_context = config.create_context(gl.current_context)

        for box in self.boxes:
            box.transform.set_translation(percent_round_int(box.center[0], WINDOW_W),
                                          percent_round_int(box.center[1], WINDOW_H))

        scale = (1,1)

        # self.transform.set_scale(*scale)

        arr = None
        win = self.viewer.window
        pyglet = self.pyglet
        gl = self.gl
        self.gl.glClearColor(1, 1, 1, 1)

        win.switch_to()
        win.dispatch_events()

        # if not self.human_render and mode != 'human':
        #     win.set_visible(False)
        if mode == 'human' and not self.human_render:
            self.human_render = True
            win.set_visible(True)

        if mode == "rgb_array" or mode == "state_pixels":
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            # t = self.transform
            if mode == 'rgb_array':
                VP_W = WINDOW_W
                VP_H = WINDOW_H
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            # t.enable()
            for geom in self.viewer.geoms:
                geom.render()
            # t.disable()
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        if mode == "rgb_array" and not self.human_render:  # agent can call or not call env.render() itself when recording video.
            win.flip()

        if mode == 'human':
            win.clear()
            # t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            # t.enable()
            for geom in self.viewer.geoms:
                geom.render()
            # t.disable()
            win.flip()

        return arr

    def close(self):
        if self.viewer:
            self.viewer.close()


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self.flush_record_write(create_new_record=False)

    def __del__(self):
        self.close()
        self.flush_record_write(create_new_record=False)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.reset_state()
        self.close()
        self.viewer = None
        return self.render('state_pixels')

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



    #
    # def render(self, mode='human'):
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
    #         for tp in self.teleporter_pairs:
    #             for teleporter in tp.ports:
    #                 r = teleporter.length[0] / 2
    #                 t = teleporter.length[1] / 2
    #                 l, b = -r, -t
    #                 teleporter.geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #                 teleporter.transform = rendering.Transform()
    #                 teleporter.transform.set_translation(percent_round_int(teleporter.center[0], WINDOW_W),
    #                                                      percent_round_int(teleporter.center[1], WINDOW_H))
    #                 teleporter.geom.add_attr(teleporter.transform)
    #                 teleporter.geom.set_color(1, .8, .8)
    #                 self.viewer.add_geom(teleporter.geom)
    #         for box in self.boxes:
    #             r = box.length[0] / 2
    #             t = box.length[1] / 2
    #             l, b = -r, -t
    #             box.geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #             box.transform = rendering.Transform()
    #             box.geom.add_attr(box.transform)
    #             box.geom.set_color(*box.color)
    #             self.viewer.add_geom(box.geom)
    #
    #     # if "t" not in self.__dict__: return  # reset() not called yet
    #
    #     for box in self.boxes:
    #         box.transform.set_translation(percent_round_int(box.center[0], WINDOW_W),
    #                                       percent_round_int(box.center[1], WINDOW_H))
    #         box.transform.set_scale(WINDOW_W/100, WINDOW_H/100)
    #
    #     for tp in self.teleporter_pairs:
    #         for teleporter in tp.ports:
    #             teleporter.transform.set_scale(WINDOW_W/100, WINDOW_H/100)
    #
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')
