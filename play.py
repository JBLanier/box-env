import multiprocessing

import cv2
import numpy as np

import gym
import pygame
import gym_boxpush
from gym_boxpush.envs.boxpush import BoxPush
from gym_boxpush.envs.boxpushsimple import BoxPushSimple


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def render_loop():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    env = gym.make('boxpush-v0')
    env.reset()
    # env.set_record_write("cool","1")

    frames = 0
    while True:

        # action = env.action_space.sample()
        pygame.event.get()
        x = joystick.get_axis(0)
        y = joystick.get_axis(1) * -1

        deadzone = 0.2

        if abs(x) < deadzone:
            x = 0
        if abs(y) < deadzone:
            y = 0

        action = np.asarray([x, y])

        env.render("human")
        step_results = env.step(action)
        # print(step_results)
        obs, reward, done, info = step_results
        # cv2.imshow("state_pixels", frame[:, :, ::-1])
        cv2.waitKey(1)

        if done or info['is_success']:
            env.reset()
            exit(0)

        frames += 1
        print(frames)


    env.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    for i in range(1):
        p = multiprocessing.Process(target=render_loop)
        print("1")
        p.start()




#
#
# p.init()
# reward = 0.0
#
# pygame.joystick.init()
#
# joystick = pygame.joystick.Joystick(0)
# joystick.init()
#
# test_image_shown = False
#
# while True:
#     frame_start_time = time.time()
#
#     if p.game_over():
#         p.reset_game()
#
#     x = joystick.get_axis(0)
#     y = joystick.get_axis(1)
#
#     # print("X,Y: ({},{})".format(x,y))
#     r, phi = cart2pol(x, y)
#
#     if abs(r) < 0.12:
#         r = 0
#     else:
#         r = min(1, r)
#         r = r + 0.12 * math.log(r)
#         r = r * 0.30
#
#     # print("r,phi: ({},{})".format(r,phi))
#
#     observation = p.getScreenRGB()
#
#     # if not test_image_shown:
#     # 	cv2.imshow("obs", observation)
#     # 	cv2.waitKey(1)
#
#     action = (0, (r, phi))
#     reward = p.act(action)
#
#     time.sleep((15.4444444 - (time.time() - frame_start_time)) * 0.001)
