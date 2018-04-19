from boxpush import BoxPush
from continousple import ContinousPLE
import pygame
import math
import numpy as np
import cv2

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

game = BoxPush()
p = ContinousPLE(game, fps=30, display_screen=True, add_noop_action=False)
print(p.getActionSet())

p.init()
reward = 0.0

pygame.joystick.init()


joystick = pygame.joystick.Joystick(0)
joystick.init()

test_image_shown = False

while True:
	if p.game_over():
	       p.reset_game()

	x = joystick.get_axis(0)
	y = joystick.get_axis(1)

	# print("X,Y: ({},{})".format(x,y))
	r,phi = cart2pol(x,y)

	if abs(r) < 0.12:
		r = 0
	else:
		r = min(1,r)
		r = r + 0.12 * math.log(r)
		r = r * 0.30


	# print("r,phi: ({},{})".format(r,phi))


	observation = p.getScreenRGB()

	if not test_image_shown:
		cv2.imshow("obs", observation)
		cv2.waitKey(1)


	action = (0, (r, phi))
	reward = p.act(action)