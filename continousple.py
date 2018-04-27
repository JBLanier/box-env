from ple import PLE

class ContinousPLE(PLE):

	def _oneStepAct(self, action):
	    """
	    Performs an action on the game. Checks if the game is over.
	    """
	    if self.game_over():
	        return 0.0

	    self._setAction(action)
	    for i in range(self.num_steps):
	        time_elapsed = self._tick()
	        self.game.step(time_elapsed)
	        self._draw_frame()

	    self.frame_count += self.num_steps

	    return self._getReward()

	def quit(self):
		self.game.quit()

