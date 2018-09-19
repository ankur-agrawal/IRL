import numpy as np
import matplotlib as plt
import pandas as pd
class cameraMDP(object):

  def __init__(self, gamma):

      self.state = np.zeros([4,1])
      self.action = np.zeros([4,1])
      self.next_state = np.zeros([4,1])
      self.gamma = gamma
      # self.tau = []

  def transition(self,state, action):
      next_state = state + action
      return next_state

  def generate_trajectories(self,data):
      

if __name__=='__main__':

    camera = cameraMDP(0.1)

    camera.state = np.array([0,0.1,0.2,0])
    camera.action = np.array([0,0,0.1,0])

    data = pd.read_csv("data/01/01_01_relabeled.csv", delim_whitespace = True, header=None)
    camera.generate_trajectories(data)
    print camera.tau
    # camera.next_state = camera.transition(camera.state, camera.action)
    # print camera.next_state
