import numpy as np
import math
from numba import vectorize

class mdp(object):
    def __init__(self, init_joint_angles):
        self.joints = init_joint_angles
        self.action_set = []
        self.gamma = 0.9
        self.beta =  0.75

        for j1 in [-0.01,0,0.01]:
            for j2 in [-0.01,0,0.01]:
                for j3 in [-0.1, 0, 0.1]:
                    # for j4 in [-0.01, 0, 0.01]:
                    self.action_set.append(np.array([j1,j2,j3,0]))


        self.projection_matrix =np.matrix([[1/math.tan(np.pi/12)*744/1301, 0,0,0],
                         [0, 1/math.tan(np.pi/12), 0,0],
                         [0, 0, -(100+0.1)/(100-0.1), -1],
                         [0, 0, -2*(100*0.1)/(100-0.1), 0]])

        self.T_base_to_rcm = np.matrix([[0,-math.sin(np.pi/3),math.cos(np.pi/3),3+5*math.cos(np.pi/3)],
                         [1,0,0,0],
                         [0,math.cos(np.pi/3),math.sin(np.pi/3),6+5*math.sin(np.pi/3)],
                         [0,0,0,1]])
        self.modelViewAdjusted = np.matrix([[0, 0, 1, 0],
                                           [1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 0, 1]])
        self.modelViewMatrix = self.getModelViewMatrix(self.joints)
    def Compute_DH_Matrix(self,alpha, a , theta, d):
        DH_matrix=np.matrix([[math.cos(theta), -math.sin(theta), 0, a],
                            [math.sin(theta)*math.cos(alpha), math.cos(theta)*math.cos(alpha), -math.sin(alpha), -d*math.sin(alpha)],
                            [math.sin(theta)*math.sin(alpha), math.cos(theta)*math.sin(alpha), math.cos(alpha), d*math.cos(alpha)],
                            [0, 0, 0, 1]])
        return DH_matrix

    def forwardKinematics(self, joints):
        DH=np.matrix([[np.pi/2, 0.0000, joints[0]+np.pi/2, 0.0000],
                     [-np.pi/2, 0.0000, joints[1]-np.pi/2, 0.0000],
                     [np.pi/2, 0.0000, 0.0, joints[2]],
                     [0.0000, 0.0000, joints[3], 0]])
        FK=np.identity(4)
        i=0
        self.Compute_DH_Matrix(DH[0,0],DH[0,1],DH[0,2],DH[0,3])
        for i in range(0, 4):
            FK=FK * self.Compute_DH_Matrix(DH[i,0],DH[i,1],DH[i,2],DH[i,3])
        FK = FK * np.matrix([[0,0,-1,0],[0,-1,0,0],[-1,0,0,0],[0,0,0,1]])
        return FK
    def getModelViewMatrix(self, joints):
        modelViewMatrix = np.transpose(np.linalg.inv(self.T_base_to_rcm*self.forwardKinematics(joints)*self.modelViewAdjusted))
        return modelViewMatrix
    def update_joints(action):
        self.joints = self.joints + action

    def get_next_state(self,state,action):
        joints = self.joints + action
        modelViewMatrix = self.getModelViewMatrix(joints)
        screen_pos = np.transpose(modelViewMatrix*self.projection_matrix)*[[0],[0],[0],[1]]
        state_position = [screen_pos[0,0]/(screen_pos[2,0]),screen_pos[1,0]/(screen_pos[2,0])]
        next_state = [state_position[0], state_position[1], 0,0,0]
        # print state_position
        return next_state

    def reward(self,state):
        x = state[0]
        y = state[1]
        # r = x
        r = np.exp(-(x**2+y**2)/(0.1**2))
        return r

    def value(self, state, iter):
        if iter == 0:
            val = self.reward(state)
        else:
            val = 0
            for action in self.action_set:
                val =   val + self.action_value(state,action,iter)
        return val

    def action_value(self, state, action, iter):
        if iter == 0:
            val = self.reward(state)
        else:
            next_state = self.get_next_state(state,action)
            val = self.reward(state) + self.gamma*self.value(next_state, iter-1)
        return val

    def calculate_z(self,state, iter):
        Z=[]
        for action in self.action_set:
            Z.append(np.exp(self.beta*self.action_value(state, action, iter)))
        # print sum(Z)
        return Z

    def policy(self,state, iter):
        Z = self.calculate_z(state,iter)
        pol = Z/sum(Z)
        # print Z
        # for action in self.action_set:
        #     pol.append([np.exp(self.beta*self.action_value(state, action, iter))]/Z)
        return pol


    # def irl(mdp, trajectories, features, n_iter, step_size, k):
#
#     '''
#     mdp : defines the Markov Decision Process of what is state space, action space, transition probabilities and discount factor
#     trajectories: defines the set of trajectory([(s1,a1),..(st,at)]  sequences, where t is the length of trajectory)
#     features : defines set of feature functions for reward mapping
#     n_iter : number of iterations to tune the feature weights
#     step_size : learning rate for feature weights
#     k : softmax activation constant
#     '''
#     feature_weights = np.random(features.number);
#
#     for t in range(0,n_iter):
#         reward = feature_weigths' * features.getVector(state)

if __name__ == '__main__':
    state = np.array([1,0,0,0,0], dtype = float32)
    # action = np.array([0.1,0,0,0])
    ecm = mdp(np.array([0,0,5,0], dtype = float32))
    Policy = ecm.policy(state, 2)
    action_index = Policy.argmax()
    action = ecm.action_set[action_index]
    print action
    # print max(Policy)
    # next_state = ecm.get_next_state(state,action)

    # print value(state, 0)
    # r = ecm.reward(state)
    # print r
