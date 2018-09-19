import numpy as np
import math
import pandas as pd


projection_matrix =np.matrix([[1/math.tan(np.pi/12)*744/1301, 0,0,0],
                 [0, 1/math.tan(np.pi/12), 0,0],
                 [0, 0, -(100+0.1)/(100-0.1), -1],
                 [0, 0, -2*(100*0.1)/(100-0.1), 0]])

T_base_to_rcm = np.matrix([[0,-math.sin(np.pi/3),math.cos(np.pi/3),3+5*math.cos(np.pi/3)],
                 [1,0,0,0],
                 [0,math.cos(np.pi/3),math.sin(np.pi/3),6+5*math.sin(np.pi/3)],
                 [0,0,0,1]])
modelViewAdjusted = np.matrix([[0, 0, 1, 0],
                                   [1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]])

def Compute_DH_Matrix(alpha, a , theta, d):
    DH_matrix=np.matrix([[math.cos(theta), -math.sin(theta), 0, a],
                        [math.sin(theta)*math.cos(alpha), math.cos(theta)*math.cos(alpha), -math.sin(alpha), -d*math.sin(alpha)],
                        [math.sin(theta)*math.sin(alpha), math.cos(theta)*math.sin(alpha), math.cos(alpha), d*math.cos(alpha)],
                        [0, 0, 0, 1]])
    return DH_matrix

def forwardKinematics(joints):
    DH=np.matrix([[np.pi/2, 0.0000, joints[0]+np.pi/2, 0.0000],
                 [-np.pi/2, 0.0000, joints[1]-np.pi/2, 0.0000],
                 [np.pi/2, 0.0000, 0.0, joints[2]],
                 [0.0000, 0.0000, joints[3], 0]])
    FK=np.identity(4)
    i=0
    Compute_DH_Matrix(DH[0,0],DH[0,1],DH[0,2],DH[0,3])
    for i in range(0, 4):
        FK=FK * Compute_DH_Matrix(DH[i,0],DH[i,1],DH[i,2],DH[i,3])
    FK = FK * np.matrix([[0,0,-1,0],[0,-1,0,0],[-1,0,0,0],[0,0,0,1]])
    return FK
def getModelViewMatrix(joints):
    modelViewMatrix = np.transpose(np.linalg.inv(T_base_to_rcm*forwardKinematics(joints)*modelViewAdjusted))
    return modelViewMatrix

folders = ['01']
files = ['01','02','03','04','05']
# files = ['01']

state_class1 = []
state_class2 = []
state_class3 = []
state_class4 = []
action_class1 = []
action_class2 = []
action_class3 = []
action_class4 = []

for folder in folders:
    for file in files:
        print 'Reading '+folder + '_' + file
        # data = pd.read_csv("data/01/01_01_relabeled.csv", delim_whitespace = True, header=None)
        data = pd.read_csv("data/" + folder + '/' + folder + '_' + file + '_relabeled.csv', delim_whitespace = True, header=None)
        print data.values.shape
        data_trimmed = data[data.values[:,50]>0]
        print data_trimmed.shape

        min_time = int(round(100*min(data_trimmed.values[:,1]),0))
        max_time = int(round(100*max(data_trimmed.values[:,1]),0))

        index = []

        for i in range(0, max_time-min_time):
            index.append(np.argmin(abs(data_trimmed.values[:,1]-0.01*(min_time+i))))
        # print index
        data_timed = data_trimmed.values[index]
        joints = data_timed[:,2:6]
        joints[:,0] = np.round(joints[:,0], 2)
        joints[:,1] = np.round(joints[:,1], 2)
        joints[:,2] = np.round(joints[:,2], 1)
        joints[:,3] = np.round(joints[:,3], 2)
        gripper_pos = np.empty([data_timed.shape[0],4])
        gripper_velocity = np.empty([data_timed.shape[0],4])
        gripper_pos[:,0:3] = data_timed[:,37:40]
        gripper_pos[:,3] = 1
        screen_pos = np.empty([data_timed.shape[0],4])
        screen_velocity = np.empty([data_timed.shape[0],4])
        state_pos = np.empty([data_timed.shape[0],2])
        state_velocity = np.empty([data_timed.shape[0],2])
        state_speed = np.empty([data_timed.shape[0],1])
        state_theta = np.empty([data_timed.shape[0],1])
        # state = np.empty([data_timed.shape[0],4])
        action = np.empty([data_timed.shape[0],4])

        for iter in range(0,data_timed.shape[0]):
            modelViewMatrix = getModelViewMatrix(joints[iter,:])
            v = np.transpose(modelViewMatrix*projection_matrix)*np.transpose([gripper_pos[iter,:]])
            screen_pos[iter, :] = v.transpose()
            state_pos[iter,:] = screen_pos[iter,[0,1]]/screen_pos[iter,3]
            if (iter==0):
                gripper_velocity[iter,:] = np.array([0,0,0,0])
            else:
                gripper_velocity[iter,:] = gripper_pos[iter,:]- gripper_pos[iter-1,:]
            v = np.transpose(modelViewMatrix*projection_matrix)*np.transpose([gripper_velocity[iter,:]])
            screen_velocity[iter,:] = v.transpose()

            state_velocity[iter,:] = screen_velocity[iter,[0,1]]/screen_pos[iter,3] - screen_velocity[iter, 3]/(screen_pos[iter, 3]**2)*screen_pos[iter,[0,1]]
            state_speed[iter,:] = np.linalg.norm(state_velocity[iter,:])
            state_theta[iter,:] = math.atan2(state_velocity[iter, 1], state_velocity[iter,0])
            if (iter == data_timed.shape[0]-1):
                action[iter,:] = np.array([0,0,0,0])
            else:
                action[iter,:] = joints[iter+1,:] - joints[iter,:]
            # print state_velocity[iter, :]
        state = np.concatenate((state_pos, state_theta, state_speed), axis=1)
        print action.shape

        # data_changed = data_timed[np.vstack(([data_timed[:-1,50]!=data_timed[1:,50]],False))]
        change_condition = data_timed[:-1,50]!=data_timed[1:,50]
        change_condition = np.atleast_2d(change_condition).transpose()
        change_condition = np.vstack((change_condition, np.array([False])))
        indices = np.where(change_condition ==True)
        data_trajectory = np.array_split(data_timed, indices[0]+1)
        state_split = np.array_split(state, indices[0]+1)
        action_split = np.array_split(action, indices[0]+1)
        flag = False

        for i in range(0,len(data_trajectory)):
            if min(data_trajectory[i][:,50]) == 1:
                flag = not flag
                if max(data_trajectory[i][:,50]) ==1:
                    flag = not flag
                    state_class1.append(state_split[i])
                    action_class1.append(action_split[i])
            if min(data_trajectory[i][:,50]) == 2:
                flag = not flag
                if max(data_trajectory[i][:,50]) ==2:
                    flag = not flag
                    state_class2.append(state_split[i])
                    action_class2.append(action_split[i])
            if min(data_trajectory[i][:,50]) == 3:
                flag = not flag
                if max(data_trajectory[i][:,50]) ==3:
                    flag = not flag
                    state_class3.append(state_split[i])
                    action_class3.append(action_split[i])
            if min(data_trajectory[i][:,50]) == 4:
                flag = not flag
                if max(data_trajectory[i][:,50]) ==4:
                    flag = not flag
                    state_class4.append(state_split[i])
                    action_class4.append(action_split[i])
            if flag:
                print 'Something is wrong with indexing...Not saving matrices'
                exit()
        print len(action_class1), len(action_class2), len(action_class3), len(action_class4)
np.savez('trajectory_class_1', state = state_class1, action = action_class1)
np.savez('trajectory_class_2', state = state_class1, action = action_class2)
np.savez('trajectory_class_3', state = state_class1, action = action_class3)
np.savez('trajectory_class_4', state = state_class1, action = action_class4)
        # trajectory = [[state, action]]
        # print trajectory[0][0]
        # print joints.shape
