import matplotlib.pyplot as plt
import numpy as np

class Plot():

    def __init__(self):
        self.ax_ptcl = plt.axes(projection ="3d")
        self.ax_ptcl.set_xlabel('X axis')
        self.ax_ptcl.set_ylabel('Y axis')
        self.ax_ptcl.set_zlabel('Z axis')

        # self.ax_ptcl.set_xlim(-100, 100)
        # self.ax_ptcl.set_ylim(-100, 100)
        # self.ax_ptcl.set_zlim(-100, 100)
        self.ax_ptcl.set_title('Point Cloud')

        fig = plt.figure()
        self.ax_traj = fig.gca(projection='3d')
        # self.ax_traj = plt.axes(projection ="3d")
        self.ax_traj.set_xlabel('X axis')
        self.ax_traj.set_ylabel('Y axis')
        self.ax_traj.set_zlabel('Z axis')
        # self.ax_traj.set_xlim(-100, 100)
        # self.ax_traj.set_ylim(-100, 100)
        # self.ax_traj.set_zlim(-100, 100)

        self.states = np.zeros(3)
        self.gt_states = []

        plt.ion()
        # plt.show()

    def plot_point_cloud(self, X):
        
        # ax_ptcl.scatter([0], [0], [0], color='red',marker=('^'))
        self.ax_ptcl.scatter(X[:,0], X[:,1], X[:,2], marker=('o'), color='green',s=20)
        self.ax_ptcl.plot([0], [0], [0], markerfacecolor='red', markeredgecolor='red', marker='^', markersize=5, alpha=0.6)
        
        
        plt.pause(0.0001)


    def plot_trajectory(self, state, gt_state):
        self.states = np.vstack((self.states, [state.x,state.y,state.z])) 
        
        self.ax_traj.plot(-self.states[:,0], -self.states[:,1],-self.states[:,2], color='blue', linestyle='solid', marker='o', markerfacecolor='blue', markersize=2, label='Predicted')
        self.ax_traj.plot(gt_state[:,0,3], gt_state[:,1,3], gt_state[:,2,3], color='red', linestyle='dashed', marker='o', markerfacecolor='red', markersize=2, label='Ground Truth')
        
        self.ax_traj.set_title('Trajectory')

        plt.pause(0.0001)

    def reset_plot():
        
        plt.ioff() 