# import matplotlib
# matplotlib.use('module://mplopengl.backend_qtgl')

import matplotlib.pyplot as plt
import numpy as np
import OpenGL.GL as gl
import pypangolin as pangolin
import threading
from threading import Thread, Lock
from multiprocessing import Process, Queue
import cv2
import random
import time
import copy

class Plot():

    def __init__(self):
        self.cnt = 0
        self.ax_ptcl = plt.axes(projection ="3d")
        self.ax_ptcl.set_xlabel('X axis')
        self.ax_ptcl.set_ylabel('Y axis')
        self.ax_ptcl.set_zlabel('Z axis')

        # self.ax_ptcl.set_xlim(-10, 100)
        # self.ax_ptcl.set_ylim(-10, 100)
        # self.ax_ptcl.set_zlim(-10, 10)
        self.ax_ptcl.set_title('Point Cloud')

        fig = plt.figure()
        self.ax_traj = fig.gca(projection='3d')
        
        # self.ax_traj = plt.axes(projection ="3d")
        self.ax_traj.set_xlabel('X axis')
        self.ax_traj.set_ylabel('Y axis')
        self.ax_traj.set_zlabel('Z axis - Altitude')
        # self.ax_traj.set_xlim(-100, 100)
        # self.ax_traj.set_ylim(-100, 100)
        # self.ax_traj.set_zlim(-100, 100)
    

        self.states = np.zeros(3)
        self.gt_states = []

        plt.ion()
        # plt.show()

    def plot_point_cloud(self, pointcloud):
        
        # ax_ptcl.scatter([0], [0], [0], color='red',marker=('^'))
        
        positions = []
        c=[]
        # print(pointcloud[0].position)
        for i in range(len(pointcloud)):
            positions.append(pointcloud[i].position)
            c.append(pointcloud[i].color)
        positions = np.array(positions)
        c = np.dstack((c,c,c))
        c=np.squeeze(c)
        # print(c.shape)

        # print(c[:][0])

        self.ax_ptcl.scatter(positions[:,2], -positions[:,0], -positions[:,1], c=c,marker=('o'),s=20)
        self.ax_ptcl.plot([0], [0], [0], markerfacecolor='red', markeredgecolor='red', marker='^', markersize=5, alpha=0.6)
        
        
        # plt.pause(0.0001)


    def plot_trajectory(self, state, gt_state):
        self.states = np.vstack((self.states, [state.x,state.y,state.z])) 
        
        # self.ax_traj.plot(-self.states[:,2], self.states[:,0],-self.states[:,1], color='blue', linestyle='solid', marker='o', markerfacecolor='blue', markersize=2, label='Predicted')
        # self.ax_traj.plot(gt_state[:,2,3], gt_state[:,0,3], gt_state[:,1,3], color='red', linestyle='dashed', marker='o', markerfacecolor='red', markersize=2, label='Ground Truth')
        
        self.ax_traj.plot(-self.states[:,2], self.states[:,0],np.zeros(len(self.states[:,1])), color='blue', linestyle='solid', marker='o', markerfacecolor='blue', markersize=2, label='Predicted')
        self.ax_traj.plot(gt_state[:,2,3], gt_state[:,0,3], np.zeros(len(gt_state[:,1,3])), color='red', linestyle='dashed', marker='o', markerfacecolor='red', markersize=2, label='Ground Truth')
        
        self.ax_traj.set_title('Trajectory')
        
        if self.cnt==0:
            self.ax_traj.legend()
            self.cnt+=1

        plt.pause(0.0001)

    def reset_plot():
        
        plt.ioff() 



class PangolinPlot():
    def __init__(self):
        self.pointcloud_positions = np.zeros(3)
        self.pointcloud_color=np.zeros(3)
        
        self.states = np.zeros(3)
        self.states = np.vstack((self.states, np.zeros(3)))
        self.key_frames = 2*np.zeros(3)
        self.key_frames = np.vstack((self.key_frames,  3*np.zeros(3)))

        self.count_key_frames = 0

        self.gt_states = np.zeros(3)
        self.gt_states = np.vstack((self.gt_states, np.zeros(3)))

        self.image = None

        self.mutex = Lock()
        x = Thread(target=self.plotting_thread)
        # x = Process(target=self.plotting_thread)
        x.daemon = True
        x.start()


    def plot_point_cloud(self, pointcloud):
        pointcloud_positions = []
        pointcloud_color=[]
        # pointcloud_positions = self.pointcloud_positions
        # pointcloud_color= self.pointcloud_color
        for i in range(len(pointcloud)):
            pointcloud_positions.append([pointcloud[i].position[2], pointcloud[i].position[0], pointcloud[i].position[1]])
            pointcloud_color.append(pointcloud[i].color)

        self.pointcloud_positions = pointcloud_positions
        self.pointcloud_color = pointcloud_color
        # self.pointcloud_positions = np.array(pointcloud_positions)
        # self.pointcloud_color = np.dstack((pointcloud_color,pointcloud_color,pointcloud_color))
        # self.pointcloud_color = np.squeeze(pointcloud_color)

    def plot_trajectory(self, state, gt_state):

        self.mutex.acquire()
        self.states =  np.vstack((self.states, self.states[-1]))
        self.states = np.vstack((self.states, [-state.z,state.x,state.y])) 
        
        # if len(self.states)%2!=0:
        #     self.states =  np.vstack((self.states, self.states[-1]))
        
        
        self.gt_states =  np.vstack((self.gt_states, self.gt_states[-1]))
        self.gt_states = np.vstack((self.gt_states, [gt_state[-1,2,3],gt_state[-1,0,3],gt_state[-1,1,3]])) 
        self.mutex.release()

        # if len(self.gt_states)%2!=0:
        #     self.gt_states =  np.vstack((self.gt_states, self.gt_states[-1]))
        

        if self.count_key_frames%random.randint(2,10) == 0:
            self.key_frames = np.vstack((self.key_frames,[-state.z,state.x,state.y]))
        self.count_key_frames+=1
        
        # self.ax_traj.plot(-self.states[:,2], self.states[:,0],np.zeros(len(self.states[:,1])), color='blue', linestyle='solid', marker='o', markerfacecolor='blue', markersize=2, label='Predicted')
        # self.ax_traj.plot(gt_state[:,2,3], gt_state[:,0,3], np.zeros(len(gt_state[:,1,3])), color='red', linestyle='dashed', marker='o', markerfacecolor='red', markersize=2, label='Ground Truth')
        

    def draw(self, map):
        self.mutex.acquire()

        # Set some random image data and upload to GPU 
        self.image = cv2.resize(cv2.cvtColor(np.flipud(map.frames[-1].image), cv2.COLOR_GRAY2BGR),(500,200))
        
        self.states = np.vstack((np.zeros(3), np.zeros(3)))
        self.gt_states = np.vstack((np.zeros(3), np.zeros(3)))
        self.key_frames = 2*np.zeros(3)
        self.key_frames = np.vstack((self.key_frames,  3*np.zeros(3)))
        pointcloud_positions = np.zeros(3)
        pointcloud_color=np.zeros(3)
        for i,frame in enumerate(map.frames):
            self.states = np.vstack((self.states, self.states[-1]))
            self.states = np.vstack((self.states, [-frame.state.z, frame.state.x, frame.state.y])) 
            
            self.gt_states =  np.vstack((self.gt_states, self.gt_states[-1]))
            self.gt_states = np.vstack((self.gt_states, [frame.gt_state[2],frame.gt_state[0],frame.gt_state[1]])) 
            
            if frame.isKeyFrame:
                self.key_frames = np.vstack((self.key_frames,[-frame.state.z, frame.state.x, frame.state.y]))

        for i in range(len(frame.pointcloud[0])):
            pointcloud_positions = np.vstack((pointcloud_positions,np.array([frame.transformed_pointcloud[0][i][2], frame.transformed_pointcloud[0][i][0], frame.transformed_pointcloud[0][i][1]])))
        pointcloud_color = np.dstack([frame.pointcloud[1][:,2],frame.pointcloud[1][:,0],frame.pointcloud[1][:,1]])
        pointcloud_color = np.squeeze(pointcloud_color)
    
        self.pointcloud_positions = pointcloud_positions
        self.pointcloud_color = pointcloud_color
        # self.pointcloud_positions = np.vstack((self.pointcloud_positions, pointcloud_positions))
        # self.pointcloud_color = np.vstack((self.pointcloud_color, pointcloud_color))
        self.mutex.release()


    def plotting_thread(self):

        pangolin.CreateWindowAndBind('Mono-V-SLAM', 1920, 720)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        # Note: 2000 is the render distance, we changed this from 200.
        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix( 1920, 720, 420, 420, 320, 240, 0.2, 5000),
            pangolin.ModelViewLookAt(-25, 25, -25, 0, 0, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(scam)

        ui_width = 180
        # Create Interactive View in window
        dcam = (
            pangolin.CreateDisplay()
            .SetBounds(
                pangolin.Attach(0),
                pangolin.Attach(1),
                pangolin.Attach.Pix(ui_width),
                pangolin.Attach(1),
                -640.0 / 480.0,
            )
            .SetHandler(handler)
        )

        pangolin.CreatePanel("ui").SetBounds(
            pangolin.Attach(0), pangolin.Attach(1), pangolin.Attach(0), pangolin.Attach.Pix(ui_width)
        )
        
        
        w, h = 500, 200
        texture = pangolin.GlTexture(w, h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    
        dimg = pangolin.Display('image')
        dimg.SetBounds(pangolin.Attach(0), pangolin.Attach(0.25), pangolin.Attach(0.3), pangolin.Attach(0))

        x_axis = np.array([[0,0,0],[5,0,0]])
        y_axis = np.array([[0,0,0],[0,5,0]])
        z_axis = np.array([[0,0,0],[0,0,5]])

        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)
            
            #Draw axes:
            gl.glLineWidth(10)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.glDrawLines(x_axis)
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.glDrawLines(y_axis)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.glDrawLines(z_axis)

            gl.glPointSize(4)
            if len(self.pointcloud_color) > 5:
                for pt_position, pt_color in zip(self.pointcloud_positions, self.pointcloud_color):
                    gl.glColor3f(pt_color[0]/255, pt_color[1]/255, pt_color[2]/255)
                    pangolin.glDrawPoints([pt_position])

            # Draw lines
            gl.glLineWidth(3)
            if len(self.states) >1:
                if not self.mutex.locked():
                    self.mutex.acquire()
                    gl.glColor3f(1.0, 0.0, 1.0)
                    pangolin.glDrawLines(np.array(self.states))
                    gl.glColor3f(0.0, 0.0, 1.0)
                    pangolin.glDrawLines(np.array(self.gt_states))
                    self.mutex.release()


            # pangolin.glDrawLines(test_traj)   # consecutive
            gl.glColor3f(0.0, 1.0, 0.0)


            #Draw Keyframes:
            gl.glPointSize(20)
            gl.glColor3f(0.9, 0.9, 0.0)
            if len(self.key_frames) > 1:
                # print(self.key_frames)
                pangolin.glDrawPoints(self.key_frames)


            if self.image is not None:
                texture.Upload(self.image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

                # display the image
                dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()
            
            # Draw camera
            # pose = np.identity(4)
            # pose[:3, 3] = np.random.randn(3)
            # gl.glLineWidth(1)
            # gl.glColor3f(0.0, 0.0, 1.0)
            # pangolin.glDrawCamera(pose, 0.5, 0.75, 0.8)

            # Draw boxes
            # poses = [np.identity(4) for i in range(10)]
            # for pose in poses:
            #     pose[:3, 3] = np.random.randn(3) + np.array([5,-3,0])
            # sizes = np.random.random((len(poses), 3))
            # gl.glLineWidth(1)
            # gl.glColor3f(1.0, 0.0, 1.0)
            # pangolin.glDrawBoxes(poses, sizes)
            # print(len(trajectory))
            # time.sleep(0.1)
            pangolin.FinishFrame()