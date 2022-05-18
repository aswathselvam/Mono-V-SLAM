class Frame():
    def __init__(self,id,isKeyFrame,image,image_coordinates,state,gt_poses,pointcloud):
        self.id=id
        self.isKeyFrame = isKeyFrame
        self.image=image
        self.image_coordinates=image_coordinates
        self.state=state
        self.gt_state=gt_poses
        self.pointcloud=pointcloud
        self.transformed_pointcloud=pointcloud
        self.transform_pointcloud()
    
    def transform_pointcloud(self):
        # print(self.state.R.shape, self.pointcloud[0].T.shape)
        # self.transformed_pointcloud[0]=(self.state.R.T@self.pointcloud[0].T).T
        # self.transformed_pointcloud[0]+=self.state.T
        # self.transformed_pointcloud[0]=self.pointcloud[0]@self.state.R
        self.transformed_pointcloud[0]+=self.state.T
    