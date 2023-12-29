import copy
import numpy as np
import open3d as o3d
import os
import h5py
OUTPUT_DIR="G:\\PointCloudAttributeCompressionArtifactsReomoval\\Dataset\\Training\\TMC13v10_split_h5_with_weight\\QP51\\"
num=0
if __name__ == "__main__":
    for i in range(1):
        filePath = 'G:\\PointCloudAttributeCompressionArtifactsReomoval\Dataset\\Training\\Split_ply\\QP51\\'
        for root, dirs, files in os.walk(filePath):
            for name in files:
                num +=1
                print(os.path.join(root, name))
                pcd=o3d.io.read_point_cloud(os.path.join(root, name))
                coordinate = np.asarray(pcd.points)
                color = np.asarray(pcd.colors)
                c_c = np.concatenate((coordinate,color),-1)
                #print(c_c.shape)
                points_dir = os.path.join(OUTPUT_DIR,name.split('.')[0]+'.h5')
                with h5py.File(points_dir, 'w') as h:
                    # print(set_points[i].dtype)
                    h.create_dataset('data', data=c_c, shape=c_c.shape)
                print(num)
        num = 0 
