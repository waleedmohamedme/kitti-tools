import numpy as np
from natsort import *
import datetime as dt
import matplotlib.pyplot as plt
import glob
import os
from collections import namedtuple

class kitti:
    def __init__(self):
        self.base_dir='/media/waleed/work/data/kitti/raw/2011_09_26/'
        self.sequences = natsorted(next(os.walk(self.base_dir))[1])
        self.list_of_velo_files = []
        self.list_of_images = []
        self.range={'x':[0,60],'y':[-20,20]}
        self.create_list_of_velo_files()

    def parse(self):
        pass

    def read_tracklet(self):
        pass

    def project_frame_2d(self,scan_points,range=None):
        if range== None:
            range=self.range
        scan_points_projected = scan_points[scan_points[:,0]>range['x'][0],:2]

        scan_points_projected = scan_points_projected[scan_points_projected[:,0]<range['x'][1] , :2]

        scan_points_projected = scan_points_projected[scan_points_projected[:,1]>range['y'][0] , :2]

        scan_points_projected = scan_points_projected[scan_points_projected[:,1]<range['y'][1] , :2]

        return scan_points_projected
    def save_scan_point(self,scan_points,type='png'):
        to_be_saved = np.zeros([600,400])
        if type == 'npy':
            pass
        elif type == 'png':
            for i in range (len(scan_points)):
                x=int((self.range['x'][1]- scan_points[i][0])/0.1)
                y=int((self.range['y'][0]+ scan_points[i][1])/0.1)
                to_be_saved[x][y]=1

        plt.imsave('testimage.png',to_be_saved)
    def extract_sequence_2d(self):
        pass

    def create_list_of_velo_files(self):
        for sequence in self.sequences:
            for path, dirs, files in os.walk(self.base_dir + sequence):
                if 'velodyne_points/data' in (path):
                    for filename in natsorted(files):
                        if filename.split('.')[-1] =='bin':
                            self.list_of_velo_files.append(os.path.join(path, filename))

    def create_list_of_rgb_files(self):
        for sequence in self.sequences:
            for path, dirs, files in os.walk(self.base_dir + sequence):
                if 'image_02/data' in (path):
                    for filename in natsorted(files):
                        if filename.split('.')[-1] =='png':
                            self.list_of_images.append(os.path.join(path, filename))

    def get_velo_scans(self):
        """Generator to parse velodyne binary files into arrays."""
        for filename in self.list_of_velo_files:
            scan = np.fromfile(filename, dtype=np.float32)
            yield scan.reshape((-1, 4))

print("hello kitti")
dataset=kitti()
scanpoint =next(dataset.get_velo_scans())
n= (dataset.project_frame_2d(scanpoint))
print(np.min(n[:,1]))
dataset.save_scan_point(n)