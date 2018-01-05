import numpy as np
from natsort import *
from tqdm import *
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
        self.velo_range={'x':[0,60],'y':[-20,20]}
        self.create_list_of_velo_files()
        self.clip_ground = True
        self.clip_ground_value = -1.7
        self.save_dir = './velo_scans/'
    def parse(self):
        pass
    def read_tracklet(self):
        pass

    def project_frame_2d(self,scan_points,range=None):
        if range== None:
            range=self.velo_range
        else:
            self.velo_range =range
        scan_points_projected = scan_points[scan_points[:,0]>range['x'][0],:3]

        scan_points_projected = scan_points_projected[scan_points_projected[:,0]<range['x'][1] , :3]

        scan_points_projected = scan_points_projected[scan_points_projected[:,1]>range['y'][0] , :3]

        scan_points_projected = scan_points_projected[scan_points_projected[:,1]<range['y'][1] , :3]

        if self.clip_ground:
            scan_points_projected = scan_points_projected[scan_points_projected[:, 2] > self.clip_ground_value, :2]
        else :
            scan_points_projected = scan_points_projected[:, :2]

        return scan_points_projected
    def save_scan_points(self,type):
        scanpoint_gen = self.get_velo_scans()
        for i in tqdm(range (len(self.list_of_velo_files))):
            scanpoints= self.project_frame_2d(next(scanpoint_gen))
            self.save_one_scan_(scanpoints,str(i),type=type)

    def save_one_scan_(self,scan_points,filename,type='png'):
        to_be_saved = np.zeros([600,400])
        if type == 'npy':
            pass
        elif type == 'png':
            for i in range (len(scan_points)):
                x=int((self.velo_range['x'][1]- scan_points[i][0])/0.1)
                y=int((self.velo_range['y'][0]- scan_points[i][1])/0.1)
                try:
                    to_be_saved[x][y]=1
                except:
                    pass
            plt.imsave(self.save_dir+filename+'.png',to_be_saved)

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
dataset.save_scan_points('png')
