import numpy as np
from natsort import *
from tqdm import *
import datetime as dt
import matplotlib.pyplot as plt
import glob
import os
from collections import namedtuple
from common import parseTrackletXML as xmlParser

class kitti:
    def __init__(self):
        self.base_dir='/media/waleed/work/data/kitti/raw/2011_09_26/'
        self.sequences = natsorted(next(os.walk(self.base_dir))[1])
        self.list_of_velo_files = []
        self.list_of_images = []
        self.list_tracklets = {}
        self.frames_per_seq = {}
        self.velo_range={'x':[0,60],'y':[-20,20]}
        self.clip_ground = True
        self.clip_ground_value = -1.7
        self.save_dir = './velo_scans/'

    def build(self):
        self.create_list_of_velo_files()
        self.create_list_of_rgb_files()
        print("[info] parsing tracklet files this may take moments ... ")
        self.frame_tracklets, self.frame_tracklets_types=self.load_tracklets_for_frames()
        print("[info] parsing finished.")
        print("[info] parsed ", len(self.sequences), " sequences.")
        print("[info] parsed ",len(self.frame_tracklets)," frames.")
        self.create_velo_labels()

    def create_velo_labels(self):
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
        for sequence in sorted(self.sequences):
            for path, dirs, files in os.walk(self.base_dir + sequence):
                if 'velodyne_points/data' in (path):
                    self.frames_per_seq[sequence]=len(files)
                    self.list_tracklets[sequence]=(path.split('velodyne_points/data')[0]+'tracklet_labels.xml')
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

    def load_tracklets_for_frames(self):
        """
        Loads dataset labels also referred to as tracklets, saving them individually for each frame.

        Parameters
        ----------
        n_frames    : Number of frames in the dataset.
        xml_path    : Path to the tracklets XML.

        Returns
        -------
        Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
        contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
        types as strings.
        """
        j=0
        frame_tracklets = {}
        frame_tracklets_types = {}
        for sequence , n_frames in sorted(self.frames_per_seq.items()):

            tracklets = xmlParser.parseXML(self.list_tracklets[sequence])

            for i in range(n_frames):
                frame_tracklets[j+i] = []
                frame_tracklets_types[j+i] = []


            # loop over tracklets
            for i, tracklet in enumerate(tracklets):
                # this part is inspired by kitti object development kit matlab code: computeBox3D
                h, w, l = tracklet.size
                # in velodyne coordinates around zero point and without orientation yet
                trackletBox = np.array([
                    [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0.0, 0.0, 0.0, 0.0, h, h, h, h]
                ])
                # loop over all data in tracklet
                for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
                    # determine if object is in the image; otherwise continue
                    if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                        continue
                    # re-create 3D bounding box in velodyne coordinate system
                    yaw = rotation[2]  # other rotations are supposedly 0
                    assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
                    rotMat = np.array([
                        [np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw), np.cos(yaw), 0.0],
                        [0.0, 0.0, 1.0]
                    ])
                    cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
                    frame_tracklets[j+absoluteFrameNumber] = frame_tracklets[j+absoluteFrameNumber] + [cornerPosInVelo]
                    frame_tracklets_types[j+absoluteFrameNumber] = frame_tracklets_types[j+absoluteFrameNumber] + [
                        tracklet.objectType]
            j = j + n_frames
        return (frame_tracklets, frame_tracklets_types)


dataset=kitti()
dataset.build()
print(dataset.frame_tracklets.get(5)[0])
#dataset.save_scan_points('png')
