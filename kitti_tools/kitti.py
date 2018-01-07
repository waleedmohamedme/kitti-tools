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
        self.saving_dtype = '.png'
        self.resolution = 0.1
        self.annotation = None
    def build(self):
        self.create_list_of_velo_files()
        self.create_list_of_rgb_files()
        print("[info] parsing tracklet files this may take moments ... ")
        self.frame_tracklets_rects, self.frame_tracklets_types ,self.frame_tracklets_dims, self.frame_tracklets_yaw =self.load_tracklets_for_frames()
        print("[sucess] parsing finished.")
        print("[info] parsed ", len(self.sequences), " sequences.")
        print("[info] parsed ",len(self.frame_tracklets_rects)," frames.")
        print("[info] processing labels ...")
        self.annotation = self.create_labels()
        print("[sucess] labels processed ...")
    def create_labels(self):
        def get_str(digits):
            mystring = ''
            for digit in digits:
                mystring = mystring + ' ' + str(digit)
            return mystring

        labels = {}
        for frame in tqdm(range(len(self.list_of_velo_files))):
            label_pre_frame = {}

            obj_id = 0
            for class_name, re, size, yaw in (
            zip(self.frame_tracklets_types[frame], self.frame_tracklets_rects[frame], self.frame_tracklets_dims[frame],self.frame_tracklets_yaw[frame])):
                label = {}
                h, w, l = size

                label['name'] = class_name
                label['yaw'] = -1 * float(yaw)
                label['h'] = h / self.resolution
                label['w'] = w / self.resolution
                label['l'] = l / self.resolution
                label['filename'] = str(frame) + self.saving_dtype
                label['x1'] = int(float(get_str(re[0]).split()[0]))
                label['x2'] = int(float(get_str(re[0]).split()[1]))
                label['x3'] = int(float(get_str(re[0]).split()[2]))
                label['x4'] = int(float(get_str(re[0]).split()[3]))
                label['x5'] = int(float(get_str(re[0]).split()[4]))
                label['x6'] = int(float(get_str(re[0]).split()[5]))
                label['x7'] = int(float(get_str(re[0]).split()[6]))
                label['x8'] = int(float(get_str(re[0]).split()[7]))

                label['y1'] = int(float(get_str(re[1]).split()[0]))
                label['y2'] = int(float(get_str(re[1]).split()[1]))
                label['y3'] = int(float(get_str(re[1]).split()[2]))
                label['y4'] = int(float(get_str(re[1]).split()[3]))
                label['y5'] = int(float(get_str(re[1]).split()[4]))
                label['y6'] = int(float(get_str(re[1]).split()[5]))
                label['y7'] = int(float(get_str(re[1]).split()[6]))
                label['y8'] = int(float(get_str(re[1]).split()[7]))

                label['z1'] = int(float(get_str(re[2]).split()[0]))
                label['z2'] = int(float(get_str(re[2]).split()[1]))
                label['z3'] = int(float(get_str(re[2]).split()[2]))
                label['z4'] = int(float(get_str(re[2]).split()[3]))
                label['z5'] = int(float(get_str(re[2]).split()[4]))
                label['z6'] = int(float(get_str(re[2]).split()[5]))
                label['z7'] = int(float(get_str(re[2]).split()[6]))
                label['z8'] = int(float(get_str(re[2]).split()[7]))

                x = np.array([label['x1'],label['x2'],label['x3'],label['x4'],label['x5'],label['x6'],label['x7'],label['x8']])
                y = np.array([label['y1'],label['y2'],label['y3'],label['y4'],label['y5'],label['y6'],label['y7'],label['y8']])
                z = np.array([label['z1'],label['z2'],label['z3'],label['z4'],label['z5'],label['z6'],label['z7'],label['z8']])

                label['cx'] = np.mean(x)
                label['cy'] = np.mean(y)
                label['cz'] = np.mean(z)

                label['velo_xmin']=np.min(x)
                label['velo_ymin']=np.min(y)
                label['velo_xmax']=np.max(x)
                label['velo_ymax']=np.max(y)

                label_pre_frame[obj_id] = label
                obj_id += 1

            labels[frame] = label_pre_frame

        return labels

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
    def save_scan_points(self):
        scanpoint_gen = self.get_velo_scans()
        for i in tqdm(range (len(self.list_of_velo_files))):
            scanpoints= self.project_frame_2d(next(scanpoint_gen))
            self.save_one_scan_(scanpoints,str(i))

    def save_one_scan_(self,scan_points,filename):
        to_be_saved = np.zeros([600,400])
        for i in range(len(scan_points)):
            x = int((self.velo_range['x'][1] - scan_points[i][0]) / self.resolution)
            y = int((self.velo_range['y'][0] - scan_points[i][1]) / self.resolution)
            try:
                to_be_saved[x][y] = 1
            except:
                pass
        if self.saving_dtype == '.npy':
            np.save(self.save_dir+filename+self.saving_dtype , to_be_saved )
        else:##image formates png or jpg
            plt.imsave(self.save_dir+filename+self.saving_dtype , to_be_saved)

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

        def bbox_from_corners(x):
            if x.shape[1] == 0:
                return [[0, 0], [0, 0]]

            x1, x2 = int(np.min(x[0])), int(np.max(x[0]))
            y1, y2 = int(np.min(x[1])), int(np.max(x[1]))
            return [(x1, y1), (x2, y2)]

        j=0
        frame_tracklets = {}
        frame_tracklets_types = {}
        dims = {}
        yawframe = {}
        imagebbox = {}

        for sequence , n_frames in sorted(self.frames_per_seq.items()):

            tracklets = xmlParser.parseXML(self.list_tracklets[sequence])

            for i in range(n_frames):
                frame_tracklets[j+i] = []
                frame_tracklets_types[j+i] = []
                dims[j+i] = []
                yawframe[j+i] = []
                imagebbox[j+i] = []

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

                    (cornerPosInVelo[1]) = ((cornerPosInVelo[1])) + abs(self.velo_range['y'][0])
                    (cornerPosInVelo) = (cornerPosInVelo) / self.resolution

                    frame_tracklets[j+absoluteFrameNumber] = frame_tracklets[j+absoluteFrameNumber] + [cornerPosInVelo]
                    frame_tracklets_types[j+absoluteFrameNumber] = frame_tracklets_types[j+absoluteFrameNumber] + [tracklet.objectType]
                    yawframe[j+absoluteFrameNumber] = yawframe[j+absoluteFrameNumber] + [yaw]
                    dims[j+absoluteFrameNumber] = dims[j+absoluteFrameNumber] + [tracklet.size]

            j = j + n_frames
        return (frame_tracklets, frame_tracklets_types ,dims ,yawframe)


dataset=kitti()
dataset.build()
print(len(dataset.annotation))
#dataset.save_scan_points(')
