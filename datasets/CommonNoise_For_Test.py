#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling CommonNoise dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018, modified by Meida
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
import torch
import math
from multiprocessing import Lock
import glob
import os

# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors

# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/

'''用于测试Test的'''


class CommonNoiseDataset(PointCloudDataset):
    """Class to handle CommonNoise dataset."""

    def __init__(self, config, set='training', use_potentials=True, load_data=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, config.datasetClass)

        ############
        # Parameters
        ############

        if 'H3D' in self.name:
            self.label_to_names = {
                0: 'unknown',
                1: 'Low vegetation',
                2: 'Impervious surfaces',
                3: 'Vehicle',
                4: 'Urban Fur',
                5: 'Roof',
                6: 'Facade',
                7: 'Shrub',
                8: 'Tree',
                9: 'Soll',
                10: 'Ver Surf',
                11: 'Chimney'
            }
            # List of classes ignored during training (can be empty)
            self.ignored_labels = np.array([0])

            # Dataset folder
            self.path = os.path.join('/data/Gaoyuan/Dataset/H3D_Dataset', config.dataset)

            # # validation file name
            # self.validationFileName = 'Mar18_val'

        elif 'ISPRS' in self.name:
            self.label_to_names = {
                0: 'unknown',
                1: 'Powerline',
                2: 'Low vegetation',
                3: 'Impervious surfaces',
                4: 'Car',
                5: 'Fence',
                6: 'Roof',
                7: 'Facade',
                8: 'Shrub',
                9: 'Tree'}
            # List of classes ignored during training (can be empty)
            self.ignored_labels = np.array([0])

            # Dataset folder
            self.path = os.path.join('/data/Gaoyuan/Dataset/isprs_Dataset', config.dataset)

            # # validation file name
            # self.validationFileName = 'Vaihingen3D_EVAL_WITH_REF'


        elif 'LASDU' in self.name:
            self.label_to_names = {
                0:'unknown',
                1: 'Ground',
                2: 'Buildings',
                3: 'Trees',
                4: 'Low vegetation',
                5: 'Artifacts'
                }

            # List of classes ignored during training (can be empty)
            self.ignored_labels = np.array([0])
            # validation Name
            # self.validationFileName =['section_1','section_4']
            # Dataset folder
            self.path = os.path.join('/data/Gaoyuan/Dataset/LasDu', config.dataset)

        elif 'DALES' in self.name:
            self.label_to_names = {0: 'unknown',
                                   1: 'Ground',
                                   2: 'Vegetation',
                                   3: 'Cars',
                                   4: 'Trucks',
                                   5: 'Power lines',
                                   6: 'Fences',
                                   7: 'Poles',
                                   8: 'Buildings'}

            # List of classes ignored during training (can be empty)
            self.ignored_labels = np.array([0])
            # validation Name
            # self.validationFileName = ['bin_5185_54485']
            # Dataset folder
            self.path = os.path.join('/data/Gaoyuan/Dataset/Dales', config.dataset)



        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        print('*************************************')
        print(self.path)
        print('*************************************')
        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes - len(self.ignored_labels)
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Path of the training files
        self.train_path = 'original_ply'

        # List of files to process
        ply_path = join(self.path, self.train_path)

        # Proportion of validation scenes
        self.cloud_names = []
        self.all_splits = []
        # self.validation_split = -1

        # validation file name
        # validationFileName = 'bin_5185_54485'

        # for root, dirs, files in os.walk(self.path):
        #     for file in files:
        #         if file.endswith(".ply"):
        #              self.cloud_names.append(os.path.join(root, file))

        for file in os.listdir(self.path):
            if file.endswith(".ply"):
                self.cloud_names.append(os.path.join(self.path, file))

        for i, val in enumerate(self.cloud_names):
            self.cloud_names[i] = os.path.basename(val).split('.')[0]
            self.all_splits.append(i)
            # if (self.cloud_names[i] == validationFileName):
            #     self.validation_split = i

        # Number of models used per epoch
        # if self.set == 'training':
        #     self.epoch_n = config.epoch_steps * config.batch_num
        if self.set in ['validation', 'test', 'ERF']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for CommonNoise data: ', self.set)

        # Stop data is not needed
        if not load_data:
            return

        ################
        # Load ply files
        ################
        if 'H3D' in self.name:
            self.prepare_CommonNoise_ply_H3D()
        elif 'ISPRS' in self.name:
            self.prepare_CommonNoise_ply_ISPRS()
        elif 'LASDU' in self.name:
            self.prepare_CommonNoise_ply_LASDU()
        elif 'DALES' in self.name:
            self.prepare_CommonNoise_ply_DALES()

        # List of training files
        self.files = []
        for i, f in enumerate(self.cloud_names):
            if self.set in ['validation', 'test', 'ERF']:
                # if self.all_splits[i] != self.validation_split:
                self.files += [join(ply_path, f + '.ply')]
            # elif self.set in ['validation', 'test', 'ERF']:
            #     if self.all_splits[i] == self.validation_split:
            #         self.files += [join(ply_path, f + '.ply')]
            else:
                raise ValueError('Unknown set for CommonNoise data: ', self.set)

        if self.set in ['validation', 'test', 'ERF']:
            self.cloud_names = [f for i, f in enumerate(self.cloud_names)]
        # elif self.set in ['validation', 'test', 'ERF']:
        #     self.cloud_names = [f for i, f in enumerate(self.cloud_names)
        #                         if self.all_splits[i] == self.validation_split]

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.input_labels1 = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []
        self.validation_labels2 = []

        if 'H3D' in self.name:
            self.load_subsampled_clouds_H3D()
        elif 'ISPRS' in self.name:
            # Start loading
            self.load_subsampled_clouds_ISPRS()
        elif 'LASDU' in self.name:
            self.load_subsampled_clouds_LASDU()
        elif 'DALES' in self.name:
            self.load_subsampled_clouds_DALES()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            N = config.epoch_steps * config.batch_num
            self.epoch_inds = torch.from_numpy(np.zeros((2, N), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()

        # For ERF visualization, we want only one cloud per batch and no randomness
        if self.set == 'ERF':
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        if self.use_potentials:
            return self.potential_item(batch_i)
        else:
            return self.random_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        l_list1 = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        info = get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None

        while True:

            t += [time.time()]

            if debug_workers:
                message = ''
                for wi in range(info.num_workers):
                    if wi == wid:
                        message += ' {:}X{:} '.format(bcolors.FAIL, bcolors.ENDC)
                    elif self.worker_waiting[wi] == 0:
                        message += '   '
                    elif self.worker_waiting[wi] == 1:
                        message += ' | '
                    elif self.worker_waiting[wi] == 2:
                        message += ' o '
                print(message)
                self.worker_waiting[wid] = 0

            with self.worker_lock:

                if debug_workers:
                    message = ''
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += ' {:}v{:} '.format(bcolors.OKGREEN, bcolors.ENDC)
                        elif self.worker_waiting[wi] == 0:
                            message += '   '
                        elif self.worker_waiting[wi] == 1:
                            message += ' | '
                        elif self.worker_waiting[wi] == 2:
                            message += ' o '
                    print(message)
                    self.worker_waiting[wid] = 1

                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Add a small noise to center point
                if self.set != 'ERF':
                    center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if self.set != 'ERF':
                    tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                    tukeys[d2s > np.square(self.config.in_radius)] = 0
                    self.potentials[cloud_ind][pot_inds] += tukeys
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
                input_labels1 = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                input_labels1 = self.input_labels1[cloud_ind][input_inds]
                input_labels1 = np.array([self.label_to_idx[l] for l in input_labels1])

            t += [time.time()]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            l_list1 += [input_labels1]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        labels1 = np.concatenate(l_list1, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        input_list += [labels1]

        # print('wfwefg :',sum(np.not_equal(labels,labels1)))
        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

        if debug_workers:
            message = ''
            for wi in range(info.num_workers):
                if wi == wid:
                    message += ' {:}0{:} '.format(bcolors.OKBLUE, bcolors.ENDC)
                elif self.worker_waiting[wi] == 0:
                    message += '   '
                elif self.worker_waiting[wi] == 1:
                    message += ' | '
                elif self.worker_waiting[wi] == 2:
                    message += ' o '
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print('\n************************\n')
            print('Timings:')
            ti = 0
            N = 5
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Pots ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Sphere .... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Collect ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Augment ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print('concat .... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('input ..... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('stack ..... {:5.1f}ms'.format(1000 * (t[ti + 1] - t[ti])))
            ti += 1
            print('\n************************\n')

        # print('取值国王而过g:', sum(np.not_equal(input_list[26], input_list[27])))
        return input_list

    def changeSemLabels(self, cloud):

        cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 2) & (cloud[:, 6:7] <= 4), 2, cloud[:, 6:7])
        cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 5) & (cloud[:, 6:7] <= 6), 3, cloud[:, 6:7])
        cloud[:, 6:7] = np.where((cloud[:, 6:7] == 8), 3, cloud[:, 6:7])
        cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 11) & (cloud[:, 6:7] <= 12), 4, cloud[:, 6:7])
        cloud[:, 6:7] = np.where((cloud[:, 6:7] == 14), 5, cloud[:, 6:7])

        cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 7) & (cloud[:, 6:7] <= 10), 1, cloud[:, 6:7])
        cloud[:, 6:7] = np.where((cloud[:, 6:7] == 13), 1, cloud[:, 6:7])
        cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 15) & (cloud[:, 6:7] <= 16), 0, cloud[:, 6:7])
        cloud[:, 6:7] = np.where((cloud[:, 6:7] == 17), 1, cloud[:, 6:7])
        cloud[:, 6:7] = np.where((cloud[:, 6:7] > 17), 0, cloud[:, 6:7])

        return cloud

    def prepare_CommonNoise_ply_H3D(self):

        print('\nPreparing ply files')
        t0 = time.time()

        ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            makedirs(ply_path)
        count = 1
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".ply"):
                    # Pass if the cloud has already been computed
                    cloud_file = join(ply_path, file)
                    if exists(cloud_file):
                        continue
                    print('Cloud %d/%d - %s' % (count, len(self.cloud_names), file))
                    count += 1

                    cloud = read_ply(join(root, file))
                    cloud = np.vstack(
                        (cloud['x'], cloud['y'], cloud['z'], cloud['red'], cloud['green'],
                         cloud['blue'], cloud['class'], cloud['oclass'])).T

                    limitMin = np.amin(cloud[:, 0:3], axis=0)
                    cloud[:, 0:3] -= limitMin
                    # 修改标签
                    # cloud = self.changeSemLabels(cloud)

                    xyz = cloud[:, :3].astype(np.float32)
                    colors = cloud[:, 3:6].astype(np.uint8)
                    labels = cloud[:, 6].astype(np.int32)
                    olabels = cloud[:, 7].astype(np.int32)

                    write_ply(cloud_file, (xyz, colors, labels, olabels),
                              ['x', 'y', 'z', 'red', 'green',
                               'blue', 'class', 'oclass'])
        print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def load_subsampled_clouds_H3D(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]
            print(cloud_name)
            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_reflectance = np.vstack((data['red'], data['green'],
                                             data['blue'])).T
                sub_labels = data['class']
                sub_labels1 = data['oclass']
                # print('取值11:', sum(np.not_equal(sub_labels, sub_labels1)))
                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                reflectance = np.vstack((data['red'], data['green'],
                                         data['blue'])).astype(np.float32).T
                labels = data['class']

                olabels = data['oclass']
                # Subsample cloud
                sub_points, sub_reflectance, sub_labels = grid_subsampling(points,
                                                                           features=reflectance,
                                                                           labels=labels,
                                                                           sampleDl=dl)
                sub_points1, sub_reflectance1, sub_labels1 = grid_subsampling(points,
                                                                              features=reflectance,
                                                                              labels=olabels,
                                                                              sampleDl=dl)
                print(len(sub_labels1) == len(sub_labels))

                # Rescale float color and squeeze label
                sub_reflectance = sub_reflectance / 50.0
                sub_labels = np.squeeze(sub_labels)
                sub_labels1 = np.squeeze(sub_labels1)
                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                # search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                # search_tree.fit(sub_points)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_reflectance, sub_labels, sub_labels1],
                          ['x', 'y', 'z', 'red', 'green',
                           'blue', 'class', 'oclass'])

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_reflectance]
            self.input_labels += [sub_labels]
            self.input_labels1 += [sub_labels1]
            size = sub_reflectance.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels, olabels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']
                    olabels = data['oclass']
                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    # dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:

                        pickle.dump([proj_inds, labels, olabels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                self.validation_labels2 += [olabels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def prepare_CommonNoise_ply_LASDU(self):

        print('\nPreparing ply files')
        t0 = time.time()

        ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            makedirs(ply_path)
        count = 1
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".ply"):
                    # Pass if the cloud has already been computed
                    cloud_file = join(ply_path, file)
                    if exists(cloud_file):
                        continue
                    print('Cloud %d/%d - %s' % (count, len(self.cloud_names), file))
                    count += 1

                    cloud = read_ply(join(root, file))
                    cloud = np.vstack(
                        (cloud['x'], cloud['y'], cloud['z'], cloud['Intensity'], cloud['class'], cloud['oclass'])).T

                    limitMin = np.amin(cloud[:, 0:3], axis=0)
                    cloud[:, 0:3] -= limitMin
                    # 修改标签
                    # cloud = self.changeSemLabels(cloud)

                    xyz = cloud[:, :3].astype(np.float32)
                    colors = cloud[:, 3].astype(np.uint8)
                    labels = cloud[:, 4].astype(np.int32)
                    olabels = cloud[:,5].astype(np.int32)

                    write_ply(cloud_file, (xyz, colors, labels, olabels),
                              ['x', 'y', 'z', 'Intensity', 'class', 'oclass'])
        print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def prepare_CommonNoise_ply_DALES(self):

        print('\nPreparing ply files')
        t0 = time.time()

        ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            makedirs(ply_path)
        count = 1
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".ply"):
                    # Pass if the cloud has already been computed
                    cloud_file = join(ply_path, file)
                    if exists(cloud_file):
                        continue
                    print('Cloud %d/%d - %s' % (count, len(self.cloud_names), file))
                    count += 1

                    cloud = read_ply(join(root, file))
                    cloud = np.vstack(
                        (cloud['x'], cloud['y'], cloud['z'], cloud['reflectance'], cloud['class'], cloud['oclass'])).T

                    limitMin = np.amin(cloud[:, 0:3], axis=0)
                    cloud[:, 0:3] -= limitMin
                    # 修改标签
                    # cloud = self.changeSemLabels(cloud)

                    xyz = cloud[:, :3].astype(np.float32)
                    colors = cloud[:, 3].astype(np.uint8)
                    labels = cloud[:, 4].astype(np.int32)
                    olabels = cloud[:,5].astype(np.int32)

                    write_ply(cloud_file, (xyz, colors, labels, olabels),
                              ['x', 'y', 'z', 'reflectance', 'class', 'oclass'])
        print('Done in {:.1f}s'.format(time.time() - t0))
        return
    def load_subsampled_clouds_LASDU(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]
            print(cloud_name)
            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_reflectance = np.expand_dims(data['Intensity'], 1)
                sub_labels = data['class']
                sub_labels1 = data['oclass']
                # print('取值11:', sum(np.not_equal(sub_labels, sub_labels1)))
                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                reflectance = np.expand_dims(data['Intensity'], 1).astype(np.float32)
                labels = data['class']

                olabels = data['oclass']
                # Subsample cloud
                sub_points, sub_reflectance, sub_labels = grid_subsampling(points,
                                                                           features=reflectance,
                                                                           labels=labels,
                                                                           sampleDl=dl)
                sub_points1, sub_reflectance1, sub_labels1 = grid_subsampling(points,
                                                                              features=reflectance,
                                                                              labels=olabels,
                                                                              sampleDl=dl)
                print(len(sub_labels1) == len(sub_labels))

                # Rescale float color and squeeze label
                sub_reflectance = sub_reflectance / 50.0
                sub_labels = np.squeeze(sub_labels)
                sub_labels1 = np.squeeze(sub_labels1)
                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                # search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                # search_tree.fit(sub_points)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_reflectance, sub_labels, sub_labels1],
                          ['x', 'y', 'z', 'Intensity', 'class', 'oclass'])

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_reflectance]
            self.input_labels += [sub_labels]
            self.input_labels1 += [sub_labels1]
            size = sub_reflectance.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels, olabels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']
                    olabels = data['oclass']
                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    # dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:

                        pickle.dump([proj_inds, labels, olabels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                self.validation_labels2 += [olabels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def load_subsampled_clouds_DALES(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]
            print(cloud_name)
            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_reflectance = np.expand_dims(data['reflectance'], 1)
                sub_labels = data['class']
                sub_labels1 = data['oclass']
                # print('取值11:', sum(np.not_equal(sub_labels, sub_labels1)))
                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                reflectance = np.expand_dims(data['reflectance'], 1).astype(np.float32)
                labels = data['class']

                olabels = data['oclass']
                # Subsample cloud
                sub_points, sub_reflectance, sub_labels = grid_subsampling(points,
                                                                           features=reflectance,
                                                                           labels=labels,
                                                                           sampleDl=dl)
                sub_points1, sub_reflectance1, sub_labels1 = grid_subsampling(points,
                                                                              features=reflectance,
                                                                              labels=olabels,
                                                                              sampleDl=dl)
                print(len(sub_labels1) == len(sub_labels))

                # Rescale float color and squeeze label
                sub_reflectance = sub_reflectance / 50.0
                sub_labels = np.squeeze(sub_labels)
                sub_labels1 = np.squeeze(sub_labels1)
                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                # search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                # search_tree.fit(sub_points)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_reflectance, sub_labels, sub_labels1],
                          ['x', 'y', 'z', 'reflectance', 'class', 'oclass'])

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_reflectance]
            self.input_labels += [sub_labels]
            self.input_labels1 += [sub_labels1]
            size = sub_reflectance.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels, olabels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']
                    olabels = data['oclass']
                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    # dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:

                        pickle.dump([proj_inds, labels, olabels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                self.validation_labels2 += [olabels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def prepare_CommonNoise_ply_ISPRS(self):

        print('\nPreparing ply files')
        t0 = time.time()

        ply_path = join(self.path, self.train_path)
        if not exists(ply_path):
            makedirs(ply_path)
        count = 1
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".ply"):
                    # Pass if the cloud has already been computed
                    cloud_file = join(ply_path, file)
                    if exists(cloud_file):
                        continue
                    print('Cloud %d/%d - %s' % (count, len(self.cloud_names), file))
                    count += 1

                    cloud = read_ply(join(root, file))
                    cloud = np.vstack(
                        (cloud['x'], cloud['y'], cloud['z'], cloud['Intensity'], cloud['return_number'],
                         cloud['number_of_returns'], cloud['class'], cloud['oclass'])).T

                    limitMin = np.amin(cloud[:, 0:3], axis=0)
                    cloud[:, 0:3] -= limitMin
                    # 修改标签
                    # cloud = self.changeSemLabels(cloud)

                    xyz = cloud[:, :3].astype(np.float32)
                    colors = cloud[:, 3:6].astype(np.uint8)
                    labels = cloud[:, 6].astype(np.int32)
                    olabels = cloud[:, 7].astype(np.int32)

                    write_ply(cloud_file, (xyz, colors, labels, olabels),
                              ['x', 'y', 'z', 'Intensity', 'return_number',
                               'number_of_returns', 'class', 'oclass'])
        print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def load_subsampled_clouds_ISPRS(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]
            print(cloud_name)
            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_reflectance = np.vstack((data['Intensity'], data['return_number'],
                                             data['number_of_returns'])).T
                sub_labels = data['class']
                sub_labels1 = data['oclass']
                # print('取值11:', sum(np.not_equal(sub_labels, sub_labels1)))
                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                reflectance = np.vstack((data['Intensity'], data['return_number'],
                                         data['number_of_returns'])).astype(np.float32).T
                labels = data['class']

                olabels = data['oclass']
                # Subsample cloud
                sub_points, sub_reflectance, sub_labels = grid_subsampling(points,
                                                                           features=reflectance,
                                                                           labels=labels,
                                                                           sampleDl=dl)
                sub_points1, sub_reflectance1, sub_labels1 = grid_subsampling(points,
                                                                              features=reflectance,
                                                                              labels=olabels,
                                                                              sampleDl=dl)
                print(len(sub_labels1) == len(sub_labels))

                # Rescale float color and squeeze label
                sub_reflectance = sub_reflectance / 50.0
                sub_labels = np.squeeze(sub_labels)
                sub_labels1 = np.squeeze(sub_labels1)
                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                # search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                # search_tree.fit(sub_points)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_reflectance, sub_labels, sub_labels1],
                          ['x', 'y', 'z', 'Intensity', 'return_number',
                           'number_of_returns', 'class', 'oclass'])

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_reflectance]
            self.input_labels += [sub_labels]
            self.input_labels1 += [sub_labels1]
            size = sub_reflectance.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels, olabels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']
                    olabels = data['oclass']
                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    # dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:

                        pickle.dump([proj_inds, labels, olabels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                self.validation_labels2 += [olabels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T


    def load_evaluation_points_color(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        if 'H3D' in self.name:
            return np.vstack((data['red'], data['green'],
                              data['blue'])).T
        elif 'ISPRS' in self.name:
            return np.vstack((data['Intensity'], data['return_number'],
                              data['number_of_returns'])).T
        elif 'LASDU' in self.name:
            return data['Intensity']
        elif 'DALES' in self.name:
            return data['reflectance']
        #默认
        return np.vstack((data['Intensity'], data['return_number'],
                          data['number_of_returns'])).T

    def load_evaluation_points_label(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['class'])).T

    def load_evaluation_points_olabel(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['oclass'])).T


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class CommonNoiseSampler(Sampler):
    """Sampler for CommonNoise"""

    def __init__(self, dataset: CommonNoiseDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        if not self.dataset.use_potentials:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int32)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(num_centers / (self.dataset.num_clouds * self.dataset.config.num_classes)))

            # Choose random points of each class for each cloud
            for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                epoch_indices = np.empty((0,), dtype=np.int32)
                for label_ind, label in enumerate(self.dataset.label_values):
                    if label not in self.dataset.ignored_labels:
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        if len(label_indices) <= random_pick_n:
                            epoch_indices = np.hstack((epoch_indices, label_indices))
                        elif len(label_indices) < 50 * random_pick_n:
                            new_randoms = np.random.choice(label_indices, size=random_pick_n, replace=False)
                            epoch_indices = np.hstack((epoch_indices, new_randoms.astype(np.int32)))
                        else:
                            rand_inds = []
                            while len(rand_inds) < random_pick_n:
                                rand_inds = np.unique(
                                    np.random.choice(label_indices, size=5 * random_pick_n, replace=True))
                            epoch_indices = np.hstack((epoch_indices, rand_inds[:random_pick_n].astype(np.int32)))

                # Stack those indices with the cloud index
                epoch_indices = np.vstack((np.full(epoch_indices.shape, cloud_ind, dtype=np.int32), epoch_indices))

                # Update the global indice container
                all_epoch_inds = np.hstack((all_epoch_inds, epoch_indices))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds[:, :num_centers])

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def fast_calib(self):
        """
        This method calibrates the batch sizes while ensuring the potentials are well initialized. Indeed on a dataset
        like Semantic3D, before potential have been updated over the dataset, there are cahnces that all the dense area
        are picked in the begining and in the end, we will have very large batch of small point clouds
        :return:
        """

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config.batch_num

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
        finer = False
        breaking = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(2)

        for epoch in range(10):
            for i, test in enumerate(self):

                # New time
                t = t[-1:]
                t += [time.time()]

                # batch length
                b = len(test)

                # Update estim_b (low pass filter)
                estim_b += (b - estim_b) / low_pass_T

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += Kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_T = 100
                    finer = True

                # Convergence
                if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                    breaking = True
                    break

                # Average timing
                t += [time.time()]
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d},  //  {:.1f}ms {:.1f}ms'
                    print(message.format(i,
                                         estim_b,
                                         int(self.dataset.batch_limit),
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

            if breaking:
                break

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.use_potentials:
            sampler_method = 'potentials'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                               self.dataset.config.in_radius,
                                               self.dataset.config.first_subsampling_dl,
                                               self.dataset.config.batch_num)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in
                              batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.cloud_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.dataset.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                          neighb_hists[layer, neighb_size],
                                                          bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            if self.dataset.use_potentials:
                sampler_method = 'potentials'
            else:
                sampler_method = 'random'
            key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                                   self.dataset.config.in_radius,
                                                   self.dataset.config.first_subsampling_dl,
                                                   self.dataset.config.batch_num)
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class CommonNoiseCustomBatch:
    """Custom batch definition with memory pinning for CommonNoise"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 7) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        input_list[ind] = np.int64(input_list[ind])
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels1 = torch.from_numpy(input_list[ind])
        ind += 1
        # print('取值jtr:',sum(np.not_equal(input_list[26],input_list[27])))
        # print('取值',torch.sum(torch.not_equal(self.labels1,self.labels)))
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.labels1 = self.labels1.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i + 1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def CommonNoiseCollate(batch_data):
    return CommonNoiseCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_upsampling(dataset, loader):
    """Shows which labels are sampled according to strategy chosen"""

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, '=>', pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0 ** 2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num
    estim_N = 0

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b,
                                     estim_N))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, loader):
    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        L = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # Print characteristics of input tensors
            print('\nPoints tensors')
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nNeigbors tensors')
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\nPools tensors')
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nStack lengths')
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nFeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nLabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\nAugment Scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\nAugment Rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nModel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nAre input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        print()

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)

            s = '\n'

            N_list = []
            n_max_list = []
            p_max_list = []
            for neighbs, pools in zip(batch.neighbors, batch.pools):
                N_list.append(neighbs.shape[0])
                n_max_list.append(neighbs.shape[1])
                p_max_list.append(pools.shape[1])

                N = neighbs.shape[0]
                n_max = neighbs.shape[1]
                p_max = pools.shape[1]

                s += '[{:6d},{:3d}, K, 3]  |  '.format(N, n_max)
                s += '[{:6d},{:3d}->K, f1]  |  '.format(N, n_max)
                s += '[{:6d}, K, f1->f2]'.format(N)
                s += '\n'

            s += '\n'
            print(s)
            print('-------------------------------------------')

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # # Console display (only one per second)
            # if (t[-1] - last_display) > 1.0:
            #     last_display = t[-1]
            #     message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
            #     print(message.format(batch_i,
            #                          1000 * mean_dt[0],
            #                          1000 * mean_dt[1]))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)
