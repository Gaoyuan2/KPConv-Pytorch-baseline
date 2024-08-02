#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018 modified by Meida Chen - 04/25/2022
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join, basename
import time
import json
from sklearn.neighbors import KDTree
from tqdm import tqdm

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion, OA, F1_score

# from sklearn.metrics import confusion_matrix

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

'''测试,把训练都过一遍并保存'''
class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, cudaDevice, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" %cudaDevice)
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return
    def cloud_segmentation_test(self, net, test_loader, config, num_votes=10, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth =0   #0.95
        test_radius_ratio = 0 #0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        # nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        #input——labels是下采样后的点云label
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # Test saving path
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            # if not exists(join(test_path, 'predictions')):
            #     makedirs(join(test_path, 'predictions'))
            # if not exists(join(test_path, 'probs')):
            #     makedirs(join(test_path, 'probs'))
            # if not exists(join(test_path, 'potentials')):
            #     makedirs(join(test_path, 'potentials'))
        else:
            test_path = None

        #得到验证集的类别权重
        # If on validation directly compute score
        # if test_loader.dataset.set == 'validation':
        #     val_proportions = np.zeros(nc_model, dtype=np.float32)
        #     # i = 0
        #     # for label_value in test_loader.dataset.label_values:
        #     #     if label_value not in test_loader.dataset.ignored_labels:
        #     #         val_proportions[i] = np.sum([np.sum(labels == label_value)
        #     #                                      for labels in test_loader.dataset.validation_labels])
        #     #         i += 1
        # else:
        #     val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        #?？?？
        last_min = -0.5

        # t = [time.time()]
        # last_display = time.time()
        # mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            with torch.no_grad():
                for i, batch in enumerate(tqdm(test_loader,desc='Testing')):

                    # New time
                    # t = t[-1:]
                    # t += [time.time()]

                    # if i == 0:
                    #     print('Done in {:.1f}s'.format(t[1] - t[0]))

                    if 'cuda' in self.device.type:
                        batch.to(self.device)

                    # Forward pass
                    outputs = net(batch, config)

                    # t += [time.time()]

                    # Get probs and labels
                    stacked_probs = softmax(outputs).cpu().detach().numpy()
                    s_points = batch.points[0].cpu().numpy() #输入点坐标(减去中心点)
                    lengths = batch.lengths[0].cpu().numpy()
                    in_inds = batch.input_inds.cpu().numpy() #输入点的索引
                    cloud_inds = batch.cloud_inds.cpu().numpy() #点云文件索引
                    torch.cuda.synchronize(self.device)

                    # Get predictions and labels per instance
                    # ***************************************

                    i0 = 0
                    for b_i, length in enumerate(lengths):

                        # Get prediction
                        # points = s_points[i0:i0 + length]
                        probs = stacked_probs[i0:i0 + length]
                        inds = in_inds[i0:i0 + length]
                        c_i = cloud_inds[b_i]

                        # #更新小范围内的点云预测值
                        # if 0 < test_radius_ratio < 1:
                        #     mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        #     inds = inds[mask]
                        #     probs = probs[mask]

                        # Update current probs in whole cloud
                        self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                        i0 += length

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print()
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Save predicted cloud

            if last_min + 1 < new_min:
                # Update last_min
                last_min += 1
                #计算预测和真实标签之间的精度
                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == 'validation':
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[i], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = test_loader.dataset.input_labels[i]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    # C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    OAs = OA(C)
                    print('OA: ',OAs)
                    F1s = F1_score(C)
                    print('F1:  ',F1s)
                    print('IoUs:')
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                # Save real IoU once in a while
                ####
                if True:
                # if int(np.ceil(new_min)) % 10 == 0:

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    # proj_probs = []
                    Confs = []
                    for i, file_path2 in enumerate(tqdm(test_loader.dataset.files,desc='File_saving')):

                        # print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)
                        # print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                        # print(test_loader.dataset.test_proj[i][:5])

                        # Reproject probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        # proj_probs += [probs]

                        # Insert false columns for ignored labels
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        file_path = join(test_loader.dataset.path,basename(file_path2))
                        points = test_loader.dataset.load_evaluation_points(file_path)
                        colors= test_loader.dataset.load_evaluation_points_color(file_path)
                        gtLabels = test_loader.dataset.load_evaluation_points_olabel(file_path)
                        # Get the predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Save plys
                        cloud_name = basename(file_path)
                        foldname='predictions'+str(int(np.floor(new_min)))
                        test_path11=join(test_path,foldname)
                        if not exists(test_path11):
                            makedirs(test_path11)
                        test_name = join(test_path11, cloud_name)
                        gtLabels = np.squeeze(gtLabels).astype(np.int32)

                        colors=np.squeeze(colors)
                        if 'H3D' in config.datasetClass:
                            write_ply(test_name,
                                      [points, colors, preds, gtLabels],
                                      ['x', 'y', 'z', 'red', 'green',
                                       'blue', 'class', 'oclass'])
                        elif 'ISPRS' in config.datasetClass:
                            write_ply(test_name,
                                      [points, colors, preds, gtLabels],
                                      ['x', 'y', 'z', 'Intensity', 'return_number',
                                       'number_of_returns', 'class', 'oclass'])
                        elif 'LASDU' in config.datasetClass:
                            write_ply(test_name,
                                      [points, colors, preds, gtLabels],
                                      ['x', 'y', 'z', 'Intensity', 'class', 'oclass'])
                        elif 'DALES' in config.datasetClass:
                            write_ply(test_name,
                                      [points, colors, preds, gtLabels],
                                      ['x', 'y', 'z', 'reflectance', 'class', 'oclass'])

                        Confs += [fast_confusion(gtLabels, preds, test_loader.dataset.label_values)]

                    C = np.sum(np.stack(Confs), axis=0)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)
                    print('***********在原始数据集上的精度**********')
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    OAs = OA(C)
                    print('OA: ', OAs)
                    F1s = F1_score(C)
                    print('F1:  ', F1s)
                    print('IoUs:')
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print('-' * len(s))
                    print(s)
                    print('-' * len(s) + '\n')
                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Show vote results
                    # if test_loader.dataset.set == 'validation':
                    #     print('Confusion on full clouds')
                    #     t1 = time.time()
                    #     Confs = []
                    #     for i, file_path in enumerate(test_loader.dataset.files):
                    #
                    #         # Get the predicted labels
                    #         preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)
                    #
                    #         # Confusion
                    #         targets = test_loader.dataset.validation_labels[i]
                    #         Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]
                    #
                    #     t2 = time.time()
                    #     print('Done in {:.1f} s\n'.format(t2 - t1))
                    #
                    #     # Regroup confusions
                    #     C = np.sum(np.stack(Confs), axis=0)
                    #
                    #     # Remove ignored labels from confusions
                    #     for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                    #         if label_value in test_loader.dataset.ignored_labels:
                    #             C = np.delete(C, l_ind, axis=0)
                    #             C = np.delete(C, l_ind, axis=1)
                    #
                    #     IoUs = IoU_from_confusions(C)
                    #     mIoU = np.mean(IoUs)
                    #     s = '{:5.2f} | '.format(100 * mIoU)
                    #     for IoU in IoUs:
                    #         s += '{:5.2f} '.format(100 * IoU)
                    #     print('-' * len(s))
                    #     print(s)
                    #     print('-' * len(s) + '\n')

                    # Save predictions
                    # print('Saving clouds')
                    # t1 = time.time()
                    # for i, file_path in enumerate(test_loader.dataset.files):
                    #
                    #     # Get file
                    #     points = test_loader.dataset.load_evaluation_points(file_path)
                    #     gtLabels = test_loader.dataset.load_evaluation_points_label(file_path)
                    #     # Get the predicted labels
                    #     preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)
                    #
                    #     # Save plys
                    #     cloud_name = basename(file_path)
                    #     test_name = join(test_path, 'predictions', cloud_name)
                    #     gtLabels = np.squeeze(gtLabels)
                    #     write_ply(test_name,
                    #               [points, preds,gtLabels],
                    #               ['x', 'y', 'z', 'preds','class'])
                    #     # test_name2 = join(test_path, 'probs', cloud_name)
                    #     # prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                    #     #               for label in test_loader.dataset.label_values]
                    #     # write_ply(test_name2,
                    #     #           [points, proj_probs[i]],
                    #     #           ['x', 'y', 'z'] + prob_names)
                    #
                    #     # Save potentials
                    #     # pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                    #     # pot_name = join(test_path, 'potentials', cloud_name)
                    #     # pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                    #     # write_ply(pot_name,
                    #     #           [pot_points.astype(np.float32), pots],
                    #     #           ['x', 'y', 'z', 'pots'])
                    #
                    #     # # Save ascii preds
                    #     # if test_loader.dataset.set == 'test':
                    #     #     if test_loader.dataset.name.startswith('Semantic3D'):
                    #     #         ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                    #     #     else:
                    #     #         ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                    #     #     np.savetxt(ascii_name, preds, fmt='%d')
                    #
                    # t2 = time.time()
                    # print('Done in {:.1f} s\n'.format(t2 - t1))

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return

