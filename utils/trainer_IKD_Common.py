#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


'''模仿IKDNet 的输出'''
import logging
from pathlib import Path

# Basic libs
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join, basename
import time
import sys

from tqdm import tqdm

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion, OA, F1_score
from utils.config import Config
from utils.IKD_semseg_metric import SemSegMetric

from sklearn.neighbors import KDTree

from models.blocks import KPConv
from torch.utils.tensorboard import SummaryWriter


# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, cudaDevice, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % cudaDevice)
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S',
                                                   time.gmtime()) + config.dataset + '_' + config.AddMessage
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        self.metric_train = SemSegMetric()

        self.val_OA = 0
        self.val_Iou = []
        train_log = 'train_log'

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs         OA       Iou       F1 \n')

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        tensorboard_dir = join(
            train_log,
            "kpConv" + '_' + config.dataset + '_torch' + '_' + config.AddMessage)
        runid = self.get_runid(tensorboard_dir)

        self.tensorboard_dir = join(train_log,
                                    runid + '_' + Path(tensorboard_dir).name)

        writer = SummaryWriter(self.tensorboard_dir)

        # Start training loop
        for epoch in range(config.max_epoch):

            print(f'=== EPOCH {epoch:d}/{config.max_epoch:d} ===')
            self.metric_train.reset()

            self.losses = []

            self.val_OA = 0
            self.val_Iou = []

            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0
            for step, batch in enumerate(tqdm(training_loader, desc='training')):

                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = net(batch, config)
                loss = net.loss(outputs, batch.labels)
                # acc = net.accuracy(outputs, batch.labels)
                t += [time.time()]

                # Backward + optimize
                loss.backward()

                predict_scores = outputs
                gt_labels = batch.labels

                self.metric_train.update(predict_scores, gt_labels)

                # 将GPU损失函数的结果转移到cpu上
                self.losses.append(loss.cpu().item())

                if config.grad_clip_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()

                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # if config.saving:
                #     with open(join(config.saving_path, 'training.txt'), "a") as file:
                #         message = '{:d} {:d} {:.3f} {:.3f} \n'
                #         file.write(message.format(self.epoch,
                #                                   self.step,
                #                                   net.output_loss,
                #                                   net.reg_loss))

                self.step += 1

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            self.valid_losses = []

            # Validation
            net.eval()
            self.validation(net, val_loader, config)
            print(' ')
            self.save_logs(writer, epoch, config)
            print(' ')
            net.train()

        print('Finished Training')
        return

    def get_runid(self, path):
        """Get runid for an experiment."""
        name = Path(path).name
        if not os.path.exists(Path(path).parent):
            return '00001'
        files = os.listdir(Path(path).parent)
        runid = 0
        for f in files:
            try:
                id, val = f.split('_', 1)
                runid = max(runid, int(id))
            except:
                pass
        runid = str(runid + 1)
        runid = '0' * (5 - len(runid)) + runid
        return runid

    def save_logs(self, writer, epoch, config):
        """Save logs from the training and send results to TensorBoard."""
        train_OA = self.metric_train.OA()

        '''论文相关变量'''
        train_ious = self.metric_train.iou()

        train_F1 = self.metric_train.F1_score()

        loss_dict = {
            'Training loss': np.mean(self.losses),
            'Validation loss': np.mean(self.valid_losses)
        }

        OA_dicts = {
            'Training OA': train_OA
        }

        iou_dicts = [{
            'Training IoU': iou
        } for iou in train_ious]

        F1_dicts = [{
            'Training F1': F1
        } for F1 in train_F1]

        for key, val in loss_dict.items():
            writer.add_scalar(key, val, epoch)

        for key, val in OA_dicts.items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)

        for key, val in iou_dicts[-1].items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)

        for key, val in F1_dicts[-1].items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)

        '''论文相关变量'''
        for i in range(config.num_classes):
            writer.add_scalars("Training Class{}".format(str(i + 1)), {"Iou": train_ious[i]},
                               global_step=epoch)

        '''验证 OA 和Iou F1'''
        writer.add_scalar("{}/ Overall".format('Validation OA'), self.val_OA, epoch)
        writer.add_scalar("{}/ Overall".format('Validation IoU'), np.mean(self.val_Iou), epoch)
        writer.add_scalar("{}/ Overall".format('Validation F1'), np.mean(self.val_F1s), epoch)
        # 看每个类的Oiou变化
        for i in range(config.num_classes):
            writer.add_scalars("Validation Class{}".format(str(i + 1)), {"Iou": self.val_Iou[i]}, global_step=epoch)

        if 'ISPRS' in config.datasetClass:
            for i in range(config.num_classes):
                writer.add_scalars("Training All Iou",
                                   {"Class 1": train_ious[0], "Class 2": train_ious[1], "Class 3": train_ious[2],
                                    "Class 4": train_ious[3],
                                    "Class 5": train_ious[4], "Class 6": train_ious[5], "Class 7": train_ious[6],
                                    "Class 8": train_ious[7], "Class 9": train_ious[8]}, global_step=epoch)

            for i in range(config.num_classes):
                writer.add_scalars("Validation All Iou",
                                   {"Class 1": self.val_Iou[0], "Class 2": self.val_Iou[1], "Class 3": self.val_Iou[2],
                                    "Class 4": self.val_Iou[3],
                                    "Class 5": self.val_Iou[4], "Class 6": self.val_Iou[5], "Class 7": self.val_Iou[6],
                                    "Class 8": self.val_Iou[7], "Class 9": self.val_Iou[8]}, global_step=epoch)
            for i in range(config.num_classes):
                writer.add_scalars("Validation All F1",
                                   {"Class 1": self.val_F1s[0], "Class 2": self.val_F1s[1], "Class 3": self.val_F1s[2],
                                    "Class 4": self.val_F1s[3],
                                    "Class 5": self.val_F1s[4], "Class 6": self.val_F1s[5], "Class 7": self.val_F1s[6],
                                    "Class 8": self.val_F1s[7], "Class 9": self.val_F1s[8]}, global_step=epoch)


        elif 'LASDU' in config.datasetClass:
            for i in range(config.num_classes):
                writer.add_scalars("Training All Iou",
                                   {"Class 1": train_ious[0], "Class 2": train_ious[1], "Class 3": train_ious[2],
                                    "Class 4": train_ious[3],
                                    "Class 5": train_ious[4]}, global_step=epoch)

            for i in range(config.num_classes):
                writer.add_scalars("Validation All Iou",
                                   {"Class 1": self.val_Iou[0], "Class 2": self.val_Iou[1], "Class 3": self.val_Iou[2],
                                    "Class 4": self.val_Iou[3],
                                    "Class 5": self.val_Iou[4]}, global_step=epoch)
            for i in range(config.num_classes):
                writer.add_scalars("Validation All F1",
                                   {"Class 1": self.val_F1s[0], "Class 2": self.val_F1s[1], "Class 3": self.val_F1s[2],
                                    "Class 4": self.val_F1s[3],
                                    "Class 5": self.val_F1s[4]}, global_step=epoch)

        print(config.AddMessage)
        print("DataName", config.dataset)
        print(f"Loss train: {loss_dict['Training loss']:.3f} "
              f" eval: {loss_dict['Validation loss']:.3f}")
        # log.info(f"Mean acc train: {acc_dicts[-1]['Training accuracy']:.3f} "
        #          f" eval: {acc_dicts[-1]['Validation accuracy']:.3f}")
        print(f"OA train: {OA_dicts['Training OA']:.3f} "
              f" eval: {self.val_OA:.3f}")

        print(f"Mean IoU train: {iou_dicts[-1]['Training IoU']:.3f} "
              f" eval: {np.mean(self.val_Iou):.3f}")
        print(f"train Iou:{train_ious}")
        print(f"val Iou:{self.val_Iou}")
        print(f"Mean F1 train: {F1_dicts[-1]['Training F1']:.3f} "
              f" eval: {np.mean(self.val_F1s):.3f}")

        print(f"train F1:{train_F1}")
        print(f"val F1:{self.val_F1s}")
        if config.saving:
            with open(join(config.saving_path, 'training.txt'), "a") as file:
                file.write('***********************************' + '\n')

                file.write(config.AddMessage + '\n')
                file.write(f"DataName:{config.dataset}" + '\n')

                message = 'Epoch{:d}  \n'
                file.write(message.format(self.epoch))

                file.write(f"Loss train: {loss_dict['Training loss']:.3f} "
                           f" eval: {loss_dict['Validation loss']:.3f}" + '\n')

                file.write(
                    f"Mean IoU train: {iou_dicts[-1]['Training IoU']:.3f} " + f" eval: {np.mean(self.val_Iou):.3f}" + '\n')

                file.write(f"train Iou:{train_ious}" + '\n')
                file.write(f"val Iou:{self.val_Iou}" + '\n')

                file.write(
                    f"Mean F1 train: {F1_dicts[-1]['Training F1']:.3f} " + f" eval: {np.mean(self.val_F1s):.3f}" + '\n')

                file.write(f"train F1:{train_F1}" + '\n')
                file.write(f"val F1:{self.val_F1s} " + '\n')

                file.write('***********************************' + '\n')

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config):
        if config.dataset_task == 'cloud_segmentation':
            self.cloud_segmentation_validation(net, val_loader, config)
        else:
            raise ValueError('No validation method implemented for this network type')

    def cloud_segmentation_validation(self, net, val_loader, config, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0  # 0.95
        softmax = torch.nn.Softmax(1)

        # # Do not validate if dataset has no validation cloud
        # if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
        #     return

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_labels]
            # self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            # i = 0
            # for label_value in val_loader.dataset.label_values:
            #     if label_value not in val_loader.dataset.ignored_labels:
            #         self.val_proportions[i] = np.sum([np.sum(labels == label_value)
            #                                           for labels in val_loader.dataset.validation_labels])
            #         i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        # last_display = time.time()
        # mean_dt = np.zeros(1)

        # t1 = time.time()

        with torch.no_grad():
            # Start validation loop
            for i, batch in enumerate(tqdm(val_loader, desc='validation')):

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                outputs = net(batch, config)

                valid_loss = net.loss(outputs, batch.labels)

                # 将GPU损失函数的结果转移到cpu上
                self.valid_losses.append(valid_loss.cpu().item())
                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                labels = batch.labels.cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):
                    # Get prediction
                    target = labels[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                       + (1 - val_smooth) * probs

                    # Stack all prediction for this epoch
                    predictions.append(probs)
                    targets.append(target)
                    if len(target) == 1:
                        print('ojofwe')
                    i0 += length

                # Average timing
                t += [time.time()]
                # mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                # if (t[-1] - last_display) > 1.0:
                #     last_display = t[-1]
                #     message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                #     print(message.format(100 * i / config.validation_size,
                #                          1000 * (mean_dt[0]),
                #                          1000 * (mean_dt[1])))

        # t2 = time.time()

        # if self.epoch >-1:
        #     if val_loader.dataset.use_potentials:
        #         val_path12 = join(config.saving_path, "potentials")
        #         if not exists(val_path12):
        #             makedirs(val_path12)
        #         pot_path = join(val_path12, "potentials_{}".format(self.epoch))
        #         if not exists(pot_path):
        #             makedirs(pot_path)
        #         files = val_loader.dataset.files
        #         for i, file_path in enumerate(files):
        #             pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
        #             ### cloud_name = file_path.split('/')[-1]
        #             cloud_name = basename(file_path)
        #             pot_name = join(pot_path, cloud_name)
        #             pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
        #             write_ply(pot_name,
        #                       [pot_points.astype(np.float32), pots],
        #                       ['x', 'y', 'z', 'pots'])

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]
            if len(np.squeeze(truth).shape) != 1:
                print('wehfohwof ')
            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        # C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)
        # Ious2=iou(C,nc_model)
        OAs = OA(C)
        F1s = F1_score(C)
        self.val_OA = OAs
        self.val_Iou = IoUs
        self.val_F1s = F1s
        t5 = time.time()
        # Saving (optionnal)
        # if config.saving:
        #
        #     # Name of saving file
        #     test_file = join(config.saving_path, 'val_IoUs.txt')
        #
        #     # Line to write:
        #     line = ''
        #     for IoU in IoUs:
        #         line += '{:.3f} '.format(IoU)
        #     line = line + '\n'
        #
        #     # Write in file
        #     if exists(test_file):
        #         with open(test_file, "a") as text_file:
        #             text_file.write(line)
        #     else:
        #         with open(test_file, "w") as text_file:
        #             text_file.write(line)

        # # Save potentials
        # if val_loader.dataset.use_potentials:
        #     pot_path = join(config.saving_path, 'potentials')
        #     if not exists(pot_path):
        #         makedirs(pot_path)
        #     files = val_loader.dataset.files
        #     for i, file_path in enumerate(files):
        #         pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
        #         ### cloud_name = file_path.split('/')[-1]
        #         cloud_name = basename(file_path)
        #         pot_name = join(pot_path, cloud_name)
        #         pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
        #         write_ply(pot_name,
        #                   [pot_points.astype(np.float32), pots],
        #                   ['x', 'y', 'z', 'pots'])

        # t6 = time.time()

        # Print instance mean
        # mIoU = 100 * np.mean(IoUs)
        # print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))
        # print('IoUs:',IoUs)
        # print('Ious2:',Ious2)
        # print('OA:',OAs)
        # # Save predicted cloud occasionally
        # if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
        #     val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch + 1))
        #     if not exists(val_path):
        #         makedirs(val_path)
        #     files = val_loader.dataset.files
        #     for i, file_path in enumerate(files):
        #
        #         # Get points
        #         points = val_loader.dataset.load_evaluation_points(file_path)
        #
        #         # Get probs on our own ply points
        #         sub_probs = self.validation_probs[i]
        #
        #         # Insert false columns for ignored labels
        #         for l_ind, label_value in enumerate(val_loader.dataset.label_values):
        #             if label_value in val_loader.dataset.ignored_labels:
        #                 sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)
        #
        #         # Get the predicted labels
        #         sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]
        #
        #         # Reproject preds on the evaluations points
        #         preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)
        #
        #         # Path of saved validation file
        #         ### cloud_name = file_path.split('/')[-1]
        #         cloud_name = basename(file_path)
        #         val_name = join(val_path, cloud_name)
        #
        #         # Save file
        #         labels = val_loader.dataset.validation_labels[i].astype(np.int32)
        #         write_ply(val_name,
        #                   [points, preds, labels],
        #                   ['x', 'y', 'z', 'preds', 'class'])



        return
