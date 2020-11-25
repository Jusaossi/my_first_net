import time
import torch
import torchvision
import pandas as pd
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
class RunManager:
    def __init__(self):       # class constructor

        self.epoch_loss = None
        self.epoch_test_loss = None
        self.epoch_l1_loss = None
        self.epoch_test_l1_loss = None
        self.epoch_num_correct_teeth = None
        self.epoch_num_all_teeth = None
        self.epoch_num_correct_no_teeth = None
        self.epoch_num_all_no_teeth = None
        self.epoch_test_num_correct_teeth = None
        self.epoch_test_num_all_teeth = None
        self.epoch_test_num_correct_no_teeth = None
        self.epoch_test_num_all_no_teeth = None
        self.epoch_count = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.train_batch_list = None
        self.batch_size = None
        self.tb = None

    def begin_run(self, run, train_batch_list, batch_size):       # first method of the class
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        # print(f'run nro = {self.run_count}')
        self.train_batch_list = train_batch_list
        self.batch_size = batch_size
        self.tb = SummaryWriter(comment=f'-{run}')

        # print(len(self.train_batch_list))
        # print(self.batch_size)
        # print((len(self.train_batch_list) * self.batch_size))
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_test_loss = 0
        self.epoch_l1_loss = 0
        self.epoch_test_l1_loss = 0
        self.epoch_num_correct_teeth = 0
        self.epoch_num_all_teeth = 0
        self.epoch_num_correct_no_teeth = 0
        self.epoch_num_all_no_teeth = 0
        self.epoch_test_num_correct_teeth = 0
        self.epoch_test_num_all_teeth = 0
        self.epoch_test_num_correct_no_teeth = 0
        self.epoch_test_num_all_no_teeth = 0
        # print(f'epoch nro = {self.epoch_count}')

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss_here = round(self.epoch_loss / len(self.train_batch_list), 4)
        test_loss_here = round(self.epoch_test_loss, 4)
        l1_loss_here = round(self.epoch_l1_loss / len(self.train_batch_list), 4)
        test_l1_loss_here = round(self.epoch_test_l1_loss, 4)

        train_accuracy = round(self.epoch_num_correct_teeth / self.epoch_num_all_teeth, 4)
        train_accuracy_no_teeth = round(self.epoch_num_correct_no_teeth / self.epoch_num_all_no_teeth, 4)

        test_accuracy = round(self.epoch_test_num_correct_teeth / self.epoch_test_num_all_teeth, 4)
        test_accuracy_no_teeth = round(self.epoch_test_num_correct_no_teeth / self.epoch_test_num_all_no_teeth, 4)

        train_wrong_teeth_voxels = round(self.epoch_num_all_teeth - self.epoch_num_correct_teeth, 4)
        train_wrong_no_teeth_voxels = round(self.epoch_num_all_no_teeth - self.epoch_num_correct_no_teeth, 4)
        train_wrong_voxels = train_wrong_teeth_voxels + train_wrong_no_teeth_voxels

        test_wrong_teeth_voxels = round(self.epoch_test_num_all_teeth - self.epoch_test_num_correct_teeth, 4)
        test_wrong_no_teeth_voxels = round(self.epoch_test_num_all_no_teeth - self.epoch_test_num_correct_no_teeth, 4)
        test_wrong_voxels = test_wrong_teeth_voxels + test_wrong_no_teeth_voxels

        self.tb.add_scalar('Train_Loss', loss_here, self.epoch_count)
        self.tb.add_scalar('Train_l1_Loss', l1_loss_here, self.epoch_count)
        self.tb.add_scalar('Train_Accuracy', train_accuracy, self.epoch_count)
        self.tb.add_scalar('Train_Accuracy_no_teeth', train_accuracy_no_teeth, self.epoch_count)
        self.tb.add_scalar('Test_Loss', test_loss_here, self.epoch_count)
        self.tb.add_scalar('Test_l1_Loss', test_l1_loss_here, self.epoch_count)
        self.tb.add_scalar('Test_Accuracy_of_teeth', test_accuracy, self.epoch_count)
        self.tb.add_scalar('Test_Accuracy_no_teeth', test_accuracy_no_teeth, self.epoch_count)

        self.tb.add_scalar('Train_wrong_teeth_voxels', train_wrong_teeth_voxels, self.epoch_count)
        self.tb.add_scalar('Train_wrong_no_teeth_voxels', train_wrong_no_teeth_voxels, self.epoch_count)
        self.tb.add_scalar('Train_wrong_voxels', train_wrong_voxels, self.epoch_count)

        self.tb.add_scalar('Test_wrong_teeth_voxels', test_wrong_teeth_voxels, self.epoch_count)
        self.tb.add_scalar('Test_wrong_no_teeth_voxels', test_wrong_no_teeth_voxels, self.epoch_count)
        self.tb.add_scalar('Test_wrong_voxels', test_wrong_voxels, self.epoch_count)

    #     results = OrderedDict()
    #     results["run"] = self.run_count
    #     results["epoch"] = self.epoch_count
    #     results['Train_loss'] = loss_here
    #     results['Test_loss'] = test_loss_here
    #     results['Train_l1_loss'] = l1_loss_here
    #     results['Test_l1_loss'] = test_l1_loss_here
    #     results['Train_sum_correct_voxels'] = self.epoch_num_correct_teeth
    #     results['Train_sum_all_teeth'] = self.epoch_num_all_teeth
    #     results['Train_sum_correct_no_teeth_voxels'] = self.epoch_num_correct_no_teeth
    #     results['Train_sum_all_no_teeth'] = self.epoch_num_all_no_teeth
    #     results['Test_sum_correct_voxels'] = self.epoch_test_num_correct_teeth
    #     results['Test_sum_all_teeth'] = self.epoch_test_num_all_teeth
    #     results['Test_sum_correct_no_teeth_voxels'] = self.epoch_test_num_correct_no_teeth
    #     results['Test_sum_all_no_teeth'] = self.epoch_test_num_all_no_teeth
    #     results["train_accuracy"] = train_accuracy
    #     results["test_accuracy"] = test_accuracy
    #     results["train_accuracy_no_teeth"] = train_accuracy_no_teeth
    #     results["test_accuracy_no_teeth"] = test_accuracy_no_teeth
    #     results["train_wrong_teeth_voxels"] = train_wrong_teeth_voxels
    #     results["train_wrong_no_teeth_voxels"] = train_wrong_no_teeth_voxels
    #     results["train_wrong_voxels"] = train_wrong_voxels
    #     results["test_wrong_teeth_voxels"] = test_wrong_teeth_voxels
    #     results["test_wrong_no_teeth_voxels"] = test_wrong_no_teeth_voxels
    #     results["test_wrong_voxels"] = test_wrong_voxels
    #     results['epoch_duration'] = round(epoch_duration, 1)
    #     results['run_duration'] = round(run_duration, 1)
    #
    #     for k, v in self.run_params._asdict().items():
    #         results[k] = v
    #     self.run_data.append(results)
    #
    # def save(self, filename):
    #     pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{filename}.csv')

    def track_loss(self, loss_item):
        self.epoch_loss += loss_item
        # print('self.epoch_loss =', self.epoch_loss)

    def track_test_loss(self, test_epoch_loss, test_count):
        self.epoch_test_loss = test_epoch_loss / test_count
        # print('self.epoch_loss =', self.epoch_loss)

    def track_l1_loss(self, l1_loss_item):
        self.epoch_l1_loss += l1_loss_item
        # print('self.epoch_l1_loss =', self.epoch_l1_loss)

    def track_test_l1_loss(self, test_l1_epoch_loss, test_count):
        self.epoch_test_l1_loss = test_l1_epoch_loss / test_count
        # print('self.epoch_loss =', self.epoch_loss)

    def track_num_correct(self, train_epoch_teeth_correct, train_epoch_all_teeth, train_epoch_no_teeth_correct, train_epoch_all_no_teeth):
        self.epoch_num_correct_teeth += train_epoch_teeth_correct
        self.epoch_num_all_teeth += train_epoch_all_teeth
        self.epoch_num_correct_no_teeth += train_epoch_no_teeth_correct
        self.epoch_num_all_no_teeth += train_epoch_all_no_teeth
        # print('self.epoch_num_correct_teeth =', self.epoch_num_correct_teeth)
        # print('self.epoch_num_all_teeth =', self.epoch_num_all_teeth)

    def track_test_num_correct(self, test_epoch_teeth_correct, test_epoch_all_teeth, test_epoch_no_teeth_correct, test_epoch_all_no_teeth):
        self.epoch_test_num_correct_teeth = test_epoch_teeth_correct
        self.epoch_test_num_all_teeth = test_epoch_all_teeth
        self.epoch_test_num_correct_no_teeth = test_epoch_no_teeth_correct
        self.epoch_test_num_all_no_teeth = test_epoch_all_no_teeth

