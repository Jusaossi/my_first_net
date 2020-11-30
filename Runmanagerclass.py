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

        self.epoch_train_recall = None
        self.epoch_train_tnr = None
        self.epoch_train_precision = None
        self.epoch_train_accuracy = None
        self.epoch_train_f1_score = None

        self.epoch_test_recall = None
        self.epoch_test_tnr = None
        self.epoch_test_precision = None
        self.epoch_test_accuracy = None
        self.epoch_test_f1_score = None

        self.epoch_count = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.train_batch_list = None
        self.test_batch_list = None
        self.batch_size = None
        self.tb = None

    def begin_run(self, run, train_batch_list, test_batch_list, batch_size):       # first method of the class
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        # print(f'run nro = {self.run_count}')
        self.train_batch_list = train_batch_list
        self.test_batch_list = test_batch_list
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

        self.epoch_train_recall = 0
        self.epoch_train_tnr = 0
        self.epoch_train_precision = 0
        self.epoch_train_accuracy = 0
        self.epoch_train_f1_score = 0

        self.epoch_test_recall = 0
        self.epoch_test_tnr = 0
        self.epoch_test_precision = 0
        self.epoch_test_accuracy = 0
        self.epoch_test_f1_score = 0

        # print(f'epoch nro = {self.epoch_count}')

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss_here = round(self.epoch_loss / len(self.train_batch_list), 4)
        test_loss_here = round(self.epoch_test_loss, 4)

        train_recall_here = round(self.epoch_train_recall / len(self.train_batch_list), 4)
        test_recall_here = round(self.epoch_test_recall / len(self.test_batch_list), 4)
        train_tnr_here = round(self.epoch_train_tnr / len(self.train_batch_list), 4)
        test_tnr_here = round(self.epoch_test_tnr / len(self.test_batch_list), 4)
        train_precision_here = round(self.epoch_train_precision / len(self.train_batch_list), 4)
        test_precision_here = round(self.epoch_test_precision / len(self.test_batch_list), 4)
        train_accuracy_here = round(self.epoch_train_accuracy / len(self.train_batch_list), 4)
        test_accuracy_here = round(self.epoch_test_accuracy / len(self.test_batch_list), 4)
        train_f1_score_here = round(self.epoch_train_f1_score / len(self.train_batch_list), 4)
        test_f1_score_here = round(self.epoch_test_f1_score / len(self.test_batch_list), 4)

        self.tb.add_scalar('Train_Loss', loss_here, self.epoch_count)
        self.tb.add_scalar('Test_Loss', test_loss_here, self.epoch_count)

        self.tb.add_scalar('Train Recall', train_recall_here, self.epoch_count)
        self.tb.add_scalar('Test Recall', test_recall_here, self.epoch_count)

        self.tb.add_scalar('Train True negative ratio', train_tnr_here, self.epoch_count)
        self.tb.add_scalar('Test True negative ratio', test_tnr_here, self.epoch_count)

        self.tb.add_scalar('Train Precision', train_precision_here, self.epoch_count)
        self.tb.add_scalar('Test Precision', test_precision_here, self.epoch_count)

        self.tb.add_scalar('Train Accuracy', train_accuracy_here, self.epoch_count)
        self.tb.add_scalar('Test Accuracy', test_accuracy_here, self.epoch_count)

        self.tb.add_scalar('Train F1-score', train_f1_score_here, self.epoch_count)
        self.tb.add_scalar('Test F1-score', test_f1_score_here, self.epoch_count)
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

    def track_num_correct(self, train_recall, train_true_negative_rate, train_precision, train_accuracy, train_f1_score):
        self.epoch_train_recall += train_recall
        self.epoch_train_tnr += train_true_negative_rate
        self.epoch_train_precision += train_precision
        self.epoch_train_accuracy += train_accuracy
        self.epoch_train_f1_score += train_f1_score

    def track_test_num_correct(self, test_recall, test_true_negative_rate, test_precision, test_accuracy, test_f1_score):
        self.epoch_test_recall = test_recall
        self.epoch_test_tnr = test_true_negative_rate
        self.epoch_test_precision = test_precision
        self.epoch_test_accuracy = test_accuracy
        self.epoch_test_f1_score = test_f1_score
