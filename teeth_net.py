import torch
import time
import numpy as np
import torch.optim as optim
import u_net_versions
import my_loss_classes  # import MyBceLOSS, MyWeightedBceLOSS
from collections import OrderedDict
from runbuilderclass import RunBuilder
from Runmanagerclass import RunManager
from MyDatasetLoader import my_data_loader
from load_my_batch import load_my_image_batch, load_my_target_batch
from l1_norm import calculate_l1, calculate_teeth_pixels, calculate_my_metrics, calculate_my_sets
from my_albumations import my_data_albumentations, my_data_albumentations2, my_data_albumentations3
import os
import platform
import csv


machine = platform.node()
torch.cuda.empty_cache()

time_str = time.strftime("%Y-%m-%d_%H-%M")
# data_folders = ['data', 'data_new', 'data_teeth']   scale=['[0,1]', '[-1,1]']  (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)
# --------------------------------------------------------------------variables for runs------------------------------
test_train_split = 5   # 20 % for test, loss_weight=[0.5, 0.9], loss_gamma=[0.5, 1, 2, 5] 'MyDiceLoss', 'MyDiceBCELoss', 'MyIoULoss', 'MyTverskyLoss', 'MyFocalTverskyLoss'
epoch_numbers = 40     # Gated_UNet  UNetQuarter 'MyFocalLoss', 'MyMixedLoss', 'MyLogDiceLoss', 'MyDiceBCELoss', 'MyLogDiceBCELoss'..... albu_prob=[(1, 1, 1)],
run_data = 'data_teeth'                       # 'Blur', 'MotionBlur', 'RandomGamma', 'MedianBlur', 'RandomBrightnessContrast'
                        # 'Resize', 'RandomCrop', 'HorizontalFlip', 'GridDistortion', 'ElasticTransform', 'ShiftScaleRotate'
                        # 'MaskDropout', 'RandomGridShuffle', 'OpticalDistortion', 'no_augmentation', 'Rotate'
                        # albu=['Transpose', 'RandomRotate90', 'VerticalFlip', 'CenterCrop', 'RandomSizedCrop'] albu=['RandomGamma_RandomCrop'], albu_prob=[0.20]
                        # '[0,1]', '[-1,1]',, 'norm'  , lower_cut=[200]
params = OrderedDict(unet=['UNetHalf'], scale=['my_shift_and_[0,1]'],
                     loss=['MyTverskyBceLoss'], lr=[0.0004], albu=['no_augmentation'], albu_prob=[0.2, 0.4], blur=[3, 5, 7])
# ----------------------------------------------------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------------------------------------------------
if machine == 'DESKTOP-K3R0DFP':
    my_parent_dir = r'C:\Users\jpkorpel\PycharmProjects\uusi_sika'
    my_parent_dir_2 = r'C:\Users\jpkorpel\PycharmProjects'
    my_save_path_help = r'C:\Users\jpkorpel\PycharmProjects\my_first_net'
    my_save_path = os.path.join(my_save_path_help, 'runs')
    # save_file = os.path.join(my_save_path, time_str + '_Teeth_net_results')
    # save_file_new = os.path.join(my_save_path, time_str + '_Teeth_net_results_train_batch.csv')
    # save_file_new_2 = os.path.join(my_save_path, time_str + '_Teeth_net_results_test_batch.csv')
    card = 'cpu'
elif machine == 'siiri-desktop':
    my_parent_dir = os.path.dirname(os.getcwd())
    # my_save_path = os.path.join(my_parent_dir, 'teeth_net/results')
    # save_file = os.path.join(my_save_path, time_str + '_Teeth_net_results')
    # save_file_new = os.path.join(my_save_path, time_str + '_Teeth_net_results_train_batch.csv')
    # save_file_new_2 = os.path.join(my_save_path, time_str + '_Teeth_net_results_test_batch.csv')
    card = 'cuda'
else:
    my_parent_dir = os.path.dirname(os.getcwd())
    my_save_path = os.path.join(my_parent_dir, 'my_first_net')
    # save_file = os.path.join(my_save_path, time_str + '_Teeth_net_results')
    # save_file_new = os.path.join(my_save_path, time_str + '_Teeth_net_results_train_batch.csv')
    # save_file_new_2 = os.path.join(my_save_path, time_str + '_Teeth_net_results_test_batch.csv')
    card = 'cuda'

# header = ['Run', 'Unet', 'Loss_function', 'Epoch', 'Batch', 'Data_set', 'Patient', 'Slices', 'Lr', 'Batch_size', 'Train_Batch_Loss', 'Average_train_loss', 'Train_Batch_l1_loss',
#           'Average_Train_l1_loss', 'Train_Batch_correct_voxels', 'Train_Batch_all_voxels', 'Train_Epoch_correct_voxels', 'Train_Epoch_all_voxels', 'Train_Epoch_accuracy']
#
# header2 = ['Run', 'Unet', 'Loss_function', 'Epoch', 'Batch', 'Data_set', 'Patient', 'Slices', 'Lr', 'Batch_size', 'Test_Batch_Loss',
#            'Test_Batch_l1_loss', 'Test_sum_Batch_correct', 'Test_sum_all_batch_voxels']


# if not os.path.isfile(save_file_new):
#     with open(save_file_new, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#
# if not os.path.isfile(save_file_new_2):
#     with open(save_file_new_2, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header2)


device = torch.device(card)
manager = RunManager()
runs_count = 0

for run in RunBuilder.get_runs(params):
    if machine == 'DESKTOP-K3R0DFP':
        my_data_folder = 'teeth_net' + '\\' + run_data
        my_path = os.path.join(my_parent_dir, my_data_folder)
    elif machine == 'siiri-desktop':
        my_parent_dir = os.path.dirname(os.getcwd())
        my_path = os.path.join(my_parent_dir, 'harjoitus_verkko/data')
    else:
        my_data_folder = 'teeth_net/' + run_data
        my_path = os.path.join(my_parent_dir, my_data_folder)

    np.random.seed(2020)
    runs_count += 1
    network = getattr(u_net_versions, run.unet)()

    print('run:', run)
    loss_function = getattr(my_loss_classes, run.loss)()
    # print(loss_function)

    network.to(device=device)

    optimizer = optim.Adam(network.parameters(), lr=run.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3, last_epoch=-1)
    train_batch_list, test_batch_list, train_dict = my_data_loader(run_data, test_train_split, data_shuffle=True, batch_size=1)
    manager.begin_run(run, train_batch_list, test_batch_list, 1)
    my_f1_max_score = 0
    my_f1_score = 0
    
    for epoch in range(1, epoch_numbers + 1):
        network.train()
        print(f'run = {runs_count}, epoch = {epoch}')
        #print('Epoch {}, lr {}'.format(
        #    epoch, optimizer.param_groups[0]['lr']))

        manager.begin_epoch()
        batch_count = 0
        epoch_loss = 0
        # epoch_l1_loss = 0
        # epoch_no_correct = 0
        # epoch_all_teeth = 0
        epoch_tp = 0
        epoch_fp = 0
        epoch_fn = 0

        for batch in train_batch_list:
            patient = train_dict[batch][0]
            patient_slices = train_dict[batch][1]
            batch_count += 1
            #print('batch nro =', batch_count)
            #print('run.batch_size', run.batch_size)
            if batch_count == 5 and machine == 'DESKTOP-K3R0DFP':
                break

            images = load_my_image_batch(batch, train_dict, my_path, train_batch_size=1, normalize=run.scale, lower_cut=200)

            targets = load_my_target_batch(batch, train_dict, my_path, train_batch_size=1)
            # print('images shape=', images.shape)
            # print('targets shape=', targets.shape)
            # if run.albu:
            #     images, targets = my_data_albumentations(images, targets, run.albu_prob)
            #     #print('albu megessä')
            if run.albu != 'no_augmentation':
                images, targets = my_data_albumentations2(images, targets, run.albu, run.albu_prob, run.blur)

            # images, targets = my_data_albumentations3(images, targets, run.albu, run.albu_prob)
            # print('images shape=', images.shape)
            # print('targets shape=', targets.shape)

            images = torch.as_tensor(images, dtype=torch.float32)
            images = images.unsqueeze(1)
            images = images.to(device)
            preds = network(images)

            targets = torch.as_tensor(targets, dtype=torch.float32)
            targets = targets.unsqueeze(1)
            targets = targets.to(device)
            # print('images shape=', images.shape)
            # print('targets shape=', targets.shape)
            # print(run.alpha)

            #if run.loss == 'MyDiceLoss':
            loss = loss_function(preds, targets, alpha=1)
            # print(loss)

            # else:
            #    loss = loss_function(preds, targets, run.loss_weight, run.loss_gamma)
            # print(loss)

            batch_loss = loss.item()
            manager.track_loss(batch_loss)

            # epoch_loss += batch_loss
            # average_epoch_loss = epoch_loss / batch_count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # batch_l1_loss = calculate_l1(preds.detach(), targets.detach())
            recall, precision, f1_score = calculate_my_metrics(preds.detach(), targets.detach())
            TP, FP, FN = calculate_my_sets(preds.detach(), targets.detach())

            epoch_tp += TP
            epoch_fp += FP
            epoch_fn += FN
            # batch_correct_teeth, batch_teeth_all, batch_correct_no_teeth, batch_no_teeth_all = calculate_teeth_pixels(preds.detach(), targets.detach())
            # batch_no_correct = batch_teeth_all - batch_correct_teeth + batch_no_teeth_all - batch_correct_no_teeth
            # print('batch_no_correct', batch_no_correct)
            # epoch_no_correct += batch_no_correct
            # epoch_teeth_correct += batch_correct_teeth
            # epoch_all_teeth += batch_teeth_all
            # epoch_accuracy = epoch_teeth_correct / epoch_all_teeth

            manager.track_num_correct(recall, precision, f1_score)


            # if batch % 1 == 0:
            #     with open(save_file_new, 'a', newline='') as f:
            #         result = [runs_count, run.unet, run.loss, epoch, batch_count, run.data, patient, str(patient_slices), run.lr, run.batch_size, round(batch_loss, 4),
            #                   round(average_epoch_loss, 4), round(batch_l1_loss, 4), round(average_l1_epoch_loss, 4), batch_correct_teeth,
            #                   batch_teeth_all,  epoch_teeth_correct, epoch_all_teeth, round(epoch_accuracy, 4)]
            #
            #         writer = csv.writer(f)
            #         writer.writerow(result)
        # print('epoch_no_correct =', epoch_no_correct)

        manager.track_test_epoch_metrics(epoch_tp, epoch_fp, epoch_fn)
        torch.cuda.empty_cache()
        test_count = 0
        test_epoch_loss = 0
        t_epoch_recall = 0
        # t_epoch_true_negative_rate = 0
        t_epoch_precision = 0
        # t_epoch_accuracy = 0
        t_epoch_f1_score = 0
        test_batch_size = 1
        test_epoch_tp = 0
        test_epoch_fp = 0
        test_epoch_fn = 0
        network.eval()
        for test_batch in test_batch_list:
            test_patient = train_dict[test_batch][0]
            test_slices = train_dict[test_batch][1]
            test_count += 1
            if test_count == 3 and machine == 'DESKTOP-K3R0DFP':
                break
            images = load_my_image_batch(test_batch, train_dict, my_path, test_batch_size, normalize=run.scale, lower_cut=200)
            targets = load_my_target_batch(test_batch, train_dict, my_path, test_batch_size)

            images = torch.as_tensor(images, dtype=torch.float32)
            images = images.unsqueeze(1)
            images = images.to(device)
            preds = network(images)

            targets = torch.as_tensor(targets, dtype=torch.float32)
            targets = targets.unsqueeze(1)
            targets = targets.to(device)

            test_loss = loss_function(preds.detach(), targets.detach(), alpha=1)

            test_epoch_loss += test_loss.item()

            t_recall, t_precision, t_f1_score = calculate_my_metrics(preds.detach(), targets.detach())
            t_epoch_recall += t_recall
            # t_epoch_true_negative_rate += t_true_negative_rate
            t_epoch_precision += t_precision
            # t_epoch_accuracy += t_accuracy
            t_epoch_f1_score += t_f1_score
            test_TP, test_FP, test_FN = calculate_my_sets(preds.detach(), targets.detach())

            test_epoch_tp += test_TP
            test_epoch_fp += test_FP
            test_epoch_fn += test_FN
            # if test_batch % 1 == 0:
            #     with open(save_file_new_2, 'a', newline='') as f:
            #         result = [runs_count, run.unet, run.loss, epoch, test_count, run.data, test_patient, str(test_slices),
            #                   run.lr, test_batch_size, round(test_loss.item(), 4),
            #                   round(test_batch_l1_loss, 4), test_batch_correct_teeth, test_batch_teeth_all]
            #
            #         writer = csv.writer(f)
            #         writer.writerow(result)

        epoch_test_recall = test_epoch_tp / (test_epoch_tp + test_epoch_fn)
        epoch_test_precision = test_epoch_tp / (test_epoch_tp + test_epoch_fp)
        epoch_test_f1_score = (2 * epoch_test_precision * epoch_test_recall) / (epoch_test_precision + epoch_test_recall)
        #print('paskaa')
        manager.track_test_loss(test_epoch_loss, test_count)

        manager.track_test_num_correct(t_epoch_recall, t_epoch_precision, t_epoch_f1_score)
        manager.track_test_true_epoch_metrics(epoch_test_recall, epoch_test_precision, epoch_test_f1_score)

        # if epoch_test_f1_score > my_f1_score:
        #     my_f1_score = epoch_test_f1_score
        #     print('model now save, epoch =', epoch)
        #     print('epoch_test_f1_score:', epoch_test_f1_score)
        #     torch.save(network, my_save_path + '\\' + 'two_augh_network_60_epoch.pth')
        # # # scheduler.step()
        manager.end_epoch()
        torch.cuda.empty_cache()
    manager.end_run()
# manager.save(save_file)






