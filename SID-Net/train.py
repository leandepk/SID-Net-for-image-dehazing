# --- Imports --- #
from __future__ import print_function  
import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
# from model.network_1 import Net
# from model.FFA import Net
 from model.network_unet import Net
#from model.new_network import Net
# from modules.UNet_arch import HDRUNet as Net
# from model.network_new1 import Net
# from model.model import Net

# from model.UNet_arch import HDRUNet as Net

from datasets.datasets import DehazingDataset,DehazingDataset_unlabel
from os.path import exists, join, basename
import time
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from loss.edg_loss import edge_loss
from utils import to_psnr, validation, print_log , MovingAvg, to_compute_loss
import os
import numpy as np
from logger import Logger
from SSIM import SSIMLoss, tv_loss_f
from vgg_loss.vgg_loss import Perceptual_loss134
# from ECLoss import ECLoss
from ECLoss import DCLoss,BCLoss
from CR import ContrastLoss
from color_loss import ColorLoss


import time



# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--patch_size', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument('--net', default='', help="path to net_Dehazing (to continue training)")
parser.add_argument('--continueEpochs', type=int, default=0, help='continue epochs')
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[0], type=list)
parser.add_argument('--category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)
parser.add_argument('--save_latest_freq', type=int, default=50000)
parser.add_argument('--checkpoints_dir', default='checkpoints')
parser.add_argument('--name', default='test_dark_channel')
parser.add_argument('--save_epoch', type=int, default=2)
parser.add_argument('--loss_data', default='Loss_data')
parser.add_argument('--log_data', default='Log_data')
opt = parser.parse_args()
print(opt)


# ---  hyper-parameters for training and testing the neural network --- #
# train_data_dir = "../data_set/SOTS/train/OTS/"
# train_data_dir = "../data_set/SOTS/train/ITS/"
# train_data_dir = "../data_set/OHAZE/train/" #OHAZE
# train_data_dir = "../data_set/NHHAZE/train/" # NHHAZE
train_data_dir = "../data_set/DENSE-HAZE/train/" # DEHAZE

train_batch_size = opt.batchSize
val_batch_size = opt.testBatchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = opt.n_GPUs
category = opt.category
continueEpochs = opt.continueEpochs
checkpoints_dir = opt.checkpoints_dir
name = opt.name
loss_data = opt.loss_data
log_data = opt.log_data
# chech_dir path
if not os.path.exists(os.path.join(checkpoints_dir,name)):
    os.makedirs(os.path.join(opt.checkpoints_dir,name))
# loss_data path
if not os.path.exists(os.path.join(checkpoints_dir,name,loss_data)):
    os.makedirs(os.path.join(opt.checkpoints_dir,name,loss_data))
# log_data path
if not os.path.exists(os.path.join(checkpoints_dir,name,log_data)):
    os.makedirs(os.path.join(opt.checkpoints_dir,name,log_data))


# --- Set category-specific hyper-parameters  --- #
if category == 'indoor':
    val_data_dir = '../data_set/SOTS/test/SOTS/indoor/'
elif category == 'outdoor':
    val_data_dir = '../data_set/SOTS/test/SOTS/outdoor/'

elif category == 'OHAZE':
    val_data_dir = '../data_set/OHAZE/test/'

elif category == 'NHHAZE':
    val_data_dir = '../data_set/NHHAZE/test/'

elif category == 'DEHAZE':
    val_data_dir = '../data_set/DENSE-HAZE/test/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')


device_ids = GPUs_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
print('===> Building model')
# model = Net(gps=3,blocks=10)
model = Net()

# define logger

logger = Logger(os.path.join(checkpoints_dir,name,log_data))

# avg_loss
avg_loss = MovingAvg(100)

# --- Define the MSE loss --- #
# MSELoss= nn.SmoothL1Loss() # mae l1 loss
MSELoss = nn.MSELoss() # mse l2 loss
MSELoss = MSELoss.to(device)

# --- Define the SSIM loss --- #

SSIMLoss = SSIMLoss()
SSIMLoss= SSIMLoss.to(device)

# dark channel loss


vgg_loss = Perceptual_loss134()
vgg_loss = vgg_loss.to(device)

# color loss

color_loss = ColorLoss()
color_loss = color_loss.to(device)

#
duibi_loss = ContrastLoss()
TVLossL1 = tv_loss_f()

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)


# --- Build optimizer and scheduler --- #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = StepLR(optimizer,step_size= 20,gamma=0.5)



# --- Load training data and validation/test data --- #
train_dataset = DehazingDataset(root_dir=train_data_dir,crop=True,crop_size=256, transform=transforms.Compose([transforms.ToTensor()]),multi_scale=False)
train_dataloader = DataLoader(dataset = train_dataset, batch_size=train_batch_size, num_workers = data_threads, shuffle=True,)

# umlabeled image path
# unlabel_path = '../data_set/un_labeled/'
# train_dataset_unlabel = DehazingDataset_unlabel(root_dir=unlabel_path,crop=True,crop_size=128, transform=transforms.Compose([transforms.ToTensor()]),multi_scale=False)
# train_dataloader_unlabel = DataLoader(dataset = train_dataset_unlabel, batch_size=1, num_workers = data_threads, shuffle=True,)

test_dataset = DehazingDataset(root_dir = val_data_dir,crop=False,crop_size=2048, transform = transforms.Compose([transforms.ToTensor()]), train=False,multi_scale=True)
test_dataloader = DataLoader(test_dataset, batch_size = val_batch_size, num_workers = data_threads, shuffle=False)


losses = []
start_epoch = 0
all_iteration = 0

# retrain model when broke
# print(opt.recontinue)
# if os.path.exists(os.path.join(checkpoints_dir, name, 'latest.pth')):
#     print('resuming from latest.pth')
#     print(os.path.join(checkpoints_dir, name, 'latest.pth'))
#     latest_info = torch.load(os.path.join(checkpoints_dir, name, 'latest.pth'))
#     start_epoch = latest_info['epoch']
#     all_iteration = latest_info['total_iter']
#     # if isinstance(model, torch.nn.DataParallel):
#     #     model.module.load_state_dict(latest_info['net_state'])
#     # else:
#     model.load_state_dict(latest_info['net_state'])
#     optimizer.load_state_dict(latest_info['optim_state'])

if opt.continueEpochs > 0:
    start_epoch = opt.continueEpochs
    all_iteration = opt.continueEpochs * len(train_dataloader)
    resume_path = os.path.join(checkpoints_dir,name, '{}_haze_{}.pth'.format(category,str(start_epoch)))
    print('resume from : %s' % resume_path)
    assert os.path.exists(resume_path), 'cannot find the resume model: %s ' % resume_path
    model.load_state_dict(torch.load(resume_path))
#
old_val_psnr, old_val_ssim = validation(model, test_dataloader, device, category)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

train_loss = []
testing_loss = []

for epoch in range(start_epoch, opt.nEpochs):
    print("Training...")
    scheduler.step(epoch)
    epoch_loss = 0
    psnr_list = []

    train_loss_epoch = []
    for iteration, inputs in enumerate(train_dataloader,1):
        all_iteration += 1
        haze1,gt = Variable(inputs['hazy_image']), Variable(inputs['clear_image'])
        # print(gt.shape[0])
        haze1 = haze1.to(device)
        gt = gt.to(device)
        optimizer.zero_grad()

        model.train()
        # detail_gt = detail_gt.to(device)
        # dehaze = model([haze1,haze1])
        dehaze = model(haze1)
        # --- Forward + Backward + Optimize --- #
        MSE_loss1 = MSELoss(dehaze, gt)
        SSIM_loss1 = SSIMLoss(dehaze,gt)
        # VGG_loss1 = vgg_loss(dehaze,gt)
        # print(dehaze.shape)
        # print(gt.shape)
        # print(haze1.shape)
        # dui bi loss
        c_loss = duibi_loss(dehaze,gt,gt)
        # c_loss_value = color_loss(dehaze,gt)
        # dui bi loss
        # print(c_loss_value)

        # Loss = MSE_loss1+0.01*c_loss+0.001*c_loss_value
        # Loss = MSE_loss1+0.001*c_loss_value
        Loss = MSE_loss1+0.1*c_loss+0.05*(1-SSIM_loss1)
        # Loss = MSE_loss1
        # Loss = MSE_loss1+0.02*VGG_loss1+0.05*(1-SSIM_loss1)
        # Loss = MSE_loss1
        # Loss = MSE_loss1+0.5*adversarial_loss+0.1*dark_loss+0.1*bright_loss
        Loss.backward()
        optimizer.step()
        loss_avg = avg_loss.set_curr_val(Loss.item())
        # if iteration % 100 == 0:
        # save training loss value
        if iteration % 100 == 0:
            train_loss_epoch.append(loss_avg)
        # train_loss_epoch.append(loss_avg)
        losses.append(loss_avg)
        if iteration % 100 == 0:
            # print("===>Epoch[{}]({}/{}):Avg Loss: {:.4f} Loss: {:.4f} MSELoss: {:.4f} SSIMLoss: {:.4f} vggLoss: {:.4f} c_loss: {:.4f} color_loss: {:.4f} lr: {:.6f}".format(epoch, iteration, len(train_dataloader), loss_avg,Loss.item(), MSE_loss1.item(), 0.05*(1-SSIM_loss1.item()),0.2*VGG_loss1.item(),0.01*c_loss.item(),0.01*c_loss_value.item(),optimizer.param_groups[0]['lr']))
            print("===>Epoch[{}]({}/{}):Avg Loss: {:.4f} Loss: {:.4f} MSELoss: {:.4f}  c_loss: {:.4f}  lr: {:.6f}".format(epoch, iteration, len(train_dataloader), loss_avg,Loss.item(), MSE_loss1.item(), 0.1*c_loss.item(),optimizer.param_groups[0]['lr']))
            # print("===>Epoch[{}]({}/{}):Avg Loss: {:.4f} Loss: {:.4f} MSELoss: {:.4f}   lr: {:.6f}".format(epoch, iteration, len(train_dataloader), loss_avg,Loss.item(), MSE_loss1.item(),optimizer.param_groups[0]['lr']))
            # TensorboardX
            info = {'loss':loss_avg}
            for tag, value in info.items():
                logger.scalar_summary(tag, value,  all_iteration)
        if all_iteration % opt.save_latest_freq == 0:
            latest_info = {'total_iter': all_iteration,
                           'epoch': epoch,
                           'optim_state': optimizer.state_dict()}
            latest_info['net_state'] = model.state_dict()
            print('save lastest model.')
            torch.save(latest_info, os.path.join(checkpoints_dir,name,'latest.pth'))
        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))


    # save loss
    avg_loss_epoch = sum(train_loss_epoch) / len(train_loss_epoch)
    train_loss.append(avg_loss_epoch)
    loss0 = np.array(train_loss)
    np.savetxt(os.path.join(checkpoints_dir,name,loss_data,'train_epoch{}'.format(epoch)),loss0)

    # print("===>start fine tine")
    #
    # for i, inputs in enumerate(train_dataloader_unlabel,1):
    #     haze_input, unlabel_clear = Variable(inputs['hazy_image']), Variable(inputs['clear_image'])
    #     # print(gt.shape[0])
    #     haze_input = haze_input.to(device)
    #     unlabel_clear = unlabel_clear.to(device)
    #     # detail_gt = detail_gt.to(device)
    #     optimizer.zero_grad()
    #     model.train()
    #     dehaze = model(haze_input)
    #     # unsp loss
    #     tv_loss = TVLossL1(dehaze)
    #     # dark channel loss
    #     dc_loss = DCLoss(dehaze)
    #     # con loss


        # loss_total = 0.1*tv_loss + 0.1*dc_loss
        # # loss_total = 0.1*tv_loss + 0.1*dc_loss + 0.01*c_loss
        #
        # loss_total.backward()
        # optimizer.step()
        # if i % 100 == 0:
        #     print("===>Epoch[{}]({}/{}):total loss: {:.4f} tv_loss: {:.4f} dc_loss: {:.4f} duibi_loss: {:.4f}".format(epoch, i, len(train_dataloader), loss_total.item(),tv_loss.item(),dc_loss.item(),c_loss.item()))
        #     # TensorboardX
        # # psnr_list.extend(to_psnr(dehaze, gt))

    train_psnr = sum(psnr_list) / len(psnr_list)
    # --- Save the network  --- #
    torch.save(model.state_dict(),
               os.path.join(checkpoints_dir, name, '{}_haze_{}.pth'.format(category, str(epoch + 1))))

    # --- Use the evaluation model in testing --- #
    # model.eval()
    val_psnr, val_ssim = validation(model, test_dataloader, device, category, save_tag=True)

    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(model.state_dict(), os.path.join(checkpoints_dir, name, '{}_haze_best.pth'.format(category)))
        old_val_psnr = val_psnr
    path = os.path.join(checkpoints_dir, name)
    print_log(path, epoch + 1, train_epoch, train_psnr, val_psnr, val_ssim, category)








    




