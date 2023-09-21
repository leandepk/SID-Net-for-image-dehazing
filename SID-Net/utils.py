

# --- Imports --- #
import torch.nn as nn
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
from torch.autograd import Variable
import os
from SSIM import SSIMLoss
from vgg_loss.vgg_loss import Perceptual_loss134


def to_compute_loss(net,DNet, val_data_loader, device):
    # define loss
    # --- Define the MSE loss --- #
    # MSELoss= nn.SmoothL1Loss() # mae l1 loss
    MSELoss = nn.MSELoss()  # mse l2 loss
    MSELoss = MSELoss.to(device)
    real_label = Variable(torch.ones([1, 1, 1, 1], dtype=torch.float)).cuda()

    # --- Define the SSIM loss --- #

    SSIM_Loss = SSIMLoss()
    SSIM_Loss = SSIM_Loss.to(device)
    vgg_loss = Perceptual_loss134()
    vgg_loss = vgg_loss.to(device)
    bce_loss = nn.BCELoss().cuda()

    loss_list = []
    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            haze1,  gt = Variable(val_data['hazy_image_s1']), Variable(val_data['clear_image_s1'])
            haze1 = haze1.to(device)
            gt = gt.to(device)
            dehaze = net(haze1)
            dehaze_label = DNet(haze1)
            MSE_loss1 = MSELoss(dehaze, gt)
            SSIM_loss1 = SSIM_Loss(dehaze, gt)
            VGG_loss1 = vgg_loss(dehaze, gt)
            adversarial_loss = bce_loss(dehaze_label,real_label)
            Loss = MSE_loss1 + 0.2 * VGG_loss1 + 0.5 * adversarial_loss + 0.5*(1-SSIM_loss1)
            loss_list.append(Loss.item())
    ave_loss = sum(loss_list) / len(loss_list)
    return ave_loss






def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]
    return ssim_list


def validation(net, val_data_loader, device, category, save_tag=True):
    """
    :param net: PCFAN
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            haze1,  gt = Variable(val_data['hazy_image_s1']), Variable(val_data['clear_image_s1'])
            haze1 = haze1.to(device)
            gt = gt.to(device)
            image_name = val_data['haze_name']
            dehaze = net(haze1)
            # dehaze = dehaze[0]
            # print("hello")
            # dehaze = net([haze1,haze1])

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(dehaze, gt))

        # --- Save image --- #
        if save_tag:
            path = './results/{}_results/'.format(category)
            if not os.path.exists(path):
                os.makedirs(path)
            save_image(dehaze, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)
    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './results/{}_results/{}'.format(category, image_name[ind][:-3] + 'png')) # nhhaze png ohaze jpg


def print_log(path,epoch, num_epochs, train_psnr, val_psnr, val_ssim, category):
    print('Epoch [{0}/{1}], Train_PSNR:{2:.2f}, Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open(os.path.join(path,'{}_log.txt'.format(category)), 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Train_PSNR: {3:.2f}, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)

class MovingAvg(object):
    def __init__(self, pool_size=100):
        from queue import Queue
        self.pool = Queue(maxsize=pool_size)
        self.sum = 0
        self.curr_pool_size = 0
        self.pool_size = pool_size

    def set_curr_val(self, val):
        if not self.pool.full():
            self.curr_pool_size += 1
            self.pool.put_nowait(val)
        else:
            last_first_val = self.pool.get_nowait()
            self.pool.put_nowait(val)
            self.sum -= last_first_val

        self.sum += val
        return self.sum / self.curr_pool_size

    def reset(self):
        from queue import Queue
        self.pool = Queue(maxsize=self.pool_size)
        self.sum = 0
        self.curr_pool_size = 0
