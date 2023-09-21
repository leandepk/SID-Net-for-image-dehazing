import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models
from pos_neg_examples_generator import get_pos_samples, get_neg_samples
from model.new_network import Net

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # 2 64 255 255
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]









# class ContrastLoss(nn.Module):
#     def __init__(self, ablation=False):
#
#         super(ContrastLoss, self).__init__()
#         self.vgg = Vgg19().cuda()
#         self.l1 = nn.L1Loss()
#         self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
#         self.ab = ablation
#
#     def forward(self, a, p, n):
#
#         a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
#
#         loss = 0
#
#         d_ap, d_an = 0, 0
#         for i in range(len(a_vgg)):
#             d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
#             if not self.ab:
#                 d_an = self.l1(a_vgg[i], n_vgg[i].detach())
#                 contrastive = d_ap / (d_an + 1e-7)
#             else:
#                 contrastive = d_ap
#
#             loss += self.weights[i] * contrastive
#         return loss

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        # self.vgg = Vgg19().cuda()
        self.vgg = Net().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.get_neg_samples = get_neg_samples
        self.get_pos_samples = get_pos_samples

    def forward(self, a, p, n):
        a = a.cuda()
        p = self.get_pos_samples(p)
        p = p.cuda()
        n = self.get_neg_samples(n)
        n = n.cuda()
        batch_size = a.shape[0]
        p_number = p.shape[1]
        n_number = n.shape[1]
        self.vgg.eval()
        a_vgg = self.vgg(a)
        p_vgg = []
        n_vgg = []
        for i in range(p_number):
            p_vgg.append(self.vgg(p[:,i]))
        # print(len(p_vgg))
        for j in range(n_number):
            n_vgg.append(self.vgg(n[:,j]))
        # print(len(n_vgg[0]))
        # print(len(n_vgg))

        loss = 0

        d_ap, d_an = 0, 0

        for k in range(len(a_vgg)):
            # print(k)
            for m in range(len(p_vgg)):
                d_ap = d_ap + self.l1(a_vgg[k], p_vgg[m][k].detach())
            d_ap = d_ap/len(p_vgg)
            if not self.ab:
                for n in range(len(n_vgg)):
                    d_an = d_an + self.l1(a_vgg[k],n_vgg[n][k].detach())
                d_an = d_an / len(n_vgg)
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap
            loss = loss + self.weights[k] * contrastive

        return loss



if __name__ == '__main__':
    inp = torch.randn(4, 3, 256, 256).cuda()
    inp1 = torch.randn(4, 3, 256, 256).cuda()
    inp2 = torch.randn(4, 3, 256, 256).cuda()
    n_es = get_neg_samples(inp1)  # [b , n, c, h, w]
    n_es = n_es.cuda()
    # n_es = get_neg_samples(inp)  # [b , n, c, h, w]
    p_es = get_pos_samples(inp2) # [b , n, c, h, w]
    p_es = p_es.cuda()


    cr_loss = ContrastLoss()
    # print(cr_loss(inp,inp1,inp2))
    print(cr_loss(inp,p_es,n_es))
    # cl_loss = InfoNCE_SingleLayer(querys=inp, positive_keys=p_es, negative_keys=n_es, temperature=0.4)
    # print(cl_loss)