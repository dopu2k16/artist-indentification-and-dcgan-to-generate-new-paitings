from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys
scriptpath = "./metric_gan.py"
# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))
# Do the import
import metric_gan
import numpy as np

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


if __name__ == '__main__':


    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    

    cudnn.benchmark = True

    
    #########################
    #### Dataset prepare ####
    #########################
    #dataset = make_dataset(dataset=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize)
    #from preprocess.py
    # Resize the image to get tensors of the same size to fed into the pretrained model
    # Augment the data by making a crop from the center of the image
    # Use mean = [0.5, 0.5, 0.5] and std = [0.5, 0.5, 0.5] because 
    # these are the values proposed to use dc-gan model
    imageSize = 64
    transform = transforms.Compose([transforms.Resize(imageSize),
                                    transforms.CenterCrop(imageSize),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    #transform = transforms.Compose([transforms.TenCrop(imageSize),
    #                                transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
    #                                transforms.Normalize(
    #                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                ])
    dataset = metric_gan.PaintingsDataset(path_csv = '../datasets/all_data_info.csv',
                              img_dir = '../datasets/paintings', genre="portrait",
                              transform = transform)

   # dataset = metric_gan.PaintingsDataset(path_csv = '../datasets/all_data_info.csv',
   #                           img_dir = '../datasets/paintings', mode='training', threshold = 300,
   #                           transform = transform)


    #assert dataset
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    
    #from preprocess.py
    dataloader =  DataLoader(dataset = dataset,
                          batch_size = 64,
                          shuffle = True,
                          num_workers = 4)  
    
    #########################
    #### Models building ####
    #########################
    dataroot = "../datasets/paintings" #set the dataroot path
    # Number of workers for dataloader
    workers = 4
    # Batch size during training
    batchSize = 64
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    imageSize = 64
    
    # Number of channels in the training images. For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)

    nz = 100
    # Size of feature maps in generator

    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Number of training epochs
    niter = 100

    # Learning rate for optimizers

    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    sampleSize = 1000 #sample size for evaluation

    outf="results" #set checkpoints and folder name


    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >0 ) else "cpu")
    

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # [emd-mmd-knn(knn,real,fake,precision,recall)]*4 - IS - mode_score - FID
    score_tr = np.zeros((niter, 4*2+3))

    # compute initial score
    s = metric_gan.compute_score_raw(dataset, imageSize, dataroot, sampleSize, 64, outf+'/real/', outf+'/fake/',
                                 netG, nz, conv_model='resnet34', workers=workers)
    score_tr[0] = s
    np.save('%s/score_tr.npy' % (outf), score_tr)

    #########################
    #### Models training ####
    #########################
    for epoch in range(niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 10 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, niter, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                        normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

        ################################################
        #### metric scores computing (key function) ####
        ################################################
        s = metric_gan.compute_score_raw(dataset, imageSize, dataroot, sampleSize, batchSize, outf+'/real/', outf+'/fake/',\
                                     netG, nz, conv_model='inception_v3', workers=workers)
        score_tr[epoch] = s

    # save final metric scores of all epoches
    np.save('%s/score_tr_ep.npy' % outf, score_tr)
    print('##### training completed :) #####')
    print('### metric scores output is scored at %s/score_tr_ep.npy ###' % outf)
