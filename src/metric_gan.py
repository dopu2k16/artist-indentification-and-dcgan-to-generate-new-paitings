import math
import os
import timeit
import math

import numpy as np
import ot
import torch
from torch import nn

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 750000000
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
import pdb
from tqdm import tqdm

from scipy.stats import entropy
from numpy.linalg import norm
from scipy import linalg


def giveName(iter):  # 7 digit name.
    ans = str(iter)
    return ans.zfill(7)
# Class to cutomize the dataset by style, genre, artist
# or a threshold based on the number of paintings by artist 
class PaintingsDataset(Dataset):
    def __init__(self, path_csv, img_dir, genre= None, style=None, artist=None, transform = None, threshold = None):
        df = pd.read_csv(path_csv)
        # filter the training rows by a threshold
        # determined by the number of paintings has an artist
        # in the data set
        if threshold is not None:
            self.df_training = df.groupby('artist').filter(lambda x: len(x)>=threshold)
        # filter the training rows based on the genre
        if genre is not None:
            self.df_training = df[df['genre']==genre]
        # filter the training rows based on the style
        if style is not None:
            self.df_training = df[df['style']==style]
        # filter the training rows based on the artist
        if artist is not None:
            self.df_training = df[df['artist']==artist]
        # labels based on the artist or on the style
        if artist is None:
            self.labels = self.df_training['artist'].values
        else:
            self.labels = self.df_training['style'].values
        # Size of the filter
        print ("Size filter trainig: {}".format(len(self.df_training)))
        # set the image directory and the path for the csv file 
        self.img_dir = img_dir
        self.path_csv = path_csv
        self.img_names = self.df_training['new_filename'].values 
        # construct a dictionary to generate
        # the numeric labels
        self.dic = {}
        idx = 0
        for key in self.labels:
            if not key in self.dic:
                self.dic[key] = idx
                idx = idx + 1

        self.transform = transform

    def __getitem__(self, index):
        #transform all the images into RGB images to use 3 channels for the training 
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index])).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        y = self.dic[self.labels[index]]
        label = torch.tensor(y, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.labels)


def sampleFake(netG, nz, sampleSize, batchSize, saveFolder):
    print('sampling fake images ...')
    saveFolder = saveFolder + '0/'

    try:
        os.makedirs(saveFolder)
    except OSError:
        pass

    noise = torch.FloatTensor(batchSize, nz, 1, 1).cuda()
    iter = 0
    for i in range(0, 1 + sampleSize // batchSize):
        noise.data.normal_(0, 1)
        fake = netG(noise)
        for j in range(0, len(fake.data)):
            if iter < sampleSize:
                vutils.save_image(fake.data[j].mul(0.5).add(
                    0.5), saveFolder + giveName(iter) + ".png")
            iter += 1
            if iter >= sampleSize:
                break


def sampleTrue(dataset, imageSize, dataroot, sampleSize, batchSize, saveFolder, workers=4):

    print('sampling real images ...')
    saveFolder = saveFolder + '0/'

    #dataset = make_dataset(dataset, dataroot, imageSize)
    #dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batchSize, num_workers=int(workers))

    imageSize = 64
    transform = transforms.Compose([transforms.Resize(imageSize),
                                    transforms.CenterCrop(imageSize),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    
    #dataset = metric_gan.PaintingsDataset(path_csv = '../datasets/all_data_info.csv',
    #                          img_dir = '../datasets/train', mode = "training",
    #                          transform = transform)

    #assert dataset
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    dataset= dataset
    


    #from preprocess.py
    dataloader =  DataLoader(dataset = dataset,
                          batch_size = 64,
                          shuffle = True,
                          num_workers = 4)  
    
    #########################
    #### Models building ####
    #########################
    
    if not os.path.exists(saveFolder):
        try:
            os.makedirs(saveFolder)
        except OSError:
            pass

    iter = 0
    for i, data in enumerate(dataloader, 0):
        img, _ = data
        for j in range(0, len(img)):

            vutils.save_image(img[j].mul(0.5).add(
                0.5), saveFolder + giveName(iter) + ".png")
            iter += 1
            if iter >= sampleSize:
                break
        if iter >= sampleSize:
            break


class ConvNetFeatureSaver(object):
    def __init__(self, model='resnet34', workers=4, batchSize=64):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''
        self.model = model
        self.batch_size = batchSize
        self.workers = workers
        if self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).cuda().eval()
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True)
            resnet.cuda().eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).cuda().eval()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained=True, transform_input=False).cuda().eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              ).cuda().eval()
            self.inception = inception
            self.inception_feature = inception_feature
            self.trans = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError

    def save(self, imgFolder, save2disk=False):
        dataset = dset.ImageFolder(root=imgFolder, transform=self.trans)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.workers)
        print('extracting features...')
        feature_pixl, feature_conv, feature_smax, feature_logit = [], [], [], []
        for img, _ in tqdm(dataloader):
            with torch.no_grad():
                input = img.cuda()
                if self.model == 'vgg' or self.model == 'vgg16':
                    fconv = self.vgg.features(input).view(input.size(0), -1)
                    flogit = self.vgg.classifier(fconv)
                    # flogit = self.vgg.logitifier(fconv)
                elif self.model.find('resnet') >= 0:
                    fconv = self.resnet_feature(
                        input).mean(3).mean(2).squeeze()
                    flogit = self.resnet.fc(fconv)
                elif self.model == 'inception' or self.model == 'inception_v3':
                    fconv = self.inception_feature(
                        input).mean(3).mean(2).squeeze()
                    flogit = self.inception.fc(fconv)
                else:
                    raise NotImplementedError
                fsmax = F.softmax(flogit)
                feature_pixl.append(img)
                feature_conv.append(fconv.data.cpu())
                feature_logit.append(flogit.data.cpu())
                feature_smax.append(fsmax.data.cpu())

        feature_pixl = torch.cat(feature_pixl, 0).to('cpu')
        feature_conv = torch.cat(feature_conv, 0).to('cpu')
        feature_logit = torch.cat(feature_logit, 0).to('cpu')
        feature_smax = torch.cat(feature_smax, 0).to('cpu')

        if save2disk:
            torch.save(feature_conv, os.path.join(
                imgFolder, 'feature_pixl.pth'))
            torch.save(feature_conv, os.path.join(
                imgFolder, 'feature_conv.pth'))
            torch.save(feature_logit, os.path.join(
                imgFolder, 'feature_logit.pth'))
            torch.save(feature_smax, os.path.join(
                imgFolder, 'feature_smax.pth'))

        return feature_pixl, feature_conv, feature_logit, feature_smax


def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd



def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd




eps = 1e-20
def inception_score(X):
    kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score

def mode_score(X, Y):
    kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score


def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
        np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


class Score:
    emd = 0
    mmd = 0
    



def compute_score_raw(dataset, imageSize, dataroot, sampleSize, batchSize,
                      saveFolder_r, saveFolder_f, netG, nz,
                      conv_model='resnet34', workers=4):

    sampleTrue(dataset, imageSize, dataroot, sampleSize, batchSize,
               saveFolder_r, workers=workers)
    sampleFake(netG, nz, sampleSize, batchSize, saveFolder_f, )

    convnet_feature_saver = ConvNetFeatureSaver(model=conv_model,
                                                batchSize=batchSize, workers=workers)
    feature_real = convnet_feature_saver.save(saveFolder_r)
    feature_fake = convnet_feature_saver.save(saveFolder_f)

    # 4 feature spaces and 7 scores + incep + modescore + fid
    score = np.zeros(4 * 2 + 3)
    for i in range(0, 4):
        print('compute score in space: ' + str(i))
        Mxx = distance(feature_real[i], feature_real[i], False)
        Mxy = distance(feature_real[i], feature_fake[i], False)
        Myy = distance(feature_fake[i], feature_fake[i], False)

        score[i * 2] = wasserstein(Mxy, True)
        score[i * 2 + 1] = mmd(Mxx, Mxy, Myy, 1)
        

    score[8] = inception_score(feature_fake[3])
    score[9] = mode_score(feature_real[3], feature_fake[3])
    score[10] = fid(feature_real[3], feature_fake[3])
    return score
