from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 700000000 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from os import listdir
from os.path import isfile, join
# Ignore warnings
import warnings
import copy
import time
warnings.filterwarnings("ignore")
#plt.ion()

#Class to customize Dataset class
class PaintingsDataset(Dataset):
    
    def __init__(self, path_csv, img_dir, mode, threshold = 300, transform = None):
        df = pd.read_csv(path_csv)
        self.mode = mode
        # get all the rows for the training images
        self.df_training_full = df[df['in_train'] == True]
        # get all the rows for the test images
        self.df_test_full =  df[df['in_train'] == False]
        # filter the training rows based on the threshold
        self.df_training = self.df_training_full.groupby('artist').filter(lambda x: len(x) >= threshold)
        # get the names of the artists
        self.artists = self.df_training['artist'].values
        # filter the test set based on the list of the artists
        self.df_test = self.df_test_full[self.df_test_full['artist'].isin(self.artists)]
        self.num_classes = self.df_training['artist'].nunique()
        num_classes_2 = self.df_test['artist'].nunique()
        # test prints 
        print ("Size filter trainig: {}".format(len(self.df_training)))
        print ("Size filter test: {}".format(len(self.df_test)))
        print ("Number of classes: {}".format(self.num_classes))
        print ("Number classes sanity check: {}".format(num_classes_2))
        # set the image directory and the path for the csv file 
        self.img_dir = img_dir
        self.path_csv = path_csv
        if mode=="training":
            self.img_names = self.df_training['new_filename'].values
        else:
            self.img_names =  self.df_test['new_filename'].values
        self.dic = {}
        idx = 0
        # construct a dictionary to generate
        # the numeric labels
        for key in self.artists:
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
        
        y = self.dic[self.artists[index]]
        label = torch.tensor(y, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.artists)
# Resize the image to get tensors of the same size to fed into the pretrained model
# Augment the data by making a random horizontal flip
# Use mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] because 
# these are the values proposed to use ResNet
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# define the training dataset and the test dataset
train_dataset = PaintingsDataset(path_csv = '../datasets/all_data_info.csv',
                              img_dir = '../datasets/train', mode = "training",
                              transform = transform)

test_dataset = PaintingsDataset(path_csv = '../datasets/all_data_info.csv',
                              img_dir = '../datasets/test', mode = "test",
                              transform = transform)

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = 32,
                          shuffle = True,
                          num_workers = 2)
test_loader = DataLoader(dataset = test_dataset,
                         batch_size = 32,
                         shuffle = False,
                         num_workers = 2)

# define the device to work with
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#future matrix used afterwards to compute precision and recall
conf_mat = np.zeros((train_dataset.num_classes, train_dataset.num_classes), dtype = int)

# function to get the f1 score
def get_f1(precision_list, recall_list):
    f1_list = 2/(1/precision_list + 1/recall_list)
    f1 = np.mean(f1_list)
    return f1, f1_list

# function to get the precision of the classifier
def get_precision():
    TP = conf_mat.diagonal()
    TP_FP = np.sum(conf_mat, axis = 1)
    precision_list = TP/TP_FP
    precision = np.mean(precision_list)
    return precision, precision_list

# function to get the recall of the classifier
def get_recall():
    TP = conf_mat.diagonal()
    TP_FN = np.sum(conf_mat, axis = 0)
    recall_list = TP/TP_FN
    recall = np.mean(recall_list)
    return recall, recall_list

# function to update the confusion matrix,
# this matrix will be used to compute
# the precision and recall measurments
def update_conf_mat(out, labels): 
    _, preds = torch.max(out, 1)
    n = preds.size(0)
    for i in range(n):
        a = preds[i].item()
        b = labels[i].item()
        conf_mat[a][b] += 1

    return None

# function to get the accuracy of the top n guesses
# Only top 1, top 3 and top 5 are used afterwards
def get_acc_topn(out, labels, n):
    correct_topn = 0
    _, preds = torch.topk(out, n, 1)
    for i in range(n):
        correct_topn += torch.sum(preds[:,i] == labels.data)

    acc_topn = correct_topn.type(torch.DoubleTensor)/labels.size(0)
    return acc_topn

# main function to train the model
def train_model(model, name_model, criterion, optimizer, num_epochs=25):
    t0 = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_top1 = 0.0
    best_acc_top3 = 0.0
    best_acc_top5 = 0.0

    # Iterate over the number of epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs ))
        print('-' * 10)

        model.train()  # Set model to training mode
        list_loss = []
        list_acc_top1 = []
        list_acc_top3 = []
        list_acc_top5 = []

        # Iterate over data
        for (idx, (X, y)) in enumerate(train_loader,0):
            X = X.to(device)
            y = y.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            with torch.set_grad_enabled(True):
                out = model(X)
                _, preds = torch.max(out, 1)
                loss = criterion(out, y)
                # backward + optimize
                loss.backward()
                optimizer.step()

                # statistics
                # store the loss for future graphics
                list_loss.append(loss.item())
                # get the top1, top3 and top5
                # for for the training
                acc_top1 = get_acc_topn(out, y, 1)
                acc_top3 = get_acc_topn(out, y, 3)
                acc_top5 = get_acc_topn(out, y, 5)
                list_acc_top1.append(acc_top1)
                list_acc_top3.append(acc_top3)
                list_acc_top5.append(acc_top5)

            # print the loss from time to time to
            # have information during training
            if idx % 20 == 0:
                print ("idx = {}, loss = {}".format(idx, loss.item()))

        #clear the cache
        torch.cuda.empty_cache()
        # get the mean of the loss and accuraries topk, k in {1,3,5}
        epoch_loss = np.mean(list_loss)
        epoch_acc_top1 = np.mean(list_acc_top1)
        epoch_acc_top3 = np.mean(list_acc_top3)
        epoch_acc_top5 = np.mean(list_acc_top5)
        # convert the lists of the loss and accuracy topk into tensor
        pt_epoch_loss = torch.FloatTensor(list_loss)
        pt_acc_top1 =  torch.FloatTensor(list_acc_top1)
        pt_acc_top3 =  torch.FloatTensor(list_acc_top3)
        pt_acc_top5 =  torch.FloatTensor(list_acc_top5)
        
        # define the names to save the loss and accuracies per epoch
        name_pt_loss = name_model+"_epoch_"+str(epoch)+"_loss.pt"
        name_pt_acc_top1 = name_model+"_epoch_"+str(epoch)+"_acc_top1.pt" 
        name_pt_acc_top3 = name_model+"_epoch_"+str(epoch)+"_acc_top3.pt"
        name_pt_acc_top5 = name_model+"_epoch_"+str(epoch)+"_acc_top5.pt"

        #save the tensors of loss and accuracy
        torch.save(pt_epoch_loss, name_pt_loss)
        torch.save(pt_acc_top1, name_pt_acc_top1)
        torch.save(pt_acc_top3, name_pt_acc_top3)
        torch.save(pt_acc_top5, name_pt_acc_top5)

        # save the best parameters learned
        if epoch_acc_top1 > best_acc_top1:
            best_acc_top1 = epoch_acc_top1
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if epoch_acc_top3 > best_acc_top3:
            best_acc_top3 = epoch_acc_top3
        
        if epoch_acc_top5 > best_acc_top5:
            best_acc_top5 = epoch_acc_top5
         
        print('Loss: {:.4f} Accuracy top 1: {:.4f}'.format(epoch_loss, epoch_acc_top1))

    time_elapsed = time.time() - t0
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc Top 1: {:4f}'.format(best_acc_top1))
    print('Best Acc Top 3: {:4f}'.format(best_acc_top3))
    print('Best Acc Top 5: {:4f}'.format(best_acc_top5))
    # load best model weights
    model.load_state_dict(best_model_wts)
    # save the model
    torch.save(model.state_dict(), name_model)
    return model

def test_model(model):
    t0 = time.time()

    # Set model to validation  mode
    model.eval()
    list_loss = []
    list_acc_top1 = []
    list_acc_top3 = []
    list_acc_top5 = []
    # Iterate over data
    for (idx, (X, y)) in enumerate(train_loader,0):
        X = X.to(device)
        y = y.to(device)            
        # forward
        with torch.set_grad_enabled(False):
            out = model(X)

            # updata the confusion matrix
            # with the current output
            update_conf_mat(out, y)
            # statistics
            acc_top1 = get_acc_topn(out, y, 1)
            acc_top3 = get_acc_topn(out, y, 3)
            acc_top5 = get_acc_topn(out, y, 5)
            list_acc_top1.append(acc_top1)
            list_acc_top3.append(acc_top3)
            list_acc_top5.append(acc_top5)
            if idx % 50 == 0:
                print ("idx = {}: accuracy top 1 = {}".format(idx, acc_top1))
                print ("idx = {}: accuracy top 3 = {}".format(idx, acc_top3))
                print ("idx = {}: accuracy top 5 = {}".format(idx, acc_top5)) 
    
    accuracy_top1 = np.mean(list_acc_top1)
    print('Acc top 1: {:.4f}'.format(accuracy_top1))
    accuracy_top3 = np.mean(list_acc_top3)
    print('Acc top 3: {:.4f}'.format(accuracy_top3))
    accuracy_top5 = np.mean(list_acc_top5)
    print('Acc top 5: {:.4f}'.format(accuracy_top5))

    time_elapsed = time.time() - t0
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return None 

# define the pretrained models to use

model18 = models.resnet18(pretrained=True)
model34 = models.resnet34(pretrained=True)
# These models were tried as well, but due to 
# hardware limitations they had to be excluded
#model50 = models.resnet50(pretrained = True)
#model101 = models.resnet101(pretrained = True)
#model152 = models.resnet152(pretrained = True)

# define a list of models to be trained
#list_model = [model18, model34, model50, model101, model152]
list_model = [model18, model34]
#names_model = ['resnet18', 'resnet34', 'model50', 'model101', 'model152']
names_model = ['resnet18', 'resnet34']
i = 0
# main loop which includes the training
# the testing and the statistics
for model in list_model:
    print ("Running model: {}".format(names_model[i]))
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_dataset.num_classes)
    model = model.to(device)
    # use cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, names_model[i], criterion, optimizer, num_epochs = 25)
    test_model(model)
    precision, precision_list = get_precision()
    recall, recall_list = get_recall()
    f1, f1_list = get_f1(precision_list, recall_list)
    print ("Precision: {}".format(precision))
    print ("Recall: {}".format(recall))
    print ("F1: {}".format(f1))
    print('=' * 30)
    i = i+1
print ("DONE!")

