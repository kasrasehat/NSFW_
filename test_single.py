from __future__ import print_function
import datetime
import os
import time

import itertools
#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import cv2
import torch.nn.functional as F

#import utils

from sklearn.metrics import confusion_matrix


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.get_cmap('Blues')):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     #plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     #plt.title(title)
#     #plt.colorbar()
#     #tick_marks = np.arange(len(classes))
#     #plt.xticks(tick_marks, classes, rotation=45)
#     #plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
def expand(patch):
    patch = torch.unsqueeze(patch, 0)
    return patch.type(torch.float32)
def main(args):
    testdir = args.data_dir
    input = cv2.imread(testdir)
    #input = torch.tensor(input).unsqueeze(0).to('cuda:0', dtype=torch.float32).permute(0,3,1,2)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("Loading test data")

    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                    expand])
    input = transform(input)

    classes = torch.load(args.checkpoint)['classes']
    print(classes)
    model = torchvision.models.__dict__[args.model](pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))

    model = nn.DataParallel(model, device_ids=args.device)
    model.cuda()
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        y_pred = model(input)

    res = torch.argmax(y_pred)
    if res==0 :
        print('nsfw')
    else:
        print('sfw')






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-dir', default='/home/sehat/dataset/single_test/test2 (32).jpg', help='dataset')
    parser.add_argument('--model', default='resnet101', help='model')
    parser.add_argument('--device', default=[0], help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--checkpoint', default='checkpoints/model_68_100.pth', help='checkpoint')

    args = parser.parse_args()

    print(args)
    main(args)
