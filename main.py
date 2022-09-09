import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn



INPUT_DIM = 30

HIDDEN_DIM = 20


def parse():

     parser = argparse.ArgumentParser()
     parser.add_argument("-n", "--num_frames", type=int, default=1)
     args = parser.parse_args()

     return args


class simpleLSTM(nn.Module):
     
     def __init__(self,input_dim,hidden_dim,output_dim):
         
         super(simpleLSTM, self).__init__()

         self.hidden_dim = hidden_dim
         
         self.lstm = nn.LSTM(input_dim,hidden_dim,batch_first=True)

         self.fc1 = nn.Linear(hidden_dim,hidden_dim)
         self.fc2 = nn.Linear(hidden_dim,output_dim)

         self.bn1 = nn.BatchNorm1d(hidden_dim)

         self.dropout = nn.Dropout(0.5)

         self.relu = nn.ReLU()


     def forword(self,x):

         _,(h,_) = self.lstm(x)

         x = self.dropout(h)
         x = self.relu(x)
         x = self.fc1(x)

         x = self.dropout(x)
         x = self.relu(x)
         x = self.bn1(x)
         x = self.fc2(x)

         return x


def train():
     return

