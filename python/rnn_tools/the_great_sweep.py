import pickle
import pandas as pd
import os
import setup

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm, trange
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
from torch.utils.data.sampler import SubsetRandomSampler
import pprint
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import MinMaxScaler
import math
from torch.profiler import profile, record_function, ProfilerActivity

from operator import itemgetter
import operator
from random import randint
# from rnn_classes import Dog, DogInput, Race, Races, GRUNet, smallGRUNet, smalll_lin_GRUNet, smalll_prelin_GRUNet
import rnn_tools.rnn_classes as rnn_classes
from rnn_tools.raceDB import build_dataset
import importlib
import datetime
from rnn_tools.model_saver import model_saver, model_saver_wandb
import rnn_tools.training_testing_gru as training_testing_gru
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence,pad_sequence, unpack_sequence, unpad_sequence
import rnn_tools.training_testing_gru_double as training_testing_gru_double
import rnn_tools.training_testing_lstm as training_testing_lstm
from goto_conversion import goto_conversion

import rnn_tools.training_testing_gru_extra_data as training_testing_gru_extra_data
import rnn_tools.training_testing_gru_ensemble as training_testing_gru_ensemble