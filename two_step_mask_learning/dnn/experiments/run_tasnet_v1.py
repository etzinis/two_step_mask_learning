"""!
@brief Run an initial CometML experiment

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys

sys.path.append('../../../')
from __config__ import API_KEY

from comet_ml import Experiment

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import two_step_mask_learning.dnn.dataset_loader.torch_dataloader as dataloader
from __config__ import WSJ_MIX_2_8K_PREPROCESSED_EVAL_P, \
    WSJ_MIX_2_8K_PREPROCESSED_TEST_P, WSJ_MIX_2_8K_PREPROCESSED_TRAIN_P
import two_step_mask_learning.dnn.losses.sisdr as sisdr_lib
import two_step_mask_learning.dnn.losses.norm as norm_lib


hparams = {
    "sequence_length": 28,
    "input_size": 28,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 10,
    "bs": 16,
    "n_jobs":3,
    "tr_get_top": 64,
    "val_get_top": 64,
    "num_epochs": 3,
    "learning_rate": 0.01
}


# experiment = Experiment(API_KEY, project_name="first_tasnet_wsj02mix")
# experiment.log_parameters(hyper_params)

n_sources = 2

# define data loaders
train_loader, eval_loader = dataloader.get_data_generators(
    [WSJ_MIX_2_8K_PREPROCESSED_TRAIN_P,
     WSJ_MIX_2_8K_PREPROCESSED_EVAL_P],
    bs=hparams['bs'], n_jobs=hparams['n_jobs'],
    get_top=[hparams['tr_get_top'], hparams['val_get_top']],
    return_items=['mixture_wav', 'clean_sources_wavs']
)

# define the losses that are going to be used
recon_loss_name, recon_loss = (
    'tr_SISDR',
    sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                 n_sources=n_sources,
                                 zero_mean=False,
                                 backward_loss=True))

val_losses = dict([
    ('val_L1', norm_lib.PermInvariantNorm(batch_size=hparams['bs'],
                                          zero_mean=False,
                                          n_sources=n_sources)),
    ('val_SISDR', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                               n_sources=n_sources,
                                               zero_mean=True,
                                               backward_loss=False)),
    ('val_SISDR_AE', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                                  n_sources=1,
                                                  zero_mean=True,
                                                  backward_loss=False))
  ])

train_losses = dict([
    ('tr_L1', norm_lib.PermInvariantNorm(batch_size=hparams['bs'],
                                         zero_mean=False,
                                         n_sources=n_sources)),
    ('tr_SISDR', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                              n_sources=n_sources,
                                              zero_mean=True,
                                              backward_loss=False)),
    ('tr_SISDR_AE', sisdr_lib.PermInvariantSISDR(batch_size=hparams['bs'],
                                                 n_sources=1,
                                                 zero_mean=True,
                                                 backward_loss=False))])

#
# # RNN Model (Many-to-One)
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         # Set initial states
#         h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
#         c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
#
#         # Forward propagate RNN
#         out, _ = self.lstm(x, (h0, c0))
#
#         # Decode hidden state of last time step
#         out = self.fc(out[:, -1, :])
#         return out
#
#
#
# rnn = RNN(hyper_params['input_size'],
#           hyper_params['hidden_size'],
#           hyper_params['num_layers'],
#           hyper_params['num_classes'])
#
# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(rnn.parameters(), lr=hyper_params['learning_rate'])
#
# # Train the Model
#
# with experiment.train():
#
#     step = 0
#     for epoch in range(hyper_params['num_epochs']):
#         experiment.log_current_epoch(epoch)
#         correct = 0
#         total = 0
#         for i, (images, labels) in enumerate(train_loader):
#             images = Variable(images.view(-1, hyper_params['sequence_length'],
#                                           hyper_params['input_size']))
#             labels = Variable(labels)
#
#             # Forward + Backward + Optimize
#             optimizer.zero_grad()
#             outputs = rnn(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # Compute train accuracy
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += float((predicted == labels.data).sum())
#
#             # Log accuracy to Comet.ml
#             experiment.log_metric("accuracy", correct / total, step=step)
#             step += 1
#
#             if (i + 1) % 100 == 0:
#                 print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
#                       % (epoch + 1, hyper_params['num_epochs'], i + 1,
#                          len(train_dataset) // hyper_params['batch_size'],
#                          loss.data.item()))
# with experiment.test():
#     # Test the Model
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = Variable(images.view(-1, hyper_params['sequence_length'],
#                                       hyper_params['input_size']))
#         outputs = rnn(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += float((predicted == labels).sum())
#
#     experiment.log_metric("accuracy", correct / total)
#     print('Test Accuracy of the model on the 10000 test images: %d %%'
#           % (100 * correct / total))