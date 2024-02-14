# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:44:41 2023

copy of srini's first jupyter notebook with CTLNS

@author: srini
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation

from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CTLN(nn.Module):
  def __init__(self,W,theta,delta_t):
    super().__init__()
    self.num_neurons = W.size(0)
    self.mask = 1-torch.eye(num_neurons)
    self.W = nn.Parameter(torch.tensor(W))
    self.theta = nn.Parameter(torch.tensor(theta))
    self.delta_t = delta_t

  # x0 is the initial condition
  # u is the time-varying external input (per time step)
  def forward(self,x0,u):
    n_steps = u.size(1)
    x = [x0]
    for step in range(n_steps):
      x_step = x[-1] + self.delta_t * (-x[-1] + torch.relu(torch.matmul(self.W,x[-1]) + self.theta + u[:,step]))
      x.append(x_step)
    x = torch.stack(x[1:],dim=-1)
    return x

delta = 0.1
eps = delta/(delta+1.)/2.

num_neurons = 5
G = torch.roll(torch.eye(num_neurons),1,0)
Gcomplement = (1-G)*(1-torch.eye(num_neurons))
W =  torch.eye(num_neurons)-1 + eps*G - delta*Gcomplement
theta = torch.tensor(1.)

T = 200
delta_t = .1

x0 = torch.tensor([1.,0,0,0,0])
u = torch.zeros(num_neurons,floor(T/delta_t))

net = CTLN(W,theta,delta_t)
print(list(net.named_parameters()))

x = net(x0,u)
plt.plot(x.detach().numpy().T)
x.shape

#let's make a pulse with a defined pulse width and pulse frequency
pulse_width = 150
pulse_frequency = pulse_width*num_neurons
x_desired = ((torch.range(1,floor(T/delta_t)).fmod(pulse_frequency))<pulse_width).to(dtype=torch.float32)
plt.plot(x_desired,'k')

num_iter=1000
lr=0.005

opt = optim.Adam(net.parameters(),lr=lr)
losses = []
for it in range(num_iter):
  x = net(x0,u)
  loss = F.mse_loss(x[0,:],x_desired)
  opt.zero_grad()
  loss.backward()
  opt.step()
  losses.append(loss.detach().numpy())
  #constrain W to be negative
  net.W.data.clamp_max_(0.)
  net.W.data.mul_(net.mask)
  if not it%100:
    print(it,loss.detach().numpy())
    plt.cla()
    plt.plot(x_desired.detach().numpy(),'k')
    plt.plot(x.detach().numpy().T)
    plt.show()
  plt.cla()
  plt.plot(losses)
  plt.xlabel('iteration')
  plt.ylabel('Loss')
  
  plt.subplot(121)
  
  
plt.imshow(W.detach().numpy())
plt.title("Original weights")
plt.colorbar()
plt.subplot(122)
plt.imshow(net.W.detach().numpy())
plt.title("Learned weights")
plt.colorbar()

W.detach().numpy(),net.W.detach().numpy()