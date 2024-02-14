# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:44:41 2023

script to fit one TLN to another TLN: takes the output of a TLN (W_0,theta_0) and 
uses it to learn a new TLN (W_1,theta_1) that can reproduce one of the attractors of
(W_0,theta_0). 

@author: juliana. Code is partially modified from srini's original code
(https://github.com/srinituraga/)
"""
# %% imports

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %% define the network dynamics:

#W and theta are the only learnable parameters
class CTLN(nn.Module):
  def __init__(self,W,theta,delta_t):
    super().__init__() 
    self.num_neurons = W.size(0)
    self.mask = 1-torch.eye(num_neurons)
    self.W = nn.Parameter(torch.tensor(W)) #learnable W
    self.theta = nn.Parameter(torch.tensor(theta)) #learnable theta
    self.delta_t = delta_t #delta_t from euler's integration

  # x0 is the initial condition
  # u is the time-varying external input (per time step)
  def forward(self,x0,u):
    n_steps = u.size(1) #as many steps as duration of external input
    x = [x0] #initial condition
    for step in range(n_steps): #euler integrate
      x_step = x[-1] + self.delta_t * (-x[-1] + \
                                       torch.relu(torch.matmul(self.W,x[-1]) \
                                                  + self.theta + u[:,step]))
      x.append(x_step)
    x = torch.stack(x[1:],dim=-1)
    return x

# %% a function that does the learning given the initial guess for (W,theta)
def TLN_fit(W,theta,x_desired,delta_t,x0,u,num_iter,lr):
    
    opt = optim.Adam(net.parameters(),lr=lr)
    losses = []
       
    for it in range(num_iter):
      x = net(x0,u) #forward
      #current_x_fit contains only as many neurons as x_desired
      current_x_fit = x[0:x_desired.size()[0],:]
      #calculate the error for all curves
      loss = F.mse_loss(current_x_fit,x_desired)
      opt.zero_grad() # zero old grads
      loss.backward() #computes dloss/dW and dLoss/dtheta
      opt.step() #updates
      #remember losses to plot them later
      losses.append(loss.detach().numpy()) 
      #constrain W to be negative and have 0 diagonal (competitive TLN)
      net.W.data.clamp_max_(0.)
      net.W.data.mul_(net.mask)
      
      if not it%1000: #plot every 100 iterations as sanity check
        plt.cla() #clear axes
        #to reset the color cycle so both neurons i in initial
        #and desired are the same color
        plt.gca().set_prop_cycle(None)
        plt.plot(x_desired.detach().numpy().T)
        #to reset the color cycle so both neurons i in initial
        #and desired are the same color
        plt.gca().set_prop_cycle(None)
        plt.plot(current_x_fit.detach().numpy().T,linestyle='dashed')
        plt.title("iter = %i" %it + ", loss = %f" %loss.detach().numpy())
        #plt.savefig("iter_%i" %it)
        plt.show()
        #plt.close()
    
    plt.cla()
    plt.gca().set_prop_cycle(None)
    plt.plot(x_desired.detach().numpy().T)
    plt.gca().set_prop_cycle(None)
    plt.plot(x.detach().numpy().T,linestyle='dashed')
    plt.title("final network output (dashed, ALL neurons)")
    plt.show()     
    
    plt.cla()
    plt.plot(losses)
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.show()       
    
    return net.W,net.theta
# %% TLN to fit: this will be the "desired" above

#shared parameters for desired and initial
eps = 0.25;
delta = 0.5;
delta_t = .1 #time step for forward euler
time = 300

#desired TLN: 4-cycu
G_desired = torch.tensor([[0,0,0,1],[1,0,1,0],[1,1,0,0],[0,1,1,0]])
num_neurons = G_desired.size()[0]
Gcomplement_desired = (1-G_desired)*(1-torch.eye(num_neurons))
W_desired =  torch.eye(num_neurons)-1 + eps*G_desired - \
    delta*Gcomplement_desired
theta_desired = torch.ones(num_neurons)
x0_desired = torch.zeros(num_neurons)
x0_desired[1]  = 0.1
# u has to have the same len as x_desired
u = torch.zeros(num_neurons,time)
#run it
net_desired = CTLN(W_desired,theta_desired,delta_t)
x_desired = net_desired(x0_desired,u).detach()
#select a few neurons to fit and permute to your liking
neurons_to_fit = [0,1,2,3]
x_desired = x_desired[neurons_to_fit,:]

#initial TLN: n-cycle
num_neurons = 6
G = torch.roll(torch.eye(num_neurons),1,0)
Gcomplement = (1-G)*(1-torch.eye(num_neurons))
W =  torch.eye(num_neurons)-1 + eps*G - delta*Gcomplement
theta = torch.ones(num_neurons)
x0 = torch.zeros(num_neurons)
x0[1]  = 0.1
# u has to have the same len as x_desired
u = torch.zeros(num_neurons,time)
#initial
net = CTLN(W,theta,delta_t)
x = net(x0,u)

#this will tell you who is being learned!
#print(list(net_desired.named_parameters()))

#plot check
plt.cla()
plt.gca().set_prop_cycle(None)
plt.plot(x_desired.detach().numpy().T)
#to reset the color cycle so both neurons i in initial
#and desired are the same color
plt.gca().set_prop_cycle(None)
plt.plot(x.detach().numpy().T,linestyle='dashed')
plt.title("x desired solid, x current dotted")
plt.show()

input("~~~~~~~~~~Press Enter to continue~~~~~~~~~~")

# %% try it out

num_iter= 10000
lr=0.0005

W_out,theta_out = TLN_fit(W,theta,x_desired,delta_t,x0,u,num_iter,lr)

# %%

#plot all the Ws
matrices_to_plot = [W.detach().numpy(),
                    theta.detach().numpy().reshape((x.shape[0],1)),
                    W_out.detach().numpy(),
                    theta_out.detach().numpy().reshape((x.shape[0],1)),
                    W_desired.detach().numpy(),
                    theta_desired.detach().numpy().reshape((x_desired.shape[0],1))]
subplot_titles = ['Initial weights','Initial theta',
                  'Learned weights','Learned theta',
                  'Desired weights','Desired theta'] 
#find bounds for consistent colormap
vmin_val = np.amin(matrices_to_plot[0])
vmax_val = np.amax(matrices_to_plot[0])
for i in range(len(matrices_to_plot)):
    if np.amin(matrices_to_plot[i]) < vmin_val:
        vmin_val = np.amin(matrices_to_plot[i])
    if np.amax(matrices_to_plot[i]) > vmax_val:
        vmax_val = np.amax(matrices_to_plot[i])

fig = plt.figure(constrained_layout=True, figsize=(11, 4))
axs = fig.subplots(1, 6, 
                   gridspec_kw={'width_ratios': [5,1,5,1,5,1]})
for i in range(len(axs)):
    pc = axs[i].pcolormesh(matrices_to_plot[i], vmin=vmin_val, vmax=vmax_val)
    axs[i].title.set_text(subplot_titles[i])
fig.colorbar(pc, shrink=1, ax=axs, location='bottom')