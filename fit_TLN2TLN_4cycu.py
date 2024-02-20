# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:44:41 2023

script to fit one TLN to another TLN: takes the output of a TLN (W_0,theta_0) and 
uses it to learn a new TLN (W_1,theta_1) that can reproduce one of the attractors of
(W_0,theta_0). 

@author: juliana. Code is partially modified from srini's original code
(https://github.com/srinituraga/)
"""
#import packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

#define the network dynamics: forward
class CTLN(nn.Module):
  def __init__(self,W,theta,delta_t):
    super().__init__() 
    self.num_neurons = W.size(0)
    self.mask = 1-torch.eye(num_neurons)
    self.W = nn.Parameter(torch.tensor(W)) #learnable W
    self.theta = nn.Parameter(torch.tensor(theta)) #learnable theta
    self.delta_t = delta_t #delta_t for euler's integration

  # x0 is the initial condition
  # u is a (time-varying) external input (per time step)
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
  
#define the desired (W,b) and the initial (W,b)
#shared parameters for desired and initial
eps = 0.25;
delta = 0.5;
delta_t = .1 #time step for forward euler
time = 1000

#desired TLN: 4-cycu
G_desired = torch.tensor([[0,0,0,1],[1,0,1,0],[1,1,0,0],[0,1,1,0]])
num_neurons = G_desired.size()[0]
Gcomplement_desired = (1-G_desired)*(1-torch.eye(num_neurons))
W_desired =  torch.eye(num_neurons)-1 + eps*G_desired - \
    delta*Gcomplement_desired
theta_desired = torch.ones(num_neurons)
x0_desired = torch.zeros(num_neurons)
x0_desired[1]  = 0.1
u = torch.zeros(num_neurons,time) # u has to have the same len as x_desired
#run it
net_desired = CTLN(W_desired,theta_desired,delta_t)
x_desired = net_desired(x0_desired,u).detach()
#select a few neurons to fit and permute to your liking
neurons_to_fit = [1,2,3,0]
x_desired = x_desired[neurons_to_fit,:]

#initial TLN: n-cycle
num_neurons = 4
#G = torch.roll(torch.eye(num_neurons),1,0)
#Gcomplement = (1-G)*(1-torch.eye(num_neurons))
#W =  torch.eye(num_neurons)-1 + eps*G - delta*Gcomplement
W = torch.tensor([[-0.0000, -1.2413, -1.8114, -1.0379],
                  [-0.4906,  0.0000, -1.3606, -0.6023],
                  [-0.9550, -0.5626, -0.0000, -1.5179],
                  [-1.5558, -1.4238, -0.7326,  0.0000]])
W_initial = W
#theta = torch.ones(num_neurons)
theta = torch.tensor([1.2706, 0.8594, 1.0083, 0.9905])
theta_initial = theta
x0 = torch.zeros(num_neurons)
x0[1]  = 0.1
# u has to have the same len as x_desired
u = torch.zeros(num_neurons,time)
#initial
net = CTLN(W,theta,delta_t)
x = net(x0,u)

#plot check
fig1 = plt.figure()
plt.gca().set_prop_cycle(None) #to reset the color cycle so both neurons i in initial
#and desired are the same color
plt.plot(x_desired.detach().numpy().T,
         label='x_desired')
plt.gca().set_prop_cycle(None) #to reset the color cycle so both neurons i in initial
#and desired are the same color
plt.plot(x.detach().numpy().T,linestyle='dashed',
         label='current')
plt.title("x_desired solid, x_current dotted")
#plt.legend()
plt.show()

#learn!
num_iter= 10000
lr=0.00005

opt = optim.Adam(net.parameters(),lr=lr)
losses = []
for it in range(num_iter):
  x = net(x0,u) #forward
  current_x_fit = x[0:x_desired.size()[0],:] #current_x_fit contains only as many neurons as x_desired
  loss = F.mse_loss(current_x_fit,x_desired) #calculate error
  opt.zero_grad() # zero out old grads
  loss.backward() #computes dloss/dW and dLoss/dtheta
  opt.step()
  losses.append(loss.detach().numpy()) #remember losses to plot
  #constrain W to be negative
  net.W.data.clamp_max_(0.)
  net.W.data.mul_(net.mask)
  #plot/print every 100 iterations
  if not it%100:
     print("iter = %i" %it + ", loss = %f" %loss.detach().numpy())
"""   if not it%100:
    plt.cla() #clear axes
    plt.gca().set_prop_cycle(None) #restart colors to match i with i
    plt.plot(x_desired.detach().numpy().T)
    plt.gca().set_prop_cycle(None)
    plt.plot(current_x_fit.detach().numpy().T,linestyle='dashed')
    plt.title("iter = %i" %it + ", loss = %f" %loss.detach().numpy())
    plt.show()
    #let me pause and see
    #input("~~~~~~~~~~Press Enter to continue~~~~~~~~~~")
    plt.close() """

fig3 = plt.figure()
ax = plt.subplot(2, 1, 1)
ax.set_title("final network output (dashed) vs desired (solid)")
ax.set_prop_cycle(None) #plt.gca().set_prop_cycle(None)
ax.plot(x_desired.detach().numpy().T)
ax.set_prop_cycle(None) #plt.gca().set_prop_cycle(None)
ax.plot(x.detach().numpy().T,linestyle='dashed')
ax = plt.subplot(2, 1, 2)
ax.set_title("Losses")
plt.plot(losses)
ax.set_xlabel('iteration')
ax.set_ylabel('Loss')
plt.show()

#save the final w and theta
W_out = net.W.detach().numpy()
theta_out = net.theta.detach().numpy()


#plot all the Ws
matrices_to_plot = [W_initial.detach().numpy(),
                    theta_initial.detach().numpy().reshape((x.shape[0],1)),
                    W_out,
                    theta_out.reshape((x.shape[0],1)),
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

fig4 = plt.figure(constrained_layout=True, figsize=(11, 4))
axs = fig4.subplots(1, 6, 
                   gridspec_kw={'width_ratios': [5,1,5,1,5,1]})
for i in range(len(axs)):
    pc = axs[i].pcolormesh(matrices_to_plot[i], vmin=vmin_val, vmax=vmax_val)
    axs[i].title.set_text(subplot_titles[i])
fig4.colorbar(pc, shrink=1, ax=axs, location='bottom')
plt.show()

np.savetxt('W_final_4cycu.csv', W_out, delimiter=',')
np.savetxt('theta_final_4cycu.csv', theta_out, delimiter=',')
