import torch
import numpy as np
from torch.distributions import MultivariateNormal, Normal, Independent, Uniform
import math
import torchvision
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
import os
import datetime

def batch_l2_norm_squared(z):
    return z.pow(2).sum(dim=tuple(range(1, len(z.shape))))    #  (100,)

def collaborative_learn(netG, 
        netD, 
        calibrator, 
        device, 
        nz=100,
        batch_size=100, 
        clen=640, 
        tau=0.1, 
        eta=0.3162):
    
    # shortcut for getting score and updated means (latent + tau/2 * grads) from latent
    def latent_grad_step(latent): 
        with torch.autograd.enable_grad():
            latent.requires_grad = True
            score = calibrator(netD(netG(latent))).squeeze() # latent -> sample -> score  d(G(z))
            obj = torch.sum(score - batch_l2_norm_squared(latent)/2) # get L2MC grads  ,  obj:(1,) == - Energy
            grads = torch.autograd.grad(obj, latent)[0]   # grads (100,100,1,1)
        mean = latent.data + tau/2 * grads
        Energy = -obj
        return score, mean #, Energy

    # initialize the chain and step once
    old_latent = torch.randn(batch_size, nz, 1, 1, device=device, requires_grad=True) # assume gaussian prior N(0,1), others also work
    #print("latent dimension: {}".format(old_latent.shape) )
    old_score, old_mean = latent_grad_step(old_latent) # current score and next-step mean
    one = old_score.new_tensor([1.0])

    loc = torch.zeros(nz).to(device)
    scale = torch.ones(nz).to(device)
    normal = Normal(loc, scale)
    diagn = Independent(normal, 1)
    
    # old Energy
    old_latent_prob = - diagn.log_prob(old_latent.squeeze())   # (100,)  -log P0(z)
    old_energy = old_latent_prob - old_score             # E(z) = -log P0(z) - d(G(z))  Dimension: (B,)->(100,)
 
    step_size = torch.from_numpy(np.full([batch_size,nz], eta, dtype=float) ).to(device)   #(100,100) 
    acc_num = torch.zeros(batch_size).to(device)   #  (100,)
    frequent = 30

    # sampler iteration
    iteratorNum = 0

    numofEnergy = 50
    DenseofEnergy =  np.full([batch_size,numofEnergy], 1.0/numofEnergy, dtype=float)   # (batch_size * numofEnergy)->(100,50)

    t0 = 400  # no change
    const = 0
    pi = 1.0 / numofEnergy
    energy_start = 130           # energy_start:95.68083953857422,energy_end:142.27020263671875
    energy_end = 210            # energy_start:95.53443908691406,energy_end:154.93214416503906

    def EnergyIndex(energy_bach):         # energy_batch: (batch_size,)->(100,)
        partition = (energy_end - energy_start) / numofEnergy    # (1,)
        IndexofEnergy = ( (energy_bach-energy_start)/partition ).int()     # IndexofEnergy:(batch_size, )
        judge_upper = IndexofEnergy > (numofEnergy-1)
        IndexofEnergy[judge_upper] = numofEnergy-1
        judge_lower = IndexofEnergy < 0
        IndexofEnergy[judge_lower] = 0
        return IndexofEnergy

    def getNormalizeTheta(index_batch):   # index_batch : (batch_size, )
        theta_norm = DenseofEnergy[ [i for i in range(batch_size)], index_batch.cpu() ]   # theta_norm: (batch_size,)
        return theta_norm     # (batch_size, )

    # return change array to DenseofEnergy
    def UpdateDenseofEnergy(index_batch):  # index_batch : (batch_size,)
        temp = iteratorNum
        gamma = t0 / max(t0,temp)
        Dense_temp = np.zeros_like(DenseofEnergy)  # zeros   B * bins
        Dense_temp -= (gamma * pi)
        Dense_temp[ [i for i in range(batch_size)], index_batch.cpu() ] = 0  # index=0
        Dense_temp[ [i for i in range(batch_size)], index_batch.cpu() ] += (gamma *(1-pi) )
        return Dense_temp


    # ###################         test             ########################
    #print("index of old_energy:")
    #energy_test = torch.ones(100).to(device)
    # energy_test = torch.from_numpy( np.full([batch_size], 300, dtype=float) ) 
    # print(energy_test)
    # inx = getNormalizeTheta(EnergyIndex(old_energy))
    # print(inx.shape)
    # print("inx:")
    # print(EnergyIndex(old_energy))
    # DenseofEnergy += UpdateDenseofEnergy(EnergyIndex(old_energy))
    # for i in DenseofEnergy:
    #     print(i)
    # exit()
    
    # MCMC transitions
    for _ in range(clen):
        iteratorNum += 1
        
        # update frequency
        if iteratorNum % frequent == 0:
            acc_num = acc_num / frequent  # accept rate
            upper_sig = acc_num > 0.45
            lower_sig = acc_num < 0.35
            step_size[upper_sig] += 0.6
            step_size[lower_sig] -= 0.2
            acc_num = acc_num * 0.0   # Reset

        # 1) proposal and step once
        #返回和old_latent大小相同的张量,由N（0,1）填充
        prop_noise = torch.randn_like(old_latent) # draw the proposed noise in q(z'|z_k)
        
        # prop_latent = old_mean + eta * prop_noise
        
        # adaptive proposal
        noise_tmp = prop_noise.squeeze() * step_size  # prop_noise : (100,100)
        noise = noise_tmp.unsqueeze(dim=2).unsqueeze(dim=3)    #  (100,100,1,1)
        noise=noise.to(device)

        prop_latent = old_mean + noise
        prop_latent = prop_latent.type(torch.FloatTensor)
        prop_latent = prop_latent.to(device)
        
        prop_score, prop_mean = latent_grad_step(prop_latent) # proposed score and next-step mean

        # prop Energy
        prop_latent_prob = - diagn.log_prob(prop_latent.squeeze())   # (100,)  -log P0(z)

        prop_energy = prop_latent_prob - prop_score             # E(z) = -log P0(z) - d(G(z))  Dimension: (B,)->(100,)

        # 2a) calculating MH ratio (in log space for stability)
        score_diff = prop_score - old_score  # note that the scores are before-sigmoid logits (also work for WGAN)
        latent_diff = batch_l2_norm_squared(old_latent)/2 - batch_l2_norm_squared(prop_latent)/2   #p0(z'),p0(zk)
        noise_diff = batch_l2_norm_squared(prop_noise)/2 - batch_l2_norm_squared(old_latent-prop_mean)/(2*eta**2)
        Ada_diff = torch.from_numpy( getNormalizeTheta(EnergyIndex(old_energy)) - getNormalizeTheta(EnergyIndex(prop_energy)) ).to(device)

        alpha = torch.min((score_diff + latent_diff + noise_diff + Ada_diff).exp(), one)   # dimension: (100,)

        # 2b) decide the acceptance with alpha
        accept = torch.rand_like(alpha) <= alpha

        # 2c) update stats with the binary mask
        old_latent.data[accept] = prop_latent.data[accept]
        old_score.data[accept] = prop_score.data[accept]
        old_mean.data[accept] = prop_mean.data[accept]
        old_energy[accept] = prop_energy[accept]
        
        # acceptance num
        acc_num[accept] += 1

        DenseofEnergy += UpdateDenseofEnergy(EnergyIndex(old_energy))   # update theta value in energy bin

    # get the final samples
    fake_samples = netG(old_latent)

    return fake_samples
