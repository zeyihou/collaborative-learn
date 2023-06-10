print("\n ===================================================================================================")

#----------------------------------------
import argparse
import os
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from torch import autograd
from torchvision.utils import save_image
from tqdm import tqdm, trange
import gc
import h5py

#----------------------------------------
from opts import gen_synth_data_opts
from utils import *
from models import *
from eval_metrics import compute_FID, compute_IS


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
args = gen_synth_data_opts()
print(args)


subsampling_method = "DDLS_nSteps_{}_alpha_{}_stepLr_{}_epsStd_{}".format(args.ddls_n_steps, args.ddls_alpha, args.ddls_step_lr, args.ddls_eps_std)

path_torch_home = os.path.join(args.root_path, 'torch_cache')
os.makedirs(path_torch_home, exist_ok=True)
os.environ['TORCH_HOME'] = path_torch_home

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
output_directory = os.path.join(args.root_path, 'output/Setting_{}'.format(args.gan_net))
os.makedirs(output_directory, exist_ok=True)

save_evalresults_folder = os.path.join(output_directory, 'eval_results')
os.makedirs(save_evalresults_folder, exist_ok=True)

dump_fake_images_folder = os.path.join(output_directory, 'dump_fake')
os.makedirs(dump_fake_images_folder, exist_ok=True)


#######################################################################################
'''                                  Load Data                                      '''
#######################################################################################
trainset_h5py_file = args.data_path + '/ImageNet_128x128_100Class.h5'
hf = h5py.File(trainset_h5py_file, 'r')
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
images_test = hf['images_valid'][:]
labels_test = hf['labels_valid'][:]
hf.close()

#######################################################################################
'''                  Load pre-trained GAN to Memory (not GPU)                       '''
#######################################################################################
if args.gan_net=="BigGANdeep":
    ckpt_g = torch.load(args.gan_gene_ckpt_path)
    ckpt_d = torch.load(args.gan_disc_ckpt_path)
    netG = BigGANdeep_Generator(G_ch=128, dim_z=args.gan_dim_g, resolution=args.img_size, G_attn='64', n_classes=args.num_classes, G_shared=True, shared_dim=128, hier=True)
    netG.load_state_dict(ckpt_g)
    netG = nn.DataParallel(netG)
    netD = BigGANdeep_Discriminator(D_ch=128, resolution=args.img_size, D_attn='64', n_classes=args.num_classes)
    netD.load_state_dict(ckpt_d)
    netD = nn.DataParallel(netD)
elif args.gan_net=="BigGAN":
    ckpt_g = torch.load(args.gan_gene_ckpt_path)
    ckpt_d = torch.load(args.gan_disc_ckpt_path)
    netG = BigGAN_Generator(G_ch=96, dim_z=args.gan_dim_g, resolution=args.img_size, G_attn='64', n_classes=args.num_classes, G_shared=True, shared_dim=128, hier=True)
    netG.load_state_dict(ckpt_g)
    netG = nn.DataParallel(netG)
    netD = BigGAN_Discriminator(D_ch=96, resolution=args.img_size, D_attn='64', n_classes=args.num_classes)
    netD.load_state_dict(ckpt_d)
    netD = nn.DataParallel(netD)
else:
    raise Exception("Not supported GAN!!")

# compute log-probability of multivariant normal distribution: logp_z
def compute_log_norm(z):
    z_dim = z.shape[1]
    prior_mean = torch.zeros(z_dim).cuda()
    prior_covm = torch.eye(z_dim).cuda()  # 2-D tensor with ones on the diagonal and zeros elsewhere
    prior_z = torch.distributions.multivariate_normal.MultivariateNormal(prior_mean, prior_covm)
    logp_z = prior_z.log_prob(z).view(-1,1)
    return logp_z.squeeze()

def batch_l2_norm_squared(z):
    return z.pow(2).sum(dim=tuple(range(1, len(z.shape))))    #  (100,), 从dim=1开始，保留第0维batch

### Langevin dynamics
def e_grad(z, given_label, netG, netD, alpha, ret_e=False):
    batch_size = z.shape[0]
    z_dim = z.shape[1]
    z = autograd.Variable(z, requires_grad=True)
    ## prior proposal for z
    prior_mean = torch.zeros(z_dim).cuda()
    prior_covm = torch.eye(z_dim).cuda()  # 2-D tensor with ones on the diagonal and zeros elsewhere
    prior_z = torch.distributions.multivariate_normal.MultivariateNormal(prior_mean, prior_covm)

    ## compute energy
    logp_z = prior_z.log_prob(z).view(-1,1)
    gen_labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
    disc = netD(netG(z, netG.module.shared(gen_labels)), gen_labels)  # d(G(z)), d is the before-sigmoid logits of D
    Energy = - logp_z - alpha * disc   # args.ddls_alpha = 1
    gradients = autograd.grad(outputs=Energy, inputs=z,
                              grad_outputs=torch.ones_like(Energy).cuda())[0]   # [0]

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    if ret_e:
        return Energy, gradients
    return gradients, logp_z.squeeze(), disc.squeeze(), Energy.squeeze()


def langevin_dynamics_colla(z, given_label, netG, netD, alpha=args.ddls_alpha, n_steps=args.ddls_n_steps, step_lr=args.ddls_step_lr, eps_std=args.ddls_eps_std):
    z_sp = []    # initial latent code z
    batch_size, z_dim = z.shape

    # 1) proposal learning
    step_size = torch.from_numpy(np.full([batch_size,z_dim], step_lr, dtype=np.float32)).cuda()   #(batch,z_dim)
    acc_num = torch.zeros(batch_size).cuda()   #  (batch,)  num of acceptance for each batch.
    acc_num_frequency = torch.zeros(batch_size).cuda()  # acceptancy per frequency for proposal learning
    frequency = 50
    # sampler iteration
    iteratorNum = 0

    # 2) target distribution learning
    numofEnergy = 70
    DenseofEnergy =  np.full([batch_size,numofEnergy], 1.0/numofEnergy, dtype=float)   # (batch_size * numofEnergy) initial:1/numofEnergy
    t0 = 300  # no change
    const = 0
    pi = 1.0 / numofEnergy
    energy_start = 150           # Burnin energy_start:162.2285766601
    energy_end = 220             # Burnin energy_start:209.42755126953125

    def EnergyIndex(energy_bach):         # energy_batch: (batch_size,)->(100,)
        partition = (energy_end - energy_start) / numofEnergy    # (1,)
        IndexofEnergy = ( (energy_bach-energy_start)/partition ).int()     # IndexofEnergy:(batch_size, )
        judge_upper = IndexofEnergy > (numofEnergy-1)
        IndexofEnergy[judge_upper] = numofEnergy-1
        judge_lower = IndexofEnergy < 0
        IndexofEnergy[judge_lower] = 0
        return IndexofEnergy

    def getTheta(index_batch):   # index_batch : (batch_size, )
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

    #  initial latent code for Langevin dynamics
    old_latent = z
    old_gradients, old_logp_z, old_score, old_Energy = e_grad(old_latent, given_label, netG, netD, alpha, ret_e=False)

    for _ in range(n_steps):

        iteratorNum += 1  # MCMC steps

        # update frequency for proposal learning
        if iteratorNum % frequency == 0 and iteratorNum < 600:
            acc_num_frequency = acc_num_frequency / frequency  # accept rate
            upper_sig = acc_num_frequency > 0.6
            lower_sig = acc_num_frequency < 0.45
            step_size[upper_sig] += 8e-4
            step_size[lower_sig] -= 2e-4
            acc_num_frequency = acc_num_frequency * 0.0   # Reset
        if iteratorNum > 600:
            step_size = 2e-4
            
        if _ % 100 == 0:
            z_sp.append(old_latent)

        old_eps = torch.randn((batch_size,z_dim)).type(torch.float).cuda()   # noise, randn-> normal distribution
        assert old_gradients.shape == z.shape
        pro_latent = old_latent - (step_size/2) * old_gradients + (step_size**0.5) * old_eps
        pro_gradients, pro_logp_z, pro_score, pro_Energy = e_grad(pro_latent, given_label, netG, netD, alpha, ret_e=False)
        
        # calculating MH ratio (in log space for stability)
        ones = old_score.new_tensor([1.0])  # torch.Size([1])
        
        log_prior_diff = pro_logp_z - old_logp_z  # logp0(z')-logp0(zk) = log( p0(z')/p0(zk) )
        log_score_diff = pro_score - old_score    #  d(G(z'))-d(G(zk))

        log_p_old2pro = compute_log_norm(old_eps)  
        log_p_pro2old = compute_log_norm( (old_latent - pro_latent + (step_size / 2)  * pro_gradients) / (step_size**0.5) )
        log_transfer_diff = log_p_pro2old - log_p_old2pro   #  log(norm(z'->zk)) - log(norm(zk->z'))

        # 2) target learning diff
        target_learn_diff = torch.from_numpy( getTheta(EnergyIndex(old_Energy)) - getTheta(EnergyIndex(pro_Energy)) ).cuda()

        acc_ratio = torch.min((log_prior_diff + log_score_diff + log_transfer_diff +target_learn_diff).exp(), ones).squeeze()  # exponent sapce

        #  decide the acceptance with acc_ratio
        accept = torch.rand_like(acc_ratio) <= acc_ratio #  rand -> uniform distribution 

        #  update stats with the binary mask
        old_latent.data[accept] = pro_latent.data[accept]
        old_gradients.data[accept] = pro_gradients.data[accept]
        old_logp_z.data[accept] = pro_logp_z.data[accept]
        old_score.data[accept] = pro_score.data[accept]
        old_Energy.data[accept] = pro_Energy.data[accept]

        # acceptance num
        acc_num[accept] += 1
        acc_num_frequency[accept] += 1

        DenseofEnergy += UpdateDenseofEnergy(EnergyIndex(old_Energy))   # update theta value in energy bin

    acc_num = acc_num / 1.0 / n_steps
    print("acc rate:{}".format(acc_num))

    z_sp.append(old_latent)

    return z_sp


def Joint_sample(given_label, netG, netD, nfake=10000, batch_size=100, verbose=True):
    netG = netG.cuda()
    netD = netD.cuda()
    netG.eval()
    netD.eval()

    fake_images = []
    if verbose:
        pb = SimpleProgressBar()
    num_taken = 0
    while num_taken < nfake:
        z = torch.randn(batch_size, args.gan_dim_g, dtype=torch.float).cuda()
        z_sp = langevin_dynamics_colla(z, given_label, netG, netD)   # the last z = z_sp[-1]
        batch_labels = (given_label*torch.ones(batch_size)).type(torch.long).cuda()
        batch_images = netG(z_sp[-1], netG.module.shared(batch_labels))
        batch_images = batch_images.detach().cpu().numpy()
        fake_images.append(batch_images)
        num_taken+=len(batch_images)   #  add num of batch
        if verbose:
            pb.update(min(float(num_taken)/nfake, 1)*100)

    fake_images = np.concatenate(fake_images, axis=0)
    fake_labels = (given_label*torch.ones(len(fake_images))).type(torch.long).view(-1)
    return fake_images[0:nfake], fake_labels[0:nfake]


###############################################################################
'''                             Compute FID and IS                          '''
###############################################################################
if args.eval or args.samp_dump_fake_data:
    if args.inception_from_scratch:
        #load pre-trained InceptionV3 (finetued on ImageNet-100)
        print("\n Load finetuned Inception V3...")
        PreNetFID = Inception3(num_classes=args.num_classes, transform_input=True, finetune=True)
        checkpoint_PreNet = torch.load(args.eval_ckpt_path)
        # PreNetFID = nn.DataParallel(PreNetFID).cuda()
        PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'],False)
    else:
        # print("\n Load pretrained Inception V3...")
        # PreNetFID = inception_v3(pretrained=True, transform_input=True)
        # PreNetFID = nn.DataParallel(PreNetFID).cuda()
        raise Exception("Not supported yet!!")
    

    ##############################################
    ''' Compute FID between real and fake images '''
    IS_scores_all = []
    FID_scores_all = []
    Intra_FID_scores_all = []

    start = timeit.default_timer()
    for nround in range(args.samp_round):   # samp_round = 1
        print("\n {}+{}, Eval round: {}/{}".format(args.gan_net, subsampling_method, nround+1, args.samp_round))

        ### generate fake images; separate h5 files
        dump_fake_images_folder_nround = os.path.join(dump_fake_images_folder, 'fake_images_{}_subsampling_{}_NfakePerClass_{}_seed_{}_Round_{}_of_{}'.format(args.gan_net, subsampling_method, args.samp_nfake_per_class, args.seed, nround+1, args.samp_round))
        os.makedirs(dump_fake_images_folder_nround, exist_ok=True)

        fake_images = []
        fake_labels = []
        for i in range(args.num_classes):
            dump_fake_images_filename = os.path.join(dump_fake_images_folder_nround, 'class_{}_of_{}.h5'.format(i+1,args.num_classes))

            if not os.path.isfile(dump_fake_images_filename):
                print("\n Start generating {} fake images for class {}/{}.".format(args.samp_nfake_per_class, i+1, args.num_classes))
                fake_images_i, fake_labels_i = Joint_sample(given_label=i, netG=netG, netD=netD, nfake=args.samp_nfake_per_class, batch_size=args.samp_batch_size)
                assert fake_images_i.max()<=1 and fake_images_i.min()>=-1
                ## denormalize images to save memory
                fake_images_i = (fake_images_i*0.5+0.5)*255.0
                fake_images_i = fake_images_i.astype(np.uint8)

                if args.samp_dump_fake_data:
                    with h5py.File(dump_fake_images_filename, "w") as f:
                        f.create_dataset('fake_images_i', data = fake_images_i, dtype='uint8', compression="gzip", compression_opts=6)
                        f.create_dataset('fake_labels_i', data = fake_labels_i, dtype='float')

            else:
                print('\n Start loading generated fake data for class {}/{}...'.format(i+1,args.num_classes))
                with h5py.File(dump_fake_images_filename, "r") as f:
                    fake_images_i = f['fake_images_i'][:]
                    fake_labels_i = f['fake_labels_i'][:]
            
            assert fake_images_i.max()>1 and fake_images_i.max()<=255.0
            fake_images.append(fake_images_i)
            fake_labels.append(fake_labels_i.reshape(-1))
        ##end for i
        fake_images = np.concatenate(fake_images, axis=0)
        fake_labels = np.concatenate(fake_labels)

        if args.eval:
            #####################
            ## Compute Intra-FID: real vs fake
            print("\n Start compute Intra-FID between real and fake images...")
            start_time = timeit.default_timer()
            intra_fid_scores = np.zeros(args.num_classes)
            for i in range(args.num_classes):
                indx_train_i = np.where(labels_train==i)[0]
                images_train_i = images_train[indx_train_i]  # train images of class i
                indx_fake_i = np.where(fake_labels==i)[0]   
                fake_images_i = fake_images[indx_fake_i]   # fake images of class i : 100
                ##compute FID within each class
                intra_fid_scores[i] = compute_FID(PreNetFID, images_train_i, fake_images_i, batch_size = args.eval_FID_batch_size, resize = (299, 299), normalize=True)
                print("\r Eval round: {}/{}; Class:{}; Real:{}; Fake:{}; FID:{}; Time elapses:{}s.".format(nround+1, args.samp_round, i+1, len(images_train_i), len(fake_images_i), intra_fid_scores[i], timeit.default_timer()-start_time))
            ##end for i
            # average over all classes
            print("\n Eval round: {}/{}; Intra-FID: {}({}); min/max: {}/{}.".format(nround+1, args.samp_round, np.mean(intra_fid_scores), np.std(intra_fid_scores), np.min(intra_fid_scores), np.max(intra_fid_scores)))

            # dump FID versus class to npy
            dump_fids_filename = save_evalresults_folder + "/{}_subsampling_{}_round_{}_of_{}_fids_finetuned_{}".format(args.gan_net, subsampling_method, nround+1, args.samp_round, args.inception_from_scratch)
            np.savez(dump_fids_filename, fids=intra_fid_scores)

            #####################
            ## Compute FID: real vs fake
            print("\n Start compute FID between real and fake images...")
            indx_shuffle_real = np.arange(len(images_train)); np.random.shuffle(indx_shuffle_real)
            indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
            fid_score = compute_FID(PreNetFID, images_train[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = args.eval_FID_batch_size, resize = (299, 299), normalize=True)
            print("\n Eval round: {}/{}; FID between {} real and {} fake images: {}.".format(nround+1, args.samp_round, len(images_train), len(fake_images), fid_score))
            
            #####################
            ## Compute IS
            print("\n Start compute IS of fake images...")
            indx_shuffle_fake = np.arange(len(fake_images)); np.random.shuffle(indx_shuffle_fake)
            is_score, is_score_std = compute_IS(PreNetFID, fake_images[indx_shuffle_fake], batch_size = args.eval_FID_batch_size, splits=10, resize=(299,299), normalize=True)
            print("\n Eval round: {}/{}; IS of {} fake images: {}({}).".format(nround+1, args.samp_round, len(fake_images), is_score, is_score_std))

            #####################
            # Dump evaluation results
            eval_results_fullpath = os.path.join(save_evalresults_folder, '{}_subsampling_{}_finetuned_{}.txt'.format(args.gan_net, subsampling_method, args.inception_from_scratch))
            if not os.path.isfile(eval_results_fullpath):
                eval_results_logging_file = open(eval_results_fullpath, "w")
                eval_results_logging_file.close()
            with open(eval_results_fullpath, 'a') as eval_results_logging_file:
                eval_results_logging_file.write("\n===================================================================================================")
                eval_results_logging_file.write("\n Separate results for {} of {} rounds; Subsampling {} \n".format(nround, args.samp_round, subsampling_method))
                print(args, file=eval_results_logging_file)
                eval_results_logging_file.write("\n Intra-FID: {}({}); min/max: {}/{}.".format(np.mean(intra_fid_scores), np.std(intra_fid_scores), np.min(intra_fid_scores), np.max(intra_fid_scores)))
                eval_results_logging_file.write("\n FID: {}.".format(fid_score))
                eval_results_logging_file.write("\n IS: {}({}).".format(is_score, is_score_std))

            ## store
            FID_scores_all.append(fid_score)
            Intra_FID_scores_all.append(np.mean(intra_fid_scores))
            IS_scores_all.append(is_score)
        ##end if args.eval
    ##end nround
    stop = timeit.default_timer()
    print("Sampling and evaluation finished! Time elapses: {}s".format(stop - start))
        
    if args.eval:
    
        FID_scores_all = np.array(FID_scores_all)
        Intra_FID_scores_all = np.array(Intra_FID_scores_all)
        IS_scores_all = np.array(IS_scores_all)

        #####################
        # Average Eval results
        print("\n Avg Intra-FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(Intra_FID_scores_all), np.std(Intra_FID_scores_all), np.min(Intra_FID_scores_all), np.max(Intra_FID_scores_all)))

        print("\n Avg FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(FID_scores_all), np.std(FID_scores_all), np.min(FID_scores_all), np.max(FID_scores_all)))

        print("\n Avg IS over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(IS_scores_all), np.std(IS_scores_all), np.min(IS_scores_all), np.max(IS_scores_all)))
        
        #####################
        # Dump evaluation results
        eval_results_fullpath = os.path.join(save_evalresults_folder, '{}_subsampling_{}_finetuned_{}.txt'.format(args.gan_net, subsampling_method, args.inception_from_scratch))
        if not os.path.isfile(eval_results_fullpath):
            eval_results_logging_file = open(eval_results_fullpath, "w")
            eval_results_logging_file.close()
        with open(eval_results_fullpath, 'a') as eval_results_logging_file:
            eval_results_logging_file.write("\n===================================================================================================")
            eval_results_logging_file.write("\n Average results over {} rounds; Subsampling {} \n".format(args.samp_round, subsampling_method))
            print(args, file=eval_results_logging_file)
            eval_results_logging_file.write("\n Avg. Intra-FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(Intra_FID_scores_all), np.std(Intra_FID_scores_all), np.min(Intra_FID_scores_all), np.max(Intra_FID_scores_all)))
            eval_results_logging_file.write("\n Avg. FID over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(FID_scores_all), np.std(FID_scores_all), np.min(FID_scores_all), np.max(FID_scores_all)))
            eval_results_logging_file.write("\n Avg. IS over {} rounds: {}({}); min/max: {}/{}.".format(args.samp_round, np.mean(IS_scores_all), np.std(IS_scores_all), np.min(IS_scores_all), np.max(IS_scores_all)))
    ## if args.eval





print("\n ===================================================================================================")