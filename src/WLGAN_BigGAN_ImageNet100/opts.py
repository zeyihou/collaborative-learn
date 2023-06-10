import argparse


def gen_synth_data_opts():
    parser = argparse.ArgumentParser()

    ''' Overall settings '''
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--eval_ckpt_path', type=str, default='')
    parser.add_argument('--gan_gene_ckpt_path', type=str, default='')
    parser.add_argument('--gan_disc_ckpt_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 2020)')
    parser.add_argument('--num_workers', type=int, default=0)

    ''' Datast Settings '''
    parser.add_argument('--num_classes', type=int, default=100, metavar='N',choices=[10, 100]) #CIFAR10 or CIFAR100
    parser.add_argument('--num_channels', type=int, default=3, metavar='N')
    parser.add_argument('--img_size', type=int, default=128, metavar='N')
    parser.add_argument('--visualize_fake_images', action='store_true', default=False)

    ''' GAN Settings '''
    parser.add_argument('--gan_net', type=str, default='BigGANdeep')
    parser.add_argument('--gan_dim_g', type=int, default=128)   

    ''' DDLS setting '''
    parser.add_argument('--ddls_n_steps', type=int, default=1000)
    parser.add_argument('--ddls_alpha', type=float, default=1)
    parser.add_argument('--ddls_step_lr', type=float, default=1e-4)  
    parser.add_argument('--ddls_eps_std', type=float, default=2e-4) 

    ''' Sampling Settings '''
    parser.add_argument('--samp_round', type=int, default=3)
    parser.add_argument('--samp_batch_size', type=int, default=1000) #also used for computing density ratios after the dre training
    parser.add_argument('--samp_burnin_size', type=int, default=5000)
    parser.add_argument('--samp_nfake_per_class', type=int, default=10000) #number of fake images per class for evaluation
    parser.add_argument('--samp_dump_fake_data', action='store_true', default=False)

    ''' Evaluation '''
    parser.add_argument('--inception_from_scratch', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_FID_batch_size', type=int, default=200)
    

    args = parser.parse_args()

    return args



