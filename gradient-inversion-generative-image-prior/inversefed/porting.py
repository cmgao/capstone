import torch
import torch.nn as nn
import yaml
import pickle
import math


def load_decoder_stylegan2_mnist():
    path_dis = f"inversefed/genmodels/stylegan2/checkpoint/GAN_DIS_2000.pth"
    path_gen = f"inversefed/genmodels/stylegan2/checkpoint/GAN_GEN_2000.pth"
    path_map = f"inversefed/genmodels/stylegan2/checkpoint/GAN_MAP_2000.pth"
    log_resolution = int(math.log2(32))
    d_latent = 128
    mapping_net_layers = 4
    from .genmodels import stylegan2_mnist
    G_synthesis = stylegan2_mnist.Network.Discriminator(log_resolution, n_features=int(32/4), max_features=int(512/4)).to('cuda')
    G = stylegan2_mnist.Network.Generator(log_resolution, d_latent, n_features=int(32/8), max_features=int(512/8)).to('cuda')
    G_mapping = stylegan2_mnist.Network.MappingNetwork(d_latent, mapping_net_layers).to('cuda')
    
    G_synthesis.load_state_dict(torch.load(path_dis))
    G.load_state_dict(torch.load(path_gen))
    G_mapping.load_state_dict(torch.load(path_map))

    G_synthesis.requires_grad_(True)
    G.requires_grad_(True)
    G_mapping.requires_grad_(True)

    return G, G_mapping, G_synthesis

def load_decoder_stylegan2(config, device, dataset='FFHQ', untrained=True, ada=True, cond=False):
    
    # if ada:
    #     if cond:
    #         if untrained:
    #             path = f'inversefed/genmodels/stylegan2/{dataset}_untrained.pkl'
    #         else:
    #             path = f'inversefed/genmodels/stylegan2/{dataset}.pkl'
    #     else:
            # if untrained:
            #     path = f'inversefed/genmodels/stylegan2/{dataset}_uc_untrained.pkl'
            # else:
            #     path = f'inversefed/genmodels/stylegan2/{dataset}_uc.pkl'
    
    # else:
    #     if dataset.startswith("FF"):
    #         path = f"inversefed/genmodels/stylegan2/Gs.pth"
    #     elif dataset.startswith("I"):
    #         path = f"inversefed/genmodels/stylegan2_ada_pytorch/cifar10u-cifar-ada-best-fid.pkl"
        
        
    # if ada:
    from .genmodels.stylegan2_ada_pytorch import legacy

    # print(path)

    # with open(path, 'wb') as f:
    #     pickle.dump(path, f)

    # with open(path, 'rb') as f:
    #     G = legacy.load_network_pkl(f)['G_ema']
    #     # G.random_noise()
    #     G_mapping = G.mapping
    #     G_synthesis = G.synthesis
    # # else:
    from .genmodels import stylegan2
    G = stylegan2.models.load(path)
    G.random_noise()
    G_mapping = G.G_mapping
    G_synthesis = G.G_synthesis

    
    G.requires_grad_(True)
    G_mapping.requires_grad_(True)
    G_synthesis.requires_grad_(True)

    if torch.cuda.device_count() > 1:
        # G = nn.DataParallel(G)
        G_mapping = nn.DataParallel(G_mapping)
        G_synthesis = nn.DataParallel(G_synthesis)

    G.requires_grad_(False)
    G_mapping.requires_grad_(False)
    G_synthesis.requires_grad_(False)


    return G, G_mapping, G_synthesis



def load_decoder_stylegan2_ada(config, device, dataset='I128'):
    from .genmodels.stylegan2_ada_pytorch import legacy
    network_pkl = ''
    if dataset.startswith('I'):
        network_pkl = f'/home/jjw/projects/inverting-quantized-gradient/models/GANs/stylegan2_ada_pytorch/output/00010-ImageNet128x128-auto2/network-snapshot-025000.pkl'

    elif dataset == 'C10':
        network_pkl = 'inversefed/genmodels/stylegan2_ada_pytorch/cifar10u-cifar-ada-best-fid.pkl'
        # with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11b-cifar10/cifar10u-cifar-ada-best-fid.pkl') as f:
        #     G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) # type: ignore
    
    with open(network_pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) 
    
    return G


def load_decoder_stylegan2_untrained(config, device, dataset='I128'):
    from .genmodels.stylegan2_ada_pytorch import legacy

    if dataset == 'I128' or dataset == 'I64' or dataset == 'I32':
        network_pkl = f'/home/jjw/projects/inverting-quantized-gradient/models/GANs/stylegan2_ada_pytorch/output/00010-ImageNet128x128-auto2/network-snapshot-025000.pkl'
        print('Loading networks from "%s"...' % network_pkl)
        G = None
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) # type: ignore

    elif dataset == 'C10':
        with open('models/GANs/stylegan2_ada_pytorch/cifar10u-untrained.pkl', 'rb') as f:
            G = pickle.load(f).requires_grad_(True).to(device) 
    
    return G


def load_decoder_dcgan(config, device, dataset='C10'):
    from inversefed.genmodels.cifar10_dcgan.dcgan import Generator as DCGAN

    G = DCGAN(ngpu=1).eval()
    G.load_state_dict(torch.load('inversefed/genmodels/cifar10_dcgan/weights/netG_epoch_199.pth'))
    G.to(device)

    return G

def load_decoder_dcgan_untrained(config, device, dataset='C10'):
    if dataset == 'PERM':
        from inversefed.genmodels.deep_image_prior.generator import Generator64 as DCGAN64
        G = DCGAN64(ngpu=1)
    else:
        from inversefed.genmodels.cifar10_dcgan.dcgan import Generator as DCGAN

        G = DCGAN(ngpu=1).eval()
    G.to(device)

    return G

