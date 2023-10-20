import math
import torch
import torchvision
from pathlib import Path
import torch.utils.data
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from Network import Generator,MappingNetwork
from sklearn.decomposition import PCA
from sklearn import preprocessing

class load_network:
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.d_latent = 128
        self.image_size = 32
        self.style_mixing_prob = 0.9
        self.mapping_net_layers = 8

        self.log_resolution = int(math.log2(self.image_size))
        # self.generator = Generator(self.log_resolution, self.d_latent).to(self.device)
        self.generator = Generator(self.log_resolution, self.d_latent, n_features=int(32/8), max_features=int(512/8)).to(self.device)
        #self.model_path = './checkpoint_mnist_150000/GAN_GEN_150000.pth'
        self.model_path = './checkpoint/GAN_GEN_2000.pth'
        self.generator.load_state_dict(torch.load(self.model_path))
        self.n_gen_blocks = self.generator.n_block
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_net_layers).to(self.device)

    def get_w(self,batch_size):
        # mix styles
        if torch.rand(()).item() < self.style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)

            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)

            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1,w2), dim=0)
        else:
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            w = self.mapping_network(z)
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)

    def get_noise(self, batch_size):
        noise = []
        resolution = 4

        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size,1,resolution, resolution, device=self.device)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

            noise.append((n1,n2))

            resolution *= 2
        return noise

    def generate_images(self, batch_size):
        w = self.get_w(batch_size)
        noise = self.get_noise(batch_size)
        image = self.generator(w,noise)
        return image, w


def sample_style_vector():
    number_sampled = 1000
    batch_size = 100
    style_matrix = []
    network_method = load_network()
    for i in range(int(number_sampled/batch_size)):
        style_vector = network_method.get_w(batch_size)[0]
        style_matrix.append(style_vector.detach().cpu().numpy())
    style_matrix = np.asarray(style_matrix)
    style_matrix = style_matrix.reshape(number_sampled, -1)
    return style_matrix

def get_eig(style_matrix):
    scalar = preprocessing.StandardScaler().fit(style_matrix)
    style_matrix = scalar.transform(style_matrix)
    pca = PCA()
    pca.fit(style_matrix)
    eig_val1 = pca.explained_variance_
    cov = np.cov(style_matrix.T)
    print(eig_val1)
    eig_val,eig_vec = np.linalg.eig(cov)
    print(eig_val)
    print(eig_val-eig_val1)
    return eig_val


def main():
    style_matrix = sample_style_vector()
    get_eig(style_matrix)
    generator = load_network()
    batch_size = 32
    test_output_dir = './test_output'
    isExist = os.path.exists(test_output_dir)
    if not isExist:
        os.makedirs(test_output_dir)
    for i in range(batch_size):
        image, w = generator.generate_images(batch_size)
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()
        plt.imshow(image[i], cmap='gray')
        plt.savefig(test_output_dir + '/graph_{}.jpg'.format(i + 1))


if __name__ == '__main__':
    main()
