import math
import torch
import torchvision
from pathlib import Path
import torch.utils.data
from PIL import Image
import os
import matplotlib.pyplot as plt
from Network import Discriminator, Generator,MappingNetwork
from Network import DiscriminatorLoss, GeneratorLoss, GradientPenalty, PathLengthPenalty


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, image_size):
        super().__init__()
        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class Config:
    def __init__(self):
        self.continue_train = False
        self.dataset = 'MNIST'
        self.thread = 4
        self.device = torch.device("cuda:0")
        self.dataset_path = './Data/MNIST/trainingSet/trainingSet'
        self.image_size = 32
        self.batch_size = 32
        self.d_latent = 128
        self.mapping_net_layers = 4
        self.learning_rate = 1e-3
        self.adam_betas = (0.0, 0.99)
        self.style_mixing_prob = 0.9
        self.gradient_accumulate_steps = 1
        self.checkpoint_save_interval = 2000
        self.training_steps = 300_000 #At least 150,000 is recommended.
        self.Gradient_Penalty_coeff = 10
        self.lazy_gradient_penalty_interval = 4
        self.lazy_path_penalty_interval = 8
        self.lazy_path_penalty_after = 2_000
        self.save_dir = './checkpoint'
        self.train_sample_dir = './sample'
        self.mapping_network_lr = self.learning_rate/100
        # Check whether the specified path exists or not
        isExist = os.path.exists(self.save_dir)
        if not isExist:
            os.makedirs(self.save_dir)
        isExist = os.path.exists(self.train_sample_dir)
        if not isExist:
            os.makedirs(self.train_sample_dir)

        if self.dataset == "MNIST":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.image_size, self.image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))
            ])
            dataset = torchvision.datasets.MNIST(root='./dataset', download=True, transform = transform)
        else:
            dataset = Dataset(self.dataset_path, self.image_size)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size,
                                                 num_workers= self.thread, shuffle= True,
                                                 pin_memory=True)
        self.loader = cycle_dataloader(dataloader)
        log_resolution = int(math.log2(self.image_size))
        self.discriminator = Discriminator(log_resolution, n_features=int(32/4), max_features=int(512/4)).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent, n_features=int(32/8), max_features=int(512/8)).to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)
        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.n_gen_blocks = self.generator.n_block
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_net_layers).to(self.device)
        self.GradientPenalty = GradientPenalty()
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        if self.continue_train:
            g_weight = self.save_dir + '/GAN_GEN_300000.pth'
            d_weight = self.save_dir + '/GAN_DIS_300000.pth'
            map_weight = self.save_dir + '/GAN_MAP_300000.pth'
            self.generator.load_state_dict(torch.load(g_weight))
            self.discriminator.load_state_dict(torch.load(d_weight))
            self.mapping_network.load_state_dict(torch.load(map_weight))

        # length penalty loss

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr = self.learning_rate, betas=self.adam_betas)

        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr = self.learning_rate,
            betas = self.adam_betas
        )

        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            self.mapping_network_lr, betas = self.adam_betas
        )

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

            noise.append([n1,n2])

            resolution *= 2
        return noise

    def generate_images(self, batch_size):
        w = self.get_w(batch_size)
        noise = self.get_noise(batch_size)
        #torch.save(noise, './noise.pth')
        image = self.generator(w,noise)
        return image, w

    def step(self, idx):
        self.discriminator_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            generated_images, _ = self.generate_images(self.batch_size)
            fake_output = self.discriminator(generated_images.detach())
            real_images = next(self.loader)[0].to(self.device)
            if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                real_images.requires_grad_()
            real_output = self.discriminator(real_images)
            real_loss, fake_loss = self.discriminator_loss(real_output,fake_output)
            disc_loss = real_loss+fake_loss
            if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                # Calculate and log gradient penalty
                gp = self.GradientPenalty(real_images, real_output)

                disc_loss = disc_loss + 0.5 * self.Gradient_Penalty_coeff * gp * self.lazy_gradient_penalty_interval

            disc_loss.backward()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        # Take optimizer step
        self.discriminator_optimizer.step()

        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            generated_images,w = self.generate_images(self.batch_size)
            fake_output = self.discriminator(generated_images)

            gen_loss = self.generator_loss(fake_output)
            # Add path length penalty
            if idx > self.lazy_path_penalty_after and (idx + 1) % self.lazy_path_penalty_interval == 0:
                # Calculate path length penalty
                plp = self.path_length_penalty(w, generated_images)
                # Ignore if `nan`
                if not torch.isnan(plp):
                    gen_loss = gen_loss + plp

            gen_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)
        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()
        print('Iteration: {}, Generator loss is:{}, Discriminator loss is: {}'.format(idx+1, gen_loss, disc_loss))
        if (idx+1) % 1000 == 0:
            with torch.no_grad():
                #img = generated_images[0].permute(1,2,0).cpu().detach().numpy()
                #plt.imshow(img, cmap='gray')
                #plt.savefig(self.train_sample_dir + '/graph_{}.png'.format(idx+1))
                #torchvision.utils.make_grid(generated_images, nrow=8, normalize=True, scale_each=True)
                torchvision.utils.save_image(generated_images, self.train_sample_dir + '/sample_{}.png'.format(str(idx+1)), nrow=8, normalize=True, scale_each = True)
                #plt.show()



        if (idx+1) % self.checkpoint_save_interval == 0:
            gen_save_file = os.path.join(self.save_dir, "GAN_GEN_"  + str(idx+1) + ".pth")
            dis_save_file = os.path.join(self.save_dir, "GAN_DIS_"  + str(idx+1) + ".pth")
            gen_optim_save_file = os.path.join(
                self.save_dir, "GAN_GEN_OPTIM_" + str(idx+1) + ".pth")
            dis_optim_save_file = os.path.join(
                self.save_dir, "GAN_DIS_OPTIM_" + str(idx+1) + ".pth")
            map_save_file = os.path.join(self.save_dir, "GAN_MAP_" + str(idx+1) + ".pth")

            torch.save(self.generator.state_dict(), gen_save_file)
            torch.save(self.discriminator.state_dict(), dis_save_file)
            torch.save(self.generator_optimizer.state_dict(), gen_optim_save_file)
            torch.save(self.discriminator_optimizer.state_dict(), dis_optim_save_file)
            torch.save(self.mapping_network.state_dict(), map_save_file)

    def train(self):
        for i in range(self.training_steps):
            self.step(i)

    def test(self):
        generated_images, _ = self.generate_images(self.batch_size)
        torchvision.utils.save_image(generated_images,  './sample.png',
                                     nrow=8, normalize=True, scale_each=True)


def cycle_dataloader(data_loader):
    """
    <a id="cycle_dataloader"></a>

    ## Cycle Data Loader

    Infinite loader that recycles the data loader after each epoch
    """
    while True:
        for batch in data_loader:
            yield batch

def main():
    config = Config()
    #config.test()
    #exit()
    config.train()


if __name__ == '__main__':
    main()
