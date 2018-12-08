import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim

from dataloader import dataloader
import utils as utils


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class CYCLEGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.datasetA = args.datasetA
        self.datasetB = args.datasetB
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62

        # load dataset
        self.data_loaderA = dataloader(self.datasetA, self.input_size, self.batch_size)
        dataA = self.data_loaderA.__iter__().__next__()[0]
        self.data_loaderB = dataloader(self.datasetB, self.input_size, self.batch_size)
        dataB = self.data_loaderB.__iter__().__next__()[0]

        # networks init
        self.G_A2B = generator(input_dim=self.z_dim,
                               output_dim=dataB.shape[1],
                               input_size=self.input_size)
        self.D_A = discriminator(input_dim=dataA.shape[1],
                                 output_dim=1,
                                 input_size=self.input_size)
        self.G_B2A = generator(input_dim=self.z_dim,
                               output_dim=dataA.shape[1],
                               input_size=self.input_size)
        self.D_B = discriminator(input_dim=dataB.shape[1],
                                 output_dim=1,
                                 input_size=self.input_size)

        self.optimizer_G = optim.Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=args.lrG,
            betas=(args.beta1, args.beta2)
        )
        self.optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(),
            lr=args.lrD,
            betas=(args.beta1, args.beta2)
        )
        self.optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(),
            lr=args.lrD,
            betas=(args.beta1, args.beta2)
        )

        if self.gpu_mode:
            self.G_A2B.cuda()
            self.D_A.cuda()
            self.G_B2A.cuda()
            self.D_B.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.criterionCycle = nn.L1Loss().cuda()
            slef.criterionIdt = nn.L1Loss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.criterionCycle = nn.L1Loss()
            slef.criterionIdt = nn.L1Loss()
            

        print('---------- Networks A architecture -------------')
        utils.print_network(self.G_A2B)
        utils.print_network(self.D_A)
        print('---------- Networks B architecture -------------')
        utils.print_network(self.G_B2A)
        utils.print_network(self.D_B)
        print('------------------------------------------------')


        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()


    def train(self):
        self.train_hist = {}
        self.train_hist['D_A_losses'] = []
        self.train_hist['D_B_losses'] = []
        self.train_hist['G_A2B_losses'] = []
        self.train_hist['G_B2A_losses'] = []
        self.train_hist['G_losses'] = []
        self.train_hist['A_cycle_losses'] = []
        self.train_hist['B_cycle_losses'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        #self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        #if self.gpu_mode:
        #    self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        #self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            #self.G.train()
            epoch_start_time = time.time()
            #for iter, (x_, _) in enumerate(self.data_loader):
            for (realA, _), (realB, _) in itertools.izip(self.data_loaderA, self.data_loaderB):
                #if iter == self.data_loader.dataset.__len__() // self.batch_size:
                #    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update G network
                self.optimizer_G.zero_grad()

                # generate real A to fake B; D_B(G_A2B(A))
                fakeB = G_A2B(realA)
                D_B_result = D_B(fakeB)
                self.y_real_ = torch.ones(self.D_B_result, 1)
                self.y_fake_ = torch.zeros(self.D_B_result, 1)
                if self.gpu_mode:
                    self.y_real_ = self.y_real_.cuda()
                    self.y_fake_ = self.y_fake_.cuda()

                G_A2B_loss = self.BCE_loss(D_B_result, self.y_real_)

                # reconstruct fake B to rec A; G_B2A(G_A2B(A))
                recA = G_B2A(fakeB)
                A_cycle_loss = self.criterionCycle(recA, realA)

                # generate real B to fake A; D_A(G_B2A(B))
                fakeA = G_B2A(realB)
                D_A_result = D_A(fakeA)
                self.y_real_ = torch.ones(self.D_A_result, 1)
                self.y_fake_ = torch.zeros(self.D_A_result, 1)
                if self.gpu_mode:
                    self.y_real_ = self.y_real_.cuda()
                    self.y_fake_ = self.y_fake_.cuda()

                G_B2A_loss = self.BCE_loss(D_A_result, self.y_real_)

                # reconstruct fake A to rec B; G_A2B(G_B2A(B))
                recB = G_A2B(fakeA)
                B_cycle_loss = self.criterionCycle(recB, realB)

                G_loss = G_A2B_loss + G_B2A_loss + A_cycle_loss + B_cycle_loss
                G_loss.backward()
                self.optimizer_G.step()

                train_hist['G_A2B_losses'].append(G_A2B_loss.data[0])
                train_hist['G_B2A_losses'].append(G_B2A_loss.data[0])
                train_hist['G_losses'].append(G_loss.data[0])
                train_hist['A_cycle_losses'].append(A_cycle_loss.data[0])
                train_hist['B_cycle_losses'].append(B_cycle_loss.data[0])
                G_A2B_losses.append(G_A2B_loss.data[0])
                G_B2A_losses.append(G_B2A_loss.data[0])
                A_cycle_losses.append(A_cycle_loss.data[0])
                B_cycle_losses.append(B_cycle_loss.data[0])


#------------------------------

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
