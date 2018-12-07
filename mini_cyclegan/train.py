import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
np.random.seed(3333)
import timeit

from mini_cyclegan import mini_cyclegan

# Hyperparams
G_LR = 0.0001
D_LR = 0.0001
Z_DIM = 50
BATCH_SIZE = 100
N_EPOCHS = 28
VIZ_EVERY = 100
D_UPDATES = 1

# Create both networks
g_net = mini_cyclegan.Generator(z_dim=Z_DIM, hidden_size=256)
d_net = mini_cyclegan.Discriminator()

print(g_net)
print(d_net)

# Loss function
criterion = nn.BCELoss()
# Optimizers
g_opt = optim.Adam(g_net.parameters(), lr=G_LR, betas=(0.5, 0.999))
d_opt = optim.Adam(d_net.parameters(), lr=D_LR, betas=(0.5, 0.999))

# store resulting losses out of training
d_rl_losses = []
d_fk_losses = []
d_losses = []
g_losses = []

# Pick a big sample from z and project it through G and compare to pdf_x (original data pdf)
# this is not data to be trained on, but to check G projections
sample_z = np.random.uniform(-1, 1, [n_samples, Z_DIM]).astype(np.float32)

# EXERCISE: NOTE THE VOLATILE=TRUE. WHAT IS IT DOING?
v_sample_z = Variable(torch.FloatTensor(sample_z), volatile=True)
batches_per_epoch = pdf_x.shape[0] / BATCH_SIZE
counter = 0
curr_epoch = -1
batch_timings = []

for counter in range(int(N_EPOCHS * batches_per_epoch)):
    if counter % batches_per_epoch == 0:
        # epoch change. First time this if is true, so also init variables.
        batch_idx = 0
        curr_epoch += 1
        # randomize the pdf_x samples
        np.random.shuffle(pdf_x)

    beg_t = timeit.default_timer()

    # sample a batch from prior pdf z
    batch_z = torch.FloatTensor(
        np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32))
    batch_z = Variable(batch_z)

    # get a batch of samples from gtruth pdf
    batch_x_real = torch.FloatTensor(pdf_x[batch_idx:(batch_idx + BATCH_SIZE)])
    batch_x_real = Variable(batch_x_real)
    d_opt.zero_grad()
    g_opt.zero_grad()

    # ------------ DISCRIMINATOR TRAINING
    # build real label
    for d_i in range(D_UPDATES):
        d_opt.zero_grad()
        labv = Variable(torch.ones(batch_x_real.size(0)))

        # (1) REAL D LOSS
        d_real = d_net(batch_x_real)
        d_real_loss = criterion(d_real, labv)
        d_real_loss.backward()

        # (2) FAKE D LOSS
        batch_x_fake = g_net(batch_z)

        # EXERCISE: NOTE THE DETACH. WHAT IS IT DOING?
        d_fake = d_net(batch_x_fake.detach())

        # build fake label
        labv.data.fill_(0.)
        d_fake_loss = criterion(d_fake, labv)
        d_fake_loss.backward()

        # Update weights with the computed gradients (all of them, no zero_grad)
        d_opt.step()

    d_loss = d_fake_loss + d_real_loss

    # ------------ GENERATOR TRAINING
    # TO DO:
    # (1) build real label `labv`
    labv = torch.ones(batch_x_real.size(0))
    # (2) forward the z batch through G
    batch_x_fake = g_net(batch_z)
    g_real = d_net(batch_x_fake)
    # (3) compute the G real loss with the label Variable
    g_real_loss = criterion(g_real, labv)
    # (4) backprop gradients
    g_real_loss.backward()
    # (5) update network parameters
    g_opt.step()

    # Gather losses to print later
    d_fk_losses.append(d_fake_loss.data.numpy())
    d_rl_losses.append(d_real_loss.data.numpy())
    d_losses.append(d_loss.data.numpy())
    g_losses.append(g_real_loss.data.numpy())

    end_t = timeit.default_timer()
    batch_timings.append(end_t - beg_t)

    if counter % VIZ_EVERY == 0:
        fig = plt.figure(figsize=(8, 8))
        fake_pred = g_net(v_sample_z).data.numpy()
        _ = plt.scatter(sample_z[:, 0], sample_z[:, 1], edgecolor='none',
                        color='orange')
        _ = plt.scatter(pdf_x[:, 0], pdf_x[:, 1], edgecolor='none')
        _ = plt.scatter(fake_pred[:, 0], fake_pred[:, 1], color='green',
                        edgecolor='none')
        plt.show()
print("Done training for {} epochs! Elapsed time: {} s".format(N_EPOCHS,
                                                               np.sum(
                                                                   batch_timings)))
print("Total amount of iterations done: ", counter)
