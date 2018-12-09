
#for folder in $(ls caricatures/Test_Cartoon);  do mv caricatures/Test_Cartoon/$folder/* ../2018-dlai-team3/datasets/celebs/testB/ ; done
#for folder in $(ls caricatures/Train_Cartoon);  do mv caricatures/Train_Cartoon/$folder/* ../2018-dlai-team3/datasets/celebs/trainB/ ; done

# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/001* ../2018-dlai-team3/datasets/celebs/trainA/
# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/002* ../2018-dlai-team3/datasets/celebs/trainA/
# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/003* ../2018-dlai-team3/datasets/celebs/trainA/
# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/004* ../2018-dlai-team3/datasets/celebs/trainA/
# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/005* ../2018-dlai-team3/datasets/celebs/trainA/

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from PIL import Image

from argparse import Namespace


import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

opt = Namespace(
    dataroot='./datasets/celebs/',
    batch_size=12, # 384s/epoch
    loadSize=71,
    fineSize=64,
    display_winsize=64,
    input_nc=3,# num channels
    output_nc=3,# num channels
    ngf=64,
    ndf=64,
    netD='basic',
    netG='resnet_9blocks',
    n_layers_D=3,
    gpu_ids=['0'],
    name='caricatures_cyclegan64',
    dataset_mode='unaligned',
    model='cycle_gan',    
    direction='AtoB',
    epoch='latest',
    load_iter=0,
    num_threads=4,
    checkpoints_dir='./checkpoints',
    norm='instance',
    serial_batches='store_true',
    no_dropout=True,
    max_dataset_size=float("inf"),
    resize_or_crop='resize_and_crop',
    no_flip=False,
    init_type='normal',
    init_gain=0.02,
    verbose=True,
    suffix='',
    isTrain=True,
    lambda_identity=0.5,
    display_freq=400,
    display_ncols=4,
    display_id=1,
    display_server='http://localhost',
    display_env='main',
    display_port=8097,
    update_html_freq=1000,
    print_freq=250,
    save_latest_freq=1000,
    save_epoch_freq=1,
    epoch_count=17,
    phase='train',
    niter=100,
    niter_decay=1,
    beta1=0.5,
    lr=0.0002,
    no_lsgan=True,
    pool_size=50,
    lr_policy='lambda',
    lr_decay_iters=12,
    continue_train=False,
    lambda_A=10.0,
    lambda_B=10.0,
    no_html=False
    
)

#!sed -i 's/net\.to(gpu_ids2/net.to(gpu_ids/g' models/networks.py
#!sed -i 's/net\.to(gpu_ids22/net.to(gpu_ids2/g' models/networks.py
#!sed -i 's/net\.to(gpu_ids2+/net.to(gpu_ids2/g' models/networks.py


from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions

# restart runtime environment
# option1 automatically done by autoreload
# option1b more autoreload
# option2 manually:
import importlib
import models
import PIL
importlib.reload(models)
importlib.reload(PIL)
from models import create_model
from models import networks
from models import test_model


#opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
model.setup(opt)
visualizer = Visualizer(opt)
total_steps = 0

print("range ",opt.epoch_count, opt.niter + opt.niter_decay + 1)
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    

    for i, data in enumerate(dataset):
        
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        #visualizer.reset()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            losses = model.get_current_losses()
            t = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
            if opt.display_id > 0:
                print(epoch, round(float(epoch_iter) / dataset_size *100),"%", opt, losses)
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save_networks('latest')

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()