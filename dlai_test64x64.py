


# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/006* ../2018-dlai-team3/datasets/celebs/testA/
# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/007* ../2018-dlai-team3/datasets/celebs/testA/
# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/008* ../2018-dlai-team3/datasets/celebs/testA/
# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/009* ../2018-dlai-team3/datasets/celebs/testA/
# mv celebs/img_align_celeba_png.7z/img_align_celeba_png/010* ../2018-dlai-team3/datasets/celebs/testA/


import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from argparse import Namespace



opt = Namespace(

    dataroot='./datasets/celebs',
    batch_size=28, # 28 for colab, 15 for local
    loadSize=71,
    fineSize=64,
    display_winsize=64,
    input_nc=3,# num channels
    output_nc=3,# num channels
    ngf=64,   # num cnn outputs (multiplier)
    ndf=64,   # num cnn outputs (multiplier)
    #ngf=32,
    #ndf=32,
    
    #netG='resnet_9blocks',
    netG='resnet_6blocks',
    
    #netD='basic',
    #n_layers_D=3,
    netD='n_layers',
    n_layers_D=1,
    
    gpu_ids=['0'],
    name='caricatures_cyclegan64',
    dataset_mode='unaligned',
    model='test', #cycle_gan    
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
    isTrain=False,
    lambda_identity=0.5,
    display_freq=400,
    display_ncols=4,
    display_id=1,
    display_server='http://localhost',
    display_env='main',
    display_port=8097,
    update_html_freq=1000,
    print_freq=100,
    save_latest_freq=5000,
    save_epoch_freq=5,
    epoch_count=1,
    phase='test',
    niter=1,
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
    results_dir='./results/',
    eval=True,
    num_test=1000, # number of pictures to test
    model_suffix="_A",
    aspect_ratio=1.0,
    ntest=float("inf"),
    
    
    
)

#opt = TestOptions().parse()
# hard-code some parameters for test
opt.num_threads = 1   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True    # no flip
opt.display_id = -1   # no visdom display
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.setup(opt)
# create a website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
# test with eval mode. This only affects layers like batchnorm and dropout.
# pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
# CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
if opt.eval:
    model.eval()
for i, data in enumerate(dataset):
    if i >= opt.num_test:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    if i % 5 == 0:
        print('processing (%04d)-th image... %s' % (i, img_path))
    save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
# save the website
webpage.save()