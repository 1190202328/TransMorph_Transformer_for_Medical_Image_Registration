import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.TransMorph_diff_rgb import Bilinear


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    visualization_save_dir = '/nfs/ofs-902-1/object-detection/jiangjing/experiments/transmorph/visualization'
    root_dir = '/nfs/s3_common_dataset/cityscapes/leftImg8bit/train/'
    crop_size = 512
    rgb_range = 1

    # need change
    city_name = 'aachen'
    source_img_numbers = ['000000', '000005', '000010', '000015']
    total_num = len(source_img_numbers)

    flow = torch.randn([2, crop_size, crop_size]).cuda()
    
    # change done
    flow = flow.repeat(total_num, 1, 1, 1)
    '''
    Initialize spatial transformation function
    '''
    reg_model = Bilinear(zero_boundary=True, mode='nearest').cuda()
    for param in reg_model.parameters():
        param.requires_grad = False
        param.volatile = True
    reg_model_bilin = Bilinear(zero_boundary=True, mode='bilinear').cuda()
    for param in reg_model_bilin.parameters():
        param.requires_grad = False
        param.volatile = True

    # get image
    images = []
    for number in source_img_numbers:
        source_img_path = f'{root_dir}/{city_name}/{city_name}_{number}_000019_leftImg8bit.png'
        image = Image.open(source_img_path).convert('RGB')
        # resize
        image = image.resize((crop_size, crop_size), Image.BICUBIC)
        input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = input_transform(image)
        images.append(image.cuda())
    images = torch.stack(images, dim=0)

    grid_img = mk_grid_img(8, 1, (images.shape[0], crop_size, crop_size))
    def_out = reg_model(images.float(), flow)
    def_grid = reg_model_bilin(grid_img.float(), flow)

    # draw
    plt.switch_backend('agg')
    pred_fig = comput_fig(def_out, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/image_warped.png')
    plt.close(pred_fig)

    grid_origin_fig = comput_fig(grid_img, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/grid_origin_fig.png')
    plt.close(grid_origin_fig)

    grid_fig = comput_fig(def_grid, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/grid_fig.png')
    plt.close(grid_fig)

    x_fig = comput_fig(images, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/image_origin.png')
    plt.close(x_fig)


def comput_fig(img, rgb_range=1, total_num=16):
    sqart_total_num = int(total_num ** 0.5)
    img = img.detach().cpu().numpy()[0:total_num, ...]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(sqart_total_num, sqart_total_num, i + 1)
        plt.axis('off')
        img_local = img[i]

        img_local = np.transpose(img_local, (1, 2, 0))
        if img_local.shape[-1] == 1:
            # gray
            plt.imshow(img_local, cmap='gray')
        else:
            if rgb_range != 1:
                # convert 0-255 to long
                img_local = img_local.astype(np.uint8)
            plt.imshow(img_local)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(64, 256, 256)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j:j + line_thickness, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i:i + line_thickness] = 1
    grid_img = grid_img[:, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # set random seed
    set_random_seed(12345)

    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
