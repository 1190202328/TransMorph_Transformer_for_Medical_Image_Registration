import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.TransMorph_diff_rgb import Bilinear, TransMorphDiffRGB
from models.TransMorph_diff_rgb import CONFIGS as CONFIGS_TM


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
    input_size = (1024, 512)
    rgb_range = 1

    # need change
    city_name = 'aachen'
    source_img_numbers = ['000000', '000005', '000010', '000015']
    # change done
    total_num = len(source_img_numbers)
    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorphDiffRGB']
    model = TransMorphDiffRGB(config, channel=3)
    model.cuda()

    def create_cylindrical_grid(H, W, focal_length):
        # 生成像素坐标网格
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        ys, xs = ys.to(torch.float32), xs.to(torch.float32)

        # 转换到圆柱坐标
        theta = xs * W / (2 * focal_length)
        xs = torch.sin(theta)
        ys = ys * W / (2 * focal_length)
        zs = torch.cos(theta)

        # 归一化坐标
        cylindrical_coords = torch.stack([xs, ys, zs], -1)
        cylindrical_coords = cylindrical_coords / cylindrical_coords.norm(dim=-1, keepdim=True)

        # 转换回图像坐标系
        grid = cylindrical_coords[..., :2] / cylindrical_coords[..., 2:]
        grid = grid.permute(1, 0, 2)  # 调整维度以符合grid_sample的要求

        return grid.unsqueeze(0)  # 增加一个批处理维度

    def create_dual_cylindrical_grid(H, W, scale=1.0):
        # 生成标准化坐标网格
        xx = torch.linspace(-1, 1, H)
        yy = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(yy, xx)

        # 水平方向上的双柱面坐标
        half_width = H // 2
        theta_x_left = torch.atan(grid_x[:, :half_width]) * 2
        theta_x_right = torch.atan(grid_x[:, half_width:]) * 2
        x_cyl_left = torch.sin(theta_x_left * scale)
        x_cyl_right = torch.sin(theta_x_right * scale)
        x_cyl = torch.cat((x_cyl_left, x_cyl_right), dim=1)

        # 垂直方向保持常规柱面变换
        y_cyl = grid_y

        # 创建网格
        grid = torch.stack((y_cyl, x_cyl), 2)
        grid = grid.unsqueeze(0)
        return grid

    def create_identity_grid(H, W):
        # 生成标准化坐标网格
        xx = torch.linspace(-1, 1, H)
        yy = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(yy, xx)
        grid = torch.stack((grid_y, grid_x), -1)
        grid = grid.unsqueeze(0)

        return grid

    # 创建单圆柱形变换网格
    # grid = create_cylindrical_grid(input_size[0], input_size[1], input_size[0] // 3)

    # # 创建双圆柱形变换网格
    # grid = create_dual_cylindrical_grid(input_size[0], input_size[1], scale=0.8)

    # # 创建identity变换网格
    # grid = create_identity_grid(input_size[0], input_size[1])

    # 创建双圆柱形变网络并且找到反向变换
    grid = create_dual_cylindrical_grid(input_size[0], input_size[1], scale=0.8)

    flow = grid.permute(0, 3, 1, 2).cuda()

    # change done
    flow = flow.repeat(total_num, 1, 1, 1)
    print(flow.shape)

    # [B, 2, H, W]

    # 正则
    def flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    L1_Loss = torch.nn.L1Loss()
    _, _, H, W = flow.shape
    flow_left_y = flow[:, :1, :, :W // 2]
    flow_left_x = flow[:, 1:, :, :W // 2]
    flow_right_y = flow[:, :1, :, W // 2:]
    flow_right_x = flow[:, 1:, :, W // 2:]
    print(L1_Loss(flow_left_y, flow_right_y))
    flow_right_x = flip(flow_right_x, 3)
    print(L1_Loss(flow_left_x, -flow_right_x))
    print(flow_left_x[(flow_left_x != -flow_right_x)].shape)
    print(flow_right_x[(flow_left_x != -flow_right_x)])
    raise Exception

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
        image = image.resize(input_size, Image.BICUBIC)
        input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = input_transform(image)
        images.append(image.cuda())
    images = torch.stack(images, dim=0)
    print(images.shape)

    # 准备计算梯度
    # 定义Sobel滤波器的水平和垂直方向的卷积核
    G_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    G_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    # 扩展滤波器的维度以匹配图片向量的通道数，这里是3
    G_x = G_x.repeat(3, 1, 1, 1).cuda()
    G_y = G_y.repeat(3, 1, 1, 1).cuda()
    # 选择使用padding来保持图片的尺寸不变
    padding = 1

    # 计算梯度
    # 对每个颜色通道应用卷积，计算水平和垂直梯度
    grad_x = F.conv2d(images, G_x, padding=padding, groups=3)
    grad_y = F.conv2d(images, G_y, padding=padding, groups=3)

    # get inverse flow
    deform_field = flow
    norm_coord = model.id_transform.clone()

    disp_field = deform_field - norm_coord
    inv_disp_field = - reg_model(disp_field, deform_field)
    inverse_flow = inv_disp_field + norm_coord
    # get done

    grid_img = mk_grid_img(16, 1, (images.shape[0], input_size[1], input_size[0]))

    # # sampling
    # images = torch.nn.UpsamplingBilinear2d(size=(input_size[1]//2, input_size[0]//2))(images)
    # flow = torch.nn.UpsamplingBilinear2d(size=(input_size[1] // 2, input_size[0] // 2))(flow)
    #
    # grid_img = torch.nn.UpsamplingNearest2d(size=(input_size[1] // 2, input_size[0] // 2))(grid_img)

    def_out = reg_model(images.float(), flow)
    def_grid = reg_model(grid_img.float(), flow)

    # inverse
    inverse_def_out = reg_model(def_out.float(), inverse_flow)
    inverse_def_grid = reg_model(grid_img.float(), inverse_flow)

    # get fusion target image
    mask = def_grid
    def_out_with_grid = def_out.clone()
    def_out_with_grid[:, 0:1, :, :][mask == 1] = 1  # red
    def_out_with_grid[:, 1:2, :, :][mask == 1] = 0
    def_out_with_grid[:, 2:3, :, :][mask == 1] = 0

    mask = inverse_def_grid
    inverse_def_out_with_grid = inverse_def_out.clone()
    inverse_def_out_with_grid[:, 0:1, :, :][mask == 1] = 1  # red
    inverse_def_out_with_grid[:, 1:2, :, :][mask == 1] = 0
    inverse_def_out_with_grid[:, 2:3, :, :][mask == 1] = 0

    # draw
    plt.switch_backend('agg')

    x_fig = comput_fig(images, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/image_origin.png')
    plt.close(x_fig)

    grad_x_fig = comput_fig(grad_x, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/grad_x_fig.png')
    plt.close(grad_x)

    grad_y_fig = comput_fig(grad_y, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/grad_y_fig.png')
    plt.close(grad_y_fig)

    pred_fig = comput_fig(def_out, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/image_warped.png')
    plt.close(pred_fig)

    pred_fig_with_grid = comput_fig(def_out_with_grid, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/image_warped_with_grid.png')
    plt.close(pred_fig_with_grid)

    inverse_pred_fig = comput_fig(inverse_def_out, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/inverse_image_warped.png')
    plt.close(inverse_pred_fig)

    inverse_pred_fig_with_grid = comput_fig(inverse_def_out_with_grid, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/inverse_image_warped_with_grid.png')
    plt.close(inverse_pred_fig_with_grid)

    grid_origin_fig = comput_fig(grid_img, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/grid_origin_fig.png')
    plt.close(grid_origin_fig)

    grid_fig = comput_fig(def_grid, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/grid_fig.png')
    plt.close(grid_fig)

    inverse_grid_fig = comput_fig(inverse_def_grid, rgb_range, total_num)
    plt.savefig(f'{visualization_save_dir}/inverse_grid_fig.png')
    plt.close(inverse_grid_fig)


def comput_fig(img, rgb_range=1, total_num=16):
    sqart_total_num = int(total_num ** 0.5)
    img = img.detach().cpu().numpy()[0:total_num, ...]
    fig = plt.figure(figsize=(12, 6), dpi=180)
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
