import glob
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from natsort import natsorted
from pytorch_msssim import SSIM
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from data import datasets
from models.TransMorph_diff_rgb import CONFIGS as CONFIGS_TM
from models.TransMorph_diff_rgb import TransMorphDiffRGB, Bilinear


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
    model_name = 'TransMorphDiffRGB'
    channel = 3  # 1 for grey or 3 for rgb
    rgb_range = 1  # 1 or 255
    use_grad = False

    # need change
    dataset_name = 'FIRE'
    batch_size = 196
    weights = [1, 10]  # loss weights

    recon_loss_fuc = None
    # recon_loss_fuc = losses.SSIM_loss(data_range=255, if_MS=False)
    # recon_loss_fuc = losses.NCC_vxm()
    # recon_loss_fuc = losses.MSE_loss_2D()

    # change done

    save_dir = '{}_rec_{}_norm_{}/'.format(model_name, weights[0], weights[1])
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    lr = 0.0001  # learning rate
    epoch_start = 0
    max_epoch = 100  # max traning epoch

    '''
    Initialize model
    '''
    config = CONFIGS_TM[model_name]
    model = TransMorphDiffRGB(config, recon_loss_fuc=recon_loss_fuc, channel=channel, use_grad=use_grad)
    model.cuda()

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

    '''
    If continue from previous training
    '''

    updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = val_composed = transforms.Compose([
        transforms.Resize(config.img_size[0]),
        transforms.ToTensor()
    ])

    '''
    Initialize dataset
    '''
    if dataset_name == 'FIRE':
        # FIRE dataset
        train_dir = '/nfs/ofs-902-1/object-detection/jiangjing/datasets/FIRE/FIRE/Images'
        val_dir = '/nfs/ofs-902-1/object-detection/jiangjing/datasets/FIRE/FIRE/Images'
        train_set = datasets.FIREDataset(train_dir, transforms=train_composed, norm=rgb_range == 1)
        val_set = datasets.FIREDataset(val_dir, transforms=val_composed, norm=rgb_range == 1)
    elif dataset_name == 'UDIS':
        # UDIS dataset
        train_dir = '/nfs/ofs-902-1/object-detection/jiangjing/datasets/UDIS/UDIS-D/training'
        val_dir = '/nfs/ofs-902-1/object-detection/jiangjing/datasets/UDIS/UDIS-D/testing'
        train_set = datasets.UDISDataset(train_dir, transforms=train_composed, norm=rgb_range == 1)
        val_set = datasets.UDISDataset(val_dir, transforms=val_composed, norm=rgb_range == 1)
    else:
        raise Exception

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    ssim = SSIM(data_range=rgb_range, size_average=True, channel=channel)

    best_ncc = 0
    writer = SummaryWriter(log_dir='logs/' + save_dir)
    info_gap = 10
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            loss_sim_iter = 0
            loss_reg_iter = 0
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]

            output = model((x, y))
            loss_sim = model.get_sim_loss() * weights[0]
            loss_sim_iter += loss_sim
            loss_reg = model.scale_reg_loss() * weights[1]
            loss_reg_iter += loss_reg
            loss = loss_sim + loss_reg

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all.update(loss.item(), y.numel())

            if idx % info_gap == 0:
                with torch.no_grad():
                    print(f'origin x & y = {ssim(x, y)}')
                    print(f'warped x & x = {ssim(output[0], x)}')
                    print(f'warped x & y = {ssim(output[0], y)}')

            del output
            output = model((y, x))
            loss_sim = model.get_sim_loss() * weights[0]
            loss_sim_iter += loss_sim
            loss_reg = model.scale_reg_loss() * weights[1]
            loss_reg_iter += loss_reg
            loss = loss_sim + loss_reg
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all.update(loss.item(), y.numel())

            if idx % info_gap == 0:
                print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                       loss.item(),
                                                                                       loss_sim_iter.item() / 2,
                                                                                       loss_reg_iter.item() / 2))
            del output

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        if epoch % info_gap != 0:
            continue
        '''
        Validation
        '''
        eval_ncc = utils.AverageMeter()
        with torch.no_grad():
            i = 1
            for data in val_loader:
                print(f'evaluating {i}/{len(val_loader)}')
                i += 1
                model.eval()
                data = [t.cuda() for t in data]
                x_rgb = data[0]
                y_rgb = data[1]

                warped_img, flow, _ = model((y_rgb, x_rgb))
                ncc = ssim(warped_img, x_rgb)
                eval_ncc.update(ncc.item(), x_rgb.numel())

                # flip image
                warped_img, flow, _ = model((x_rgb, y_rgb))
                ncc = ssim(warped_img, y_rgb)
                eval_ncc.update(ncc.item(), x_rgb.numel())

            grid_img = mk_grid_img(8, 1, (x_rgb.shape[0], config.img_size[0], config.img_size[1]))
            def_out = reg_model(x_rgb.cuda().float(), flow)
            def_grid = reg_model_bilin(grid_img.float(), flow)
            print(flow[0][:, 50:55, 50:55])
        print(f'result = {eval_ncc.avg}')
        best_ncc = max(eval_ncc.avg, best_ncc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_ncc': best_ncc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_ncc.avg))
        writer.add_scalar('DSC/validate', eval_ncc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out, rgb_range)
        grid_origin_fig = comput_fig(grid_img, rgb_range)
        grid_fig = comput_fig(def_grid, rgb_range)
        x_fig = comput_fig(x_rgb, rgb_range)
        tar_fig = comput_fig(y_rgb, rgb_range)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('Grid_origin', grid_origin_fig, 0)
        plt.close(grid_origin_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()


def comput_fig(img, rgb_range=1):
    img = img.detach().cpu().numpy()[0:16, ...]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
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


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(64, 256, 256)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j:j + line_thickness, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i:i + line_thickness] = 1
    grid_img = grid_img[:, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


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
