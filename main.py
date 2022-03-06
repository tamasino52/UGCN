"""
This code is unofficial implementation of
'Motion Guided 3D Pose Estimation from Videos' (Authors : Jingbo Wang1, Sijie Yan1, Yuanjun Xiong2, and Dahua Lin1)
Also, this code is based on https://github.com/Vegetebird/MHFormer repository.
"""

import os
import glob
import torch
import random
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from model.ugcn import Model

def show3DposePair(realt3d, faket3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
                   gt=True, pred=False):  # blue, orange
  """
  Visualize a 3d skeleton pair

  Args
  channels: 96x1 vector. The pose to plot.
  ax: matplotlib 3d axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  realt3d = np.reshape(realt3d, (17, -1))
  faket3d = np.reshape(faket3d, (17, -1))

  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  for idx, vals in enumerate([realt3d, faket3d]):
    # Make connection matrix
    for i in np.arange(len(I)):
      x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
      if idx == 0:
        ax.plot(x, z, -y, lw=2, c='k')
      #        ax.plot(x,y, z,  lw=2, c='k')

      elif idx == 1:
        ax.plot(x, z, -y, lw=2, c='r')
      #        ax.plot(x,y, z,  lw=2, c='r')

      else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 1  # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")

  # Get rid of the ticks and tick labels
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #  ax.set_zticks([])
  #
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])
  #  ax.set_zticklabels([])
  #     ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)


def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    loss_all = {'loss': AccumLoss()}

    error_sum = AccumLoss()
    action_error_sum = define_error_list(actions)

    if split == 'train':
        model.train()
    else:
        model.eval()

    t = tqdm(dataLoader, 0)
    for i, data in enumerate(t):

        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])
    
        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels) 
        out_target[:, :, 0] = 0
        gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.test_augmentation and split =='test':
            input_2D, output_3D = input_augmentation(input_2D, model)
        else:
            input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor) # N, C, T, J, M
            output_3D = model(input_2D) 

        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.out_joints, opt.out_channels)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), opt.out_joints, opt.out_channels)
        
        output_3D_single = output_3D[:, opt.pad].unsqueeze(1) 

        if split == 'train':
            pred_out = output_3D 
        elif split == 'test':
            pred_out = output_3D_single

        loss = mpjpe_cal(pred_out, out_target)
        motion_loss = motion_cal(pred_out, out_target)
        loss_np = loss.detach().cpu().numpy()
        loss_all['loss'].update(loss_np * N, N)

        # loss prompting
        t.set_description('Loss({0:,.4f})'.format(loss_np))
        t.refresh()

        if split == 'train':
            optimizer.zero_grad()
            (loss + opt.alpha * motion_loss).backward()
            optimizer.step()

            pred_out[:, :, 0, :] = 0
            joint_error = mpjpe_cal(pred_out, out_target).item()
            error_sum.update(joint_error*N, N)
        elif split == 'test':
            pred_out[:, :, 0, :] = 0
            action_error_sum = test_calculation(pred_out, out_target, action, action_error_sum, opt.dataset, subject)

        if split == 'train' and i % 1000 == 0:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1, projection='3d')
            ax1.set_title('Pair-3D')

            show3DposePair(out_target[0, 48].cpu().detach().numpy(), pred_out[0, 48].cpu().detach().numpy(), ax=ax1)
            plt.savefig(os.path.join(opt.checkpoint, 'viz.png'), dpi=100)
            plt.close(fig)

    if split == 'train':
        return loss_all['loss'].avg, error_sum.avg*1000
    elif split == 'test':
        mpjpe = print_error(opt.dataset, action_error_sum, opt.train)

        return mpjpe


def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13] 
    joints_right = [1, 2, 3, 14, 15, 16]

    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)

    output_3D_flip = model(input_2D_flip)
    output_3D_flip[:, 0] *= -1
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D


if __name__ == '__main__':
    opt = opts().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    opt.manualSeed = 4321
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    if opt.test:
        test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path =root_path)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    model = Model(opt).cuda()

    model_dict = model.state_dict()
    if opt.reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))[0]
        print(model_path)

        pre_dict = torch.load(model_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]

        model.load_state_dict(model_dict)

    all_param = []
    lr = opt.lr
    all_param += list(model.parameters())

    optimizer = optim.AdamW(all_param, lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90, 100], gamma=opt.lr_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)

    for epoch in range(1, opt.nepoch):
        lr = scheduler.get_last_lr()[0]
        if opt.train: 
            loss, error = train(opt, actions, train_dataloader, model, optimizer, epoch)

        if opt.test:
            mpjpe = val(opt, actions, test_dataloader, model)
            data_threshold = mpjpe

            if opt.train and data_threshold < opt.previous_best_threshold: 
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model)
                opt.previous_best_threshold = data_threshold

            if opt.train == 0:
                print('mpjpe: %.2f' % (mpjpe))
                break
            else:
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, mpjpe: %.2f' % (epoch, lr, loss, mpjpe))
                print('e: %d, lr: %.7f, loss: %.4f, mpjpe: %.2f' % (epoch, lr, loss, mpjpe))

        scheduler.step()

    print(opt.checkpoint)








