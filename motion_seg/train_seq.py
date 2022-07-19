# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Train the trajectory-based motion segmentation network.
"""
import os
import shutil
import argparse
import torch
import torchvision
from core.utils.utils import load_config_file, save_model, cls_iou
from core.dataset.flythings3d_seq import flyingthings3d_seq
from core.network.traj_oa_depth import traj_oa_depth
from core.network import loss_func

def setup_dataset(cfg):
    train_transform = torchvision.transforms.ToTensor()
    test_transform = torchvision.transforms.ToTensor()
    if cfg.train_dataset == 'flyingthings3d_seq':
        train_dataset = flyingthings3d_seq(cfg.train_root, transform=train_transform, split='train', gap=1, load_flow=True, input_size=cfg.resolution)
    else:
        raise NotImplementedError
    if cfg.test_dataset == 'flyingthings3d_seq':
        test_dataset = flyingthings3d_seq(cfg.test_root, transform=test_transform, split='test', gap=1, load_flow=True, input_size=cfg.resolution)
    else:
        raise NotImplementedError
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        pin_memory=True, num_workers=cfg.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=cfg.num_workers)
    return train_loader, test_loader

def setup_model(cfg):
    if cfg.model_name == 'traj_oa_depth':
        model = traj_oa_depth(cfg.window_size, cfg.resolution)
    else:
        raise NotImplementedError
    return model

def train_epoch(cfg, model, optimizer, train_loader):
    # Train the model for one epoch over the given loader
    model.train()
    for idx, sample in enumerate(train_loader):
        # load batch
        img, flow, gt, depth = sample["imgs"], sample["flows"], sample["gts"], sample["depths"]
        traj, mask, traj_label = sample["point_traj"], sample["mask"], sample["label"]
        gt_t, depth_t = gt.float().cuda(), depth.float().cuda()
        traj_t, mask_t, traj_label_t = traj.float().cuda(), mask.float().cuda(), traj_label.float().cuda()
        input_batch = {"traj": traj_t, "mask": mask_t, "depth": depth_t}
        
        # forward
        pred = model(input_batch)
        b, _, h, w, l = gt_t.shape
        gt_t = gt_t.permute(0,4,1,2,3).reshape(b*l, _, h, w)
        traj_label_t = traj_label_t.unsqueeze(1)
        
        # weighted BCEloss
        N = traj_label_t.shape[-1]
        scale = (N - traj_label_t.sum((1,2))) / (traj_label_t.sum((1,2)) + 1e-6)
        scale = scale.unsqueeze(-1).unsqueeze(-1).repeat(1,1,N)
        weight = scale * traj_label_t + (1.0 - traj_label_t)
        loss = loss_func.BCELoss(pred, traj_label_t, weight)
        cur_iou = cls_iou(pred, traj_label_t)
        
        # optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % cfg.print_freq == 0:
            print("Train iter {}/{}, Loss {}, train iou {}".format(idx, len(train_loader), loss.item(), cur_iou.item()))
    return loss

@torch.no_grad()
def test_epoch(cfg, epoch, model, test_loader):
    model.eval()
    sum_iou = 0
    for idx, sample in enumerate(test_loader):
        # load batch
        img, flow, gt, depth = sample["imgs"], sample["flows"], sample["gts"], sample["depths"]
        traj, mask, traj_label = sample["point_traj"], sample["mask"], sample["label"]
        gt_t, depth_t = gt.float().cuda(), depth.float().cuda()
        traj_t, mask_t, traj_label_t = traj.float().cuda(), mask.float().cuda(), traj_label.float().cuda()
        input_batch = {"traj": traj_t, "mask": mask_t, "depth": depth_t}

        # inference
        pred = model(input_batch)
        b, _, h, w, l = gt_t.shape
        gt_t = gt_t.permute(0,4,1,2,3).reshape(b*l, _, h, w)
        traj_label_t = traj_label_t.unsqueeze(1)
        
        # evaluate
        cur_iou = cls_iou(pred, traj_label_t)
        sum_iou += cur_iou
    mean_iou = sum_iou / len(test_loader)
    mean_iou = mean_iou.cpu().numpy()
    print("Test epoch {}, mean iou {}".format(epoch, mean_iou))
    save_model(model, cfg.log_dir, epoch, mean_iou)
    return mean_iou

def main(cfg):
    # initialize dataloader
    train_loader, test_loader = setup_dataset(cfg)
    print('Data loader ready...Training samples number %d' % (len(train_loader)))
    # initialize model
    model = setup_model(cfg)
    model.cuda()
    # initialize training
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(cfg.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Load model from {}'.format(cfg.resume_path))
    else:
        print('training from scratch')
    
    # log file
    test_metric_log = open(os.path.join(cfg.log_dir, 'test_metrics.txt'), 'w')
    for epoch in range(cfg.max_epochs):
        train_loss = train_epoch(cfg, model, optimizer, train_loader)
        test_iou = test_epoch(cfg, epoch, model, test_loader)
        test_metric_log.write(str(epoch) + ' ' + str(test_iou) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trajectory-based motion segmentation network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config_file', metavar='DIR', help='path to config file')
    args = parser.parse_args()
    cfg = load_config_file(args.config_file)

    # copy the config file into log dir
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    shutil.copy(args.config_file, cfg.log_dir)
    
    main(cfg)
