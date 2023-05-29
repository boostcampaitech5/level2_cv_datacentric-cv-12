import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--json_dir', type=str , default=None) # json_dir 추가
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
    
    parser.add_argument('--resume', type=str , default=None) # pth 추가
    parser.add_argument('--best_loss', type=float , default=None) # best loss 추가


    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

# json_dir 인자 추가, resume, best_loss 추가
def do_training(data_dir, json_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, resume, best_loss):
    dataset = SceneTextDataset(
        data_dir,
        json_dir, # json_dir 추가
        # split='train', ## split 제거
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    
    # resume 시 pth를 model에 장착
    if resume:
        model.load_state_dict(torch.load(resume))
    model.to(device)
    
    # optimizer 설정
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    
    # scheduler 설정
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=0.001
    )

    # mean_loss 저장
    mean_loss = best_loss if best_loss else 999
    
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        cls_loss, angle_loss, iou_loss = 0, 0, 0 # loss 저장
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
                
                # batch 별 loss 더하기
                cls_loss += extra_info['cls_loss']
                angle_loss += extra_info['angle_loss']
                iou_loss += extra_info['iou_loss']
                
        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        # batch 별 평균 loss 저장
        wandb.log({
            'Cls_epoch_loss': cls_loss / num_batches,
            'Angle_epoch_loss': angle_loss / num_batches,
            'IoU_epoch_loss': iou_loss / num_batches,
            "Mean_epoch_loss": epoch_loss / num_batches,
        })

        
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        # best 모델 저장
        if mean_loss > (epoch_loss/num_batches):
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'best_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)       
            mean_loss = epoch_loss/num_batches



def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()

    wandb.login()
    wandb.init(
        project = 'OCR',
        name='baseline_img300',
        entity='ganddddi_datacentric',
        resume= True if args.resume else False
    )
    
    main(args)
