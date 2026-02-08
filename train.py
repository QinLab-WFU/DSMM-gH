import json
import os
import time
import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from timm.utils import AverageMeter
from torch.cuda import device
from torch.utils.data import DataLoader
# from _data_ri import RandomSampler, ImageDataset, get_class_num, get_topk, init_dataset
from _data import build_loader, get_topk, get_class_num, build_default_trans
from _network import build_model
from build import build_models,freeze_backbone
# from build import build_model
from _utils import (
    build_optimizer,
    calc_learnable_params,
    EarlyStopping,
    init,
    mean_average_precision,
    save_checkpoint,
    seed_everything,
    validate_smart,
    rename_output,
    print_in_md,
)
from config import get_config
from loss import DMMLLoss
from util import RandomErasing
from torch.nn import CrossEntropyLoss
from save_mat import Save_mat
# def train_epoch(args, dataloader, net, criterion,criterion1, optimizer, epoch):
def get_dataset_features(net, dataloader, device):
    """获取数据集的哈希码和标签"""
    net.eval()
    hash_list = []
    label_list = []
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            cls_out, hash_out = net(images)  # 假设模型输出为 (cls_out, hash_out)
            hash_list.append(hash_out.cpu().numpy())
            label_list.append(labels.cpu().numpy())
    return np.concatenate(hash_list, axis=0), np.concatenate(label_list, axis=0)

def train_epoch(args, dataloader, net, criterion, criterion_cls, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["dm_loss", "l2_loss", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    # load_pretrained(args.pretrained_dir, net, logger)
    net.train()
    net=net.to(args.device)
    # print(f"DataLoader length: {len(dataloader)}")
    for images, labels, _ in dataloader:
        '''
        Training process
        '''

       
    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )
    return net  # 返回更新后的模型，用于特征提取

def train_init(args):
   
    criterion = DMMLLoss(args)
    criterion_cls = CrossEntropyLoss().to(args.device)
    
    logger.info(f"number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, criterion, criterion_cls, optimizer
    # return net, criterion, criterion1, optimizer

def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, criterion_cls, optimizer = train_init(args)
   
    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):

       '''
        Training process
        '''

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def get_trans(is_train=True):

    # Size of the images during training
    SIZE_TRAIN = [128, 256]
    # SIZE_TRAIN = [224, 224]
    # Size of the images during test
    SIZE_TEST = [128, 256]
    # SIZE_TEST = [224, 224]
    # Random probability for images horizontal flip
    PROB = 0.5
    # Random probability for random erasing
    RE_PROB = 0.5
    # Values to be used for images normalization
    PIXEL_MEAN = [0.5, 0.5, 0.5]
    # Values to be used for images normalization
    PIXEL_STD = [0.5, 0.5, 0.5]
    # Value of padding size
    PADDING = 0

    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    if is_train:
        trans = T.Compose(
            [
                T.Resize(SIZE_TRAIN),
                T.RandomHorizontalFlip(p=PROB),
                T.Pad(PADDING),
                T.RandomCrop(SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=RE_PROB, mean=PIXEL_MEAN),
            ]
        )
    else:
        trans = T.Compose([T.Resize(SIZE_TEST), T.ToTensor(), normalize_transform])
    return trans



def prepare_loaders(args, bl_func):
    # trans_train = get_trans(is_train=True)
    # trans_test = get_trans(is_train=False)
    # trans_train = build_default_trans("train")
    # trans_test = build_default_trans("query")
    train_loader, query_loader, dbase_loader = (
        bl_func(
            args.data_dir,
            args.dataset,
            "train",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=False,
        ),

        bl_func(
            args.data_dir,
            args.dataset,
            "query",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
        bl_func(
            args.data_dir,
            args.dataset,
            "dbase",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
    )
    return train_loader, query_loader, dbase_loader

def main():
    init()
    args = get_config()
    # 11111111111111111111
    args.save_dir = f"./output/{args.backbone}/{args.dataset}/{args.n_bits}"
    os.makedirs(args.save_dir, exist_ok=True)
    # 1111111111111111111111111
    # rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["aid"]:
        '''


           '''


            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)

            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})

    print_in_md(rst)


if __name__ == "__main__":
    main()

