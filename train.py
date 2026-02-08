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
        images, labels = images.to(args.device), labels.to(args.device)
        # print(f"Total samples: {len(labels)}")  # 总样本数
        # print("images:",images)
        # q_cnt = labels.shape[0] // 10
        # print(f"q_cnt = {q_cnt} (labels.shape[0] // 10)")  # 查询集大小

        # hash_out, cls_out, embeddings = net(images)

        # embeddings = net(images)
        cls_out, hash_out = net(images)
        # print("hash_out:", hash_out.shape)
        # print("cls:", cls_out.shape)
        # output_s, output, output_t = net(images, is_train=True, DEVICE=args.device)
        # print("embedding:", embeddings.shape)
        # print("embedding:", embeddings)
        # print("Independent test embeddings mean:", embeddings.mean().item())
        # print("label:", labels.shape)
        # with torch.cuda.amp.autocast(enabled=args.amp):
            # logits = net(x)

            # embeddings = net(images)


        one_labels=torch.argmax(labels,dim=1)
        # one_labels = labels.argmax(dim=1)
        # print("one:",one_labels.shape)
        # print("one:", one_labels)
        # dm_loss, l2_loss = criterion(embeddings, labels)
        # hash_loss = torch.mean((torch.abs(hash_out) - 1) ** 2)

        dm_loss, l2_loss = criterion(hash_out, one_labels)
        stat_meters["dm_loss"].update(dm_loss)
        stat_meters["l2_loss"].update(l2_loss)
        # cls_loss=criterion1(embeddings,one_labels)
        cls_loss = criterion_cls(cls_out, one_labels)

        # 3. 哈希码正则化（迫使接近 ±1）
        hash_reg_loss = torch.mean((torch.abs(hash_out) - 1) ** 2)

        # 总损失：哈希损失 + 分类损失 + 正则化（权重需调参）
        loss = dm_loss + args.weight * l2_loss+ 0.4* cls_loss

        # loss = dm_loss + args.weight * l2_loss+ 0.3 * cls_loss + 1 * hash_reg_loss
        # loss = dm_loss + args.weight * l2_loss + 0.3 * cls_loss
        # loss = dm_loss + args.weight * l2_loss+ 0.3*cls_loss+ 1.0*hash_reg_loss
        # loss = dm_loss + args.weight * l2_loss+0.1 * hash_loss
        stat_meters["loss"].update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        q_cnt = labels.shape[0] // 10
        # print(q_cnt)
        if q_cnt > 0:
            idxes = torch.randperm(len(labels))

            map_v = mean_average_precision(
                hash_out[idxes[:q_cnt]].sign(),
                hash_out[idxes[q_cnt:]].sign(),

                labels[idxes[:q_cnt]],
                labels[idxes[q_cnt:]],
            )
            # print("ql.shape[0]:",labels[idxes[:q_cnt]].shape[0])
            stat_meters["mAP"].update(map_v)
        # print("**********************************")
            l = hash_out.shape[1]


    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )
    return net  # 返回更新后的模型，用于特征提取

def train_init(args):
    # setup net

    # net = build_model(args)
    net = build_models(args,args.n_classes)
    # freeze_backbone(net, args.freeze_backbone)
    # net = MainNet(proposalN=6, num_classes=args.n_classes, channels=args.channels,
    #                 bit=args.n_bits)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net =MainNet(num_classes=args.n_classes, channels=args.channels, bit=args.n_bits)
    # state_dict = torch.load('checkpoint/aid_model_ft_20.t', map_location=device)['model_state_dict']
    # net.load_state_dict(state_dict, strict=False)
    # asymmetric_loss = AsyLoss(args.beta_param, args.bits, num_database)
    # setup criterion
    criterion = DMMLLoss(args)
    criterion_cls = CrossEntropyLoss().to(args.device)
    # criterion1 = torch.nn.CrossEntropyLoss()
    logger.info(f"number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, criterion, criterion_cls, optimizer
    # return net, criterion, criterion1, optimizer

def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, criterion_cls, optimizer = train_init(args)
    # net, criterion, criterion1, optimizer = train_init(args)
    # net.snapshot(args.dataset, 4, args.n_bits)
    # state_dict = torch.load('checkpoint/aid_model_4_16.t', map_location=args.device)['model_state_dict']
    # net.load_state_dict(state_dict, strict=False)
    # state_dict = torch.load('checkpoint/aid_model_ft_20.t', map_location=args.device)['model_state_dict']
    # net.load_state_dict(state_dict, strict=False)
    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):

        # train_epoch(args, train_loader, net, criterion,criterion_cls, optimizer, epoch)
        net=train_epoch(args, train_loader, net, criterion, criterion_cls, optimizer, epoch)
        # 11111111111111111111111111111111111
        train_hash, train_labels = get_dataset_features(net, train_loader, args.device)
        # 2. 获取完整测试集特征和标签（合并 query 和 dbase）
        query_hash, query_labels = get_dataset_features(net, query_loader, args.device)
        dbase_hash, dbase_labels = get_dataset_features(net, dbase_loader, args.device)
        test_hash = np.concatenate([query_hash, dbase_hash], axis=0)
        test_labels = np.concatenate([query_labels, dbase_labels], axis=0)
        # 3. 保存 .mat 文件（训练集作为检索库，测试集作为查询集）
        # Save_mat(
        #     epoch=epoch,
        #     output_dim=args.n_bits,
        #     datasets=args.dataset,
        #     query_labels=test_labels,  # 查询集：所有测试样本
        #     retrieval_labels=train_labels,  # 检索库：所有训练样本
        #     query_img=test_hash,  # 查询集特征
        #     retrieval_img=train_hash,  # 检索库特征
        #     save_dir=args.save_dir,  # 保存路径
        #     mode_name='ours',  # 模式名称
        #     mAP=early_stopping.best_map
        # )
        # 11111111111111111111111111111111111
        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:

            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                parallel_val=args.parallel_val,
            )
            if early_stop:
                break



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
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args,build_loader)
        # train_loader, query_loader, dbase_loader = prepare_loaders(args)

        for hash_bit in [128]:
            # hash_bit=args.n_bits
            print(f"processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pkl") for x in os.listdir(args.save_dir)):
                print(f"*.pkl exists in {args.save_dir}, will pass")
                continue

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
