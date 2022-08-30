import os
import time, datetime
from pathlib import Path
import argparse
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import utils, presets
from coco_utils import get_coco
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler


def get_dataset(name, image_set, transform, data_path, is_filtered=None):
    if name == "coco":
        return get_coco(root=data_path, image_set=image_set, transforms=transform, is_filtered=is_filtered)


def get_transform(train, data_augmentation):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=data_augmentation)
    else:
        return presets.DetectionPresetEval()


def main(args):
    output_dir = "output/"
    data_path = "COCO"
    dataset_name = "coco"
    data_augmentation = "hflip"
    resume_train_model = output_dir + "model_19.pth"
    filtered = True
    print_freq = 1000
    start_epoch = 0
    lr_gamma = 0.1
    lr_steps = [10, 15, 20]
    lr_scheduler = "multisteplr"
    weights = None
    weights_backbone = "ResNet50_Weights.IMAGENET1K_V1"
    weight_decay = 1e-4
    momentum = 0.9
    learning_rate = 0.02
    workers = 8
    aspect_ratio_group_factor = 3
    batch_size = 5
    trainable_backbone_layers = None
    rpn_score_thresh = None
    model_name = "fasterrcnn_resnet50_fpn"
    num_classes = 90 + 1
    mixed_precision_training = True

    if output_dir:
        utils.mkdir(output_dir)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    # dataset = get_dataset(dataset_name, "train", get_transform(True, data_augmentation), data_path, is_filtered=filtered)
    # train_sampler = torch.utils.data.RandomSampler(dataset)
    #
    # if aspect_ratio_group_factor >= 0:
    #     group_ids = create_aspect_ratio_groups(dataset, k=aspect_ratio_group_factor)
    #     train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    # else:
    #     train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    #
    # train_collate_fn = utils.collate_fn
    #
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_sampler=train_batch_sampler, num_workers=workers, pin_memory=True, collate_fn=train_collate_fn
    # )

    kwargs = {"trainable_backbone_layers": trainable_backbone_layers}
    if data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in model_name:
        if rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = rpn_score_thresh

    model = torchvision.models.detection.__dict__[model_name](
        weights=weights,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        **kwargs
    )
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    lr_scheduler = lr_scheduler.lower()
    if lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)
    elif lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if Path(resume_train_model).exists():
        checkpoint = torch.load(resume_train_model, )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        start_epoch = checkpoint["epoch"] + 1
        args.epochs = args.epochs + start_epoch
        if mixed_precision_training:
            scaler.load_state_dict(checkpoint["scaler"])

    print(f"Start training from {start_epoch} to {args.epochs}")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):

        # train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler)
        # lr_scheduler.step()

        # if output_dir:
        #     checkpoint = {
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "lr_scheduler": lr_scheduler.state_dict(),
        #         "args": args,
        #         "epoch": epoch,
        #     }
        #     if mixed_precision_training:
        #         checkpoint["scaler"] = scaler.state_dict()
        #
        #     utils.save_on_master(checkpoint, os.path.join(output_dir, f"model_{epoch+1}.pth"))
        #     utils.save_on_master(checkpoint, os.path.join(output_dir, "checkpoint.pth"))

        if args.test:
            # Test
            dataset_test = get_dataset(dataset_name, "val", get_transform(False, data_augmentation), data_path, is_filtered=filtered)

            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=workers, pin_memory=True, collate_fn=utils.collate_fn
            )
            evaluate(model, data_loader_test, device=device)
            del dataset_test, test_sampler, data_loader_test

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    parser.add_argument("--epochs","--e" , default=1, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--test", "--t", action="store_true", help="evaluate model on test set")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args)
    main(args)
