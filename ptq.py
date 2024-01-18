# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------
import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from train.config import get_config
from data import build_loader
from train.lr_scheduler import build_scheduler
from train.logger import create_logger
from utils import load_checkpoint, load_weights, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, update_model_ema, unwrap_model
import copy
from train.optimizer import build_optimizer
from repvggplus import create_RepVGGplus_by_name
from hmquant.qat_torch.apis import trace_model_for_qat, set_calib, set_fake_quant
from repvgg import RepVGGBlock
from hmquant.observers.utils import pure_diff
from rich.table import Table
from rich.console import Console

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--arch', default=None, type=str, help='arch name')
    parser.add_argument('--batch-size', default=128, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='/your/path/to/dataset', type=str, help='path to dataset')
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],  #TODO Note: use amp if you have it
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='/your/path/to/save/dir', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.ARCH}")

    fp_model = create_RepVGGplus_by_name(config.MODEL.ARCH, deploy=False, use_checkpoint=args.use_checkpoint)
    

    logger.info(str(fp_model))
    fp_model.cuda()
    
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')
    
    if config.TRAIN.EMA_ALPHA > 0 and (not config.EVAL_MODE) and (not config.THROUGHPUT_MODE):
        fp_model_ema = copy.deepcopy(fp_model)
    else:
        fp_model_ema = None
    if (not config.THROUGHPUT_MODE) and config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, fp_model, logger)

        
    # QAT
    quant_cfg = dict(
        global_wise_cfg=dict(
            o_cfg=dict(calib_metric="minmax", dtype="int8", use_grad_scale=True, symmetric=False), 
            # o_cfg=dict(calib_metric="percent-0.99999", dtype="int8"), 
            # o_cfg=dict(calib_metric="KL", dtype="int8", use_grad_scale=True), 
            # freeze_bn=False,
            freeze_bn=True,
            w_cfg=dict(calib_metric="minmax", dtype="int8", use_grad_scale=True)
            # w_cfg=dict(calib_metric="minmax", dtype="int16", use_grad_scale=True)
            # w_cfg=dict(dtype="int8")
        )
    )
    
    model = trace_model_for_qat(copy.deepcopy(fp_model.train()), quant_cfg, domain="xh2")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module=model)
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    if torch.cuda.device_count() > 1:
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK],
                                                          broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    max_accuracy = 0.0

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        calib_one_epoch(model=model, data_loader=data_loader_train)
        print("start val")
        if dist.get_rank() == 0:
            if epoch % config.SAVE_FREQ == 0:
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, model_ema=fp_model_ema)

        if epoch % config.SAVE_FREQ == 0:

            if data_loader_val is not None:
                acc1, acc5, loss = validate(config, data_loader_val, model, fp_model)
                logger.info(f"Accuracy of the network at epoch {epoch}: {acc1:.3f}%")
                max_accuracy = max(max_accuracy, acc1)
                logger.info(f'Max accuracy: {max_accuracy:.2f}%')
                if max_accuracy == acc1 and dist.get_rank() == 0:
                    save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger,
                                    is_best=True, model_ema=fp_model_ema)
                    
        break
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

@torch.no_grad()
def calib_one_epoch(model, data_loader):
    set_calib(model=model)
    print("calibrating")
    model.eval()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        outputs = model(samples)
        break
    set_fake_quant(model=model)
    model.train()
    torch.cuda.synchronize()

@torch.no_grad()
def validate(config, data_loader, model, fp_model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    fp_model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    quant_model_dict = dict(model.module.named_modules())
    val_mean_ab_tensor, val_mean_re_tensor, val_related_mse, val_cos = {}, {}, {}, {}
    count = 0.0
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        
        previous_key = "input"
        fp_block_ouputs = {"input": images}
        quant_block_ouputs = {"input": images}
        for key, fp_module in fp_model.named_modules():
            if isinstance(fp_module, RepVGGBlock):
                # print(key)
                fp_block_ouputs[key] = fp_module(fp_block_ouputs[previous_key])
                quant_block_ouputs[key] = quant_model_dict[key](quant_block_ouputs[previous_key])
                del fp_block_ouputs[previous_key]
                del quant_block_ouputs[previous_key]
                mean_ab_tensor, mean_re_tensor, related_mse, cos = pure_diff(raw_tensor=fp_block_ouputs[key], quanted_tensor=quant_block_ouputs[key])
                previous_key = key
                val_mean_ab_tensor[key] = val_mean_ab_tensor.get(key, 0.0) + mean_ab_tensor
                val_mean_re_tensor[key] = val_mean_re_tensor.get(key, 0.0) + mean_re_tensor
                val_related_mse[key] = val_related_mse.get(key, 0.0) + related_mse
                val_cos[key] = val_cos.get(key, 0.0) + cos
    
        #   =============================== deepsup part
        if type(output) is dict:
            output = output['main']

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
        
        count += 1.0
        
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    
    activation_error_record_ops_dict = {}
    for k in val_mean_ab_tensor.keys():
        val_mean_ab_tensor[k] /= count
        val_mean_re_tensor[k] /= count
        val_related_mse[k] /= count
        val_cos[k] /= count
        activation_error_record_ops_dict[k] = {}
        activation_error_record_ops_dict[k]["w/a"] = "activation"
        activation_error_record_ops_dict[k]["mean L1 Error"] = str(val_mean_ab_tensor[k].item())
        activation_error_record_ops_dict[k]["mean related L1 Error"] = str(val_mean_re_tensor[k].item())
        activation_error_record_ops_dict[k]["related mse"] = str(val_related_mse[k].item())
        activation_error_record_ops_dict[k]["cosine dist"] = str(val_cos[k].item())
    
    table = Table(title="Sequencer Analyse Report -- activation")
    table.add_column("Module")
    table_data = []
    for node_key, node_metric_dict in activation_error_record_ops_dict.items():
        node_key_list = [node_key]
        for metric_value in node_metric_dict.values():
            node_key_list.append(metric_value)
        table_data.append(node_key_list)
    for metric_key in activation_error_record_ops_dict[next(iter(activation_error_record_ops_dict))].keys():
        table.add_column(metric_key)
    for row in table_data:
        table.add_row(*row)
        
    console = Console(record=True)
    console.print(table)
    
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg




import os

if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if not config.EVAL_MODE:
        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 256.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    print('==========================================')
    print('real base lr: ', config.TRAIN.BASE_LR)
    print('==========================================')

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0 if torch.cuda.device_count() == 1 else dist.get_rank(), name=f"{config.MODEL.ARCH}")

    if torch.cuda.device_count() == 1 or dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
