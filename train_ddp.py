import torch
import torch.optim as optim
import time, sys, os, random, glob
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
import torch.distributed as dist

import util.utils as utils

def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ##### adjust learning rate
        utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

        ##### prepare input and forward
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)

        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if cfg.local_rank == 0:
            sys.stdout.write(
                "epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
                (epoch, cfg.epochs, i + 1, len(train_loader), am_dict['loss'].val, am_dict['loss'].avg,
                 data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
            if (i == len(train_loader) - 1): print()

    # f = utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5] + '_%d'%os.getpid(), epoch, cfg.save_freq)
    if cfg.local_rank == 0:
        logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

        f = utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq)
        logger.info('Saving {}'.format(f))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k+'_train', am_dict[k].avg, epoch)


def eval_epoch(val_loader, model, model_fn, epoch):
    if cfg.local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            ##### prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)

            ##### merge(allreduce) multi-gpu
            if cfg.dist:
                for k, v in visual_dict.items():
                    count = meter_dict[k][1]
                    # print("[PID {}] Before allreduce: key {} value {} count {}".format(os.getpid(), k, float(v), count))

                    v = v * count
                    count = loss.new_tensor([int(count)], dtype=torch.long)
                    dist.all_reduce(v), dist.all_reduce(count)
                    count = count.item()
                    v = v / count
                    # print("[PID {}] After allreduce: key {} value {} count {}".format(os.getpid(), k, float(v), count))

                    visual_dict[k] = v
                    meter_dict[k] = (float(v), count)

            ##### meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])

            ##### print
            if cfg.local_rank == 0:
                sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))
                if (i == len(val_loader) - 1): print()

        if cfg.local_rank == 0:
            logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)


def main(gpu, cfgs):
    ##### config
    global cfg
    cfg = cfgs
    cfg.local_rank = gpu

    ##### logger & summary writer
    if cfg.local_rank == 0:
        # logger
        global logger
        from util.log import get_logger
        logger = get_logger(cfg)

        # summary writer
        global writer
        writer = SummaryWriter(cfg.exp_path)

    ##### distributed training setting
    if cfg.dist:
        cfg.rank = cfg.node_rank * cfg.ngpu_per_node + gpu
        print('[PID {}] rank: {}  world_size: {}'.format(os.getpid(), cfg.rank, cfg.world_size))
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % cfg.tcp_port, world_size=cfg.world_size, rank=cfg.rank)

        torch.cuda.set_device(gpu)

        assert cfg.batch_size % cfg.world_size == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (cfg.batch_size, cfg.world_size)
        cfg.batch_size = cfg.batch_size // cfg.world_size

    if cfg.local_rank == 0:
        logger.info(cfg)

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')

    if model_name == 'pointgroup':
        from model.pointgroup.pointgroup import PointGroup as Network
        from model.pointgroup.pointgroup import model_fn_decorator
    else:
        print("Error: no model - " + model_name)
        exit(0)

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    if cfg.local_rank == 0:
        logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda

    model = model.to(gpu)
    if cfg.dist:
        if cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    if cfg.local_rank == 0:
        # logger.info(model)
        logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(cfg)

    ##### dataset
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            import data.scannetv2_inst
            dataset = data.scannetv2_inst.Dataset(cfg)
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
        dataset.trainLoader()
        dataset.valLoader()
        if cfg.local_rank == 0:
            logger.info('Training samples: {}'.format(len(dataset.train_file_names)))
            logger.info('Validation samples: {}'.format(len(dataset.val_file_names)))

    # f = utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5] + '_%d' % os.getpid(), 0, cfg.save_freq)

    ##### resume
    start_epoch, f = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], dist=cfg.dist, f=cfg.pretrain, gpu=gpu)  # resume from the latest epoch, or specify the epoch to restore
    if cfg.local_rank == 0:
        logger.info('Restore from {}'.format(f) if len(f) > 0 else 'Start from epoch {}'.format(start_epoch))

    ##### train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.dist:
            dataset.train_sampler.set_epoch(epoch)
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)

        if cfg.validation:
            if utils.is_multiple(epoch, cfg.save_freq) or utils.is_power2(epoch):
                if cfg.dist:
                    dataset.val_sampler.set_epoch(epoch)
                eval_epoch(dataset.val_data_loader, model, model_fn, epoch)


if __name__ == '__main__':
    ##### config
    from util.config import get_parser
    cfg = get_parser()
    print('[PID {}] {}'.format(os.getpid(), cfg))

    ##### backup
    print('[PID {}] Copying backup files ...'.format(os.getpid()))
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train_ddp.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    ##### shared memory
    if cfg.cache:
        if cfg.dataset == 'scannetv2':
            train_file_names = sorted(glob.glob(os.path.join(cfg.data_root, cfg.dataset, 'train', '*' + cfg.filename_suffix)))
            val_file_names = sorted(glob.glob(os.path.join(cfg.data_root, cfg.dataset, 'val', '*' + cfg.filename_suffix)))
            utils.create_shared_memory(train_file_names, wlabel=True)
            utils.create_shared_memory(val_file_names, wlabel=True)

    ##### main
    cfg.world_size = cfg.nodes * cfg.ngpu_per_node
    cfg.dist = True if cfg.world_size > 1 else False
    if cfg.dist:
        mp.spawn(main, nprocs=cfg.ngpu_per_node, args=(cfg,))
    else:
        main(0, cfg)

    ##### delete shared momery
    # if cfg.cache:
    #     if cfg.dataset == 'scannetv2':
    #         utils.delete_shared_memory(train_file_names, wlabel=True)
    #         utils.delete_shared_memory(val_file_names, wlabel=True)