from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from vehiclereid.data_manager import ImageDataManager
from vehiclereid import models
from vehiclereid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from vehiclereid.utils.iotools import check_isfile
from vehiclereid.utils.avgmeter import AverageMeter
from vehiclereid.utils.loggers import Logger, RankLogger
from vehiclereid.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from vehiclereid.utils.visualtools import visualize_ranked_results
from vehiclereid.utils.generaltools import set_random_seed
from vehiclereid.eval_metrics import evaluate
from vehiclereid.optimizers import init_optimizer
from vehiclereid.lr_schedulers import init_lr_scheduler
import xml.etree.ElementTree as ET

# global variables
parser = argument_parser()
args = parser.parse_args()


def compute_distance(query_features, test_features, as_numpy=True):
    m, n = query_features.size(0), test_features.size(0)
    distmat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(test_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, query_features, test_features.t())
    if as_numpy:
        distmat = distmat.cpu().numpy()
    return distmat


def compute_ortho_loss(features):
    l = len(features)
    loss = torch.zeros([]).float()
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            loss = loss + torch.mean(features[i] * features[j], dim=1, keepdim=True)
    return torch.abs(loss.sum())


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu, vertical=args.vertical,
                              attn=args.attn, multi=args.multi, num_prototypes=args.num_prototypes, channel=args.channel,
                              discard=args.discard, num_dim=args.num_dim, has_global=args.has_global, stride3=args.stride3)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    # if args.ssl_weights and len(args.load_weights) == 0:
    #     assert check_isfile(args.ssl_weights)
    #     load_pretrained_weights(model, args.ssl_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_color = CrossEntropyLoss(num_classes=12, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_orientation = CrossEntropyLoss(num_classes=6, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_type = CrossEntropyLoss(num_classes=11, use_gpu=use_gpu, label_smooth=args.label_smooth)

    criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)
        scheduler.last_epoch = args.start_epoch

    if args.evaluate:
        print('Evaluate only')

        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            distmat = test(model, queryloader, galleryloader, use_gpu, name, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print('=> Start training')
    '''
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
    '''
    for epoch in range(args.start_epoch, args.max_epoch):
        # print(epoch)
        # print(args.start_eval)
        # print(args.eval_freq)
        args.eval_freq = 1
        train(epoch, model, [criterion_xent, criterion_color, criterion_orientation, criterion_type], criterion_htri, optimizer, trainloader, use_gpu)

        scheduler.step()
        
        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            print('=> Test')

            ##### skip eval
            for name in args.target_names:
                print('Evaluating {} ...'.format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1 = test(model, queryloader, galleryloader, use_gpu, name)
                ranklogger.write(name, epoch + 1, rank1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'arch': args.arch,
            'optimizer': optimizer.state_dict(),
        }, args.save_dir)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    ranklogger.show_summary()


def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    ent_losses = AverageMeter()
    div_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    criterion_xent, criterion_color, criterion_orientation, criterion_type = criterion_xent

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, camid, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        camid = camid[1:]
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs = model(imgs)

        is_attn = len(outputs) == 4

        if is_attn:
            outputs, features, g_features, prod = outputs
            total_iterator = features + g_features
            total_length = len(total_iterator)

            xent_loss = sum([criterion_xent(o, pids) for o in outputs])
            htri_loss = sum([criterion_htri(f, pids)[0] for f in total_iterator])
            loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss + prod#+ ortho * args.lambda_orth# + spatial
        elif isinstance(outputs[0], list):
            outputs, features = outputs

            xent_loss = sum([criterion_xent(o, pids) for o in outputs])
            htri_loss = sum([criterion_htri(f, pids)[0] for f in features])
            loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        else:
            outputs, features = outputs

            xent_loss = criterion_xent(outputs, pids)
            htri_loss, pre = criterion_htri(features, pids)
            loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss# + ortho

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        if is_attn:
            htri_losses.update(htri_loss.item(), pids.size(0))
            xent_losses.update(xent_loss.item(), pids.size(0))
            ent_losses.update(prod.item(), pids.size(0))
            # div_losses.update(diversity.item(), pids.size(0))
        else:
            htri_losses.update(htri_loss.item(), pids.size(0))
            xent_losses.update(xent_loss.item(), pids.size(0))
        
        accs.update(accuracy(outputs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR {lr}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                  'Orth {ortho.val:.4f} ({ortho.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                lr=optimizer.param_groups[0]['lr'],
                batch_time=batch_time,
                data_time=data_time,
                xent=xent_losses,
                htri=htri_losses,
                ortho=ent_losses,
                acc=accs
            ))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, target_name, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, q_areas = [], [], [], []
        for batch_idx, (imgs, pids, camids, paths) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features, area = model(imgs)

            batch_time.update(time.time() - end)
            
            qf.append(features)
            q_areas.append(area)
            q_pids.extend(pids)
            q_camids.extend(camids)

        if torch.is_tensor(area[0]):
            q_areas = [torch.cat(f, 0).cpu() for f in zip(*q_areas)]
        qf = [torch.cat(f, 0).cpu() for f in zip(*qf)]
        # qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        
        s = qf[0].size(0) if isinstance(qf, list) else qf.size(0)
        ss = qf[0].size(1) if isinstance(qf, list) else qf.size(1)
        print('Extracted features for query set, obtained {}-by-{} matrix'.format(s, ss))

        gf, g_pids, g_camids, g_areas = [], [], [], []
        for batch_idx, (imgs, pids, camids, files) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features, area = model(imgs)
            batch_time.update(time.time() - end)
            # [print(a.shape) for a in area]
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            g_areas.append(area)

        gf = [torch.cat(f, 0).cpu() for f in zip(*gf)]
        if torch.is_tensor(area[0]):
            g_areas = [torch.cat(f, 0).cpu() for f in zip(*g_areas)]
        # gf = torch.cat(gf, 0)
        test_image_id = np.array([str(g.item()) for g in g_pids])
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        s = gf[0].size(0) if isinstance(gf, list) else gf.size(0)
        ss = gf[0].size(1) if isinstance(gf, list) else gf.size(1)
        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(s, ss))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))
    
    # cat = [torch.cat([q, g], dim=0) for q, g in zip(qf, gf)]
    # mean = [c.mean(dim=0, keepdim=True) for c in cat]
    # stdev = [(c.var(dim=0, keepdim=True) + 1e-05).sqrt() for c in cat]
    # qf = [(q - m) / s for q, m, s in zip(qf, mean, stdev)]
    # gf = [(g - m) / s for g, m, s in zip(gf, mean, stdev)]
    distmats = torch.stack([compute_distance(q, g, False) for q, g in zip(qf, gf)], dim=-1)
    # del qf, gf
    if torch.is_tensor(area[0]):
        areas_mat = torch.stack([q.unsqueeze(1) * g.unsqueeze(0) for q, g in zip(q_areas, g_areas)], dim=-1)
        areas_mat = areas_mat
        # distmats = distmats.cpu()
        areas_mat = F.normalize(areas_mat, dim=-1, p=1)
        distmats = distmats * areas_mat
    distmat = distmats.sum(dim=-1).cpu().numpy()

    distmat_argsort = distmat.argsort(axis=1)[:, :100]
    if not os.path.exists(os.path.join(args.save_dir, 'viz')):
        os.mkdir(os.path.join(args.save_dir, 'viz'))
    f = open(os.path.join(args.save_dir, 'track2.txt'), 'w')
    for index, (dm, distance) in enumerate(zip(distmat, distmat_argsort)):
        top_n = test_image_id[distance.tolist()]
        f.write(' '.join(top_n) + '\n')
        with open(os.path.join(args.save_dir, 'viz', str(index+1).zfill(6) + '.txt'), 'w') as vizf:
            for t in top_n[:50]:
                vizf.write(t + '\n')
    f.close()
    if target_name != 'aicity':
        print('Computing CMC and mAP')
        # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, dataset=args.target_names[0])

        print('Results ----------')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        print('------------------')

        if return_distmat:
            return distmat
        return cmc[0]


if __name__ == '__main__':
    main()
