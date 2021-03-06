import gc
import os
import glob
import math

from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from decoder_planercnn.modules import *
from decoder_planercnn.utils import *
from decoder_planercnn.visualize_utils import *
from decoder_planercnn.evaluate_utils import *
from decoder_planercnn.config import InferenceConfig
from decoder_planercnn.model import *

from decoder_yolo import test
from decoder_yolo.utils.utils import *
from decoder_yolo.utils import torch_utils

from model.model import *
from data.dataset import *
from loss.ssim import SSIM

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    # print('Apex recommended for mixed precision and faster training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

import warnings
warnings.filterwarnings("ignore")

# Hyperparameters https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])

def train(plane_args, yolo_args, midas_args, 
            plane_weightage, yolo_weightage, midas_weightage,
            resume_train=False, model_path=''):

    options = plane_args
    config = InferenceConfig(options)

    ## Start yolo train setup 
    opt = yolo_args

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)

    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)

    # Image Sizes
    gs = 64  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = grid_min * gs, grid_max * gs
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob('*_batch*.png') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Model(yolo_cfg=cfg, midas_cfg=None, planercnn_cfg=config, path=midas_args.weights).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    start_epoch = 0
    best_loss = 1000

    if resume_train:
        last_model = torch.load(model_path + 'model_last.pt', map_location=device)
        model.load_state_dict(last_model['state_dict'])
        optimizer.load_state_dict(last_model['optimizer'])
        best_loss = last_model['best_loss']
        start_epoch = last_model['epoch'] + 1
        del last_model
    else:
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
            chkpt = torch.load(weights, map_location=device)

            # load model
            try:
                #chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
                model.load_state_dict(chkpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                    "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
                raise KeyError(s) from e

            # load optimizer
            if chkpt['optimizer'] is not None:
                print('loading Optimizer')
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            start_epoch = chkpt['epoch'] + 1
            del chkpt

        elif len(weights) > 0:  # darknet format
            # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
            load_darknet_weights(model, weights)

        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'),strict=False)
        midas_parameters = torch.load(midas_args.weights)

        if "optimizer" in midas_parameters:
            midas_parameters = midas_parameters["model"]

        model.load_state_dict(midas_parameters,strict=False)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    lf = lambda x: (((1 + math.cos(
        x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [round(epochs * x) for x in [0.8, 0.9]], 0.1, start_epoch - 1)

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    yolo_params = dict(path = train_path, img_size = img_size, batch_size = batch_size, augment=True,hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)

    planercnn_params = dict(options=options, config=config, random=False)
    midas_params=None

    # Dataset
    train_dataset = create_data(yolo_params, planercnn_params, midas_params)

    # Dataloader
    batch_size = min(batch_size, len(train_dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    trainloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=nw,
                            shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=train_dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(trainloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    loss_list=[]

    for epoch in range(start_epoch, start_epoch+epochs):
        
        model.train()
        print(('\n' + '%10s' * 6) % ('Epoch', 'Dp_loss', 'bbx_loss', 'pln_loss', 'All_loss', 'img_size'))
        pbar = tqdm(enumerate(trainloader))
        mloss = torch.zeros(4).to(device)  # mean losses

        for i, (plane_data, yolo_data, depth_data) in pbar:

            optimizer.zero_grad()
            imgs, targets, _, _ = yolo_data

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn * 2:
                model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou yolo_loss ratio (obj_loss = 1.0 or giou)
                if ni == n_burn:  # burnin complete
                    print_model_biases(model)

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

            # Multi-Scale training
            if opt.multi_scale:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            yolo_ip = imgs

            depth_img_size, depth_img, depth_target = depth_data #######
            depth_sample = torch.from_numpy(depth_img).to(device).unsqueeze(0) ######

            midas_ip = depth_sample ####

            data_pair, plane_img, plane_np = plane_data
            sample = data_pair

            plane_losses = []            

            input_pair = []
            detection_pair = []

            camera = sample[30][0].cuda()

            #for indexOffset in [0, ]:
            indexOffset=0
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()
            
            masks = (gt_segmentation == torch.arange(gt_segmentation.max() + 1).cuda().view(-1, 1, 1)).float()
            input_pair.append({'image': images, 'depth': gt_depth, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera, 'plane': planes[0], 'masks': masks, 'mask': gt_masks})
            
            plane_ip = dict(input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='inference_detection', use_nms=2, use_refinement=True, return_feature_map=False)

            yolo_op, midas_op, plane_op = model.forward(yolo_ip, midas_ip, plane_ip)

            pred = yolo_op
            depth_prediction = midas_op           

            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = plane_op
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_parameter_loss = compute_losses(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters)

            plane_losses =[rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_parameter_loss]

            if depth_np_pred.shape != gt_depth.shape:
                depth_np_pred = torch.nn.functional.interpolate(depth_np_pred.unsqueeze(1), size=(512, 512), mode='bilinear',align_corners=False).squeeze(1)
                pass

            if config.PREDICT_NORMAL_NP:
                normal_np_pred = depth_np_pred[0, 1:]                    
                depth_np_pred = depth_np_pred[:, 0]
                gt_normal = gt_depth[0, 1:]                    
                gt_depth = gt_depth[:, 0]
                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                normal_np_loss = l2LossMask(normal_np_pred[:, 80:560], gt_normal[:, 80:560], (torch.norm(gt_normal[:, 80:560], dim=0) > 1e-4).float())
                plane_losses.append(depth_np_loss)
                plane_losses.append(normal_np_loss)
            else:
                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                plane_losses.append(depth_np_loss)
                normal_np_pred = None
                pass

            if len(detections) > 0:
                detections, detection_masks = unmoldDetections(config, camera, detections, detection_masks, depth_np_pred, normal_np_pred, debug=False)
                if 'refine_only' in options.suffix:
                    detections, detection_masks = detections.detach(), detection_masks.detach()
                    pass
                XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
                detection_mask = detection_mask.unsqueeze(0)                        
            else:
                XYZ_pred = torch.zeros((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                detection_mask = torch.zeros((1, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                plane_XYZ = torch.zeros((1, 3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda() 
                detections = torch.zeros((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                detection_masks = torch.zeros((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()                        
                pass

            detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'plane_XYZ': plane_XYZ, 'depth_np': depth_np_pred})
            loss_fn = nn.MSELoss()

            try:
                plane_parameters = torch.from_numpy(plane_np['plane_parameters']).cuda()
                plane_masks = torch.from_numpy(plane_np['plane_masks']).cuda()
                plane_parameters_pred = detection_pair[0]['detection'][:, 6:9]
                plane_masks_pred = detection_pair[0]['masks'][:, 80:560]

                if plane_parameters_pred.shape != plane_parameters.shape:
                    plane_parameters_pred = torch.nn.functional.interpolate(plane_parameters_pred.unsqueeze(1).unsqueeze(0), size=plane_parameters.shape, mode='bilinear',align_corners=True).squeeze()
                    pass
                if plane_masks_pred.shape != plane_masks.shape:
                    plane_masks_pred = torch.nn.functional.interpolate(plane_masks_pred.unsqueeze(1).unsqueeze(0), size=plane_masks.shape, mode='trilinear',align_corners=True).squeeze()
                    pass
                plane_params_loss = loss_fn(plane_parameters_pred,plane_parameters) + loss_fn(plane_masks_pred,plane_masks)
            except:
                plane_params_loss = 1

            predicted_detection = visualizeBatchPair(options, config, input_pair, detection_pair, indexOffset=i)
            predicted_detection = torch.from_numpy(predicted_detection)

            if predicted_detection.shape != plane_img.shape:
                predicted_detection = torch.nn.functional.interpolate(predicted_detection.permute(2,0,1).unsqueeze(0).unsqueeze(1), size=plane_img.permute(2,0,1).shape).squeeze()
                pass

            plane_img = plane_img.permute(2,0,1)
            
            #https://github.com/Po-Hsun-Su/pytorch-ssim
            #https://github.com/jorge-pessoa/pytorch-msssim
            pln_ssim = torch.clamp(1 - SSIM(predicted_detection.unsqueeze(0).type(torch.cuda.FloatTensor), plane_img.unsqueeze(0).type(torch.cuda.FloatTensor)), min=0, max=1)
            
            plane_loss = sum(plane_losses) + pln_ssim
            plane_losses = [l.data.item() for l in plane_losses] #train_planercnn.py 331
            depth_prediction = (torch.nn.functional.interpolate(
                            depth_prediction.unsqueeze(1),
                            size=tuple(depth_img_size[:2]),
                            mode="bicubic",
                            align_corners=False))
            bits=2

            depth_min = depth_prediction.min()
            depth_max = depth_prediction.max()

            max_val = (2**(8*bits))-1

            if depth_max - depth_min > np.finfo("float").eps:
                depth_out = max_val * (depth_prediction - depth_min) / (depth_max - depth_min)
            else:
                depth_out = 0
            
            depth_target = torch.from_numpy(np.asarray(depth_target)).to(device).type(torch.cuda.FloatTensor).unsqueeze(0)
            depth_target = (torch.nn.functional.interpolate(
                                depth_target.unsqueeze(1),
                                size=depth_img_size[:2],
                                mode="bicubic",
                                align_corners=False
                            ))

            depth_pred = Variable( depth_out,  requires_grad=True)
            depth_target = Variable( depth_target, requires_grad = False)

            ssim_out = torch.clamp(1- SSIM(depth_pred,depth_target),min=0,max=1)
            
            RMSE_loss = torch.sqrt(loss_fn(depth_pred, depth_target))
            depth_loss = (0.0001*RMSE_loss) + ssim_out

            yolo_loss, yolo_loss_items = compute_loss(pred, targets, model)

            if not torch.isfinite(yolo_loss):
                print('WARNING: non-finite yolo_loss, ending training ', yolo_loss_items)
                return results

            total_loss = (plane_weightage * plane_loss) + (yolo_weightage * yolo_loss) + (midas_weightage * depth_loss)

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()
                pass

            optimizer.step()
            gc.collect()
            ema.update(model)

            # Print batch results
            mloss = (mloss * i + yolo_loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

            s = ('%10s'+ '%10.3g' * 5) % ('%g/%g' % (epoch, (start_epoch+epochs) - 1), depth_loss.item(), yolo_loss.item(), plane_loss.item(), total_loss.item(), img_size)
            pbar.set_description(s)

        scheduler.step()

        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs

        model_chkpt = {'best_loss':best_loss,
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }

        torch.save(model_chkpt, model_path + 'model_last.pt')

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model_chkpt, model_path + 'model_best.pt')

        loss_list.append(total_loss.item())

    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return loss_list