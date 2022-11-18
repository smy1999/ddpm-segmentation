import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc
import numpy as np
import logging

from torch.utils.data import DataLoader

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions, \
    pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset3, make_transform
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev


def evaluation(args, models):
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageLabelDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'],
                            generator=rnd_gen, device=dev())
    else:
        noise = None

    preds, gts, uncertainty_scores = [], [], []
    for img, label in tqdm(dataset):
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )
        gts.append(label.numpy())
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item())
    # print(len(preds))
    # print(preds[0].shape)
    save_predictions(args, dataset.image_paths, preds)
    miou = compute_iou(args, preds, gts)
    logging.info(f'Overall mIoU: ' +str(miou))
    # print(f'Overall mIoU: ', miou)
    logging.info(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')
    # print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置打印级别
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s \n%(message)s')
    # 设置屏幕打印的格式
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 设置log保存
    fh = logging.FileHandler("test.log", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    # Prepare the experiment folder
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    # print('Experiment folder: %s' % (path))
    logging.info('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models in ensemble are trained
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth'))
                  for i in range(opts['model_num'])]

    opts['start_model_num'] = sum(pretrained)

    logging.info('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)
