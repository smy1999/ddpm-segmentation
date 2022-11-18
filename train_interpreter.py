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


def prepare_data(args):
    feature_extractor = create_feature_extractor(**args)

    logging.info(f"Preparing the train set for {args['category']}...")
    # print(f"Preparing the train set for {args['category']}...")
    dataset = ImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        num_images=args['training_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )
    # X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)
    # y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'],
                            generator=rnd_gen, device=dev())
    else:
        noise = None

    d = 0
    file_len = []
    for row, (img, label) in enumerate(tqdm(dataset)):
        logging.info("Processing Image" + str(row))
        # print("Processing Image", row)
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        X = collect_features(args, features).cpu()

        for target in range(args['number_class']):
            if target == args['ignore_label']: continue
            if 0 < (label == target).sum() < 20:
                logging.info(f'Delete small annotation from image {dataset.image_paths[row]} | label {target}')
                # print(f'Delete small annotation from image {dataset.image_paths[row]} | label {target}')
                label[label == target] = args['ignore_label']
        y = label
        d = X.shape[0]
        X = X.reshape(d, -1).permute(1, 0)
        y = y.flatten()
        X = X[y != args['ignore_label']]
        y = y[y != args['ignore_label']]
        file_len.append(X.shape[0])

        x_mem = np.memmap('processing/features_' + str(row) + '.npy', dtype='float32', mode='w+', shape=X.shape)
        y_mem = np.memmap('processing/labels_' + str(row) + '.npy', dtype='uint8', mode='w+', shape=y.shape)
        x_mem[:] = X[:]
        y_mem[:] = y[:]
        logging.info('Saving features and labels' + str(row))
        # print('Saving features and labels', row)

    logging.info(f'Total dimension {d}')
    # print(f'Total dimension {d}')
    return file_len, d
    # return X[y != args['ignore_label']], y[y != args['ignore_label']]


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

    save_predictions(args, dataset.image_paths, preds)
    miou = compute_iou(args, preds, gts)
    logging.info(f'Overall mIoU: ' +str(miou))
    # print(f'Overall mIoU: ', miou)
    logging.info(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')
    # print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
    # file_len, dim = prepare_data(args)
    # print(file_len)
    file_len = [65536, 65536, 65536, 65536, 65536, 65536, 65528, 65536, 65536, 65536, 65521, 65536, 65536, 65536,
                65536, 65536, 65536, 65536, 65536, 65536, 65536, 65522, 65536, 65536, 65517, 65518, 65536, 65536, 65525,
                65520]
    dim = 8448
    memory_len = 20
    train_data = FeatureDataset3(file_len, dim, memory_len)
    data_len = 0
    for l in file_len:
        data_len += l

    resume = 'pixel_classifiers/cat_15/ddpm/50_150_250_5_6_7_8_12/model_9_13000_0_10000000_0.pth'

    logging.info(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    # print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    logging.info(f" *********************** Current number data {data_len} ***********************")
    # print(f" *********************** Current number data {data_len} ***********************")

    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    logging.info(" *********************** Current dataloader length " + str(len(train_loader)) + " ***********************")
    # print(" *********************** Current dataloader length " + str(len(train_loader)) + " ***********************")
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):

        gc.collect()
        classifier = pixel_classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])

        if MODEL_NUMBER == args['start_model_num'] and resume != '':
            logging.info(f" Loading model from checkpoint {resume}")
            state_dict = dict(torch.load(resume)['model_state_dict'])
            classifier.load_state_dict(state_dict)
            iteration, break_count, best_loss, stop_sign =  \
                [int(x) for x in resume.split('/')[-1].split('.')[0].split('_')[2:]]
            logging.info("Loading model succeed")
        else:
            classifier.init_weights()
            iteration = 0
            break_count = 0
            best_loss = 10000000
            stop_sign = 0

        # classifier = nn.DataParallel(classifier).cuda(device=1)
        classifier = classifier.cuda(device=1)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        for epoch in range(100):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                y_batch = y_batch.type(torch.long)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)
                acc = multi_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 500 == 0:
                    # print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    logging.info('Epoch : ' + str(epoch) + ' iteration ' + str(iteration) + ' loss ' +
                                 str(loss.item()) + ' acc ' + str(acc))
                if iteration % 1000 == 0:
                    model_path = os.path.join(args['exp_dir'],
                                              'model_' + str(MODEL_NUMBER) + '_' + str(iteration) + '_' +
                                              str(break_count) + '_' + str(best_loss) + '_' + str(stop_sign) + '.pth')
                    logging.info('save to: ' + str(model_path))
                    torch.save({'model_state_dict': classifier.state_dict()},
                                model_path)

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        logging.info("*************** Break, Total iters, " + str(iteration) + " , at epoch" +
                                     str(epoch) + "***************")
                        # print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch),
                        #       "***************")
                        break

            if stop_sign == 1:
                break

        model_path = os.path.join(args['exp_dir'],
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        logging.info('save to: ' + str(model_path))
        # print('save to:', model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)


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

    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train(opts)

    # print('Loading pretrained models...')
    logging.info('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)
