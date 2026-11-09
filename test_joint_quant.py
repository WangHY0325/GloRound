# Joint LS-ViT entry point.
#
# Model construction, quantisation wrapping, calibrator, bit-widths, seed,
# calibration set size, KL temperature and evaluation rules are taken from the
# original LS-ViT entry point.  The single difference is the optimisation stage:
# every reconstruction module is initialised once and trained jointly by
# utils.joint_lsvit.JointLSViTReconstructor instead of block by block.
#
# The FP32 starting point is an explicit file; there is no fallback to another
# checkpoint or to remote pretrained weights.
import argparse
import importlib
import logging
import os
import sys
from datetime import datetime

import numpy as np
import timm
import torch
from torch import nn
from torchvision.datasets import ImageFolder

import utils.datasets as mydatasets
from utils.calibrator import QuantCalibrator
from utils.joint_lsvit import JointLSViTReconstructor
from utils.test_utils import validate
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net


FP32_WEIGHTS = r'E:\111WHY_code\model\deit_tiny\deit_tiny_patch16_224-a1311bcf.pth'
DEFAULT_DATA_ROOT = r'E:\111WHY_code\data'
JOINT_ITERS = 20000

MODEL_ZOO = {
    'vit_tiny': 'vit_tiny_patch16_224',
    'vit_small': 'vit_small_patch16_224',
    'vit_base': 'vit_base_patch16_224',
    'vit_large': 'vit_large_patch16_224',
    'deit_tiny': 'deit_tiny_patch16_224',
    'deit_small': 'deit_small_patch16_224',
    'deit_base': 'deit_base_patch16_224',
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base': 'swin_base_patch4_window7_224',
    'swin_base_384': 'swin_base_patch4_window12_384',
}


class EmptySafeViTImageNetLoaderGenerator(mydatasets.ViTImageNetLoaderGenerator):
    """ViT loader whose training folder tolerates empty class directories.

    The local ImageNet subset keeps all 1000 class folders but only 1024
    training images, so the plain ImageFolder call raises on the empty ones.
    allow_empty=True keeps the standard ImageNet label mapping unchanged.  The
    calibration sampling, caching and DataLoader arguments stay untouched.
    """

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = ImageFolder(
                os.path.join(self.root, 'train'),
                self.train_transform,
                allow_empty=True,
            )
        return self._train_set


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deit_tiny', choices=list(MODEL_ZOO.keys()),
                        help='model architecture')
    parser.add_argument('--config', default='./configs/w3a3.py',
                        help='file path to import Config class from')
    parser.add_argument('--dataset', default=DEFAULT_DATA_ROOT,
                        help='path to ImageNet dataset')
    parser.add_argument('--val-batch-size', default=500, type=int,
                        help='batchsize of validation set')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='number of data loading workers')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--seed', default=3407, type=int, help='seed')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--lambda-kl', type=float, default=None,
                        help='explicit weight of the global KL term (required to train)')
    parser.add_argument('--diagnose-scales', action='store_true',
                        help='only measure L_LS, L_KL and alpha gradient norms')
    return parser


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cur_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def save_model(model, args, cfg, root_path):
    auto_name = '{}_w{}_a{}_jointsize_{}_lambda_{}.pth'.format(
        args.model, cfg.w_bit, cfg.a_bit, cfg.optim_size, args.lambda_kl)
    save_path = os.path.join(root_path, auto_name)
    logging.info('Saving checkpoint to {}'.format(save_path))
    torch.save(model.state_dict(), save_path)
    return save_path


def save_mse_checkpoint(model, args, cfg, root_path):
    """Save the MSE-calibrated, reparameterised model before AdaRound init.

    Stage: after QuantCalibrator.batching_quant_calib() and
    wrap_reparamed_modules_in_net(), before any AdaRound wrapping, LS
    sensitivity fit or joint optimiser step.  The model is only read here.
    """
    out_dir = os.path.join(root_path, 'mse_calibration')
    os.makedirs(out_dir)
    save_path = os.path.join(out_dir, 'post_mse_reparam.pth')

    modes = set()
    modules_with_mode = 0
    without_calibrated = []
    without_inited = []
    for name, m in model.named_modules():
        if hasattr(m, 'mode'):
            modules_with_mode += 1
            modes.add(m.mode)
            if not getattr(m, 'calibrated', False):
                without_calibrated.append(name)
        for attr in ('a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer'):
            quantizer = getattr(m, attr, None)
            if quantizer is not None and not getattr(quantizer, 'inited', False):
                without_inited.append('{}.{}'.format(name, attr))

    checkpoint = {
        'stage': 'post_mse_calibration_after_reparam',
        'model': args.model,
        'timm_arch': MODEL_ZOO[args.model],
        'cfg': vars(cfg),
        'seed': args.seed,
        'calib_size': cfg.calib_size,
        'calib_batch_size': cfg.calib_batch_size,
        'fp32_weights': FP32_WEIGHTS,
        'state_dict': model.state_dict(),
        'quantizer_state': {
            'modules_with_mode': modules_with_mode,
            'modes_seen': sorted(modes),
            'modules_without_calibrated': without_calibrated,
            'quantizers_without_inited': without_inited,
        },
    }
    torch.save(checkpoint, save_path)
    logging.info('Saving post-MSE checkpoint (stage: {}) to {}'.format(
        checkpoint['stage'], save_path))
    logging.info('quantizer state: modules_with_mode={} modes={} '
                 'modules_without_calibrated={} quantizers_without_inited={}'.format(
                     modules_with_mode, sorted(modes),
                     len(without_calibrated), len(without_inited)))
    return save_path


def main(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    root_path = './checkpoints/quant_result/joint_{}'.format(timestamp)
    os.makedirs(root_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler('{}/output.log'.format(root_path)),
                            logging.StreamHandler()
                        ])

    logging.info('{} - start the joint LS-ViT process.'.format(get_cur_time()))
    logging.info(str(args))

    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    cfg = getattr(imported_module, 'Config')()
    logging.info('Successfully imported Config class!')
    for name, value in vars(cfg).items():
        logging.info('{}: {}'.format(name, value))

    if args.device.startswith('cuda:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[1]
        args.device = 'cuda:0'
    device = torch.device(args.device)

    seed_all(args.seed)

    logging.info('Loading FP32 weights from {}'.format(FP32_WEIGHTS))
    if not os.path.isfile(FP32_WEIGHTS):
        raise FileNotFoundError('FP32 weights not found: {}'.format(FP32_WEIGHTS))
    model = timm.create_model(MODEL_ZOO[args.model], pretrained=False,
                              checkpoint_path=FP32_WEIGHTS)
    model.to(device)
    model.eval()

    g = EmptySafeViTImageNetLoaderGenerator(args.dataset, args.val_batch_size,
                                            args.num_workers, kwargs={'model': model})
    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    logging.info('Wrapping quantisation modules (reparam: True) ...')
    model = wrap_modules_in_net(model, cfg, reparam=True)
    model.to(device)
    model.eval()

    logging.info('{} - start {} guided calibration'.format(get_cur_time(), cfg.calib_metric))
    calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size,
                                  seed=args.seed)
    QuantCalibrator(model, calib_loader).batching_quant_calib()
    model = wrap_reparamed_modules_in_net(model)
    model.to(device)
    logging.info('{} - {} guided calibration finished.'.format(get_cur_time(), cfg.calib_metric))
    save_mse_checkpoint(model, args, cfg, root_path)

    optim_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size,
                                  seed=args.seed)
    joint = JointLSViTReconstructor(model, optim_loader,
                                    batch_size=cfg.optim_batch_size,
                                    temp=cfg.temp,
                                    iters=JOINT_ITERS,
                                    lambda_kl=args.lambda_kl)

    logging.info('{} - initialising all AdaRound variables, fitting LS sensitivities '
                 'and fixing normalisers'.format(get_cur_time()))
    joint.prepare()
    logging.info('{} - joint initialisation finished.'.format(get_cur_time()))

    if args.diagnose_scales:
        joint.diagnose()
        logging.info('{} - scale diagnosis finished, no optimiser step was taken.'.format(
            get_cur_time()))
        return

    if args.lambda_kl is None:
        raise ValueError('--lambda-kl must be given explicitly for joint training')

    logging.info('{} - start joint optimisation ({} steps, lambda_KL={})'.format(
        get_cur_time(), JOINT_ITERS, args.lambda_kl))
    joint.train()
    logging.info('{} - joint optimisation finished, switching to hard rounding.'.format(
        get_cur_time()))
    joint.finalize()

    save_model(model, args, cfg, root_path)
    logging.info('Validating on optimisation calibration set after joint quantisation ...')
    validate(optim_loader, model, criterion, print_freq=args.print_freq, device=device)
    logging.info('Validating on test set after joint quantisation ...')
    validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    logging.info('{} - finished the process.'.format(get_cur_time()))


if __name__ == '__main__':
    main(get_args_parser().parse_args())
