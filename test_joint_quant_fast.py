# Fast variant of the Joint LS-ViT entry point.
#
# Identical to test_joint_quant.py in every stage -- FP32 loading, MSE
# calibration, reparameterisation, post-MSE checkpoint, LS sensitivity fit,
# fixed normalisers, joint optimisation, hard-rounding conversion, evaluation
# and checkpoint saving -- except that the joint training loop reads the
# optimisation images and the FP32 teacher from caches built once beforehand.
#
# The original entry point and utils/joint_lsvit.py are untouched and keep
# working on their own.
import importlib
import logging
import os
import sys
from datetime import datetime

import timm
import torch
from torch import nn

import test_joint_quant as base
from utils.calibrator import QuantCalibrator
from utils.joint_lsvit_fast import JointLSViTReconstructorFast
from utils.test_utils import validate
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net


def get_args_parser():
    parser = base.get_args_parser()
    parser.add_argument('--cache-device', default='cpu', choices=['cpu', 'cuda'],
                        help='where the cached FP32 teacher lives')
    return parser


def main(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    root_path = './checkpoints/quant_result/joint_fast_{}'.format(timestamp)
    os.makedirs(root_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler('{}/output.log'.format(root_path)),
                            logging.StreamHandler()
                        ])

    logging.info('{} - start the FAST joint LS-ViT process.'.format(base.get_cur_time()))
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

    base.seed_all(args.seed)

    logging.info('Loading FP32 weights from {}'.format(base.FP32_WEIGHTS))
    if not os.path.isfile(base.FP32_WEIGHTS):
        raise FileNotFoundError('FP32 weights not found: {}'.format(base.FP32_WEIGHTS))
    model = timm.create_model(base.MODEL_ZOO[args.model], pretrained=False,
                              checkpoint_path=base.FP32_WEIGHTS)
    model.to(device)
    model.eval()

    g = base.EmptySafeViTImageNetLoaderGenerator(args.dataset, args.val_batch_size,
                                                 args.num_workers, kwargs={'model': model})
    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    logging.info('Wrapping quantisation modules (reparam: True) ...')
    model = wrap_modules_in_net(model, cfg, reparam=True)
    model.to(device)
    model.eval()

    logging.info('{} - start {} guided calibration'.format(base.get_cur_time(), cfg.calib_metric))
    calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size,
                                  seed=args.seed)
    QuantCalibrator(model, calib_loader).batching_quant_calib()
    model = wrap_reparamed_modules_in_net(model)
    model.to(device)
    logging.info('{} - {} guided calibration finished.'.format(base.get_cur_time(), cfg.calib_metric))
    base.save_mse_checkpoint(model, args, cfg, root_path)

    optim_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size,
                                  seed=args.seed)
    joint = JointLSViTReconstructorFast(model, optim_loader,
                                        batch_size=cfg.optim_batch_size,
                                        temp=cfg.temp,
                                        iters=base.JOINT_ITERS,
                                        lambda_kl=args.lambda_kl,
                                        cache_device=args.cache_device)

    logging.info('{} - initialising all AdaRound variables, fitting LS sensitivities '
                 'and fixing normalisers'.format(base.get_cur_time()))
    joint.prepare()
    logging.info('{} - joint initialisation finished.'.format(base.get_cur_time()))

    if args.diagnose_scales:
        joint.diagnose()
        logging.info('{} - scale diagnosis finished, no optimiser step was taken.'.format(
            base.get_cur_time()))
        return

    if args.lambda_kl is None:
        raise ValueError('--lambda-kl must be given explicitly for joint training')

    logging.info('{} - building input and FP32 teacher caches ...'.format(base.get_cur_time()))
    joint.build_caches()
    logging.info('{} - caches built.'.format(base.get_cur_time()))

    logging.info('{} - start joint optimisation ({} steps, lambda_KL={})'.format(
        base.get_cur_time(), base.JOINT_ITERS, args.lambda_kl))
    joint.train()
    logging.info('{} - joint optimisation finished, switching to hard rounding.'.format(
        base.get_cur_time()))
    joint.finalize()

    base.save_model(model, args, cfg, root_path)
    logging.info('Validating on optimisation calibration set after joint quantisation ...')
    validate(optim_loader, model, criterion, print_freq=args.print_freq, device=device)
    logging.info('Validating on test set after joint quantisation ...')
    validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    logging.info('{} - finished the process.'.format(base.get_cur_time()))


if __name__ == '__main__':
    main(get_args_parser().parse_args())
