#!/usr/bin/env python
"""
debug_train.py: 用于单卡调试的训练脚本，不使用分布式。
用法:
    python tools/debug_train.py <config_file> --work-dir <work_dir> [其他参数]
"""

import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

# 如果你使用了 dynamo 优化，保留下面这行（如不需要可删除）
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Debug train a detector (single GPU, no distributed training)'
    )
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='保存日志和模型的目录')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='是否启用混合精度训练'
    )
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='是否自动缩放学习率'
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='若指定 checkpoint 路径，则从该路径恢复；若不指定，则自动从 work_dir 中恢复最新 checkpoint'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='覆盖配置文件中的设置，格式为 key=value'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='local rank, 默认值为 0'
    )
    args = parser.parse_args()
    # 单卡调试时，确保 local_rank 环境变量存在
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # 降低 dynamo 重复编译次数，提高训练速度
    setup_cache_size_limit_of_dynamo()

    # 加载配置文件
    cfg = Config.fromfile(args.config)
    # 固定使用单卡训练，不采用分布式启动
    cfg.launcher = 'none'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 优先使用命令行指定的 work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 启用混合精度训练（AMP）
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                '配置中已启用 AMP 训练。',
                logger='current',
                level=logging.WARNING
            )
        else:
            assert optim_wrapper == 'OptimWrapper', (
                f'使用 --amp 参数时，优化器包装器类型必须为 OptimWrapper，但配置中为 {optim_wrapper}.'
            )
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # 启用自动缩放学习率（auto scale lr）
    if args.auto_scale_lr:
        if ('auto_scale_lr' in cfg and 
            'enable' in cfg.auto_scale_lr and 
            'base_batch_size' in cfg.auto_scale_lr):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('配置文件中未找到 "auto_scale_lr" 或其必要的键值，请检查配置。')

    # 配置恢复训练
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # 构建 Runner 对象
    if 'runner_type' not in cfg:
        # 使用默认 Runner 构建方式
        runner = Runner.from_cfg(cfg)
    else:
        # 如果 cfg 中设置了 runner_type，则从注册器中构建自定义 Runner
        runner = RUNNERS.build(cfg)

    # 开始训练
    runner.train()


if __name__ == '__main__':
    main()
