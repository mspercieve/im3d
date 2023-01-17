# Main file for training and testing
# author: ynie
# date: Feb, 2020

import argparse
from configs.config_utils import CONFIG
import os
import train, test,demo
from utils.logger import setup_logger
from net_utils.utils import write_info


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Total 3D Understanding.')
    parser.add_argument('--config', type=str, default='configs/total3d_ldif_gcnn3.yaml',
                        help='configure file for training or testing.')   # out/total3d/20110611514267/out_config_ours.yaml     'out/total3d/22122910024042/out_config.yaml'   'configs/total3d_ldif_gcnn_joint.yaml'
    parser.add_argument('--mode', type=str, default='qtrain', help='train, test, demo or qtrain, qtest')
    parser.add_argument('--demo_path', type=str, default='demo/inputs/1', help='Please specify the demo path.')
    parser.add_argument('--name', type=str, default=None, help='wandb exp name.')
    parser.add_argument('--sweep', action='store_true')
    return parser

if __name__ == '__main__':
    parser = parse_args()
    cfg = CONFIG(parser)

    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)

    '''Run'''
    if cfg.config['mode'] == 'train':
        try:
            train.run(cfg)
        except KeyboardInterrupt:
            pass
        except:
            raise
        cfg.update_config(mode='test', resume=True, weight=os.path.join(cfg.save_path, 'model_best.pth'))
    if cfg.config['mode'] == 'test':
        test.run(cfg)
    if cfg.config['mode'] == 'demo':
        logger = setup_logger('Demo', cfg.config['demo_path'])
        write_info(logger, True)
        demo.run(cfg)

