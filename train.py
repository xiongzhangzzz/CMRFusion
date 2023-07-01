import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from dataloader import Dataset
import warnings
warnings.filterwarnings("ignore")
from models.select_model import define_Model

def main(json_path='opt.json'):
    
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    
    # get parsers
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='/home/user/文档/xz/mymethod/crosssrfuse/options/opt.json', help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    init_epoch, init_path = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_net'] = init_path
    init_iter_optimizer, init_path_optimizer = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_optimizer'] = init_path_optimizer
    cur_epoch = max(init_epoch, init_iter_optimizer)
    

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info('random seed: %d' %seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        # elif phase == 'test':
        #     test_set = Dataset(dataset_opt)
        #     test_loader = DataLoader(test_set, batch_size=1,
        #                              shuffle=False, num_workers=1,
        #                              drop_last=False, pin_memory=True)
        # else:
        #     raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()

    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    current_step = 0
    
    for epoch in range(200):  # keep running
        cur_epoch += 1
        print('epoch %d start'% cur_epoch )

        for i, train_data in enumerate(train_loader):

            current_step += 1
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(cur_epoch)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------

            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(cur_epoch)
        # -------------------------------
        #  training information
        # -------------------------------
            if current_step % 20 == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(cur_epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

        if (cur_epoch % opt['train']['checkpoint_print'] == 0) and opt['rank'] == 0:
        # if True:
            logs = model.current_log()  # such as loss
            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(cur_epoch, current_step, model.current_learning_rate())
            for k, v in logs.items():  # merge log information into message
                message += '{:s}: {:.3e} '.format(k, v)
            logger.info(message)

        # -------------------------------
        #  save model
        # -------------------------------
        if cur_epoch % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
            save_dir = opt['path']['models'] 
            save_filename = '{}_{}.pth'.format(cur_epoch, 'G')
            save_path = os.path.join(save_dir, save_filename)
            logger.info('Saving the model. Save path is:{}'.format(save_path))
            model.save(cur_epoch)

if __name__ == '__main__':
    main()
