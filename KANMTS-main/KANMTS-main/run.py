import argparse
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.set_num_threads(6)
    parser = argparse.ArgumentParser(description='KANMTS')
    # basic config
    parser.add_argument('--grid_size', type=int, default=4, help='status')

    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='KANMTS',
                        help='model name, options: [KAN_Stock,KANLinear1,Linear2,KAN_Channel,Autoformer, Transformer, TimesNet,SOFTS,StockMixer,SOFTS_Stock_patch,SOFTS_Stock,SOFTS_Stock_KAN]')
    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # model define,
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    # parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')

    parser.add_argument('--d_core', type=int, default=128, help='dimension of core<d_model>,下采样')
    # parser.add_argument('--d_core', type=int, default=512, help='dimension of core')

    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--num_layers', type=int, default=2, help='num of channel layers')

    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    # parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0., help='dropout')
    # parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--attention_type', type=str, default="full", help='the attention type of transformer')
    # parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--use_norm', type=int, default=1, help='use norm and denorm')



    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    # parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    # parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--save_model', action='store_true')

    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--n_layers', type=int, default=2)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    print('Args in experiment:')
    print(args)
    Exp = Exp_Long_Term_Forecast


    def train(args=args):
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_dc{}_el{}_dl{}_df{}_fc{}_dt{}_{}_nl{}_hd{}_bs{}_lr{}_te{}_gs{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_core,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.distil,
            args.des,
            args.num_layers,
            args.hidden_dim,
            args.batch_size,
            args.learning_rate,
            args.train_epochs,
            args.grid_size

        )

        exp = Exp(args)  # set experiments

        with open('./result/results.txt', 'a', encoding='utf-8', errors='replace') as file:
            file.write(f'setting: {setting}\n')

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))


        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()

    # def train(args=args):
    #     #
    #     learning_rates = [0.0001, 0.0003, 0.0005]  #
    #     batch_sizes = [16,32, 64]  #
    #     dmodel = [64,128,256,512,2048]
    #     dcore = [64, 128, 256, 512, 2048]
    #     dff = [64, 128, 256, 512, 2048]
    #     grid_sizes = [4,5,6,7,10]  #
    #     pred_len = [96,196,336,720]
    #
    #     for el in [1,2,3,4,5]:
    #         for pred_len in pred_len:
    #             for lr in learning_rates:
    #                 for bs in batch_sizes:
    #                     for dmodel in  dmodel:
    #                         for dcore in dcore:
    #                             for dff in dff:
    #                                 for grid_sizes in grid_sizes:
    #                                     args.pred_len = pred_len
    #                                     args.grid_size = grid_sizes
    #                                     args.e_layers = el
    #                                     args.learning_rate = lr
    #                                     args.batch_size = bs
    #                                     args.d_model = dmodel
    #                                     args.d_core = dcore
    #                                     args.d_ff = dff
    #                                     setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_dc{}_el{}_dl{}_df{}_fc{}_dt{}_{}_nl{}_hd{}_bs{}_lr{}_te{}_gs{}'.format(
    #                                         args.task_name,
    #                                         args.model_id,
    #                                         args.model,
    #                                         args.data,
    #                                         args.features,
    #                                         args.seq_len,
    #                                         args.label_len,
    #                                         args.pred_len,
    #                                         args.d_model,
    #                                         args.d_core,
    #                                         args.e_layers,
    #                                         args.d_layers,
    #                                         args.d_ff,
    #                                         args.factor,
    #                                         args.distil,
    #                                         args.des,
    #                                         args.num_layers,
    #                                         args.hidden_dim,
    #                                         args.batch_size,
    #                                         args.learning_rate,
    #                                         args.train_epochs,
    #                                         args.grid_size
    #                                     )
    #
    #                                     exp = Exp(args)  # set experiments
    #
    #                                     with open('./result/results.txt', 'a', encoding='utf-8', errors='replace') as file:
    #                                         file.write(f'setting: {setting}\n')
    #
    #                                     print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    #                                     exp.train(setting)
    #
    #
    #
    #
    #                                     print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #                                     exp.test(setting)
    #
    #                                     torch.cuda.empty_cache()

    if args.is_training:
        train(args)
    else:
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des)
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        exp.test(setting, test=1)
        torch.cuda.empty_cache()

        # 查看GPU使用情况
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())


