import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import argparse
from train import train_and_eval

'''
true : 1462, false : 1383
prepare train dataset with length 2845
true : 212, false : 194
prepare val dataset with length 406
true : 424, false : 389
prepare test dataset with length 813
'''
def get_args():
    parser = argparse.ArgumentParser(description='Depression Detection')

    # DataLoader
    parser.add_argument('--data_path', type=str, default='data',
                        help='path for storing the dataset')  # 没用到，作为保留
    parser.add_argument('--batch-size', type=int, default=8, help='choose a batch size')
    parser.add_argument('--split-ratio', type=float, default=0.2, help='take a part of the whole dataset out')
    parser.add_argument('--loader-mode', type=str, default='splice', choices=['splice', 'split'], help='A: splice tweets in one sentence; B: split tweets')

    # Training Setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--data-parallel', type=bool, default=False)
    # Optimizer
    parser.add_argument('--optim', type=str, default='AdamW',
                        help='optimizer for training, please follow type like [Adam, SGD...]')
    # TODO: different lr for different modules
    parser.add_argument('--lr', type=float, default='6e-6', help='learning rate for optimizer')
    parser.add_argument('--factor', type=float, default=0.5, help='lr * factor')

    # Models
    parser.add_argument('--name', type=str, default='best_val', help='name for the saved model')
    parser.add_argument('--text-only', type=bool, default=False, help='if the model uses text only')
    parser.add_argument('--image-only', type=bool, default=False, help='if the model uses image only')
    parser.add_argument('--baseline', type=str, default='finetune', choices=['finetune', 'gru', 'split'], help='choose a baseline')
    # Model hyperparams

    # Results save
    parser.add_argument('--if-save-results', type=bool, default=False, help='if save results in ./save_results/name.txt')
    parser.add_argument('--if-save-model', type=bool, default=False, help='if save the best model in ./save_models/')
    parser.add_argument('--save-name', type=str, default='test', help='name.txt')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    args.data_parallel = True
    args.name = 'test'
    args.text_only = True
    
    if args.optim not in ['Adam', 'AdamW', 'SGD']:
        raise KeyError('Please set a valid optimizer, such as "Adam, AdamW ..."')
    #
    # args.if_save_results = True
    args.save_name = 'test'  #
    if args.if_save_results:
        record_file_path = f'save_results/{args.save_name}.txt'
        if not os.path.exists(record_file_path):
            with open(record_file_path, 'w') as f:
                f.close()
                
    val_loss = train_and_eval(args)
