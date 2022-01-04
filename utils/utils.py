import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import json

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        file = json.load(f)
    return file


def print_results(args, pred, truth, split='test'):
    return binary_cls(args, pred, truth, split)
    
    
def binary_cls(args, pred, truth, split='test'):
    y_targets = truth.cpu().detach().numpy()
    y_preds = pred.argmax(dim=1).cpu().detach().numpy()
    acc2 = accuracy_score(y_targets, y_preds)
    f1 = f1_score(y_targets, y_preds)
    precision = precision_score(y_targets, y_preds)
    recall = recall_score(y_targets, y_preds)
    print(f'{split} binary classification :')
    print('Acc2 :', acc2)
    print('F1-score :', f1)
    print('precision :', precision)
    print('recall :', recall)
    return {'Acc2':acc2, 'F1-score':f1, 'precision':precision, 'recall':recall}



def save_model(args, model, name='best_val'):
    if not os.path.exists('save_models'):
        os.mkdir('save_models')
    if args.data_parallel:
        torch.save(model.module.state_dict(), f'save_models/{name}.pth')
    else:
        torch.save(model.state_dict(), f'save_models/{name}.pth')

def load_model(args, model, name='best_val'):
    # TODO
    checkpoint = torch.load(f'save_models/{name}.pth')
    if args.data_parallel:      
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)