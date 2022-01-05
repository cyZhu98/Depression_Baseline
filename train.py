import os

import numpy as np
import torch
from torch import optim
from torch import nn
from models.baselines import *
import time
from utils.utils import *
from tqdm import tqdm
import transformers
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def select_model(args):
    if args.text_only:
        if args.loader_mode == 'splice' and args.baseline == 'finetune':
            model = Text_Splice_Finetune().cuda()
        elif args.loader_mode == 'splice' and args.baseline == 'gru':
            model = Text_Splice_GRU().cuda()
        else:
            model = Text_Split().cuda()
    elif args.image_only:
        model = Image_Only().cuda()
    else:
        model = Fusion_Basic().cuda()
    if args.data_parallel:
        model = nn.DataParallel(model)
    return model


def train_and_eval(args):
    fix_seed(args.seed)
    split = ['train', 'val', 'test']  # fixed
    # Loader
    if args.loader_mode == 'splice':
        from dataloader_splice import get_dataloader
    else:
        from dataloader import get_dataloader

    train_loader = get_dataloader(args, 'train')
    val_loader = get_dataloader(args, 'val')
    test_loader = get_dataloader(args, 'test')
    # Model
    model = select_model(args)
    # for name, param in model.named_parameters():
    #     if 'text_encoder' in name:
    #         param.requires_grad = False

    # Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    optimizer = getattr(optim, args.optim)(
        optimizer_grouped_parameters, lr=args.lr)
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=args.factor,
                                                     verbose=True)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss()  # cls

    best_val = -1
    record_test = 0
    best_epoch = 0
    best_preds = 0
    best_targets = 0
    for epoch in tqdm(range(args.epochs)):
        train_one_epoch(epoch, model, train_loader, optimizer,
                        criterion, scheduler=lr_scheduler)
        val_loss, test_loss, test_preds, test_targets = validate(
            args, epoch, model, val_loader, test_loader, criterion, scheduler)
        '''
        select the best epoch according to the val F1-score
        you can change it to loss / accuracy
        '''
        if val_loss > best_val or (val_loss == best_val and test_loss > record_test):
            best_val = val_loss
            record_test = test_loss
            print('save model')
            best_epoch, best_preds, best_targets = epoch, test_preds, test_targets
            if args.if_save_model:
                save_model(args, model, args.name)

    print('Conclusion : ')
    print('Best Epoch is : {}'.format(best_epoch))
    print('Test Result is ')
    use_saved_data = True
    if use_saved_data:
        results = print_results(args, best_preds, best_targets)
    else:
        model = select_model(args)
        load_model(model, args.name)
        _, preds, targets = get_pred(model, test_loader, criterion)
        results = print_results(args, preds, targets)
    if args.if_save_results:
        with open(f'save_results/{args.save_name}.txt', 'a+') as f:
            f.write('lr {} , cls ratio {} , bert ratio {}\n'.format(
                args.lr, args.cls_ratio, args.bert_ratio))
            f.write('F1-score: {}\n'.format(results['F1-score']))
            f.write('Accuracy : {}\n'.format(results['Acc2']))
            f.write('Recall : {}\n'.format(results['recall']))
            f.write('Precision : {}\n'.format(results['precision']))
            f.write('\n')
    return record_test


def train_one_epoch(epoch, model, loader, optimizer, criterion, scheduler=None):
    model.train()
    if epoch == 0:
        print(model)
    training_loss = 0
    batch = 0
    start_time = time.time()
    # id, audio ,text, target
    for idx, data in enumerate(tqdm(loader, mininterval=10)):
        text, text_att, img, target = data
        text, text_att, target = text.cuda(), text_att.cuda(), target.cuda()
        if img is not None:
            img = img.cuda()
        optimizer.zero_grad()
        pred = model(text, img, text_att)
        loss = criterion(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)  # TODO:set a number
        optimizer.step()
        scheduler.step()
        batch_loss = loss.item() * target.shape[0]  # loss * batch_size
        training_loss += batch_loss
        batch += target.shape[0]
    avg_loss = training_loss / batch
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    print('Epoch {} | Time/Epoch(s) {:.3f} | Train Loss {:.5f}'.
          format(epoch, time.time() - start_time, avg_loss))


@torch.no_grad()
def get_pred(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    batch_size = 0
    for idx, data in enumerate(loader):
        text, text_att, img, target = data
        text, text_att, target = text.cuda(), text_att.cuda(), target.cuda()
        if img is not None:
            img = img.cuda()
        pred = model(text, img, text_att)
        loss = criterion(pred, target).item()
        batch_size += target.shape[0]
        total_loss += loss * target.shape[0]
        preds.append(pred)
        targets.append(target)

    total_loss /= batch_size
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return total_loss, preds, targets


@torch.no_grad()
def validate(args, epoch, model, val_loader, test_loader, criterion, scheduler):
    start_time = time.time()
    val_loss, val_preds, val_targets = get_pred(model, val_loader, criterion)
    test_loss, test_preds, test_targets = get_pred(
        model, test_loader, criterion)
    end_time = time.time()
    print('Epoch {:2d} | Time {:.5f} sec | Valid Loss {:.5f} | Test Loss {:.5f}'.format(epoch, end_time - start_time,
                                                                                        val_loss, test_loss))
    val_f1 = print_results(args, val_preds, val_targets,
                           split='val')['F1-score']
    print('Val F1-score :', val_f1)
    print("-" * 20)
    test_f1 = print_results(args, test_preds, test_targets, split='test')[
        'F1-score']
    return val_f1, test_f1, test_preds, test_targets
