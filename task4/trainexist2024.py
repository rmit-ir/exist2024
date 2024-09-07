import os
from dataset import MyDataset
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data, test_data, processor):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              collate_fn=MyDataset.collate_func,
                              shuffle=True)
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)

    clip_params = list(map(id, model.model.parameters()))
    base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
    optimizer = AdamW([
            {"params": base_params},
            {"params": model.model.parameters(),"lr": args.clip_learning_rate}
            ], lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                            num_training_steps=total_steps)

    #max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list = batch 
            inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            labels = torch.tensor(label_list).to(device)

            loss, score = model(inputs,labels=labels)
            
            #print("score training:", score)
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            if args.optimizer_name == 'adam':
                scheduler.step() 
            optimizer.zero_grad()

        #### here, only use entropy 
        #val_targets, val_predictions, sumval_loss, sumval_step = eval_fn(args, model, device, dev_data, processor, mode='dev')
        #wandb.log({'dev_loss': sumval_loss/sumval_step}) #, 'dev_targets': val_targets, 'dev_predictions': val_predictions
        #dev_acc, dev_f1 ,dev_precision,dev_recall = evaluate_acc_f1(args, model, device, dev_data, processor, mode='dev')
        #wandb.log({'dev_acc': dev_acc, 'dev_f1': dev_f1, 'dev_precision': dev_precision, 'dev_recall': dev_recall})
        #logging.info("i_epoch is {}, dev_loss is {}, dev_targets is {}, dev_predictions is {}".format(i_epoch, sumval_loss/sumval_step, val_targets, val_predictions))
        
        
        ##khi nao thi save model? va make prediction on test dataset?
        path_to_save = os.path.join(args.output_dir, args.model)
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        #if (i_epoch+1) == int(args.num_train_epochs):
        model_to_save = (model.module if hasattr(model, "module") else model)
        torch.save(model_to_save.state_dict(), os.path.join(path_to_save, str(i_epoch+1)+'model.pt'))
        
        val_targets, val_predictions, sumval_loss, sumval_step = eval_fn(args, model, device, dev_data, processor, mode='dev')
        logging.info("i_epoch is {}, dev_loss is {}, dev_targets is {}, dev_predictions is {}".format(i_epoch, sumval_loss/sumval_step, val_targets, val_predictions))
        
            ### make prediction on test dataset? - check code da,
        torch.cuda.empty_cache()
    logger.info('Train done')

def eval_fn(args, model, device, data, processor, macro=False,pre = None, mode='test'):
    data_loader = DataLoader(data, batch_size=args.dev_batch_size, collate_fn=MyDataset.collate_func,shuffle=False)
    model.eval()
    fin_targets = []
    fin_predictions = []
    
    sum_loss = 0
    sum_step = 0
    
    with torch.no_grad():
        for i_batch, t_batch in enumerate(data_loader):
            text_list, image_list, label_list, id_list = t_batch
            inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            labels = torch.tensor(label_list).to(device)
            loss, t_outputs = model(inputs,labels=labels)
            sum_loss += loss.item()
            sum_step += 1
            
            targets = labels
            #outputs = torch.argmax(t_outputs, -1) ## co the chinh sua o day 
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_predictions.extend(t_outputs.cpu().detach().numpy().tolist())
    return fin_targets, fin_predictions, sum_loss, sum_step