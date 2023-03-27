import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from trainutil import OnlineTripletLoss, SemihardNegativeTripletSelector

from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer)

import torch.nn as nn
import torch    

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 label

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.label = label


def convert_examples_to_features(dataline,tokenizer,args):
    """convert examples to token ids"""

    datajson = json.loads(dataline)
    code = ""
    codelines = datajson['bug_trace']
    for codeline in codelines:
        codeline = codeline.strip(" ")
        code += codeline
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    warningjson = json.loads(datajson['warning_json'])
    bugtype = warningjson['bug_type']
    qualifier = warningjson['qualifier']
    procedure = warningjson['procedure']
    filename = warningjson['file']
    if ".cpp" in filename:
        filename = filename.replace(".cpp","").split('/')[-1]
    if ".c" in filename:
        filename = filename.replace(".c","").split('/')[-1]
    nl = bugtype + qualifier + procedure + filename
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    score = int(datajson['score'])
    if score >= 1:
        label = 1
    else:
        label = 0

    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,label)

class WarningDataset(Dataset):
    def __init__(self, tokenizer, args, truedatapath, falsedatapath):
        self.examples = []


        with open(truedatapath)as f:
            truedata = f.readlines()
        with open(falsedatapath)as f:
            falsedata = f.readlines()

        self.oversample = args.oversampling
        self.undersample = args.undersampling

        truedata = truedata*self.oversample

        tmplen = round(len(falsedata)*self.undersample)
        random.shuffle(falsedata)
        falsedata = falsedata[0:tmplen]

        data_all = truedata + falsedata

        for data in data_all:       
            self.examples.append(convert_examples_to_features(data, tokenizer, args))


                         
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # if self.examples[i].label == 0:
        #     positive_sample = random.choice(self.falsedata)
        #     negative_sample = random.choice(self.truedata)
        # else:
        #     positive_sample = random.choice(self.truedata)
        #     negative_sample = random.choice(self.falsedata)
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids),torch.tensor(self.examples[i].label))

class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768*2,2)

      
    def forward(self, code_inputs, nl_inputs): 
        code_outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
        code_outputs = (code_outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
        code_outputs = torch.nn.functional.normalize(code_outputs, p=2, dim=1)
        
        nl_outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
        nl_outputs = (nl_outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
        nl_outputs = torch.nn.functional.normalize(nl_outputs, p=2, dim=1)

        feature = torch.cat((nl_outputs, code_outputs), 1)

        logits = self.fc1(feature)
        logits = self.dropout(logits)
        return logits, feature



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    #get training dataset
    train_dataset = WarningDataset(tokenizer, args, f"{args.data_file}/warning_true_train_old.txt", f"{args.data_file}/warning_false_train_old.txt")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)

    crossentropyloss = CrossEntropyLoss()


    # # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    # logger.info("  Total train batch size  = %d", args.train_batch_size)
    # logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss = 0,0
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            labels = batch[2].to(args.device)

            #get code and nl vectors
            predictes, features = model(code_inputs=code_inputs, nl_inputs=nl_inputs)
            
            #calculate scores and loss
            loss = crossentropyloss(predictes, labels)

            
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
            if (step+1)%10 == 0:
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
        # if (idx+1)%5 == 0:
        model_to_save = model.module if hasattr(model,'module') else model
        output_dir = f'{args.output_dir}/unixcoder_{idx+1}'
        torch.save(model_to_save.state_dict(), output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default="/home/zhipengxue/warning_elimination/dataset", type=str, 
                        help="The input training data file.")
    parser.add_argument("--output_dir", default="/data1/zhipengxue/model", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_reload", default=1, type=int,
                        help="Reload num.")
    
    parser.add_argument("--model_name_or_path", default="/home/zhipengxue/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=100, type=int,
                        help="Optional nl input sequence length after tokenization.") 
    parser.add_argument("--code_length", default=300, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument('--oversampling', type=int, default=20,
                        help="oversampling for true")
    parser.add_argument('--undersampling', type=float, default=1,
                        help="undersampling for false")





    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)


    # Set seed
    set_seed(args.seed)


    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path) 
 
    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)

    # reload_dir = f'{args.output_dir}/unixcoder_{args.num_reload}'
    # model_to_load = model.module if hasattr(model, 'module') else model
    # model_to_load.load_state_dict(torch.load(reload_dir)) 
    model.to(args.device)

    train(args, model, tokenizer)




if __name__ == "__main__":
    main()