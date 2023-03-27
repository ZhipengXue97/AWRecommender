import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import copy
import math
from numpy import mean
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
    # if bugtype != 'NULL_DEREFERENCE':
    #     return 0
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

    label = float(datajson['score'])

    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,label)

class WarningDataset(Dataset):
    def __init__(self, tokenizer, args, truedatapath, falsedatapath):
        self.examples = []

        with open(truedatapath)as f:
            truedata = f.readlines()
        with open(falsedatapath)as f:
            falsedata = f.readlines()
        data_all = truedata + falsedata
        for data in data_all:
            feature = convert_examples_to_features(data, tokenizer, args)
            if feature != 0:
                self.examples.append(convert_examples_to_features(data, tokenizer, args))
        
                         
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids),torch.tensor(self.examples[i].label))


class WarningtestDataset(Dataset):
    def __init__(self, tokenizer, args, path):
        self.examples = []

        with open(path)as f:
            self.data = f.readlines()
 
        for data in self.data:
            self.examples.append(convert_examples_to_features(data, tokenizer, args))
                         
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids), self.data[i])


class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768*2,4)


      
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

        return logits


def get_max_id(m, l):
    max_index = []
    t = copy.deepcopy(m)
    for _ in range(l):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_index.append(index)
    t = []

    return max_index

def nDCG(top, label_list, k):
    DCG = 0
    num = 0
    iDCG = 0
    for i in range(k):
        id = top[i]
        if label_list[id] >= 2:
            num += 1
            DCG += 1/math.log(2+i)

    for i in range(num):
        iDCG += 1/math.log(2+i)

    if iDCG == 0:
        return 0
    else:
        return DCG/iDCG

def get_recall(top, labels_list, k):
    num = 0
    all = labels_list.count(2)+labels_list.count(3)
    for i in range(k):
        id = top[i]
        if labels_list[id] >= 2:
            num += 1

    return num/all


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def val(args, model, tokenizer):
    """ Test the model """
    #get testing dataset
    test_dataset = WarningDataset(tokenizer, args, f"{args.data_file}/warning_true_test_old.txt", f"{args.data_file}/warning_false_test_old.txt")
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle = False)
    
    model.eval()
    count = 0
    predicts_list = []
    labels_list = []
    predicts_score_list = []


    for batch in test_dataloader:
        #get inputs
        code_inputs = batch[0].to(args.device)    
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        #get code and nl vectors
        with torch.no_grad():
            predicts = model(code_inputs=code_inputs, nl_inputs=nl_inputs)
            predicts_score = predicts.max(1)[0]
            predicts = predicts.max(1)[1]
            
            predicts_list.extend(predicts.tolist())
            labels_list.extend(labels.tolist())
            predicts_score_list.extend(predicts_score.tolist())



    ap_1 = []
    ap_3 = []
    ap_5 = []
    mrr_list = []
    ndcgs_1 = []
    ndcgs_3 = []
    ndcgs_5 = []
    recall = []

    idlist = [i for i in range(len(labels_list))]
    r = 0
    while r < 100:

        random.shuffle(idlist)
        tmp_idlist = idlist[:1000]
        tmp_predicts_list = [predicts_list[j] for j in tmp_idlist]
        tmp_labels_list = [labels_list[j] for j in tmp_idlist]
        tmp_predicts_score_list = [predicts_score_list[j] for j in tmp_idlist]
        if tmp_labels_list.count(2)+tmp_labels_list.count(3)<10:
            continue
        
        r += 1
        high_id, medium_id, low_id, false_id = [], [], [], []
        high_score, medium_score, low_score, false_score = [], [], [], []

        for i in range(len(tmp_predicts_list)):
            if tmp_predicts_list[i] == 0:
                false_id.append(i)
                false_score.append(tmp_predicts_score_list[i])
            if tmp_predicts_list[i] == 1:
                low_id.append(i)
                low_score.append(tmp_predicts_score_list[i])
            if tmp_predicts_list[i] == 2:
                medium_id.append(i)
                medium_score.append(tmp_predicts_score_list[i])
            if tmp_predicts_list[i] == 3:
                high_id.append(i)
                high_score.append(tmp_predicts_score_list[i])

        top = []
        tmp_id = get_max_id(high_score, len(high_score))
        top.extend([high_id[i] for i in tmp_id])
        tmp_id = get_max_id(medium_score, len(medium_score))
        top.extend([medium_id[i] for i in tmp_id])
        tmp_id = get_max_id(low_score, len(low_score))
        top.extend([low_id[i] for i in tmp_id])
        tmp_id = get_max_id(false_score, len(false_score))
        tmp_id.reverse()
        top.extend([false_id[i] for i in tmp_id])
            

        ap = 0
        t=0
        f=0

        for i in range(len(top)):
            id = top[i]
            if tmp_labels_list[id] >= 2:
                mrr_list.append(1/(i+1))
                break

        ndcgs_1.append(nDCG(top, tmp_labels_list, 1))
        ndcgs_3.append(nDCG(top, tmp_labels_list, 3))
        ndcgs_5.append(nDCG(top, tmp_labels_list, 5))
        recall.append(get_recall(top, tmp_labels_list, 300))

    print(mean(mrr_list), mean(ndcgs_1), mean(ndcgs_3), mean(ndcgs_5))
    print(mean(recall))



    #     for id in top:
    #         if tmp_labels_list[id] >= 1:
    #             t += 1
    #             ap += t/(t+f)
    #         if tmp_labels_list[id] < 1:
    #             f += 1

    #         if t+f == 1:
    #             ap_1.append(ap/1)
    #         if t+f == 3:
    #             ap_3.append(ap/3)
    #         if t+f == 5:
    #             ap_5.append(ap/5)
    # print(mean(ap_1), mean(ap_3), mean(ap_5), mean(nDCGs))


def identify_val(args, model, tokenizer):
    """ Test the model """
    #get testing dataset
    test_dataset = WarningDataset(tokenizer, args, f"{args.data_file}/warning_true_test_old.txt", f"{args.data_file}/warning_false_test_old.txt")
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    model.eval()
    predicts_all = []
    labels_all =[]

    for batch in test_dataloader:
        #get inputs
        code_inputs = batch[0].to(args.device)    
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        #get code and nl vectors
        with torch.no_grad():
            predicts = model(code_inputs=code_inputs, nl_inputs=nl_inputs)
            predicts = predicts.max(1)[1]
            predicts_all.extend(predicts.tolist())
            labels_all.extend(labels.tolist())
        
    test_len = len(labels_all)
    TP, FP, TN, FN = 0, 0, 0, 0 
    for i in range(test_len):
        thread = 1
        if labels_all[i] >= thread and predicts_all[i] >= thread:
            TP += 1
        if labels_all[i] < thread and predicts_all[i] >= thread:
            FN += 1
        if labels_all[i] >= thread and predicts_all[i] < thread:
            FP += 1
        if labels_all[i] < thread and predicts_all[i] < thread:
            TN += 1
    precesion = TP/(TP+FP)
    recall = TP/(TP+FN)
    acc = (TP+TN)/(TP+TN+FP+FN)
    F1 = 2*precesion*recall/(precesion+recall)
    print(TP,TN,FP,FN,F1)



def test(args, model, tokenizer):
    """ Test the model """
    #get testing dataset
    test_dataset = WarningtestDataset(tokenizer, args, f"{args.data_file}//warning_test.txt")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle = False)
    
    model.eval()
    count = 0
    predicts_list = []
    warning_jsons_list = []
    predicts_score_list = []

    for batch in test_dataloader:
        #get inputs
        code_inputs = batch[0].to(args.device)    
        nl_inputs = batch[1].to(args.device)
        warning_json = batch[2][0]
        #get code and nl vectors
        with torch.no_grad():
            predicts = model(code_inputs=code_inputs, nl_inputs=nl_inputs)
            predicts_score = predicts.max(1)[0]
            predicts = predicts.max(1)[1]
            
            predicts_list.extend(predicts.tolist())
            warning_jsons_list.append(warning_json)
            predicts_score_list.extend(predicts_score.tolist())

    high_id, medium_id, low_id, false_id = [], [], [], []
    high_score, medium_score, low_score, false_score = [], [], [], []

    for i in range(len(predicts_list)):
        if predicts_list[i] == 0:
            false_id.append(i)
            false_score.append(predicts_score_list[i])
        if predicts_list[i] == 1:
            low_id.append(i)
            low_score.append(predicts_score_list[i])
        if predicts_list[i] == 2:
            medium_id.append(i)
            medium_score.append(predicts_score_list[i])
        if predicts_list[i] == 3:
            high_id.append(i)
            high_score.append(predicts_score_list[i])

        top = []
        tmp_id = get_max_id(high_score, len(high_score))
        top.extend([high_id[i] for i in tmp_id])
        tmp_id = get_max_id(medium_score, len(medium_score))
        top.extend([medium_id[i] for i in tmp_id])
        tmp_id = get_max_id(low_score, len(low_score))
        top.extend([low_id[i] for i in tmp_id])
        tmp_id = get_max_id(false_score, len(false_score))
        tmp_id.reverse()
        top.extend([false_id[i] for i in tmp_id])

    with open(f'{args.data_file}/res_rank.txt',"a+")as f:
        for id in top:
            f.write(warning_jsons_list[id])








def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default="/home/zhipengxue/warning_elimination/dataset", type=str, 
                        help="The input training data file.")
    parser.add_argument("--output_dir", default="/data1/zhipengxue/model_rank", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--num_reload", default=12, type=int,
                        help="Reload num.")#11#5#8#12MRR
    
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

    parser.add_argument('--oversampling', type=int, default=5,
                        help="oversampling for true")
    parser.add_argument('--undersampling', type=float, default=0.2,
                        help="undersampling for false")


    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
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

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
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

    reload_dir = f'{args.output_dir}/unixcoder_{args.num_reload}'
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(reload_dir)) 
    model.to(args.device)

    val(args, model, tokenizer)
    # identify_val(args, model, tokenizer)
    # test(args, model, tokenizer)
    

if __name__ == "__main__":
    main()