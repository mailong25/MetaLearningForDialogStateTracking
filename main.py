import  torch
import os
import  numpy as np
from    metadataset import MetaDataset
from    torch.utils.data import DataLoader
import  random, sys, pickle, time
from transformers import BertTokenizer
from random import shuffle
from reptile import Meta
#from maml import Meta
import json

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]

class Args:
    def __init__(self):
        self.meta_epoch=10
        self.k_spt=60
        self.k_qry=30
        self.k_qry_test = 30
        self.task_num=3
        self.meta_lr=8e-5
        self.inner_update_lr = 8e-5
        self.inner_update_step = 10
        self.inner_eval_update_step = 40
        self.bert_model = 'bert-base-uncased'
        self.train_batchsz = 1000
        self.test_batchsz = 5
        self.inner_train_batch_size = 24
        self.max_grad_norm = 1.0
        self.adam_eps = 1e-8
        self.save_path = './models'

args = Args()

def compute_joint_f1(tasks_metrics):
    #tasks_metrics = [[0.5, 3, 5, 10], [0.5, 3, 5, 10], [0.5, 3, 5, 10]]
    true_predict = sum([metric[1] for metric in tasks_metrics])
    total_predict = sum([metric[2] for metric in tasks_metrics])
    total_truth = sum([metric[3] for metric in tasks_metrics])
    if total_predict==0 or true_predict==0:
        return 0
    precision = float(true_predict) / total_predict
    recall    = float(true_predict) / total_truth
    f1 = 2 * precision * recall / (precision + recall)
    return f1


tokenizer = BertTokenizer.from_pretrained(args.bert_model, lower_case = False, do_lower_case = False)
mini_test = MetaDataset(data_path = './dataset', batchsz = args.test_batchsz, k_shot = args.k_spt, 
                 k_query = args.k_qry, k_query_test = args.k_qry_test, tokenizer = tokenizer, mode = 'test')
mini = MetaDataset(data_path = './dataset', batchsz = args.train_batchsz, k_shot = args.k_spt, 
              k_query = args.k_qry, k_query_test = args.k_qry_test, tokenizer = tokenizer, mode = 'train')

maml = Meta(args)

global_step = 0

EXP_NAME = 'reptile'
LOG_NAME = 'logs/' + EXP_NAME + '.txt'
SAVE_PATH = 'model_logs/' + EXP_NAME
OPTIMIZER_PATH = SAVE_PATH + '-optimizer'
os.mkdir(OPTIMIZER_PATH)

for epoch in range(args.meta_epoch):
    
    train_f1 = []
    db = create_batch_of_tasks(mini, is_shuffle = True, batch_size = args.task_num)

    for step, task_batch in enumerate(db):
        
        start = time.time()
        f = open(LOG_NAME, 'a')
        
        tasks_metrics = maml(task_batch)
        print(tasks_metrics)
        train_f1.append(tasks_metrics)

        print('Step:', step, '\ttraining F1:', compute_joint_f1(tasks_metrics))
        f.write('F1 : ' + str(compute_joint_f1(tasks_metrics)) + '\n')

        if global_step % 40 == 0:
            random_seed(123)
            print("\n-----------------Testing Mode-----------------\n")
            db_test = create_batch_of_tasks(mini_test, is_shuffle = False, batch_size = 1)
            f1_all_test = []

            for test_batch in db_test:
                f1s = maml(test_batch, training = False)
                f1_all_test.append(f1s[0])

            test_f1 = compute_joint_f1(f1_all_test)
            print('Test F1:', test_f1)
            f.write('Test F1 : ' + str(test_f1) + '\n')
            
            random_seed(int(time.time() % 10))
            
        global_step += 1
        
        print("Elapse time: ", time.time() - start)
        f.close()
        
        if global_step % 40 == 0:
            save_dir = SAVE_PATH + str(global_step)
            os.mkdir(save_dir)
            model_to_save = (maml.model.module if hasattr(maml.model, "module") else maml.model)
            model_to_save.save_pretrained(save_dir)
            torch.save(args, os.path.join(save_dir, "training_args.bin"))
            torch.save(maml.meta_optimizer, os.path.join(OPTIMIZER_PATH, "optimizer.pt"))
    
    train_f1 = [j for i in train_f1 for j in i]
    train_f1 = compute_joint_f1(train_f1)
    
    f = open(LOG_NAME, 'a')
    f.write("AVG-Train F1: " + str(train_f1) + '\n')
    print('---------- Next iteration ----------\n')
    f.write('---------- Next iteration ----------\n')
    f.close()