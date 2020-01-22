import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader, RandomSampler
from    torch import optim
import  numpy as np
from torch.optim import Adam, SGD
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from learner import BertForStateTracking
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
import time
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import gc

LABEL_MAP  = {'no':0, 'dontcare':1, 'span':2, 0:'no', 1:'dontcare', 2:'span'}

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Meta, self).__init__()
        self.inner_update_step = args.inner_update_step
        self.inner_eval_update_step = args.inner_eval_update_step
        self.inner_update_lr = args.inner_update_lr
        self.meta_lr = args.meta_lr
        self.inner_train_batch_size = args.inner_train_batch_size
        self.bert_model = args.bert_model
        self.train_batchsz = args.train_batchsz
        self.meta_epoch = args.meta_epoch
        self.max_grad_norm = args.max_grad_norm
        self.adam_eps = args.adam_eps
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForStateTracking.from_pretrained(self.bert_model,cache_dir='',
                                                          num_labels = int(len(LABEL_MAP) / 2))
        
        if os.path.exists(os.path.join(self.bert_model, 'optimizer.pt')):
            print("Load existed optimizer")
            self.meta_optimizer = torch.load(os.path.join(self.bert_model, 'optimizer.pt'))
        else:
            self.meta_optimizer = Adam(self.model.parameters(), lr=self.meta_lr)
        
        self.loss_fn = CrossEntropyLoss()
        self.model.train()

    def forward(self, batch_tasks, training = True):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]
                 
        # support #TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_valid_ids,
                                 all_slot_embeddings, all_overall_ids, all_start_ids, all_end_ids)
        """ 
        
        task_f1s = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step if training else self.inner_eval_update_step

        for task_id, dataset in enumerate(batch_tasks):
            support = dataset[0]
            query   = dataset[1]
            
            fast_model = deepcopy(self.model)
            fast_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                  batch_size=self.inner_train_batch_size)
            
            inner_optimizer   = Adam(fast_model.parameters(), lr=self.inner_update_lr)
            fast_model.train()
            
            for i in range(0,num_inner_update_step):
                all_gate_loss = []
                all_span_loss = []
                for inner_step, batch in enumerate(support_dataloader):
                    
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, segment_ids, valid_id, slot_embedding, overall_id, start_id, end_id = batch
                    
                    gate_logits, start_logits, end_logits = fast_model(input_ids, attention_mask, segment_ids, None,
                                                                       valid_id, slot_embedding)
                    
                    total_loss, gate_loss, span_loss = self.compute_loss(gate_logits, start_logits, end_logits, 
                                                                         overall_id, start_id, end_id, valid_id)
                    
                    total_loss.backward()
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()
                    
                    all_gate_loss.append(gate_loss.item())
                    if span_loss.item() != 0: all_span_loss.append(span_loss.item())
                
                if i % 4 == 0:
                    print(round(np.mean(all_span_loss),3), round(np.mean(all_gate_loss),3))

            query_dataloader = DataLoader(query, sampler=RandomSampler(query), batch_size=int(len(query)/2) + 1)
            for query_batch in query_dataloader:
                query_batch = tuple(t.to(self.device) for t in query_batch)
                input_ids, attention_mask, segment_ids, valid_id, slot_embedding, overall_id, start_id, end_id = query_batch
                gate_logits, start_logits, end_logits = fast_model(input_ids, attention_mask, 
                                                                   segment_ids, None, valid_id, slot_embedding)

                if training:
                    q_loss, q_gate_loss, q_span_loss = self.compute_loss(gate_logits, start_logits, end_logits, 
                                                                         overall_id, start_id, end_id, valid_id)
                    q_loss.backward()
                    fast_model.to(torch.device('cpu'))

                    for i, params in enumerate(fast_model.parameters()):
                        if task_id == 0:
                            sum_gradients.append([deepcopy(params.grad)])
                        else:
                            sum_gradients[i].append(deepcopy(params.grad))
                    
                    fast_model.to(torch.device('cuda'))

                f1 = self.compute_f1(gate_logits, start_logits, end_logits, overall_id, start_id, end_id, valid_id)
                print(f1)
                task_f1s.append(f1)
            
            fast_model.to(torch.device('cpu'))
            del fast_model, inner_optimizer
            torch.cuda.empty_cache()
            gc.collect()
            
        if training:
            # Average gradient across task
            for i in range(0,len(sum_gradients)):
                sum_gradients[i] = [g for g in sum_gradients[i] if type(g) != type(None)]
                if len(sum_gradients[i]) == 0:
                    sum_gradients[i] = None
                else:
                    total_gradient = sum_gradients[i][0]
                    for j in range(1,len(sum_gradients[i])):
                        total_gradient += sum_gradients[i][j]
                    sum_gradients[i] = total_gradient / len(sum_gradients[i])

            #Assign gradient for original model and using optimizer to update the weights
            for i, params in enumerate(self.model.parameters()):
                params.grad = sum_gradients[i]

            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()
            
            del sum_gradients
            gc.collect()
            torch.cuda.empty_cache()
        
        return task_f1s

    def compute_loss(self, gate_logits, start_logits, end_logits, overall_id, start_id, end_id, valid_id):
        
        gate_loss = self.loss_fn(gate_logits, overall_id)
        total_loss = gate_loss
        span_loss, span_count = torch.Tensor([0]), 0

        for i in range(0,len(start_id)):
            if overall_id[i] == LABEL_MAP['span']:
                start_logit = start_logits[i][valid_id[i][1:] == 1]
                end_logit   = end_logits[i][valid_id[i][1:] == 1]
                start_loss = self.loss_fn(start_logit.unsqueeze(0), start_id[i].unsqueeze(0))
                end_loss   = self.loss_fn(end_logit.unsqueeze(0), end_id[i].unsqueeze(0))
                if span_count == 0:
                    span_loss = start_loss + end_loss
                else:
                    span_loss = span_loss + start_loss + end_loss
                span_count += 1
        
        if span_count > 0:
            span_loss = span_loss / (span_count * 2)
            total_loss = total_loss + span_loss
                    
        return total_loss, gate_loss, span_loss
    
    def compute_f1(self, gate_logits, start_logits, end_logits, overall_id, start_id, end_id, valid_id):
        
        total_predict, true_predict, total_ground_truth = 0, 0, 0
        gate_logits = F.softmax(gate_logits, dim = 1)
        predict_gate_id = torch.argmax(gate_logits, dim = 1)

        for i in range(0,len(predict_gate_id)):
            if overall_id[i] == predict_gate_id[i]:
                if overall_id[i] == LABEL_MAP['dontcare']:
                    true_predict += 1
                    total_predict += 1
                    total_ground_truth += 1
                elif overall_id[i] == LABEL_MAP['span']:
                    total_predict += 1
                    total_ground_truth += 1

                    start_logit = start_logits[i][valid_id[i][1:] == 1]
                    end_logit   = end_logits[i][valid_id[i][1:] == 1]
                    start_logit = torch.argmax(F.softmax(start_logit.unsqueeze(0))[0])
                    end_logit   = torch.argmax(F.softmax(end_logit.unsqueeze(0))[0])

                    if start_logit == start_id[i] and end_logit == end_id[i]:
                        true_predict += 1
            else:
                if overall_id[i] == LABEL_MAP['no']:
                    total_predict += 1
                elif overall_id[i] == LABEL_MAP['dontcare']:
                    total_ground_truth += 1
                    if predict_gate_id[i] == LABEL_MAP['span']:
                        total_predict += 1
                else:
                    total_ground_truth += 1
                    if predict_gate_id[i] == LABEL_MAP['dontcare']:
                        total_predict += 1
        
        if true_predict == 0 or total_predict == 0:
            return 0, 0, 1, total_ground_truth
        
        precision = float(true_predict) / total_predict    
        recall    = float(true_predict) / total_ground_truth
        f1 = 2 * precision * recall / (precision + recall)
        return f1 , true_predict, total_predict, total_ground_truth