import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import json, pickle
from torch.utils.data import TensorDataset
from collections import Counter

from feature_extraction import convert_examples_to_features

random.seed(42)

LABEL_MAP  = {'no':0, 'dontcare':1, 'span':2, 0:'no', 1:'dontcare', 2:'span'}

class MetaDataset(Dataset):
    """
    put mini-imagenet files as :
    data_path :
        |- train.json includes all trainning samples
        |- train.json includes all test samples
        |- schema_embedding.pkl: dictionary of {slot_embedding : numpy vector}
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains k_shot for meta-train set, k_query for meta-test set.
    """

    def __init__(self, data_path, batchsz, k_shot, k_query, k_query_test, tokenizer, mode = None):
        """
        :param examples: list of samples
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param k_shot:
        :param k_query
        """
        self.mode = mode
        
        with open(os.path.join(data_path, self.mode + '.json')) as f:
            self.examples = json.load(f)
            random.shuffle(self.examples)
        
        with open(os.path.join(data_path, 'target.json')) as f:
            self.targets = json.load(f)
            
        with open(os.path.join(data_path, 'schema_embedding.pkl'),'rb') as f:
            self.schema_embedding = pickle.load(f)
        
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.k_query_test = k_query_test
        self.tokenizer = tokenizer
        self.max_seq_length = 100

        self.create_batch(self.batchsz)

    def create_feature_set(self,examples):
        
        flatten_features = convert_examples_to_features(examples, LABEL_MAP, self.max_seq_length, 
                                                        self.tokenizer, self.schema_embedding)
        
        all_input_ids = torch.tensor([f.input_ids for f in flatten_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in flatten_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in flatten_features], dtype=torch.long)
        all_overall_ids = torch.tensor([f.overall_id for f in flatten_features], dtype=torch.long)
        all_start_ids = torch.tensor([f.start_id for f in flatten_features], dtype=torch.long)
        all_end_ids = torch.tensor([f.end_id for f in flatten_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in flatten_features], dtype=torch.long)
        all_slot_embeddings = torch.tensor([f.slot_embedding for f in flatten_features], dtype=torch.float32)
        
        tensor_set = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_valid_ids,
                                    all_slot_embeddings, all_overall_ids, all_start_ids, all_end_ids)
        
        return tensor_set
        
    
    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        
        for b in range(batchsz):  # for each batch
            # 1.select domain randomly
            randIdx = random.choice(range(0,len(self.examples)))
            domain  = self.examples[randIdx]['domain']
            domainExamples = [e for e in self.examples if e['domain'] == domain]
            
            # 1.select k_shot + k_query examples from domain randomly
            if self.mode == 'train':
                selected_examples = random.sample(domainExamples,self.k_shot + self.k_query)
            else:
                selected_examples = random.sample(domainExamples,self.k_shot + self.k_query_test)
            
            random.shuffle(selected_examples)
            exam_Dtrain = selected_examples[:self.k_shot]
            exam_Dtest  = selected_examples[self.k_shot:]
            
            #Adding target examples into meta-test set
            if self.mode == 'train':
                exam_Dtest.extend(self.targets)
            
            self.support_x_batch.append(exam_Dtrain)
            self.query_x_batch.append(exam_Dtest)

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        support_set = self.create_feature_set(self.support_x_batch[index])
        query_set   = self.create_feature_set(self.query_x_batch[index])
        
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz
