import torch
from transformers import BertModel, BertTokenizer,BertPreTrainedModel
from torch.nn.functional import gelu, tanh, elu
import torch.nn.functional as F
import torch.nn as nn
from foward_bert import functional_bert

class BertForStateTracking(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super(BertForStateTracking, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        
        transition_hidden_size = config.hidden_size + 768*2
        last_hidden_size = 500
        #schema_embedding_size = 768

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        
        self.gate_linear1 = nn.Linear(transition_hidden_size, last_hidden_size)
        self.gate_linear2 = nn.Linear(last_hidden_size, config.num_labels)

        self.start_linear1 = nn.Linear(transition_hidden_size, last_hidden_size)
        self.start_linear2 = nn.Linear(last_hidden_size, 1)
        
        self.end_linear1 = nn.Linear(transition_hidden_size, last_hidden_size)
        self.end_linear2 = nn.Linear(last_hidden_size, 1)
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                valid_ids=None, schema_embedding = None, head_mask=None, inputs_embeds=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds, )[0]
                
        batch_size,max_len,feat_dim = outputs.shape
        schema_embedding = schema_embedding.unsqueeze(1).repeat(1,max_len,1)
        
        outputs = torch.cat((outputs,schema_embedding),dim = 2)
        #outputs  = (outputs + schema_embedding) / 2
        feat_dim = outputs.shape[-1]
        
        if outputs.is_cuda == True:
            outputs[valid_ids == 0] = torch.zeros(feat_dim, dtype=torch.float32).cuda()
        else:
            outputs[valid_ids == 0] = torch.zeros(feat_dim, dtype=torch.float32)
            
        outputs = self.dropout(outputs)
        
        r_gate  = outputs[:,0,:]
        r_start = outputs[:,1:,:]
        r_end   = outputs[:,1:,:]
        
        r_gate = gelu(self.gate_linear1(r_gate))
        r_gate = self.dropout(r_gate)
        r_gate = self.gate_linear2(r_gate)
        
        r_start = gelu(self.start_linear1(r_start))
        r_start = self.dropout(r_start)
        r_start = self.start_linear2(r_start)
        r_start = r_start.squeeze(dim=-1)
        
        r_end = gelu(self.end_linear1(r_end))
        r_end = self.dropout(r_end)
        r_end = self.end_linear2(r_end)
        r_end = r_end.squeeze(dim=-1)
        
        return r_gate, r_start, r_end
    
    def functional_forward(self, fast_weights, input_ids=None, attention_mask=None, token_type_ids=None, 
                           position_ids=None, valid_ids=None, schema_embedding = None, head_mask=None, 
                           inputs_embeds=None, is_train = True):
        
        outputs = functional_bert(fast_weights, self.config, input_ids=input_ids, attention_mask=attention_mask, 
                                  token_type_ids=token_type_ids, position_ids=position_ids,head_mask=head_mask, 
                                  inputs_embeds=inputs_embeds, encoder_hidden_states=None, 
                                  encoder_attention_mask=None, is_train = is_train)[0]
        
        batch_size,max_len,feat_dim = outputs.shape
        schema_embedding = schema_embedding.unsqueeze(1).repeat(1,max_len,1)
        
        outputs = torch.cat((outputs,schema_embedding),dim = 2)
        feat_dim = outputs.shape[-1]
        
        if outputs.is_cuda == True:
            outputs[valid_ids == 0] = torch.zeros(feat_dim, dtype=torch.float32).cuda()
        else:
            outputs[valid_ids == 0] = torch.zeros(feat_dim, dtype=torch.float32)
        
        outputs = F.dropout(outputs, p = 0.1, training = is_train)
        
        r_gate  = outputs[:,0,:]
        r_start = outputs[:,1:,:]
        r_end   = outputs[:,1:,:]
        
        r_gate = gelu(F.linear(r_gate, fast_weights['gate_linear1.weight'], fast_weights['gate_linear1.bias']))
        r_gate = F.dropout(r_gate,p=0.1, training = is_train)
        r_gate = F.linear(r_gate, fast_weights['gate_linear2.weight'], fast_weights['gate_linear2.bias'])
        
        r_start = gelu(F.linear(r_start, fast_weights['start_linear1.weight'], fast_weights['start_linear1.bias']))
        r_start = F.dropout(r_start,p=0.1, training = is_train)
        r_start = F.linear(r_start, fast_weights['start_linear2.weight'], fast_weights['start_linear2.bias'])
        r_start = r_start.squeeze()
        
        r_end = gelu(F.linear(r_end, fast_weights['end_linear1.weight'], fast_weights['end_linear1.bias']))
        r_end = F.dropout(r_end,p=0.1, training = is_train)
        r_end = F.linear(r_end, fast_weights['end_linear2.weight'], fast_weights['end_linear2.bias'])
        r_end = r_end.squeeze()
        
        return r_gate, r_start, r_end
