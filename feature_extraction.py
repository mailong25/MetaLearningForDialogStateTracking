class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, segment_ids, overall_id, start_id, 
                 end_id, slot_embedding, valid_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.overall_id = overall_id
        self.start_id = start_id
        self.end_id = end_id
        self.valid_ids = valid_ids
        self.slot_embedding = slot_embedding

def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, schema_embedding):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index,example) in enumerate(examples):
        
        agentTokens = example['agent'].split(' ')
        domain = example['domain']
        slots = example['slot']
        userTokens = example['user'].split(' ')
        
        input_ids = [tokenizer._convert_token_to_id('[CLS]')]
        attention_mask = [1]
        segment_ids = [0]
        valid_ids = [1]
        all_tokens = ['[CLS]']
        
        for token in agentTokens:
            sub_tokens = tokenizer._tokenize(token)
            valid_ids.append(1)
            for sub in sub_tokens:
                input_ids.append(tokenizer._convert_token_to_id(sub))
                attention_mask.append(1)
                segment_ids.append(0)
                valid_ids.append(0)
                all_tokens.append(sub)
            valid_ids = valid_ids[:-1]

        input_ids.append(tokenizer._convert_token_to_id('[SEP]'))
        attention_mask.append(1)
        segment_ids.append(0)
        valid_ids.append(1)
        all_tokens.append('[SEP]')

        for token in userTokens:
            sub_tokens = tokenizer._tokenize(token)
            valid_ids.append(1)
            for sub in sub_tokens:
                input_ids.append(tokenizer._convert_token_to_id(sub))
                attention_mask.append(1)
                segment_ids.append(1)
                valid_ids.append(0)
                all_tokens.append(sub)
            valid_ids = valid_ids[:-1]

        input_ids.append(tokenizer._convert_token_to_id('[SEP]'))
        attention_mask.append(1)
        segment_ids.append(1)
        valid_ids.append(1)
        all_tokens.append('[SEP]')

        #Add padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            segment_ids.append(0)
            valid_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(valid_ids) == max_seq_length
        
        for name in slots:
            slot_embedding = schema_embedding[domain + '.' + name]
            overall_id, start_id, end_id = (-1, -1, -1)
            
            if len(slots[name]) == 0:
                overall_id = label_map['no']
            else:
                if slots[name]['value'] == 'dontcare':
                    overall_id = label_map['dontcare']
                else:
                    overall_id = label_map['span']
                    start_offset = slots[name]['start']
                    end_offset   = slots[name]['end']
                    
                    if start_offset == 0:
                        start_id = 0
                    else:
                        start_id = len(example['user'][:start_offset-1].split(' '))
                    
                    end_id = start_id + slots[name]['value'].count(' ')
                    
                    start_id += len(agentTokens) + 1
                    end_id   += len(agentTokens) + 1

            features.append(InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  segment_ids=segment_ids,
                                  overall_id=overall_id,
                                  start_id=start_id,
                                  end_id=end_id,
                                  slot_embedding = slot_embedding,
                                  valid_ids=valid_ids))
    return features
