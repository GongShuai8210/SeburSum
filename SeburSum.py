import torch
import json

import glob
import numpy as np

from torch import nn
from tqdm import tqdm
from transformers import BertModel, RobertaModel
from pytorch_pretrained_bert import BertTokenizer
from collections import Counter


model_path = glob.glob("/opt/data/default/gongshuai/gongshuai/program/CL_checkpoint_3/cnndm062300")
data_file = "/opt/data/default/gongshuai/gongshuai/SCEDS/pacsum_cnndm/cnndm_lead/lead/test.ccnndm.bert.json"

encoder_model_total = ["Roberta", "Bert"]
encoder_model = encoder_model_total[1]
print("The encoder is ", encoder_model)
data = []
# index_neg = []
index_neg = 0
with open(data_file, 'r') as f:
    for line in f:
        example = json.loads(line)
        data.append(example)

for model in model_path:
    save_file = 'sup_test_reddit1.json'
    fout = open(save_file, 'w')

    # load the model
    if encoder_model == 'Bert':
        encoder = BertModel.from_pretrained(model)
        pad_id = 0  # for BERT
    else:
        encoder = RobertaModel.from_pretrained(model)
        pad_id = 1  # for Roberta
    encoder.to("cuda:1")

    hidden_size = 768
    final_score_total = []
    final_score_single = 0

    for ex_index in tqdm(range(len(data))):

        ex = data[ex_index]

        if len(ex['text'])< 5:
            continue
        indices_ = ex['indices']
        document_id = ex['text_id']
        candidate = ex['candidate_id']

        if encoder_model == 'Roberta':
            candidate_id = np.ones([len(candidate), len(max(candidate, key=lambda x: len(x)))], dtype=np.int)
        else:
            candidate_id = np.zeros([len(candidate), len(max(candidate, key=lambda x: len(x)))], dtype=np.int)

        for i, j in enumerate(candidate):
            candidate_id[i][0:len(j)] = j

        # cal_loss document_id
        # document_id = torch.tensor(document_id)
        # document_id = document_id.unsqueeze(0)
        # document_id = document_id.to("cuda:1")

        candidate_id = torch.tensor(candidate_id)
        candidate_id = candidate_id.unsqueeze(0)
        candidate_id = candidate_id.to("cuda:1")

        batch_size = document_id.size(0)


        # one document loss_total
        with torch.no_grad():
            # get candidate embedding

            candidate_id = candidate_id.view(-1, candidate_id.size(-1))
            candidate_num = candidate_id.size(0)
            input_mask = ~(candidate_id == pad_id)
            input_mask = input_mask.to("cuda:1")

            out = encoder(candidate_id, attention_mask=input_mask).pooler_output
            candidate_emb = out.view(batch_size, candidate_num, hidden_size)  # [batch_size, candidate_num, hidden_size]
            assert candidate_emb.size() == (batch_size, candidate_num, hidden_size)

            score = []
            for pos in range(candidate_num):
                score_pos = []
                index_neg = []

                positive = indices_[pos]
                # if len(positive) == 3:
                #     continue
                pos_emb = candidate_emb[:, pos:pos + 1]
                for neg in range(candidate_num):
                    negtive = indices_[neg]
                    set_negtive = set(negtive)
                    set_positive = set(positive)
                    is_negtive = set_negtive.isdisjoint(set_positive)

                    if (is_negtive and len(positive) == 3):
                        index_neg.append(neg)
                        neg_emb = candidate_emb[:, neg:neg + 1]
                        score_pos_neg = torch.cosine_similarity(pos_emb, neg_emb, dim=-1)
                        score.append(score_pos_neg)
                        break

                    if (is_negtive and len(positive) != 3):
                        index_neg.append(neg)
                        neg_emb = candidate_emb[:, neg:neg + 1]
                        score_pos_neg = torch.cosine_similarity(pos_emb, neg_emb, dim=-1)
                        score_pos.append(score_pos_neg)

                    if neg == candidate_num - 1:
                        if score_pos:
                            min_score = min(score_pos)
                            max_score = max(score_pos)
                            score.append(max_score)
                            # loss_total.append(sum(loss)/len(loss))
        
            score = score.index(max(score))


        src_txt_ = ex['text']
        tgt_ = ex['summary']
        ext_idx = ex['ext_idx']
        ext_idx = sorted(ext_idx)
        indices = ex['indices'][index_total]
        src_txt = []
        for ext_index in ext_idx:
            # if len(ext_idx) > len(src_txt_):
            #     continue
            src_txt.append(src_txt_[ext_index])
        tgt_txt_ = '<q>'.join([tt for tt in tgt_])
        data_dict = {'src_txt': src_txt, 'tgt_txt': tgt_txt_}
        fout.write(json.dumps(data_dict) + "\n")

    fout.close()

