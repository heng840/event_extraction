import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import torchtext.vocab as vocab
import pickle

import bcolz
from transformers import BertModel
from data_load import idx2trigger, argument2idx, word_embedding, word_x_2d, word2idx, idx2word, embedding_dim,all_words
from consts import NONE
from utils import find_triggers


class Net(nn.Module):#Net类
    def __init__(self, trigger_size=None, entity_size=None, all_postags=None, postag_embedding_dim=50, argument_size=None,
                 entity_embedding_dim=50, device=torch.device("cpu"), all_words=None,word_size = None,
                 word_embedding_dim = embedding_dim, all_triggers=None, triggers_embedding_dim=50):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')#基于英文，区分大小写，bert类
        self.entity_embed = MultiLabelEmbeddingLayer(num_embeddings=entity_size, embedding_dim=entity_embedding_dim, device=device)#!
        self.postag_embed = nn.Embedding(num_embeddings=all_postags, embedding_dim=768+postag_embedding_dim)#!
        """Input: (*)(∗), IntTensor or LongTensor of arbitrary shape containing the indices to extract
            Output: (*, H)(∗,H), where * is the input shape and H=embedding_dim"""

        # num_embeddings个词，每个词用embedding_dim维词向量表示
        # 输入： LongTensor(N, W), N = mini - batch, W = 每个mini - batch中提取的下标数
        # 输出： (N, W, embedding_dim)
        self.trigger_embed=nn.Embedding(num_embeddings=all_triggers, embedding_dim=768+triggers_embedding_dim)
        #self.word_embed=nn.Embedding(num_embeddings=all_words, embedding_dim=768+words_embedding_dim)
        #word2d
        #初始化
        # num_embeddings (int) - 嵌入字典的大小，embedding_dim (int) - 每个嵌入向量的大小（两个非optional）
        # 这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。
        # 输入： LongTensor (N, W), N = mini-batch, W = 每个mini-batch中提取的下标数；输出： (N, W, embedding_dim)
        # self.lstm = nn.LSTM(bidirectional=True, num_layers=1, input_size=768, hidden_size=768 // 2, batch_first=True)
        #self.lstm = LSTMClass()
        # inputsize=818
        #循环卷积LSTM
        #batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
        # 需要input:(input,(h0,c0))  而output，(hn,cn)=lstm(input,(h0,c0))
        # batch_first=True:使得input和output=(batch, seq, feature)
        # 这里是对LSTM的初始化
        # 级联bert
        """级联的思路：首先用bert获取一个矩阵，其output输入到lstm中"""

        "glove"
        # glove = vocab.GloVe(name='twitter.27B', dim=50, cache=glove_dir)
        # print(glove.vectors.size())
        #使用词汇表的向量信息初始化nn.embedding

        self.word_emb = word_embedding
        self.lstm = nn.LSTM(bidirectional=True, num_layers=1, input_size=word_embedding_dim + 768,
                            hidden_size=768 // 2, batch_first=True)



        # hidden_size = 768 + entity_embedding_dim + postag_embedding_dim
        hidden_size = 768
        #各类卷积操作
        self.fc1 = nn.Sequential(#顺序容器。模块将按照它们在构造函数中传递的顺序添加到它
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
        )
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),#对输入数据做线性变换：y=Ax+b,学习weight和bias.A
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size),
        )
        self.device = device
        """问题：这里定义fc以后，后面传入tensor是怎么计算的"""

    def predict_triggers_LSTM(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d, words_x_2d):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)  # 变为一维，后续要截取enc,token即word
        # postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        # print(words_x_2d.shape)
        # words_x_2d = torch.LongTensor(words_x_2d.numpy()).to(self.device)

        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)  # 复制到cuda上,head_index:在json文件中，是包含了text，start，end的词典
        # 在data_load中，由__getitem__得到的列表

        # postags_x_2d = self.postag_embed(postags_x_2d)
        # entity_x_2d = self.entity_embed(entities_x_3d)
        # words_x_3d = self.word_emb(words_x_2d)
        # print(words_x_2d)
        # print(words_x_3d.shape)
        tokens_x_3d = self.word_emb(tokens_x_2d)

        # model = self.lstm()
        output,(hn,cn) = self.lstm(tokens_x_3d)

        """"enc = encoded_layers[-1]，这也是错误的提取，bert输出的late_hidden_state应该是三维的。问题：为什么要用到三维？✔"""

    # else:
    #     self.bert.eval()
    #     with torch.no_grad():
    #         model = self.bert(tokens_x_2d)
    #         encoded_layers = model[0]
    #         output,(hn,cn) = self.lstm(encoded_layers)

                # enc = encoded_layers[-1]
        # x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        # x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]
        # x = encoded_layers
        # print(encoded_layers.shape)
        # logits = self.fc2(x + enc)


        batch_size = tokens_x_2d.shape[0]

        trigger_logits = self.fc_trigger(output)  # nn.linear()是用来设置网络中的全连接层的
        trigger_hat_2d = trigger_logits.argmax(-1)

        "以下预测了argument，可以从predict_triggers中独立，用到了data_load里的idx2trigger"
        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = output[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = output[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys


    "predict_triggers也用到了arguments,但这应该是为了下一步的预测。"
    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d):#预测触发词
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)#变为一维，后续要截取enc
        # postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)#复制到cuda上,head_index:在json文件中，是包含了text，start，end的词典
                                                                            #在data_load中，由__getitem__得到的列表

        # postags_x_2d = self.postag_embed(postags_x_2d)
        # entity_x_2d = self.entity_embed(entities_x_3d)

        if self.training:
            self.bert.train()#train的用法
            """encoded_layers=self.bert(tokens_x_2d)是错误的用法，要整个用bert(*),再提取第几维"""
            model= self.bert(tokens_x_2d)#bert()方法输入的是idx索引，唯一的输入非optional，词汇中输入序列令牌的索引：input_ids (torch.LongTensor of shape (batch_size, sequence_length))
            encoded_layers=model[0]#输出有两个非optional：[0]:last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
                                   #

            """"enc = encoded_layers[-1]，这也是错误的提取，bert输出的late_hidden_state应该是三维的。问题：为什么要用到三维？✔"""

            #print(enc)
        else:
            self.bert.eval()
            with torch.no_grad():
                model = self.bert(tokens_x_2d)
                encoded_layers = model[0]

                #enc = encoded_layers[-1]
        # x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        # x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]
        x = encoded_layers
        print(encoded_layers.shape)
        # logits = self.fc2(x + enc)

        batch_size = tokens_x_2d.shape[0]

        for i in range(batch_size):#分batch进行。
            #print(x[i].shape)，输出：torch.Size([74, 768])，batch_size=24，x[i]即是x按照第一维选取的
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])#is_head则选取（句子长度+维度）

        trigger_logits = self.fc_trigger(x)#nn.linear()是用来设置网络中的全连接层的
        trigger_hat_2d = trigger_logits.argmax(-1)

        "以下预测了argument，可以从predict_triggers中独立，用到了data_load里的idx2trigger"
        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = x[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = x[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

    def predict_triggers_bert2lstm(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d):#级联预测
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)#变为一维，后续要截取enc
        postags_x_2d = torch.LongTensor(postags_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)#复制到cuda上,head_index:在json文件中，是包含了text，start，end的词典
                                                                            #在data_load中，由__getitem__得到的列表

        postags_x_2d = self.postag_embed(postags_x_2d)
        entity_x_2d = self.entity_embed(entities_x_3d)
        """if else语句：training==true，去查看，始终没有变化。
        注释了index_select语句，变量不会改变，使得loss.backword()可用"""
        # if self.training:
        #     self.bert.train()#train的用法
        """encoded_layers=self.bert(tokens_x_2d)是错误的用法，要整个用bert(*),再提取第几维"""
        model_bert= self.bert(tokens_x_2d)#bert()方法输入的是idx索引，唯一的输入非optional，词汇中输入序列令牌的索引：input_ids (torch.LongTensor of shape (batch_size, sequence_length))
        encoded_layers = model_bert[0]#输出有两个非optional：[0]:last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
                               #
        print(encoded_layers.shape)
        # self.lstm.train()

        model_lstm, (hn,cn) = self.lstm(encoded_layers)#N,L,H  when batch_first=True    batch_size,sequence_len,input_size

        """"enc = encoded_layers[-1]，这也是错误的提取，bert输出的late_hidden_state应该是三维的。问题：为什么要用到三维？✔"""

        #print(enc)
        # else:
        #     self.bert.eval()
        #     with torch.no_grad():
        #         model_bert = self.bert(tokens_x_2d)
        #         encoded_layers = model_bert[0]
        #     self.lstm.eval()
        #     with torch.no_grad():
        #         model_lstm, (hn,cn) = self.lstm(encoded_layers)

                #enc = encoded_layers[-1]
        # x = torch.cat([enc, entity_x_2d, postags_x_2d], 2)
        # x = self.fc1(enc)  # x: [batch_size, seq_len, hidden_size]
        #x = encoded_layers
        # logits = self.fc2(x + enc)
        # x = model_lstm

        batch_size = tokens_x_2d.shape[0]

        outputs = []
        # for i in range(batch_size):#分batch进行。
        #     #print(x[i].shape)，输出：torch.Size([74, 768])，batch_size=24，x[i]即是x按照第一维选取的
        #     model_lstm[i] = torch.index_select(model_lstm[i], 0, head_indexes_2d[i])#is_head则选取（句子长度+维度）

        trigger_logits = self.fc_trigger(model_lstm)#nn.linear()是用来设置网络中的全连接层的
        trigger_hat_2d = trigger_logits.argmax(-1)

        "以下预测了argument，可以从predict_triggers中独立，用到了data_load里的idx2trigger"
        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = model_lstm[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = model_lstm[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):#预测输入
        argument_hidden = torch.stack(argument_hidden)#stack：沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状，即是扩维拼接
        argument_logits = self.fc_argument(argument_hidden)#linear
        argument_hat_1d = argument_logits.argmax(-1)#返回最大值

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:#与json文件可以对应
            a_label = argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys, argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):#定义一个卷积层，在entity_embed中应用
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)#一个保存了固定字典和大小的简单查找表
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x

"""Glove_word_emd"""


def get_glove_vector(txt_file_path):
    embeddings_dict = {}
    with open(file=txt_file_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector#dict
    print(embeddings_dict)




class LSTMClass(nn.Module):
    def __init__(self):
        self.rnn_cell = nn.LSTM(bidirectional=True, num_layers=1, input_size=768, hidden_size=768 // 2, batch_first=True)
        self.hidden = self.init_hidden()
        super(LSTMClass,self).__init__()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


    def forward(self, x):

        self.hidden = self.init_hidden()  #
        out, self.hidden = self.rnn_cell(x, self.hidden)
        return out

