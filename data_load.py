import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, CLS, SEP, UNK, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS
from utils import build_vocab,  word_embedding_fun
from pytorch_pretrained_bert import BertTokenizer

# init vocab初始化词汇表
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)

#words
# file_path = 'glove.twitter.27B.50d.txt'
# all_words, all_vector, word2idx , idx2word ,words_vector = get_word_vector(file_path)
embed_path = '.vector_cache/glove/glove.6B.50d.txt'
word_embedding, word_x_2d, word2idx, idx2word, embedding_dim, all_words = word_embedding_fun(embed_path=embed_path, over_writte=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))


class ACE2005Dataset(data.Dataset):#数据类
    def __init__(self, fpath):
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li = [], [], [], [], []#sent：句子，sentence；arguments：元素
        self.word_li = []
        with open(fpath, 'r') as f:
            data = json.load(f)# dict:"sentence"..."word"
            for item in data:#json数据包括了：words，拆分sentence
                words = item['words']
                entities = [[NONE] for _ in range(len(words))]
                triggers = [NONE] * len(words)
                postags = item['pos-tags']#词性
                arguments = {#argument，创建空词典
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }

                for entity_mention in item['golden-entity-mentions']:#在data里，建立arguments，在candidates中添加三个
                    arguments['candidates'].append((entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))#data里的词典entity_mention

                    for i in range(entity_mention['start'], entity_mention['end']):
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)#format：将entity_type填入{}中
                        else:
                            entity_type = 'I-{}'.format(entity_type)

                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            entities[i].append(entity_type)

                for event_mention in item['golden-event-mentions']:#data里的词典
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):#trigger
                        trigger_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)

                    event_key = (event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:#arguments词典
                        role = argument['role']
                        if role.startswith('Time'):#startswith() 方法用于检查字符串是否是以指定子字符串开头
                            role = role.split('-')[0]
                        arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))

                self.sent_li.append([CLS] + words + [SEP])
                self.entities_li.append([[PAD]] + entities + [[PAD]])
                self.postags_li.append([PAD] + postags + [PAD])
                self.triggers_li.append(triggers)
                self.arguments_li.append(arguments)

                self.word_li.append([PAD]+ words +[UNK])

    def __len__(self):#返回长度
        return len(self.sent_li)

    def __getitem__(self, idx):#把类中的属性定义为序列，可以使用__getitem__()函数输出序列属性中的某个元素，这个方法返回与指定键关联的值
        words, entities, postags, triggers, arguments = self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], self.triggers_li[idx], self.arguments_li[idx]
        #这对应json数据里的词典
        words_lstm = self.word_li[idx]
        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        words_lstm_x = []
        # print(words)
        # print("=========")
        # print(postags)
        # tmp = zip(words, entities, postags, words_lstm)
        # tmp2= zip(words, entities, postags)
        # print(tmp)
        # print("++++++++")
        # print(tmp2)
        for w, e, p ,w_lstm in zip(words, entities, postags, words_lstm):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            # # 将句子拆分为word,即tokens
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens) # words被转化为了id，即tokens_x
            # print(tokens_xx)
            # words_lstm = tokenizer.tokenize(w_lstm) if w_lstm not in [PAD,UNK] else [w_lstm]
            # w_lstm = [word2idx[word] for word in words_lstm]


            # word = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            # word_xx = tokenizer.convert_ids_to_tokens(word)
            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(w) - 1)#补0

            p = [p] + [PAD] * (len(w) - 1)
            e = [e] + [[PAD]] * (len(w) - 1)  # <PAD>: no decision,因为entity是词典，与postag不同
            p = [postag2idx[postag] for postag in p]
            e = [[entity2idx[entity] for entity in entities] for entities in e]


            #用法 append（）用于在列表末尾添加新的对象，输入参数为对象； extend（）用于在列表末尾追加另一个序列中的多个值，输入对象为元素队列
            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)
            # words_lstm_x.extend(w_lstm)


        triggers_y = [trigger2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, words_lstm_x, triggers_y, arguments, seqlen, head_indexes, words, triggers

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def pad(batch):
    tokens_x_2d, entities_x_3d, postags_x_2d, words_lstm_x_2d ,triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = list(map(list, zip(*batch)))#矩阵转置
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        """[0]*以后为什么还有数值：[0]*得到的是一个张量，用来补齐batch"""
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))#补零语句，batch
        postags_x_2d[i] = postags_x_2d[i] + [0] * (maxlen - len(postags_x_2d[i]))
        words_lstm_x_2d[i] = words_lstm_x_2d[i] + [0] * (maxlen - len(words_lstm_x_2d[i]))#
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        #print(head_indexes_2d[i])
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))]


    return tokens_x_2d, entities_x_3d, postags_x_2d, words_lstm_x_2d,\
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d

