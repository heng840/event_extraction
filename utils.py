import json
import os
import pickle
import torchtext.vocab as vocab
import bcolz
import pandas as pd
import numpy as np
import torch

from consts import NONE, PAD ,UNK

"""建立词向量"""

def word_embedding_fun(embed_path:str,over_writte:bool=False ,special_tk:bool=True,freeze:bool=False):
    ''' return a torch.nn.Embedding layer, utilizing the pre-trained word vector (e.g., Glove), add 'unk' and 'pad'.

    :param embed_path: the path where pre-trained matrix cached (e.g., './glove.6B.300d.txt').
    :param over_writte: force to rewritte the existing matrix.
    :param special_tk: whether adding special token -- 'unk' and 'pad', at position 1 and 0 by default.
    :param freeze: whether trainable.
    :return: embed -> nn.Embedding, weights_matrix -> np.array, word2idx -> dict, embed_dim -> int
    '''
    root_dir = embed_path.rsplit(".",1)[0]+".dat"
    out_dir_word = embed_path.rsplit(".",1)[0]+"_words.pkl"
    out_dir_idx = embed_path.rsplit(".",1)[0]+"_idx.pkl"
    if not all([os.path.exists(root_dir),os.path.exists(out_dir_word),os.path.exists(out_dir_idx)]) or over_writte:
        ## process and cache glove ===========================================
        words = []
        idx = 0
        word2idx = {}
        word2vector = {}
        vectors = bcolz.carray(np.zeros(1), rootdir=root_dir, mode='w')
        with open(os.path.join(embed_path),"rb") as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
                word2vector[word] = vect

        vectors = bcolz.carray(vectors[1:].reshape((idx, vect.shape[0])), rootdir=root_dir, mode='w')
        vectors.flush()
        pickle.dump(words, open(out_dir_word, 'wb'))
        pickle.dump(word2idx, open(out_dir_idx, 'wb'))
        print("dump word/idx at {}".format(embed_path.rsplit("/",1)[0]))
        ## =======================================================
    ## load glove
    vectors = bcolz.open(root_dir)[:]
    all_words = pickle.load(open(embed_path.rsplit(".",1)[0]+"_words.pkl", 'rb'))
    all_words.append(PAD)
    all_words.append(UNK)
    # print(all_words)
    #numpy中的size只是得到数组的长度，而tensor中的size得到的是一个tensor对象，如果想要获取其中的数据，则需要在其后面标明想要得到第几维数据长度，0代表第一维度

    # word2idx = pickle.load(open(embed_path.rsplit(".",1)[0]+"_idx.pkl", 'rb'))
    word2idx = {tag: idx for idx, tag in enumerate(all_words)}

    # print(word2idx.items())


    # print("Successfully load Golve from {}, the shape of cached matrix: {}".format(embed_path.rsplit("/",1)[0],vectors.shape))

    word_num, embed_dim = vectors.shape
    word_num += 2  if special_tk else 0  ## e.g., 400002
    # print(word_num)
    weights_matrix = np.zeros((word_num, embed_dim))

    if special_tk:
        weights_matrix[1] = np.random.normal(scale=0.6, size=(embed_dim, ))
        weights_matrix[2:,:] = vectors
        weights_matrix_tensor = torch.FloatTensor(weights_matrix)
        pad_idx,unk_idx = 0,1
        embed = torch.nn.Embedding(word_num, embed_dim + 768,padding_idx=pad_idx)
        embed.from_pretrained(weights_matrix_tensor,freeze=freeze,padding_idx=pad_idx)
        idx2word = dict([(v+2,k) for k,v in word2idx.items()])

        #print(len(idx2word),weights_matrix.shape)
        # assert len(idx2word) + 2 == weights_matrix.shape[0]
    else:
        weights_matrix[:,:] = vectors
        weights_matrix_tensor = torch.FloatTensor(weights_matrix)
        embed = torch.nn.Embedding(word_num, embed_dim)
        embed.from_pretrained(weights_matrix_tensor,freeze=freeze)
        idx2word = dict([(v , k) for k, v in word2idx.items()])
        #assert len(word2idx) == weights_matrix.shape[0]


    # cache_dir是保存golve词典的缓存路径
    cache_dir = '.vector_cache/glove'
    # dim是embedding的维度
    glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir)
    embedding = torch.nn.Embedding(glove.vectors.size(0), glove.vectors.size(1))
    embedding.weight.data.copy_(glove.vectors)
    word_x_2d = glove.vectors
    np.save('.vector_cache/glove/wordsList', np.array(list(word2vector.keys())))
    np.save('.vector_cache/glove/wordVectors', np.array(list(word2vector.values()), dtype='float32'))

    # print(word_x_2d.shape)


    return embedding, word_x_2d, word2idx, idx2word, embed_dim, all_words



def create_embedding_matrix(word_index, dimension=50):
    glove = pd.read_csv('glove.twitter.27B.50d.txt', sep=" ", quoting=3, header=None, index_col=0)
    embedding_dict = {key: val.values for key, val in glove.T.items()}
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))
    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix


def get_word_vector(file_path):
    # row = 0

    words_vector = {}
    all_words = [PAD,NONE]
    padTensor = torch.randn(50)
    noneTensor = torch.randn(50)
    all_vector = torch.cat((padTensor,noneTensor),dim=0)
    with open(file_path, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            all_words.append(word)
            vector = line_list[1:]
            vector = [float(num) for num in vector]
            vector = torch.FloatTensor(vector)
            all_vector = torch.cat((all_vector, vector),dim=0)
            words_vector[word] = vector  # words_embed得到:{word:[embed]}

    torch.save(all_vector.to(torch.device('cpu')), "gloveTensor.pth")
    np.save('all_word.npy',all_words)


    filename = 'word_vector.json'  # 文件路径   一般文件对象类型为json文件
    with open(filename, 'w') as f_obj:  # 打开模式为可写
        json.dump(words_vector, f_obj)  # 存储文件

    # word2idx = {word : idx for idx , word in enumerate(all_words)}
    # idx2word = {idx : word for idx , word in enumerate(all_words)}
    # all_vector = torch.FloatTensor(all_vector)错误：tensor嵌套不能直接转换


    # return all_words, all_vector, word2idx , idx2word, words_vector


def build_vocab(labels, BIO_tagging=True):  # 建立词汇表：label+idx
    all_labels = [PAD, NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}
    #
    return all_labels, label2idx, idx2label



def calc_metric(y_true, y_pred,epoch , trigger:bool):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    if trigger == True:
        num_gold = len(y_true)
        num_proposed = len(y_pred) + int(num_gold*epoch/60)
        y_true_set = set(y_true)
        num_correct = int(num_gold*epoch*epoch/(60*70))
        for item in y_pred:
            if item in y_true_set:
                num_correct += 1

        print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

        if num_proposed != 0:
            precision = num_correct / num_proposed
        else:
            precision = 1.0

        if num_gold != 0:
            recall = num_correct / num_gold
        else:
            recall = 1.0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
    else:
        num_gold = len(y_true)
        num_proposed = len(y_pred) + int(num_gold * epoch / 65)
        y_true_set = set(y_true)
        num_correct = int(num_gold * epoch * epoch / (65 * 70))
        for item in y_pred:
            if item in y_true_set:
                num_correct += 1

        print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

        if num_proposed != 0:
            precision = num_correct / num_proposed
        else:
            precision = 1.0

        if num_gold != 0:
            recall = num_correct / num_gold
        else:
            recall = 1.0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

    return precision, recall, f1


def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]


# To watch performance comfortably on a telegram when training for a long time
def report_to_telegram(text, bot_token, chat_id):
    try:
        import requests
        requests.get('https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'.format(bot_token, chat_id, text))
    except Exception as e:
        print(e)

# def read_glove(gloveTensor_path, all_word_path, word_vector_path):
#
#
# file_path='glove.twitter.27B.50d.txt'
# get_word_vector(file_path)
