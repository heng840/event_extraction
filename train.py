import os

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from model import Net

from data_load import ACE2005Dataset, pad, all_triggers, all_entities, all_postags, all_arguments, tokenizer,\
    word_embedding, word_x_2d, word2idx, idx2word, embedding_dim,all_words
from utils import report_to_telegram
from eval import eval
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #用在GPU上运行
# device_ids = [0,1]# 编号不同，123映射到编码里的012

"""那么使用nn.DataParallel后，事实上DataParallel也是一个Pytorch的nn.Module，那么你的模型和优化器都需要使用.module来得到实际的模型和优化
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
保存模型：
torch.save(net.module.state_dict(), path)
加载模型：
net=nn.DataParallel(Resnet18())
net.load_state_dict(torch.load(path))
net=net.module
优化器使用：
optimizer.step() --> optimizer.module.step()"""


def train(model, iterator, optimizer, criterion):#train(model, train_iter, optimizer, criterion)
    model.train()

    for i, batch in enumerate(iterator):#函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        tokens_x_2d, entities_x_3d, postags_x_2d, words_lstm_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = batch
        optimizer.zero_grad()#意思是把梯度置零
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module.predict_triggers_LSTM(tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                           postags_x_2d=postags_x_2d, words_lstm_x_2d= words_lstm_x_2d,
                                                                                                                           head_indexes_2d=head_indexes_2d,
                                                                                                                           triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d,)
        #来自model.py，预测触发词
        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))

        if len(argument_keys) > 0:
            argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(argument_hidden, argument_keys, arguments_2d)
            argument_loss = criterion(argument_logits, arguments_y_1d)
            loss = trigger_loss + 2 * argument_loss
            if i == 0:
                print("=====sanity check for arguments======")
                print('arguments_y_1d:', arguments_y_1d)
                print("arguments_2d[0]:", arguments_2d[0]['events'])
                print("argument_hat_2d[0]:", argument_hat_2d[0]['events'])
                print("=======================")
        else:
            loss = trigger_loss

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)#
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()#求导

        # optimizer.module.step()
        optimizer.step()


        if i == 0:
            print("=====sanity check======")
            print("tokens_x_2d[0]:", tokenizer.convert_ids_to_tokens(tokens_x_2d[0])[:seqlens_1d[0]])
            print("entities_x_3d[0]:", entities_x_3d[0][:seqlens_1d[0]])
            print("postags_x_2d[0]:", postags_x_2d[0][:seqlens_1d[0]])
            print("head_indexes_2d[0]:", head_indexes_2d[0][:seqlens_1d[0]])
            print("triggers_2d[0]:", triggers_2d[0])
            print("triggers_y_2d[0]:", triggers_y_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
            print('trigger_hat_2d[0]:', trigger_hat_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
            print("seqlens_1d[0]:", seqlens_1d[0])
            print("arguments_2d[0]:", arguments_2d[0])
            print("=======================")

        if i % 10 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))

        """if loss.item() <= min_loss_val:
            min_loss_val = loss.item()
            torch.save(model, "best_model.pt")
            print("the best_model has been saved")"""




if __name__ == "__main__":
    parser = argparse.ArgumentParser()#声明一个parser
    #parser.add_argument添加参数：各个参数的含义
    parser.add_argument("--batch_size", type=int, default=24)#batch_size=24
    parser.add_argument("--lr", type=float, default=0.00002)#学习率lr
    parser.add_argument("--n_epochs", type=int, default=50)#epoch=50
    parser.add_argument("--logdir", type=str, default="logdir")
    parser.add_argument("--trainset", type=str, default="data/train.json")#数据集
    parser.add_argument("--devset", type=str, default="data/dev.json")
    parser.add_argument("--testset", type=str, default="data/test.json")

    parser.add_argument("--telegram_bot_token", type=str, default="")
    parser.add_argument("--telegram_chat_id", type=str, default="")
    #用hp调用
    hp = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(
        device=device,
        trigger_size=len(all_triggers),#data_load里初始化得到的（调用utils中的build_vocab)
        entity_size=len(all_entities),
        argument_size=len(all_arguments),
        all_postags=len(all_postags),
        all_triggers=len(all_triggers),
        all_words=len(all_words)
    )
    if device == 'cuda':
        model = model.cuda()

    # model = nn.DataParallel(model,device_ids=device_ids)# 多GPU
    model = nn.DataParallel(model)

    train_dataset = ACE2005Dataset(hp.trainset)#创建实例:
    dev_dataset = ACE2005Dataset(hp.devset)
    test_dataset = ACE2005Dataset(hp.testset)

    samples_weight = train_dataset.get_samples_weight()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_iter = data.DataLoader(dataset=train_dataset,#ACE返回的量用pad方法
                                 batch_size=hp.batch_size,
                                 shuffle=False,#打乱
                                 sampler=sampler,# 报错：sampler option is mutually exclusive with shuffle，因为shuffle和sampler不能同时为真，如果sampler不为默认的None的时候，不用设置shuffle属性了
                                 num_workers=4,#线程，在Windows系统中，num_workers参数建议设为0，在Linux系统则不需担心
                                 collate_fn=pad)#
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    # optimizer = nn.DataParallel(optimizer,device_ids=device_ids)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(ignore_index=0)#loss函数

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    "保存模型以继续训练"
    start_epoch = 1
    log_dir = "best_model_LSTM.pt"
    log_dir_latest = "latest_model_LSTM.pt"
    # log_dir = "best_model_lr=0.00001.pt"
    # log_dir_latest="latest_model_lr=0.00001.pt"
    # if os.path.exists(log_dir_latest):
    #     checkpoint = torch.load(log_dir_latest)
    #     model.load_state_dict(checkpoint['model'])
    #     # model = model.module
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch']
    #     print('加载 epoch {} 成功！'.format(start_epoch))
    # else:
    #     start_epoch = 1
    print('无保存模型，将从头开始训练！')

    min_trigger_f1_dev = 0.1
    fname = None
    for epoch in range(start_epoch, hp.n_epochs + 1):
        if start_epoch < 50:
            train(model, train_iter, optimizer, criterion)
            fname = os.path.join(hp.logdir, str(epoch))
        elif start_epoch == 50:
            """直接加载best_model"""
            checkpoint = torch.load(log_dir)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            fname = os.path.join(hp.logdir, str(epoch))  #
        print(f"=========eval dev at epoch={epoch}=========")
        metric_dev, trigger_f1_dev = eval(model, dev_iter, fname + '_dev',epoch)
        if trigger_f1_dev >= min_trigger_f1_dev:
            state_best = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            min_trigger_f1_dev = trigger_f1_dev
            torch.save(state_best, log_dir)
            print("the best model has been saved")

        print(f"=========eval test at epoch={epoch}=========")
        metric_test, trigger_f1_test = eval(model, test_iter, fname + '_test',epoch)

        if hp.telegram_bot_token:
            report_to_telegram('[epoch {}] dev\n{}'.format(epoch, metric_dev), hp.telegram_bot_token,
                               hp.telegram_chat_id)
            report_to_telegram('[epoch {}] test\n{}'.format(epoch, metric_test), hp.telegram_bot_token,
                               hp.telegram_chat_id)
        state_lastest = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state_lastest, log_dir_latest)  # 需要保存最优的模型
        print("the lastest_model has been saved")
        # print(f"weights were saved to {fname}.pt")
