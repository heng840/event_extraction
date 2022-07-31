# bert-event-extraction
Pytorch Solution of Event Extraction Task using BERT on ACE 2005 corpus

## Prerequisites

1. Prepare **ACE 2005 dataset**. 

2. Use [nlpcl-lab/ace2005-preprocessing](https://github.com/nlpcl-lab/ace2005-preprocessing) to preprocess ACE 2005 dataset in the same format as the [data/sample.json](https://github.com/nlpcl-lab/bert-event-extraction/blob/master/data/sample.json). Then place it in the data directory as follows:
    ```
    ├── data
    │     └── test.json
    │     └── dev.json
    │     └── train.json
    │...
    ```
   这是data的分布格式。
3. change into ere dataset
   ```
    ├── data
    │     └── test.json
    │     └── dev.json
    │     └── train.json
    │...
   ```
   tokens：[] list形式，将sentence拆分为一维的list[]
"tokens": ["Con", "respecto", "a", "la", "pregunta", "que", "se", "deben", "estar", "haciendo", "..."]
   
4. setence与tokens可以对应
   "sentence": "Con respecto a la pregunta que se deben estar haciendo..."
   
5. entity_mentions 实体提及，是文本中指代实体（enetity）的词
实体：先列出来BIOES分别代表什么意思：

B，即Begin，表示开始

I，即Intermediate，表示中间

E，即End，表示结尾

S，即Single，表示单个字符

O，即Other，表示其他，用于标记无关字符
其中，PER代表人名， LOC代表位置， ORG代表组织. B-PER、I-PER代表人名首字、人名非首字，
B-LOC、I-LOC代表地名(位置)首字、地名(位置)非首字，B-ORG、I-ORG代表组织机构名首字、组织机构名非首字，O代表该字不属于命名实体的一部分
[{"id": "c93832992e8ca0020c806137834bdd38-0-42-303", "start": 6, 
"end": 7, "entity_type": "PER", "mention_type": "PRO", "text": "se"}]
与ace对比：
"golden-entity-mentions": [
      {
        "text": "we",
        "entity-type": "ORG:Media",
        "head": {
          "text": "we",
          "start": 2,
          "end": 3
        },







4. Install the packages.
   ```
   pip install pytorch==1.0 pytorch_pretrained_bert==0.6.1 numpy
   ```

## Usage

### Train
```
python train.py
```

### Evaluation
```
python eval.py --model_path=latest_model.pt
```

## Result	

### Performance	

<table>	
  <tr>	
    <th rowspan="2">Method</th>	
    <th colspan="3">Trigger Classification (%)</th>	
    <th colspan="3">Argument Classification (%)</th>	
  </tr>	
  <tr>	
    <td>Precision</td>	
    <td>Recall</td>	
    <td>F1</td>	
    <td>Precision</td>	
    <td>Recall</td>	
    <td>F1</td>	
  </tr>	
  <tr>	
    <td>JRNN</td>	
    <td>66.0</td>	
    <td>73.0</td>	
    <td>69.3</td>	
    <td>54.2</td>	
    <td>56.7</td>	
    <td>55.5</td>	
  </tr>	
  <tr>	
    <td>JMEE</td>	
    <td>76.3</td>	
    <td>71.3</td>	
    <td>73.7</td>	
    <td>66.8</td>	
    <td>54.9</td>	
    <td>60.3</td>	
  </tr>	
  <tr>	
    <td>This model (BERT base)</td>	
    <td>63.4</td>	
    <td>71.1</td>	
    <td>67.7</td>	
    <td>48.5</td>	
    <td>34.1</td>	
    <td>40.0</td>	
  </tr>	
</table>	

The performance of this model is low in argument classification even though pretrained BERT model was used. The model is currently being updated to improve the performance.

## Reference
* Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation (EMNLP 2018), Liu et al. [[paper]](https://arxiv.org/abs/1809.09078)
* lx865712528's EMNLP2018-JMEE repository [[github]](https://github.com/lx865712528/EMNLP2018-JMEE)
* Kyubyong's bert_ner repository [[github]](https://github.com/Kyubyong/bert_ner)

## train.py
train(model, train_iter, optimizer, criterion)