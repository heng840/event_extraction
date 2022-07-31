NONE = 'O'
PAD = "[PAD]"
UNK = "[UNK]"

# for BERT
CLS = '[CLS]'
SEP = '[SEP]'

"""[CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 C 可以用于后续的分类任务。
[SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。
[UNK]标志指的是未知字符
[MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么
"""

# 34 event triggers
TRIGGERS = ['Business:Merge-Org',
            'Business:Start-Org',
            'Business:Declare-Bankruptcy',
            'Business:End-Org',
            'Justice:Pardon',
            'Justice:Extradite',
            'Justice:Execute',
            'Justice:Fine',
            'Justice:Trial-Hearing',
            'Justice:Sentence',
            'Justice:Appeal',
            'Justice:Convict',
            'Justice:Sue',
            'Justice:Release-Parole',
            'Justice:Arrest-Jail',
            'Justice:Charge-Indict',
            'Justice:Acquit',
            'Conflict:Demonstrate',
            'Conflict:Attack',
            'Contact:Phone-Write',
            'Contact:Meet',
            'Personnel:Start-Position',
            'Personnel:Elect',
            'Personnel:End-Position',
            'Personnel:Nominate',
            'Transaction:Transfer-Ownership',
            'Transaction:Transfer-Money',
            'Life:Marry',
            'Life:Divorce',
            'Life:Be-Born',
            'Life:Die',
            'Life:Injure',
            'Movement:Transport']

"""
    28 argument roles
    
    There are 35 roles in ACE2005 dataset, but the time-related 8 roles were replaced by 'Time' as the previous work (Yang et al., 2016).
    ['Time-At-End','Time-Before','Time-At-Beginning','Time-Ending', 'Time-Holds', 'Time-After','Time-Starting', 'Time-Within'] --> 'Time'.
"""
#argument是事件元素，事件元素是指事件的参与者
ARGUMENTS = ['Place',
             'Crime',
             'Prosecutor',
             'Sentence',
             'Org',
             'Seller',
             'Entity',
             'Agent',
             'Recipient',
             'Target',
             'Defendant',
             'Plaintiff',
             'Origin',
             'Artifact',
             'Giver',
             'Position',
             'Instrument',
             'Money',
             'Destination',
             'Buyer',
             'Beneficiary',
             'Attacker',
             'Adjudicator',
             'Person',
             'Victim',
             'Price',
             'Vehicle',
             'Time']

# 54 entities
ENTITIES = ['VEH:Water',
            'GPE:Nation',
            'ORG:Commercial',
            'GPE:State-or-Province',
            'Contact-Info:E-Mail',
            'Crime',
            'ORG:Non-Governmental',
            'Contact-Info:URL',
            'Sentence',
            'ORG:Religious',
            'VEH:Underspecified',
            'WEA:Projectile',
            'FAC:Building-Grounds',
            'PER:Group',
            'WEA:Exploding',
            'WEA:Biological',
            'Contact-Info:Phone-Number',
            'WEA:Chemical',
            'LOC:Land-Region-Natural',
            'WEA:Nuclear',
            'LOC:Region-General',
            'PER:Individual',
            'WEA:Sharp',
            'ORG:Sports',
            'ORG:Government',
            'ORG:Media',
            'LOC:Address',
            'WEA:Shooting',
            'LOC:Water-Body',
            'LOC:Boundary',
            'GPE:Population-Center',
            'GPE:Special',
            'LOC:Celestial',
            'FAC:Subarea-Facility',
            'PER:Indeterminate',
            'VEH:Subarea-Vehicle',
            'WEA:Blunt',
            'VEH:Land',
            'TIM:time',
            'Numeric:Money',
            'FAC:Airport',
            'GPE:GPE-Cluster',
            'ORG:Educational',
            'Job-Title',
            'GPE:County-or-District',
            'ORG:Entertainment',
            'Numeric:Percent',
            'LOC:Region-International',
            'WEA:Underspecified',
            'VEH:Air',
            'FAC:Path',
            'ORG:Medical-Science',
            'FAC:Plant',
            'GPE:Continent']

# 45 pos tags
POSTAGS = ['VBZ', 'NNS', 'JJR', 'VB', 'RBR',
           'WP', 'NNP', 'RP', 'RBS', 'VBP',
           'IN', 'UH', 'JJS', 'NNPS', 'PRP$',
           'MD', 'DT', 'WP$', 'POS', 'LS',
           'CC', 'VBN', 'EX', 'NN', 'VBG',
           'SYM', 'FW', 'TO', 'JJ', 'VBD',
           'WRB', 'CD', 'PDT', 'WDT', 'PRP',
           'RB', ',', '``', "''", ':',
           '.', '$', '#', '-LRB-', '-RRB-']
