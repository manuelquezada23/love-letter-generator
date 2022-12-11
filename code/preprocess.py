import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
from functools import reduce
from sklearn.utils import shuffle 
import os
import re
import pickle
import glob

"""
reads poem corpus
"""
def read_poems(data_path):
    # https://stackoverflow.com/questions/7099290/how-to-ignore-hidden-files-using-os-listdir
    poems = glob.glob(os.path.join(data_path, '*'))
    print('Processing:', len(poems), 'files\n')

    poem_data = []
    purge = 0
    ## read through all files
    for poem in poems:
        # print(poem)
      ## with open(path + file, "rb") as train_file:
        poem_file_path = poem
        print("reading from path: ", poem_file_path)
            # print(file)
        with open(poem_file_path, "r", encoding='utf-8') as poem_file:
            print(poem_file_path, "poem being read")
            file = poem_file.read()
            # print(file)
            file = file.encode('utf-8', errors='ignore').decode('utf-8')
            # file = file.decode('utf8')
            # lower case
            file = file.lower()
            # removes leading, ending and duplicates whitespaces
            # duplicate whitespaces especially important since some poems might be wacky
            file = re.sub(' +', ' ', file).strip()

            chars_remove = ['\t', '"', '%', '&', '\*', '\+', '/', '0', '1', '2', '3', '4', '¦',
                            '5', '6', '7', '8', '9', '=', '\[', '\]', '_',  '`', '{',
                            '¨', 'ª', '«', '²', 'º', '»', 'à', 'â', 'ã', 'ä', 'ç',
                            'è', 'ê', 'ì', 'ï', 'ò', 'ô', 'õ', 'ö', 'ü', '\xa0', '°', 
                            '~','¤','©','®','·','á','æ','é','ë','í','î','ð','ñ','ó','ø','ù','ú','û','侄', 'ý','þ','ā','ă','ć','č','đ','ī','œ','ś','ş','š','ţ','ũ','ū','ž','ơ','ư','ș','ț̣̀́̃̉','ά','έ','ή','ί','α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω','ό','ύ','ώ','а','б','в','г','д','е','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','х','ц','ч','ш','ы','ь','э','ю','я','ё','،','آ','ئ','ا','ب','ت','ج','ح','خ','د','ر','ز','س','ش','ص','ط','ع','غ','ف','ك','ل','م','ن','وِ','ٹ','پ','چ','ڑ','ک','گ','ں','ھ','ہ','ی','ے','۔ँं','अ','आ','इ','ई','उ','ए','ऐ','ओ','औ','क','ख','ग','घ','च','ज','झ','ट','ठ','ड','ण','त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह़ािीुूृेैॉोौ्','ग़','ज़','ड़','ढ़','फ़','।ঁং','অ','আ','ই','উ','এ','ও','ক','খ','গ','ঙ','চ','ছ','জ','ঝ','ঞ','ট','ড','ণ','ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','য','র','ল','শ','স','হ়ািীুূেোৌ্','ৎଁଂ','ଅ','ଆ','ଉ','ଏ','କ','ଗ','ଚ','ଛ','ଜ','ଟ','ଠ','ଡ','ଣ','ତ','ଥ','ଦ','ଧ','ନ','ପ','ବ','ଭ','ମ','ଯ','ର','ଳ','ଶ','ସ','ହାିୀୁେୋ୍','ୟ','ୱ','ạ','ấ','ầ','ậ','ắ','ằ','ẻ','ế','ỉ','ọ','ỏ','ố','ồ','ộ','ớ','ờ','ở',' ','​','‌','–','—','―','“','”','„','•','€','™','−','─','│','▌','▐','▲','◆','○','●','★','☆','☻','♥','。','《','》','【','】','一','七','三','上','下','不','与','世','东','两','丧','中','丰','丹','为','乐','乘','九','书','争','二','于','云','五','交','享','京','亲','人','今','他','仙','代','们','优','会','传','伤','但','位','体','余','作','你','俏','儿','先','光','入','全','八','共','内','写','冶','冷','几','出','分','切','则','创','初','别','刻','前','剑','力','动','化','北','区','十','千','午','半','华','印','卷','厄','去','县','参','古','句','叶','号','合','同','后','吼','咆','和','咬','咽','哀','品','响','哭','哮','哽','唐','喘','喷','嚼','四','回','园','国','图','在','地','城','堪','境','墨','士','声','处','夕','外','多','夜','大','天','太','夫','失','夹','女','妈','姚','娥','子','字','孙','孟','安','宋','官','定','客','宣','宵','容','寂','富','对','寺','导','寿','少','尔','尘','就','山','岁','岸','己','已','市','师','希','席','常','平','年','广','府','座','开','弋','形','影','待','律','得','循','心','忆','志','快','怀','怒','思','恩','息','恸','悠','悲','情','惜','意','愿','慈','成','我','或','战','才','扭','承','抖','拉','挥','挽','描','摧','摸','撼','改','故','教','数','文','新','方','施','无','日','旦','早','时','旷','星','春','是','景','暖','暴','曲','更','曾','月','有','朝','未','朵','机','朽','李','束','条','来','板','构','林','枝','枯','柔','柳','树','栖','格','桃','桐','梦','梨','欲','歌','正','残','殡','每','比','毕','毫','氏','水','求','汉','江','池','汩','沉','沧','河','泊','法','泣','注','泪','泳','泼','洋','洞','津','活','流','浑','浓','海','涌','淌','淙','深','淹','温','游','湖','源','满','潺','灯','灵','炉','点','烟','然','照','熔','熠','爱','牡','狂','玉','珀','琢','瓦','瓶','生','由','男','画','界','疏','痛','白','百','的','皆','盘','盛','目','相','省','看','眼','睛','知','矫','磬','祁','祈','神','祭','秃','秋','种','稿','穴','空','穿','窗','章','端','笔','第','等','简','精','紫','繁','纸','经','绝','缘','缠','罗','美','翠','翻','联','育','胆','脉','臣','自','至','舌','色','芒','花','若','英','草','荣','荷','莱','莲','菊','萎','萦','落','著','蘸','虚','蛇','血','行','裴','裸','襄','西','见','规','觅','触','言','认','记','论','词','译','诗','谦','谷','象','赋','赝','走','起','轮','载','辈','达','过','运','这','进','远','连','迟','透','通','逝','速','遵','邀','酒','酬','酸','醉','醒','里','野','钟','银','铺','锁','锋','错','镜','长','门','间','闻','阳','阴','阵','陆','陶','雄','雅','雕','雪','雾','霜','霭','露','青','静','韧','音','韵','题','颤','风','飘','飙','饱','首','馨','马','魂','魄','默','龙','','！','（','）','，','：','？','💏','💐','💓'
            ]

            chars_allowed = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                            '\'','!','"','#','$','%','&','','(',')','*','+',',',' ','-','.','/',':',';','<','=','>','?','@','[',']','^','_','`','{','|','}','~'
            ]

            valid = False
            for char in chars_remove:
                if char in file:

                    print(poem_file_path, "contains invalid character, so it is purged from system")
                    purge = purge + 1
                    break

                else:
                    valid = True

            if valid == True:
                # print('\nErase from docs non meaningful characters:', chars_remove)
                # as reggex expression
                chars_remove = '[' + ''.join(chars_remove) + ']'
                # erase from docs non meaninful characters
                file = re.sub(chars_remove, ' ', file)
                file = re.sub(' +', ' ', file).strip()

                poem_data.append({'file': poem_file_path,
                            'corpus': file})

    print("poems purged: ",purge)
    return poem_data

def clean_data(corpus):
    ############ Character per doc.
    chars = [len(x['corpus']) for x in corpus]
    print('\nCharacters per doc:\n', pd.Series(chars).describe())
    # remove outliers, less than 260 characters
    idx = [x > 260 for x in chars]
    print('docs removed:', len(corpus) - sum(idx))
    corpus = list(np.array(corpus)[np.array(idx)])

    ############ lines per doc, remove single liners
    lines = [len(x['corpus'].split('\n')) for x in corpus]
    print('\nLines per doc:\n', pd.Series(lines).describe())
    # remove outliers. less than 10 lines
    idx = [((x > 8) & (x < 50)) for x in lines]
    print('docs removed:', len(corpus) - sum(idx))
    corpus = list(np.array(corpus)[np.array(idx)])

    ############ words count.
    words = [len(re.findall(r'\w+', x['corpus'])) for x in corpus]
    print('\nwords per doc:\n', pd.Series(words).describe())
    # remove outliers
    # less than 100 words
    idx = [((x > 30) & (x < 200)) for x in words]
    print('docs removed:', len(corpus) - sum(idx))
    corpus = list(np.array(corpus)[np.array(idx)])

    words = [re.findall(r'\w+', x['corpus']) for x in corpus]
    # flat list into single array to count frequency
    words = [item for sublist in words for item in sublist]

    print('\nTotal words in corpus', len(words))
    print('\nUnique words:', len(set(words)))

    # most common words couont
    words_counter = nltk.FreqDist(words)
    words_counter = pd.DataFrame(words_counter.most_common())
    words_counter.columns = ['words', 'freq']
    words_counter['len'] = [len(x) for x in words_counter['words']]
    print('\Most common words:', '\nat least 4 digits:\n',
          words_counter[['words','freq']][words_counter['len']>=4].head(10),
          '\nat least 6 digits:\n',
          words_counter[['words','freq']][words_counter['len']>=6].head(10))
    

    ##### Clean less frecuent characters
    # want all docs as a single string to get unique chars
    all_text = str()
    for x in corpus:
        all_text = all_text + x['corpus']
  
    # unique characters
    characters = sorted(list(set(all_text)))
    print('unique characters:', len(characters))

    # count of characters
    # print('Number of appereance per unique character in corpus')
    # chars_count = []
    # for digit in characters:
    #     counter = {'digit': digit,
    #                         'count': sum([digit in char for char in all_text])}
    #     print(counter)
    #     chars_count.append(counter)
    # chars_count = pd.DataFrame(chars_count)

    # remove non meaningfull characters
    # manually make the cut at 766
    # chars_count['digit'][chars_count['count'] <= 766].values
    chars_remove = ['\t', '"', '%', '&', '\*', '\+', '/', '0', '1', '2', '3', '4', '侄', '¦'
                    '5', '6', '7', '8', '9', '=', '\[', '\]', '_',  '`', '{',
                    '¨', 'ª', '«', '²', 'º', '»', 'à', 'â', 'ã', 'ä', 'ç',
                    'è', 'ê', 'ì', 'ï', 'ò', 'ô', 'õ', 'ö', 'ü', '\xa0', '°', 
                    '~','¤','©','®','·','á','æ','é','ë','í','î','ð','ñ','ó','ø','ù','ú','û','ý','þ','ā','ă','ć','č','đ','ī','œ','ś','ş','š','ţ','ũ','ū','ž','ơ','ư','ș','ț̣̀́̃̉','ά','έ','ή','ί','α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','ς','σ','τ','υ','φ','χ','ψ','ω','ό','ύ','ώ','а','б','в','г','д','е','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','х','ц','ч','ш','ы','ь','э','ю','я','ё','،','آ','ئ','ا','ب','ت','ج','ح','خ','د','ر','ز','س','ش','ص','ط','ع','غ','ف','ك','ل','م','ن','وِ','ٹ','پ','چ','ڑ','ک','گ','ں','ھ','ہ','ی','ے','۔ँं','अ','आ','इ','ई','उ','ए','ऐ','ओ','औ','क','ख','ग','घ','च','ज','झ','ट','ठ','ड','ण','त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह़ािीुूृेैॉोौ्','ग़','ज़','ड़','ढ़','फ़','।ঁং','অ','আ','ই','উ','এ','ও','ক','খ','গ','ঙ','চ','ছ','জ','ঝ','ঞ','ট','ড','ণ','ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','য','র','ল','শ','স','হ়ািীুূেোৌ্','ৎଁଂ','ଅ','ଆ','ଉ','ଏ','କ','ଗ','ଚ','ଛ','ଜ','ଟ','ଠ','ଡ','ଣ','ତ','ଥ','ଦ','ଧ','ନ','ପ','ବ','ଭ','ମ','ଯ','ର','ଳ','ଶ','ସ','ହାିୀୁେୋ୍','ୟ','ୱ','ạ','ấ','ầ','ậ','ắ','ằ','ẻ','ế','ỉ','ọ','ỏ','ố','ồ','ộ','ớ','ờ','ở',' ','​','‌','–','—','―','“','”','„','•','€','™','−','─','│','▌','▐','▲','◆','○','●','★','☆','☻','♥','。','《','》','【','】','一','七','三','上','下','不','与','世','东','两','丧','中','丰','丹','为','乐','乘','九','书','争','二','于','云','五','交','享','京','亲','人','今','他','仙','代','们','优','会','传','伤','但','位','体','余','作','你','俏','儿','先','光','入','全','八','共','内','写','冶','冷','几','出','分','切','则','创','初','别','刻','前','剑','力','动','化','北','区','十','千','午','半','华','印','卷','厄','去','县','参','古','句','叶','号','合','同','后','吼','咆','和','咬','咽','哀','品','响','哭','哮','哽','唐','喘','喷','嚼','四','回','园','国','图','在','地','城','堪','境','墨','士','声','处','夕','外','多','夜','大','天','太','夫','失','夹','女','妈','姚','娥','子','字','孙','孟','安','宋','官','定','客','宣','宵','容','寂','富','对','寺','导','寿','少','尔','尘','就','山','岁','岸','己','已','市','师','希','席','常','平','年','广','府','座','开','弋','形','影','待','律','得','循','心','忆','志','快','怀','怒','思','恩','息','恸','悠','悲','情','惜','意','愿','慈','成','我','或','战','才','扭','承','抖','拉','挥','挽','描','摧','摸','撼','改','故','教','数','文','新','方','施','无','日','旦','早','时','旷','星','春','是','景','暖','暴','曲','更','曾','月','有','朝','未','朵','机','朽','李','束','条','来','板','构','林','枝','枯','柔','柳','树','栖','格','桃','桐','梦','梨','欲','歌','正','残','殡','每','比','毕','毫','氏','水','求','汉','江','池','汩','沉','沧','河','泊','法','泣','注','泪','泳','泼','洋','洞','津','活','流','浑','浓','海','涌','淌','淙','深','淹','温','游','湖','源','满','潺','灯','灵','炉','点','烟','然','照','熔','熠','爱','牡','狂','玉','珀','琢','瓦','瓶','生','由','男','画','界','疏','痛','白','百','的','皆','盘','盛','目','相','省','看','眼','睛','知','矫','磬','祁','祈','神','祭','秃','秋','种','稿','穴','空','穿','窗','章','端','笔','第','等','简','精','紫','繁','纸','经','绝','缘','缠','罗','美','翠','翻','联','育','胆','脉','臣','自','至','舌','色','芒','花','若','英','草','荣','荷','莱','莲','菊','萎','萦','落','著','蘸','虚','蛇','血','行','裴','裸','襄','西','见','规','觅','触','言','认','记','论','词','译','诗','谦','谷','象','赋','赝','走','起','轮','载','辈','达','过','运','这','进','远','连','迟','透','通','逝','速','遵','邀','酒','酬','酸','醉','醒','里','野','钟','银','铺','锁','锋','错','镜','长','门','间','闻','阳','阴','阵','陆','陶','雄','雅','雕','雪','雾','霜','霭','露','青','静','韧','音','韵','题','颤','风','飘','飙','饱','首','馨','马','魂','魄','默','龙','','！','（','）','，','：','？','💏','💐','💓'
    ]

    print('\nErase from docs non meaningful characters:', chars_remove)
    # as reggex expression
    chars_remove = '[' + ''.join(chars_remove) + ']'
    # erase from docs non meaninful characters
    for doc in corpus:
        doc['corpus'] = re.sub(chars_remove, ' ', doc['corpus'])
        doc['corpus'] = re.sub(' +', ' ', doc['corpus']).strip()

    # line space: '\r\n ' '\r\n' to '\n', '\r\r\n'
    for doc in corpus:
        for pattern in ['\r\r\n', '\r\n ', '\r\n', '\n\n', '\r']:
            doc['corpus'] = re.sub(pattern, '\n', doc['corpus'])

    # add special charater for at the begining and final of text.
    # Model will learn when to start/end
    for doc in corpus:
        doc['corpus'] = doc['corpus'] + '.$'

    return corpus


def get_vocabulary(corpus):
    '''
    Mapping is a step in which we assign an arbitrary number to a character/word
    in the text. In this way, all unique characters/words are mapped to a number.
    This is important, because machines understand numbers far better than text,
    and this subsequently makes the training process easier.
    '''
    # create dictionaries for character/number mapping
    print('\n---\nCreating word mapping dictionaries')

    # want all docs as a single string to get unique chars
    all_text = str()
    for x in corpus:
        all_text = all_text + x['corpus']
  
    # unique characters
    characters = sorted(list(set(all_text)))
    print('unique characters after cleansing', len(characters))
    print(''.join(characters))
    
    # dictionaries to be used as index mapping
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}
    
    return characters, n_to_char, char_to_n  

# =============================================================================
# split train and test
# =============================================================================
def corpus_split(corpus, split):
    # Train/Test per doc
    # create corpus index
    idx = [i for i in range(len(corpus))]
    # random random for train
    idx_train = np.random.choice(idx, size=int(len(corpus)*split), replace=False)
    # index not in train
    idx_test = [i for i in idx if i not in idx_train]
    # split corpus by index
    corpus_train = [corpus[i] for i in idx_train]
    corpus_test = [corpus[i] for i in idx_test]    
    # Docs stats corpus
    print('\n--- Total docs in corpus', len(corpus))
    print('split in:')
    print('Train:', len(corpus_train))
    print('Test:', len(corpus_test))
    return corpus_train, corpus_test

# Create tensor data from corpus
def build_data(corpus, char_to_n, max_seq = 100, stride = [1,6]):
    '''
    Transform list of documents into tensor format to be fed to a lstm network
    outout shape: (sequences, max_lenght)
    max_seq: maximum length of sequence
    stride: steps apply in rolling window over text. next windows could be next character(1) or 6 characters ahead
    '''
    # place holder to  save results
    data_x = []
    data_y = []
    sequences = []
    # target sequence is lagged by 1, hence sequence length = max_seq+1
    max_seq+=1  
    # for each document in corpus
    for i in range(len(corpus)):
        if (i % max(1, int(len(corpus)/10)) == 0):
            print('\n--- Progress %:{0:.2f}'.format(i/len(corpus)))
        text = corpus[i]['corpus']
        text_length = len(text)
        # iterate for all text in rolling windows of size max_seq
        j = max_seq
        while j < text_length + stride[0]:
            k_to = min(j, text_length) # 
            k_from = (k_to - max_seq)            
            # print(j, ':', k_from, '-', k_to)
            #♠ slice text
            sequence = text[k_from:k_to] 
            # print("sequence", sequence)
            # print("char_to_n", char_to_n)
            # characters to int
            sequence_encoded = np.array([char_to_n[x] for x in sequence]) 
            # append results
            sequences.append(sequence)
            data_x.append(sequence_encoded[:-1])
            data_y.append(sequence_encoded[1:])
            # random stride between 1-6
            j+=int(np.random.uniform(stride[0], stride[1]+1))       
    # Tensor structure
    data_x = np.array(data_x) 
    data_y = np.array(data_y) 
    
    # Shuffle data
    data_x, data_y = shuffle(data_x, data_y)

    # output
    print('Outupt shape -', 'X:', data_x.shape, '- Y:', data_y.shape)
    # size = data_x.nbytes*1e-6 + data_y.nbytes*1e-6
    # size = print(int(size), 'Megabytes')
    return data_x, data_y

def get_tensor_data(corpus_train, corpus_test, char_to_n, max_seq, stride ):
  # Create tensor data from corpus
  print('\n---\nBuild Tensor data')
  # Train datasets
  print('\nBuild Train data:')
  train_x, train_y = build_data(corpus_train, char_to_n, 
                                max_seq = max_seq, stride=stride)

  if len(corpus_test):
      print('\nBuild Test data:')
      test_x, test_y = build_data(corpus_test, char_to_n, 
                                    max_seq = max_seq, stride=stride)
  else:
      test_x, test_y = None, None
  
  return train_x, train_y, test_x, test_y

def get_data():
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of:
        train (1-d list or array with training words in vectorized/id form), 
        test (1-d list or array with testing words in vectorized/id form), 
        vocabulary (Dict containg word->index mapping)
    """
    # is it a test? if True only 25 songs are used
    TEST_MODE = False

    # Split train/test
    SPLIT = 1

    # NLP
    MAX_SEQ = 120 # sequence length
    STRIDE = [MAX_SEQ/2, MAX_SEQ] 

    # output path
    OUTPUT_PATH = './data/data_processed/'
    # OUTPUT_FILE = 'NLP_data_poems_120_no-split'

    # read all .txt poems files
    poem_dir_path = '../data/poetry_data'
    poem_corpus = read_poems(poem_dir_path)

    corpus = clean_data(poem_corpus)

    characters, n_to_char, char_to_n =  get_vocabulary(corpus)

    # mapping used as part of the model
    words_mapping = {'characters': characters,
                    'n_to_char': n_to_char,
                    'char_to_n': char_to_n}

    corpus_train, corpus_test = corpus_split(corpus, split=SPLIT)

    train_x, train_y, test_x, test_y = get_tensor_data(corpus_train=corpus_train, corpus_test=corpus_test, char_to_n = char_to_n, max_seq=MAX_SEQ, stride=STRIDE)

    # Merge all data in one dict
    data = {'corpus': corpus,
            'words_mapping': words_mapping,
            'train_x': train_x ,
            'train_y': train_y,
            'test_x' : test_x,
            'test_y' : test_y}

    # save file
    with open('../data/processed/processed_poems.pickle', 'wb') as file:
        pickle.dump(data, file)
    print('Data saved in:', 'data/processed/processed_poems.pickle')
  

    return corpus_train, corpus_test


    # want to end with this:
    # return train_data, test_data, vocabulary

    # # If it is a test mode just take first 25 songs
    # if TEST_MODE:
    #     print("TEST MODE: ON")
    #     corpus = corpus[:250]
    # else:
    #     print("TEST MODE: OFF")


    
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    ## TODO: Implement pre-processing for the data files. See notebook for help on this.
    with open(train_file) as f1:
        train_data = f1.readlines()
    with open(test_file) as f2:
        test_data = f2.readlines()

    train_data = ' '.join(train_data)
    # print("train_data", train_data)
    train_data = train_data.split()

    test_data = ' '.join(test_data)
    test_data = test_data.split()

    ## get unique words
    unique_words = sorted(set(train_data))
    vocabulary = {w:i for i, w in enumerate(unique_words)}
    
    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)

    # Vectorize, and return output tuple.
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data  = list(map(lambda x: vocabulary[x], test_data))

    # print("train_data", train_data)
    return train_data, test_data, vocabulary