# Editor : Giwon Nam
# last edit date: 2018-11-29 22:13

import tensorflow as tf
import numpy as np
import pickle
tf.set_random_seed(10)

# 초성 리스트. 00 ~ 18 --> 19개
CHOSUNG_LIST = ['ᄀ', 'ᄁ', 'ᄂ', 'ᄃ', 'ᄄ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄈ', 'ᄉ', 'ᄊ', 'ᄋ', 'ᄌ', 'ᄍ', 'ᄎ', 'ᄏ', 'ᄐ', 'ᄑ', 'ᄒ']

# 중성 리스트. 00 ~ 20 --> 21개
JUNGSUNG_LIST = ['ᅡ', 'ᅢ', 'ᅣ', 'ᅤ', 'ᅥ', 'ᅦ', 'ᅧ', 'ᅨ', 'ᅩ', 'ᅪ', 'ᅫ', 'ᅬ', 'ᅭ', 'ᅮ', 'ᅯ', 'ᅰ', 'ᅱ', 'ᅲ', 'ᅳ', 'ᅴ',
                 'ᅵ']

# 종성 리스트. 00 ~ 27 + 1(1개는 종성없음코드) --> 28개
JONGSUNG_LIST = [' ', 'ᆨ', 'ᆩ', 'ᆪ', 'ᆫ', 'ᆬ', 'ᆭ', 'ᆮ', 'ᆯ', 'ᆰ', 'ᆱ', 'ᆲ', 'ᆳ', 'ᆴ', 'ᆵ', 'ᆶ', 'ᆷ', 'ᆸ', 'ᆹ', 'ᆺ',
                 'ᆻ', 'ᆼ', 'ᆽ', 'ᆾ', 'ᆿ', 'ᇀ', 'ᇁ', 'ᇂ']

# 독립 자소 리스트. --> 51개
INDI_LIST = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ', 'ㅅ',
             'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
             'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

def syllable(char):
    s = ord(char) - 44032
    cho = (s//21)//28
    jung = (s%(21*28))//28
    jong = (s%28)
    
    return CHOSUNG_LIST[cho], JUNGSUNG_LIST[jung], JONGSUNG_LIST[jong]

# 자소의 차원은 119
JASO_DIM = len(CHOSUNG_LIST)+len(JUNGSUNG_LIST)+len(JONGSUNG_LIST)+len(INDI_LIST)

hangul_johab = range(44032,55204)
hangul_jaeum = range(12593,12623)
hangul_moeum = range(12623,12644)
hangul_chosung = range(4352,4371)
hangul_jungsung = range(4449,4470)
hangul_jongsung = range(4520,4547)
english1 = range(65,91)
english2 = range(97,123)
digit = range(48,58)
special_char = [ord('.'), ord('\''), ord('?'), ord(','), ord('!'), ord('%')] # 형태소 분석에 필요하다고 생각하는 특수문자 추가

def read_data(file_path):
    sentence = []
    data = []
    label = []
    d_append = data.append
    with open(file_path,"r") as f:
        for line in f.readlines():
            if line != '\n':
                w = line.split('\t')
                label.append(w[1].replace('\n',''))
                word = []
                w_append = word.append
                w_extend = word.extend
                for c in w[0]:
                    sign_unk = 0
                    
                    if ord(c) in hangul_johab or ord(c) in hangul_chosung or \
                       ord(c) in hangul_jungsung or ord(c) in hangul_jongsung or \
                       ord(c) in hangul_jaeum or ord(c) in hangul_moeum or \
                       ord(c) in english1 or ord(c) in english2 or \
                       ord(c) in digit or ord(c) in special_char: pass
                    else: sign_unk = 1 # 지정된 한글, 영어, 숫자, 특수문자 이외에 전부 UNK태그 지정

                    if sign_unk == 1:
                        w_append('<UNK>')
                    else:
                        if ord(c) in hangul_johab: # 조합형 한글은 자모를 분리
                            jaso = syllable(c)
                            w_extend(jaso)
                        else:
                            w_append(c) # 한글자모, 영어, 숫자는 그대로
                sentence.append((word,w[1].replace('\n',''))) # ([분리된 형태소],태그) 형태로 저장
            else:
                if sentence != []:
                    d_append(sentence) # sentence마다 구분지어서 저장
                    sentence = []
    return data,label
print("\n==Read Data==\n")
data, label = read_data("Data/data_v5_edit.txt") # data는 3차원 리스트로
                                                       # 전체 데이터 -> 문장 -> 형태소 순으로 저장됨
pickle.dump(data, open('Data/data.pkl','wb'))
pickle.dump(label, open('Data/label.pkl','wb'))

word_max_length = 0
for sen in data:
    for word in sen:
        if word_max_length < len(word[0]): word_max_length = len(word[0])

char_list = ['<PAD>','<UNK>']+CHOSUNG_LIST+JUNGSUNG_LIST+JONGSUNG_LIST+INDI_LIST+[chr(i) for i in english1]\
               + [chr(i) for i in english2] + [chr(i) for i in digit] + [chr(i) for i in special_char]
dictionary_char = dict()

for i in char_list:
    dictionary_char[i] = len(dictionary_char)

label_list = sorted(list(set(label)))
dictionary_label= dict()
for i in label_list:
    dictionary_label[i] = len(dictionary_label)

# 위에서 정의된 dictionary에 따라 데이터를 index로 치환
def make_dataSet(data, dictionary_char):
    indexed_data = [] 
    d_append = indexed_data.append
    for sentence in data:
        sen = []
        lab = []
        s_append = sen.append
        l_append = lab.append
        for word in sentence:
            s_append(([dictionary_char[char] for char in word[0]], dictionary_label[word[1]]))
        d_append(sen)
    
    return indexed_data

indexed_data = make_dataSet(data,dictionary_char)

pickle.dump(indexed_data,open('Data/indexed_data.pkl','wb'))
pickle.dump(dictionary_char,open('Data/char_dict.pkl','wb'))
pickle.dump(dictionary_label,open('Data/label_dict.pkl','wb'))