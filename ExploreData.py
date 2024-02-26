import string

import nltk
import numpy as np
import pandas
import pandas as pd
import random

from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import MatplotUtils
import PdHelper
import Utils
import re
from nltk.tag.perceptron import PerceptronTagger
from nltk.stem import *


nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


LEMMATIZER = WordNetLemmatizer()    # revove s es
STEMMER = PorterStemmer()

DASH = '-'
SPACE = ' '
PUNCTUATIONS_LIKE_SPACE = ['\n', '_', '\x89', '&', '+', '\x81', ',', '\\', '\x9d', '%', '÷', '=', '$', '|', '*', '(',
                           ')', ';', '[', ']', '.', ',']
LESS_COMMON_CHARACTERS = [ '_', '\x89', 'û', ';', '*', 'ª', '|', '[', ']', 'å', '+', 'ï', 'ê', 'ò',
                          '$', '=', '%', '÷', '\x9d', 'ó', '~', 'ì', '¢', '©', '£', '^', '¨', '\\', '}', '¼', 'è',
                          '{', 'â', 'ñ', '¡', 'à', 'á', '¤', '`', 'ä', 'ã', 'ü', 'ç', '«', '\x81', '>', '´', '¬']
LESS_COMMON_CHARACTER_THRESHOLD = 1400
PEOPLE_TAG = '@someonetag'
LINK_TAG = '@linktag'
KEYWORD_TAG = '@keytag'
LOCATION_TAG = '@loctag'
QUESTION_TAG='@questiontag'
FREQ_PEOPLE_TAG_MAP = {"a": 0}
FREQ_LINK_TAG_MAP = {}
VOCABULARY_FREQ ={}
STOP_WORDS=['the', 'a', 'to', 'and', 'is','-','by','are','be','was','just', 'but', 'so', 'will', 'an','more','or','were']
LOCATION_2_COUNTRY_MAP = {'USA': 'USA', 'New York': 'USA', 'United States': 'USA', 'London': 'UK', 'Canada': 'Canada', 'Nigeria': 'Nigeria', 'UK': 'UK', 'Los Angeles, CA': 'USA', 'India': 'India', 'Mumbai': 'India', 'Washington, DC': 'USA', 'Kenya': 'Kenya', 'Worldwide': 'Worldwide', 'Australia': 'Australia', 'Chicago, IL': 'USA', 'California': 'USA', 'New York, NY': 'USA', 'California, USA': 'USA', 'Everywhere': 'Worldwide', 'San Francisco': 'USA', 'Florida': 'USA', 'United Kingdom': 'UK', 'Indonesia': 'Indonesia', 'Los Angeles': 'USA', 'Washington, D.C.': 'USA', 'Toronto': 'Canada', 'NYC': 'USA', 'Ireland': 'UK', 'San Francisco, CA': 'USA', 'Chicago': 'USA', 'Earth': 'Worldwide', 'Seattle': 'USA', 'London, UK': 'UK', 'Texas': 'USA', 'New York City': 'USA', 'ss': '', 'Atlanta, GA': 'USA', 'Sacramento, CA': 'USA', 'London, England': 'UK', 'Nashville, TN': 'USA', '304': 'USA', 'US': 'USA', 'Dallas, TX': 'USA', 'World': 'Worldwide', 'Manchester': 'UK', 'Denver, Colorado': 'USA', 'San Diego, CA': 'USA', 'South Africa': 'South-Africa', 'Scotland': 'UK'}
HARDCODE_KEYWORD_LEMMA =[]
KEYWORD_MAPPING = {
    'terrorism':['bioterrorism','bioterror','terrorism']
    ,'buildings-burning':['burning-buildings','buildings-burning']
    ,'emergency':['radiation-emergency','emergency','emergency-plan']
    ,'collapse':['collapse','collapsed']
    ,'death':['death','deaths']
}
SUFFIX_STEM = ['s','ed'] # -ing reduce 0.2 acc ?
SUFFIX_STEM = ['s']
WRONG_STEM = ['building','united','morning','wanted','added','evening','tired','willing','this','does','buildings','rights','thus','pros']
WRONG_LEMMATIZE= ['was','has','ass','times','does','lives','media','nws','diss','cos','hrs','pros','as','us','vs','dies','less','pass','boss','ms','vegas','cites','isis','rights']
NOISE_KEYWORD = ['bioterror','radiation-emergency','rioting','snowstorm','war-zone','terrorism','mass-murderer','deaths','collapsed','burning-buildings','buildings-burning']
KEYWORD_ACCURACY = [('aftershock', 1.0), ('blazing', 1.0), ('body-bag', 1.0), ('body-bags', 1.0), ('debris', 1.0), ('derailment', 1.0), ('drown', 1.0), ('electrocute', 1.0), ('fatal', 1.0), ('inundation', 1.0), ('oil-spill', 1.0), ('outbreak', 1.0), ('razed', 1.0), ('rescue', 1.0), ('ruin', 1.0), ('suicide-bomber', 1.0), ('suicide-bombing', 1.0), ('wreckage', 1.0), ('wrecked', 1.0), ('armageddon', 0.9230769230769231), ('harm', 0.9230769230769231), ('ambulance', 0.9166666666666666), ('blaze', 0.9166666666666666), ('crush', 0.9166666666666666), ('curfew', 0.9166666666666666), ('explode', 0.9166666666666666), ('panic', 0.9166666666666666), ('quarantined', 0.9166666666666666), ('sandstorm', 0.9166666666666666), ('typhoon', 0.9166666666666666), ('wounded', 0.9166666666666666), ('army', 0.9090909090909091), ('arsonist', 0.9090909090909091), ('attacked', 0.9090909090909091), ('bloody', 0.9090909090909091), ('collide', 0.9090909090909091), ('derail', 0.9090909090909091), ('destruction', 0.9090909090909091), ('evacuation', 0.9090909090909091), ('flattened', 0.9090909090909091), ('hazard', 0.9090909090909091), ('military', 0.9090909090909091), ('nuclear-disaster', 0.9090909090909091), ('rescuers', 0.9090909090909091), ('screaming', 0.9090909090909091), ('smoke', 0.9090909090909091), ('suicide-bomb', 0.9090909090909091), ('tragedy', 0.9090909090909091), ('traumatised', 0.9090909090909091), ('blew-up', 0.9), ('blight', 0.9), ('body-bagging', 0.9), ('crushed', 0.9), ('destroyed', 0.9), ('meltdown', 0.9), ('obliterate', 0.9), ('panicking', 0.9), ('stretcher', 0.9), ('terrorist', 0.9), ('thunderstorm', 0.9), ('wild-fires', 0.9), ('', 0.8947368421052632), ('avalanche', 0.8888888888888888), ('bombing', 0.8888888888888888), ('demolished', 0.8888888888888888), ('mayhem', 0.8888888888888888), ('blizzard', 0.875), ('bush-fires', 0.875), ('lava', 0.8636363636363636), ('quarantine', 0.8636363636363636), ('deluge', 0.8461538461538461), ('sinking', 0.8461538461538461), ('collided', 0.8333333333333334), ('collision', 0.8333333333333334), ('famine', 0.8333333333333334), ('fear', 0.8333333333333334), ('fire', 0.8333333333333334), ('flames', 0.8333333333333334), ('flooding', 0.8333333333333334), ('hostages', 0.8333333333333334), ('injury', 0.8333333333333334), ('sinkhole', 0.8333333333333334), ('siren', 0.8333333333333334), ('twister', 0.8333333333333334), ('upheaval', 0.8333333333333334), ('weapon', 0.8333333333333334), ('ablaze', 0.8181818181818182), ('airplane-accident', 0.8181818181818182), ('attack', 0.8181818181818182), ('bleeding', 0.8181818181818182), ('blood', 0.8181818181818182), ('bomb', 0.8181818181818182), ('bridge-collapse', 0.8181818181818182), ('burning', 0.8181818181818182), ('catastrophe', 0.8181818181818182), ('cliff-fall', 0.8181818181818182), ('danger', 0.8181818181818182), ('demolish', 0.8181818181818182), ('detonate', 0.8181818181818182), ('displaced', 0.8181818181818182), ('drowning', 0.8181818181818182), ('dust-storm', 0.8181818181818182), ('evacuated', 0.8181818181818182), ('hail', 0.8181818181818182), ('heat-wave', 0.8181818181818182), ('injured', 0.8181818181818182), ('inundated', 0.8181818181818182), ('massacre', 0.8181818181818182), ('rainstorm', 0.8181818181818182), ('rescued', 0.8181818181818182), ('riot', 0.8181818181818182), ('screamed', 0.8181818181818182), ('screams', 0.8181818181818182), ('structural-failure', 0.8181818181818182), ('tornado', 0.8181818181818182), ('arson', 0.8), ('blown-up', 0.8), ('buildings-on-fire', 0.8), ('burned', 0.8), ('chemical-emergency', 0.8), ('detonation', 0.8), ('devastated', 0.8), ('eyewitness', 0.8), ('forest-fires', 0.8), ('hailstorm', 0.8), ('hijacking', 0.8), ('injuries', 0.8), ('mass-murder', 0.8), ('obliterated', 0.8), ('survive', 0.8), ('survived', 0.8), ('trouble', 0.8), ('violent-storm', 0.8), ('wildfire', 0.8), ('sunk', 0.7916666666666667), ('catastrophic', 0.7777777777777778), ('dead', 0.7777777777777778), ('deluged', 0.7777777777777778), ('first-responders', 0.7777777777777778), ('obliteration', 0.7777777777777778), ('seismic', 0.7777777777777778), ('sirens', 0.7777777777777778), ('survivors', 0.7777777777777778), ('volcano', 0.7777777777777778), ('floods', 0.7727272727272727), ('battle', 0.75), ('bombed', 0.75), ('derailed', 0.75), ('destroy', 0.75), ('drowned', 0.75), ('earthquake', 0.75), ('epicentre', 0.75), ('fatalities', 0.75), ('fatality', 0.75), ('hellfire', 0.75), ('hurricane', 0.75), ('pandemonium', 0.75), ('threat', 0.75), ('thunder', 0.75), ('windstorm', 0.75), ('wreck', 0.75), ('accident', 0.7272727272727273), ('annihilated', 0.7272727272727273), ('casualties', 0.7272727272727273), ('casualty', 0.7272727272727273), ('crashed', 0.7272727272727273), ('demolition', 0.7272727272727273), ('desolation', 0.7272727272727273), ('devastation', 0.7272727272727273), ('drought', 0.7272727272727273), ('electrocuted', 0.7272727272727273), ('engulfed', 0.7272727272727273), ('flood', 0.7272727272727273), ('hijacker', 0.7272727272727273), ('loud-bang', 0.7272727272727273), ('nuclear-reactor', 0.7272727272727273), ('refugees', 0.7272727272727273), ('storm', 0.7272727272727273), ('tsunami', 0.7272727272727273), ('apocalypse', 0.7), ('crash', 0.7), ('cyclone', 0.7), ('emergency-services', 0.7), ('exploded', 0.7), ('hijack', 0.7), ('hostage', 0.7), ('landslide', 0.7), ('lightning', 0.7), ('trapped', 0.7), ('trauma', 0.7), ('wounds', 0.7), ('burning-buildings', 0.6666666666666666), ('desolate', 0.6666666666666666), ('emergency', 0.6666666666666666), ('evacuate', 0.6666666666666666), ('explosion', 0.6666666666666666), ('forest-fire', 0.6666666666666666), ('mudslide', 0.6666666666666666), ('weapons', 0.6666666666666666), ('whirlwind', 0.6666666666666666), ('buildings-burning', 0.6363636363636364), ('collapse', 0.6363636363636364), ('collapsed', 0.6363636363636364), ('death', 0.6363636363636364), ('disaster', 0.6363636363636364), ('emergency-plan', 0.6363636363636364), ('hazardous', 0.6363636363636364), ('natural-disaster', 0.6363636363636364), ('damage', 0.6153846153846154), ('fire-truck', 0.6), ('mass-murderer', 0.6), ('bioterror', 0.5833333333333334), ('police', 0.5833333333333334), ('annihilation', 0.5555555555555556), ('rubble', 0.5555555555555556), ('snowstorm', 0.5555555555555556), ('terrorism', 0.5454545454545454), ('deaths', 0.5), ('war-zone', 0.5), ('rioting', 0.45454545454545453), ('bioterrorism', 0.4444444444444444), ('radiation-emergency', 0.3333333333333333)]
# collect words having target-1-rate >= threshold and topic rate >=threshold
HOT_WORDS = {
    #['bioterrorism','bioterror','terrorism']
    'terrorism':['terrorism:','#bioterrorism','colluded','anthrax','w/bioterrorism','idis','iran','military','israeli','hostage','caught','pakistan','arrest','id','@usagov','wht','auth','terrorism','90blks','8whts','red-handed','bioterrorism?']
    #['burning-buildings','buildings-burning']
    ,'buildings-burning':['down']
    #['radiation-emergency','emergency','emergency-plan']
    ,'emergency': ['@cityofcalgary','activates','municipal','#abstorm','calgary','city']
    #['collapse','collapsed']
    ,'collapse':['greece\'s','crisis']
    #['forest-fire']
    ,'forest-fire':['pisgah','forest']
    #['rioting']
    ,'rioting':['riots']
    #['police']
    ,'police':['suspect','shooting']
    #['hazardous']
    ,'hazardous':['health','issues']
    #['desolate']
    ,'desolate':['hope:','the']
    #['mudslide']
    ,'mudslide':['#tajikistan']
    #['weapons']
    ,'weapons': ['nuclear','kill','give','sell']
    #['whirlwind']
    ,'whirlwind':['news']
    # ['damage']
    ,'damage': ['vandalized','#sandy']
}
# collect words having target-0-rate >= threshold topic rate >=threshold
COLD_WORDS ={
    'terrorism':['@fedex','omega','regional','religion','automation','kyle']
    ,'emergency':['plan?','kit','download']
    ,'collapse':['ashes']
    #['death','deaths']
    ,'death':['her','like']
    ,'rioting':['if']
    #['explosion']
    ,'explosion':['explosion-proof']
    #['evacuate']
    ,'evacuate':['my']
    #['mudslide']
    ,'mudslide': ['eat','cake','#bakeofffriends','lady','loving','chewing','chocolate']
    #['weapons']
    ,'weapons': ['@dannyonpc','helpful','benefits']
    #['natural-disaster']
    ,'natural-disaster':['via']
    #['disaster']
    ,'disaster':['not','what','https://t']
    #['whirlwind']
    ,'whirlwind':['romance','liked','trip','my']
    #['damage']
    ,'damage':['zero','my','goodwill','called','media']
    #['annihilation']
    ,'annihilation':['stop','save','please']
    #['rubble']
    ,'rubble':['rubble?:','china']
    #['snowstorm']
    ,'snowstorm':['that','like','answers','imagine']
}
# harcode Pandemonium in aba decrease score / fedex no longer increase scores
# HARDCODE_TARGET_TESTDATA={7596:1,7615:1,7624:1,7625:1,7637:1,7618:1}

HARDCODE_TARGET_TESTDATA ={}
CHANGED_TARGET_TRAINDATA={8698:0,10583:0,6165:0,516:0,8218:0,6274:0,10552:0,3879:0,6788:0,8905:0,2332:0,4624:0,2337:0,3810:0,7844:0,6032:0,3866:0}
def show_intersect_keyword(train_df, test_df):
    train_keywords = train_df['keyword'].dropna().tolist()
    test_keywords = test_df['keyword'].dropna().tolist()

    train_freq_keyword = dict(Utils.group_list(train_keywords))
    test_freq_keyword = dict(Utils.group_list(test_keywords))

    intersect_keywords = set(train_keywords).intersection(set(test_keywords))

    intersect_freq = [(k, (0 if k not in test_freq_keyword else test_freq_keyword.get(k))) for k in intersect_keywords]

    intersect_freq.sort(key=lambda x: x[1], reverse=True)

    print(f"train data: {len(train_keywords)} keywords with unique {len(set(train_keywords))} keywords")
    print(f"test data: {len(test_keywords)}/{len(test_df)} keywords with unique {len(set(test_keywords))} keywords")
    print(f"intersect data: {len(intersect_keywords)} unique keywords")
    for e in intersect_freq:
        print(e)


def show_disaster_distribute_on_keyword(df):
    print(len(df))
    df['keyword']=df['keyword'].fillna('').apply(normalize_keyword)
    distributions = PdHelper.get_values_distribute('keyword', 'target', df)
    print(distributions)
    MatplotUtils.show_histogram([x[2] for x in distributions])

    # keyword_2_acc = dict(KEYWORD_ACCURACY)
    # keyword_2_target_rate={}
    # for (k,tar,r) in distributions:
    #     if k not in keyword_2_target_rate:
    #         keyword_2_target_rate.update({k:r})
    #     else:
    #         keyword_2_target_rate.update({k: max(keyword_2_target_rate.get(k),r)})
    #
    # list_diff = []
    # print(keyword_2_acc.items())
    # for (k,r) in keyword_2_target_rate.items():
    #     list_diff.append((k,keyword_2_acc.get(k)-r))
    #
    # list_diff.sort(key=lambda x:abs(x[1]),reverse=True)
    #
    # for i in list_diff:
    #     print(i)

def normalize_keywords(keywords):
    for i in range(0, len(keywords)):
        keywords[i] = keywords[i].replace("%20", DASH)
    return keywords
def normalize_keyword(keyword):
    keyword = keyword.replace("%20", DASH).lower()
    #without this acc 0.798,with this acc 0.8
    keyword = mapping_keyword(keyword)
    return keyword

def mapping_keyword(keyword):
    for k,v in KEYWORD_MAPPING.items():
        if keyword in v:
            return k
    return keyword


def normalize_texts(texts):
    for i in range(0, len(texts)):
        texts[i] = texts[i].lower()
        for p in PUNCTUATIONS_LIKE_SPACE:
            texts[i] = texts[i].replace(p, SPACE)
        for c in LESS_COMMON_CHARACTERS:
            texts[i] = texts[i].replace(c, '')
        # remove redundant spaces
        texts[i] = re.sub(' +', SPACE, texts[i]).strip()
    return texts


def normalize_text(text):
    text=text.lower()



    text = map_tags(text)

    for p in PUNCTUATIONS_LIKE_SPACE:
        text=text.replace(p,SPACE)
    for c in LESS_COMMON_CHARACTERS:
        text=text.replace(c,'')

    # text = stemmerize(text)  # reduce 0.2 acc
    text = lemmatize(text)

    #without this got 0.826, with this got 0.825
    # text = remove_single_numeric(text)
    # text = remove_single_character(text)

    #remove redundant spaces
    text=re.sub(' +', SPACE, text).lower().strip()
    return text

def normalize_location(loc):
    if loc in LOCATION_2_COUNTRY_MAP:
        loc=LOCATION_2_COUNTRY_MAP.get(loc)
    return loc

def map_link(text):
    tmp = text
    for p in ', \n':  # exclude @
        tmp = tmp.replace(p, SPACE)
    words = tmp.split(SPACE)

    candidate = []
    for w in words:
        if 'https://' in w or 'http://' in w:
            candidate.append(w)
    for c in candidate:
        if FREQ_LINK_TAG_MAP.get(c)==1:
            text = text.replace(c, '')

    #without this got 0.798693, with this got 0.798726 score
    if 'http://t.co/' in text:
        text = text.replace('http://t.co/',LINK_TAG)
    if 'https://t.co/' in text:
        text = text.replace('https://t.co/',LINK_TAG)

    return text

def extract_link(text):
    tmp = text
    for p in ', \n':  # exclude @
        tmp = tmp.replace(p, SPACE)
    words = tmp.split(SPACE)

    candidate = []
    for w in words:
        if 'https://' in w or 'http://' in w:
            candidate.append(w)

    return candidate

def map_people_tag(text):
    tmp = text
    punc = string.punctuation
    punc = punc.replace('@', '')  # exclude @
    for p in punc:
        tmp = tmp.replace(p, SPACE)
    words = tmp.split(SPACE)

    candidate = []
    for w in words:
        if w.startswith('@') and not KEYWORD_TAG:
            candidate.append(w)

    for c in candidate:
        if FREQ_PEOPLE_TAG_MAP.get(c) == 1:
            text = text.replace(c, '')

    return text

def map_question_mark(text):
    for p in ' \n':  # exclude @
        text = text.replace(p, SPACE)

    words = text.split(SPACE)
    candidate=[]
    for w in words:
        unique_chars = list(set(w))
        if len(unique_chars)==1 and unique_chars[0]=='?':
            candidate.append(w)
    text=' '+text+' '
    for c in candidate:
        text = text.replace(' '+c+' ', ' '+QUESTION_TAG+' ')
    return text

def lemmatize(text):
    raw_text=text
    text=text.lower()
    for p in ' \n':  # exclude @
        text = text.replace(p, SPACE)

    words = text.split(SPACE)
    candidate = []
    for w in words:
        lemv = LEMMATIZER.lemmatize(w)
        if lemv != w and len(w) >= 4 and w not in WRONG_LEMMATIZE and lemv in VOCABULARY_FREQ:
            candidate.append((w,lemv))

    text = ' ' + text + ' '
    for c0,c1 in candidate:
        text = text.replace(' ' + c0 + ' ', ' ' + c1 + ' ')

    return text

loc_test =[]
def stemmerize(text):
    raw_text=text
    text=text.lower()
    for p in ' \n':  # exclude @
        text = text.replace(p, SPACE)

    words = text.split(SPACE)
    candidate = []
    for w in words:
        stemv = STEMMER.stem(w)

        yes_suffix = False
        for suf in SUFFIX_STEM:
            if w.endswith(suf):
                yes_suffix = True

        if stemv != w and stemv in VOCABULARY_FREQ and len(w) >= 4 and yes_suffix and stemv not in WRONG_STEM and not stemv.startswith('#'):
            candidate.append((w, stemv))

    text = ' ' + text + ' '
    for c0,c1 in candidate:
        text = text.replace(' ' + c0 + ' ', ' ' + c1 + ' ')

    if len(candidate)>=3:
        loc_test.append((raw_text,text))

    return text
def map_tags(text):

    text = map_people_tag(text)
    text = map_link(text)   #remove this got 0.7964, add this got 0.7956
    text = map_question_mark(text)

    return text

def remove_single_numeric(text):
    for p in ' \n':  # exclude @
        text = text.replace(p, SPACE)

    words = text.split(SPACE)
    candidate=[]
    for w in words:
        if w.isnumeric():
            candidate.append(w)

    for c in candidate:
        text = text.replace(c, '')
    return text

def remove_single_character(text):
    for p in ' \n':  # exclude @
        text = text.replace(p, SPACE)

    words = text.split(SPACE)
    candidate=[]
    for w in words:
        if len(w)==1 and w!='?':
            candidate.append(w)
    text=' '+text+' '
    for c in candidate:
        text = text.replace(' '+c+' ', ' ')
    return text

def add_keyword_to_text(df):
    df['keyword'] = df['keyword'].fillna('')

    df.loc[df['keyword']!='','text']=df['text']+SPACE+KEYWORD_TAG+df['keyword']
    return df

def add_location_to_text(df):
    df['location'] = df['location'].fillna('')
    df['location']=df['location'].apply(normalize_location)
    df.loc[df['location'].isin(LOCATION_2_COUNTRY_MAP),'text']=df['text']+SPACE+LOCATION_TAG+df['location']
    # df.loc[df['location'] != "",'text']=df['text']+SPACE+LOCATION_TAG+df['location']  bad score
    return df


def show_character_distribute(keywords):
    count_map = {}
    keywords = normalize_keywords(keywords)
    for k in keywords:
        for c in str(k).lower():
            if c not in count_map:
                count_map.update({c: 0})
            v = count_map.get(c)
            count_map.update({c: v + 1})
    list_char_freq = list(count_map.items())
    list_char_freq.sort(key=lambda x: x[1], reverse=True)

    print(f"num of chars = {len(list_char_freq)}")
    less_common_chars = []
    for e in list_char_freq:
        if e[1] < LESS_COMMON_CHARACTER_THRESHOLD:
            less_common_chars.append(e[0])
        print(e)
    print("less common chars:")
    print(less_common_chars)


def show_num_words_distribute(texts):
    arr = [len(t.split(SPACE)) for t in texts]
    # MatplotUtils.show_histogram(arr)
    print(f"sum of words = {sum(arr)}")

    words = []
    for t in texts:
        t=t.lower()
        for p in ', \n':  # exclude @
            t = t.replace(p, SPACE)
        words += t.split(SPACE)
    print(f"sum of unique words = {len(set(words))}")

    freqs = Utils.group_list(words)

    for f in freqs[:300]:
        print(f)

def get_id_2_keyword_map():
    raw_train_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/train.csv")
    raw_test_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/test.csv")
    raw_train_df['keyword'] = raw_train_df['keyword'].fillna('')
    raw_train_df['keyword'] = raw_train_df['keyword'].apply(normalize_keyword)

    raw_test_df['keyword'] = raw_test_df['keyword'].fillna('')
    raw_test_df['keyword'] = raw_test_df['keyword'].apply(normalize_keyword)
    id_2_keyword = {}

    for idx in raw_train_df.index:
        id_2_keyword.update({raw_train_df['id'][idx]: raw_train_df['keyword'][idx]})

    for idx in raw_test_df.index:
        id_2_keyword.update({raw_test_df['id'][idx]: raw_test_df['keyword'][idx]})

    return id_2_keyword

def explore_dup_text(df):
    raw_df=df.copy()
    print(len(df))
    df['text']=df['text'].apply(normalize_text)
    print(len(df))
    df['count']=df.groupby(['text','target'])['target'].transform('count')
    print(len(df))
    df['summ']=df.groupby(['text'])['text'].transform('count')
    df=df[df['summ']>1].drop_duplicates(['text','target'])
    df['rate']=df['count']/df['summ']
    PdHelper.print_full(df.head(10))
    # MatplotUtils.show_histogram(df['rate'].tolist())
    # PdHelper.print_full(df[df['rate']==0.5].sort_values('text'))
    ids=df[df['rate']==0.5]['id'].tolist()
    raw_df=raw_df[raw_df['id'].isin(ids)]
    PdHelper.print_full(raw_df.sort_values('text'))
    print(len(df))

def try_perceptontagger(df):
    df['text']=df['text']
    pretrain = PerceptronTagger()
    print(pretrain.tag('The quick brown fox jumps over the lazy dog'.split()))
    list_adj=[]
    for text in df['text'].tolist():
        text=text.lower()
        for p in ', \n':
            text=text.replace(p,SPACE)
        tags = pretrain.tag(text.split(SPACE))
        for w,kind in tags:
            if kind=='JJ':
                list_adj.append(w)

    list_adj=Utils.group_list(list_adj)
    list_adj.sort(key=lambda x:x[1],reverse=True)

    print(f"detect adj len {len(list_adj)}")

    for e in list_adj[:100]:
        print(e)

def explore_text(df):
    # df['text'] = df['text'].apply(normalize_text)
    # examples =df.groupby(['keyword','target'])['text'].agg(lambda x: list(x)[0])
    #
    # PdHelper.print_full(examples)
    #
    # show_num_words_distribute(df.text.tolist())

    arr_link = []
    count_numeric=0
    for text in df['text'].tolist():
        tmp = text
        for p in ', \n':  # exclude @
            tmp = tmp.replace(p, SPACE)
        # tmp = tmp.replace('\n', SPACE)
        words = tmp.split(SPACE)

        for w in words:
            if w.isnumeric():
                count_numeric+=1
            if 'https://' in w or 'http://' in w:
                arr_link.append(len(w))
                if len(w)>23:
                    print(w)
                #arr_link.append(w)
    link_freq = Utils.group_list(arr_link)
    print(f"len = {len(arr_link)}")
    # for l in link_freq[:100]:
    #     print(l)

    print(link_freq)

    # df['text']=df['text'].apply(map_link)
    # duplicate_values = df['text'].duplicated().reset_index(name="cc")
    # print(len(duplicate_values[duplicate_values['cc']==True]))
    # duplicate_values=duplicate_values[duplicate_values['cc'] == True]

    print(count_numeric)
    # PdHelper.print_full(duplicate_values.head(10))
    # PdHelper.print_full(df[df['id'].isin( duplicate_values['index'].tolist())].head(100))

    voca=[]
    # df['text']=df['text'].apply(normalize_text)
    for t in df['text'].tolist():
        t=t.lower()
        for p in ' \n':
            t=t.replace(p,SPACE)
        voca+=t.split(SPACE)

    group_voca=Utils.group_list(voca)

    print(len(group_voca))
    print(group_voca[:300])

    freq_map = dict(group_voca)

    lem_words = [LEMMATIZER.lemmatize(e[0]) for e in group_voca]

    change_list =[]
    for e in group_voca:
        lemv = LEMMATIZER.lemmatize(e[0])

        # if lemv != e[0] and len(e[0])>=4 and e[0] not in WRONG_LEMMATIZE:
        #     change_list.append((e[0],lemv,e[1]))
        stemv = STEMMER.stem(e[0])
        yes_suffix =False
        for suf in SUFFIX_STEM:
            if e[0].endswith(suf):
                yes_suffix = True
        if stemv != e[0] and stemv in VOCABULARY_FREQ and len(e[0])>=4 and yes_suffix:
            change_list.append((e[0],stemv,e[1]))

    group_lem_words = Utils.group_list(lem_words)
    print(len(group_lem_words))

    print(f"list change len ={len(change_list)}")

    for e in change_list:
        print(e)

    # df['text']=df['text'].apply(normalize_text)
    # cnt0=0
    # cnt1=1
    # for idx in df.index:
    #     if QUESTION_TAG in df['text'][idx]:
    #         if df['target'][idx]==1:
    #             cnt1+=1
    #         else:
    #             cnt0+=1
    #
    # print(f"{cnt0}  {cnt1}")

def bruteforce_lemma(sentences,min_root_remove=2,min_neighbor_remove=4,min_word_len=3,not_lemma=[],hardcode_lemma=[]):
    #norm
    voca=list()
    punc = string.punctuation
    punc= punc.replace(SPACE,'')
    punc = punc.replace(DASH, '')
    for s in sentences:
        s=str(s).lower()
        for p in punc:
            s=s.replace(p,'')
        tokens = s.split(SPACE)
        for t in tokens:
            if t.isnumeric() or len(t)<min_word_len:
                continue
            voca.append(t)

    voca = list(np.unique(voca))
    print(f"all unique vocabulary {len(voca)}")

    lemma={}
    added=set()
    voca.sort(key=lambda x:len(x))

    for w in voca:
        if w in added:
            continue
        for w2 in voca:
            if len(w2)<len(w) or w2 in added or w==w2:
                continue
            if((w,w2) in not_lemma or (w2,w) in not_lemma):
                continue

            yes=False
            for iroot in range(0,min_root_remove+1):
                for ineighbor in range(0,min_neighbor_remove+1):
                    if len(w)-iroot <min_word_len or len(w2)-ineighbor<min_word_len:
                        continue
                    if w[:len(w)-iroot]==w2[:len(w2)-ineighbor]:
                        yes=True
                        break
                if yes:
                    break
            if yes:
                if w not in lemma:
                    lemma.update({w:[]})
                l=lemma.get(w)
                l.append(w2)
                lemma.update({w:l})
                added.add(w)
                added.add(w2)


    #hard code lemma
    for k,v in hardcode_lemma:
        if k not in lemma:
            lemma.update({k:[]})
        l = lemma.get(k)
        l+=v
        lemma.update({k:l})
    print(f"detects {len(lemma)} lemma")

    list_lemma=list(lemma.items())
    list_lemma.sort(key=lambda x:x[0])
    # for i in list_lemma:
    #     print(i)

    return lemma


def explore_lemma(df):
    uni_keywords=np.unique(df['keyword'].tolist())

    for e in uni_keywords:
        print(e)

def explore_location(df):
    list_loc = Utils.group_list(df['location'].tolist())
    dict_loc={}
    print(list_loc[:100])
    for i in list_loc[:50]:
        dict_loc.update({i[0]:i[0]})
    print(dict_loc)
    # PdHelper.print_full(df[df['location']=='USA'])

    df['location'] = df['location'].fillna("0")
    df['location']= df['location'].apply(lambda x:"1" if x!="0" else "0")


    df['sum']=df.groupby(['target','location'])['text'].transform('count')
    df=df.drop_duplicates(subset=['target','location'])
    PdHelper.print_full(df.head(10))



def explore_link(df):
    df['keyword']=df['keyword'].fillna('')
    link_2_keyword_map ={}

    for idx in df.index:
        links = extract_link(df['text'][idx])

        for l in links:
            if l not in link_2_keyword_map:
                link_2_keyword_map.update({l:[]})
            keys=link_2_keyword_map.get(l)
            if df['keyword'][idx] != '':
                keys.append(df['keyword'][idx])
            link_2_keyword_map.update({l:keys})

    links = list(link_2_keyword_map.items())
    print(len(links))
    for i in range(0,len(links)):
        links[i]=(links[i][0],np.unique(links[i][1]))
    links.sort(key=lambda x:len(x[1]),reverse=True)
    links =list(filter(lambda x:len(x[1])>=2,links))
    print(links[:30])
    print(len(links))

def calc_word_rate_on_target(df):
    #should normalize text before pass to func

    voca_freq = {}
    word_on_target = {}
    for idx in df.index:
        text = df['text'][idx]
        tar = df['target'][idx]
        for w in list(np.unique(text.split(SPACE))):

            if (w,tar) not in word_on_target:
                word_on_target.update({(w,tar):1})
            else:
                word_on_target.update({(w,tar):word_on_target.get((w,tar))+1})

            if w not in voca_freq:
                voca_freq.update({w:1})
            else:
                voca_freq.update({w:voca_freq.get(w)+ 1})

    result=[]
    for w in voca_freq.keys():
        freq1= word_on_target.get((w,1))
        freq0= word_on_target.get((w,0))
        if freq1 ==None:
            freq1=0
        if freq0==None:
            freq0=0

        if freq0>=freq1:
            result.append((w, 0, freq0, freq0 / voca_freq.get(w)))
        else:
            result.append((w, 1, freq1, freq1 / voca_freq.get(w)))

    return result

def calc_word_overall_rate(df,topn=5):
    #for each word, take top $topn most freq topics
    # should normalize text before pass to func
    word_2_topic = {}
    for idx in df.index:
        text = df['text'][idx]
        topic = df['keyword'][idx]
        for w in list(np.unique(text.split(SPACE))):
            if w not in word_2_topic:
                word_2_topic.update({w: [topic]})
            else:
                word_2_topic.update({w: word_2_topic.get(w) + [topic]})

    result=[]
    for w in word_2_topic.keys():
        topic_freq=Utils.group_list(word_2_topic.get(w))
        _sum=sum(map(lambda x:x[1],topic_freq))
        topic_rate=[]
        for t,f in topic_freq:
            topic_rate.append((t,f,f/_sum))
        topic_rate.sort(key=lambda x:x[1],reverse=True)
        result.append((w,topic_rate[:min(len(topic_rate),topn)]))

    return result
def calc_word_rate_on_topic(sub_df,total_df):
    # should normalize text before pass to func
    #return rate word w in sub_df / w in total_df
    total_df_freq ={}
    for text in total_df['text'].tolist():
        for w in list(np.unique(text.split(SPACE))):
            if w not in total_df_freq:
                total_df_freq.update({w: 1})
            else:
                total_df_freq.update({w: total_df_freq.get(w) + 1})

    sub_df_freq={}
    for text in sub_df['text'].tolist():
        for w in list(np.unique(text.split(SPACE))):
            if w not in sub_df_freq:
                sub_df_freq.update({w: 1})
            else:
                sub_df_freq.update({w: sub_df_freq.get(w) + 1})

    result=[]
    for w,sub_freq in sub_df_freq.items():
        assert total_df_freq.get(w)!=None
        r=sub_freq/total_df_freq.get(w)
        assert r<=1.0
        result.append((w,r))


    return result


def explore_word_freq(df):
    df['text'] = df['text'].apply(lambda x: x.replace('\n', SPACE).lower())
    df['keyword'] = df['keyword'].fillna('').apply(normalize_keyword)
    freq_dict = {}
    voca_freq = {}
    word_on_target ={}
    for idx in df.index:
        text = df['text'][idx]
        topic = df['keyword'][idx]
        tar = df['target'][idx]
        for w in list(np.unique(text.split(SPACE))):
            if w in VOCABULARY_FREQ and VOCABULARY_FREQ.get(w)==1:
                continue
            if (w,tar) not in word_on_target:
                word_on_target.update({(w,tar):1})
            else:
                word_on_target.update({(w,tar):word_on_target.get((w,tar))+1})

            if w not in voca_freq:
                voca_freq.update({w:1})
            else:
                voca_freq.update({w:voca_freq.get(w)+ 1})

            if (w,topic) not in freq_dict:
                freq_dict.update({(w,topic):0})
            v= freq_dict.get((w,topic))
            freq_dict.update({(w,topic):v+1})

    THRESHOLD_RATE=0.8

    valuable_words=[]
    valuable_target_words=[]
    samples=[]

    for (w,topic),freq in freq_dict.items():
        if w=='dangerous':
            samples.append((w,topic,freq))
        if w in voca_freq and voca_freq.get(w)>1:
            if w=='dangerous':
                print(freq/voca_freq.get(w))
            if freq/voca_freq.get(w)>=THRESHOLD_RATE:
                valuable_words.append((w,topic,freq))

    for (w,tar),freq in word_on_target.items():
        if freq/voca_freq.get(w)>=THRESHOLD_RATE and freq>=2:
            valuable_target_words.append((w,tar,freq,freq/voca_freq.get(w)))


    valuable_words.sort(key=lambda x:x[2],reverse=True)
    # print(f"len of valuable words {len(valuable_words)}")

    # print(samples)

    valuable_target_words.sort(key=lambda x:x[2],reverse=True)
    print(f"len of valuable_target_words {len(valuable_target_words)}")
    for e in valuable_target_words[:300]:
        print(e)


"""
terrorism based on Pakistan,iran,israel
war-zone based on look like
mass-murderer based on white

normal keywords = photography , insurance, Story, Listen,Track, purchase, How To, why, funny, history, learn, Experts, sport, quiz
"""
def take_examples(raw_df):

    df=raw_df.copy()
    df['keyword']=df['keyword'].dropna().apply(normalize_keyword)
    df['text']=df['text'].apply(lambda x: x.replace('\n',SPACE).lower())
    # df['text'] = df['text'].apply(normalize_text)
    df=df[df['keyword'].isin( ['sinkhole'])]
    df=df.sort_values(by=['keyword',"target"]).reset_index()
    PdHelper.print_full(df)

def explore_test_df(df):
    good_keywords = [x[0] for x in list(filter(lambda x: x[1] < 0.7, KEYWORD_ACCURACY))]
    df['keyword']=df['keyword'].fillna('').apply(normalize_keyword)
    df = df[df['keyword'].isin(good_keywords)]
    PdHelper.print_all_column_stats(df)
    show_disaster_distribute_on_keyword(df)

def explore_train_df2(df):
    good_keywords = [x[0] for x in list(filter(lambda x: x[1] < 0.7, KEYWORD_ACCURACY))]
    df['keyword'] = df['keyword'].fillna('').apply(normalize_keyword)
    df = df[df['keyword'].isin(good_keywords)]
    # PdHelper.print_all_column_stats(df)
    # show_disaster_distribute_on_keyword(df)
    df=df[df['keyword'].isin(['bioterrorism','bioterror','terrorism'])]
    explore_word_freq(df)

def explore_outlier(train_df):
    bad_data_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/wrong-ids.csv")
    train_df=train_df[train_df['id'].isin(bad_data_df['id'].tolist())]
    train_df=train_df.groupby(['target'])['id'].agg('count')
    PdHelper.print_full(train_df)
def split_data_by_keyword(df,train_rate,random_seed=100):
    raw_df=df.copy()
    #should be norm before run this func
    # df['keyword']=df['keyword'].fillna("")

    df = df.groupby('keyword')['id'].agg(list).reset_index(name='list_ids')
    train_ids=set()
    test_ids=set()
    for idx in df.index:
        ids=list(df['list_ids'][idx])
        random.Random(random_seed).shuffle(ids)
        p1,p2= np.split(ids,[int(len(ids)*train_rate)])
        train_ids.update(list(p1))
        test_ids.update(list(p2))

    train_df=raw_df[raw_df['id'].isin(train_ids)]
    test_df=raw_df[raw_df['id'].isin(test_ids)]

    print(f"size of train data = {len(train_df)}")
    print(f"size of test data = {len(test_df)}")
    return (train_df,test_df)

def disjoin_set(list_id,list_edge):
    #return list components

    dfs_visited = set()
    dfs_map_neighbors ={}
    for (u,v) in list_edge:
        if u not in dfs_map_neighbors:
            dfs_map_neighbors.update({u:[]})
        nbr=dfs_map_neighbors.get(u)
        nbr.append(v)
        dfs_map_neighbors.update({u: nbr})
        if v not in dfs_map_neighbors:
            dfs_map_neighbors.update({v:[]})
        nbr=dfs_map_neighbors.get(v)
        nbr.append(u)
        dfs_map_neighbors.update({v: nbr})

    def color_dfs(curv):
        if curv in dfs_visited:
            return []
        res =[curv]
        dfs_visited.add(curv)
        if curv not in dfs_map_neighbors:
            return res
        for nxt in dfs_map_neighbors.get(curv):
            res+=color_dfs(nxt)
        return res

    list_components=[]
    for id in list_id:
        nrbs=color_dfs(id)
        if len(nrbs)!=0:
            list_components.append(nrbs)

    list_components.sort(key=lambda x:len(x),reverse=True)
    return list_components


def explore_knn(raw_train_df,raw_test_df):
    bad_data_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/wrong-ids.csv")
    bad_data_df = bad_data_df[['id']]
    print(f"len of bad data = {len(bad_data_df)}")
    bad_data_df = bad_data_df.assign(outlier=[1] * len(bad_data_df))

    df = pd.merge(raw_train_df, bad_data_df, how='left', on='id')
    df['outlier'] = df['outlier'].fillna(0)
    df['outlier'] = df['outlier'].astype(int)
    print(f"len outlier in df = {len(df[df['outlier'] == 1])}")
    print(f"len non-outlier in df = {len(df[df['outlier'] == 0])}")

    df['text'] = df['text'].apply(normalize_text)
    df = df.drop_duplicates(subset=['text'], keep='first')
    documents = (df['text'].tolist())
    vectorizer = CountVectorizer(min_df=2, stop_words=STOP_WORDS, ngram_range=(1, 2))
    vectorizer.fit(documents)
    X = vectorizer.transform(df['text'].tolist())
    nbrs = NearestNeighbors(n_neighbors=5,algorithm="brute",metric='cosine').fit(X)
    distances, indices = nbrs.kneighbors(X)
    ids=df['id'].tolist()
    outliers = df['outlier'].tolist()
    target=df['target'].tolist()
    id_2_target ={}
    for idx in df.index:
        id_2_target.update({df['id'][idx]:df['target'][idx]})
    good_dis=[]
    bad_dis=[]

    cntdiff=0
    cnt_same_target=0
    cnt_diff_target = 0

    for i in range(len(ids)):
        # if target[i]==1:
        #     continue


        if id_2_target[ids[i]] != id_2_target[ids[indices[i][1]]] and distances[i][1]==0:
            cntdiff+=1
        if outliers[i]==0 or outliers[indices[i][1]]==1:
            continue

        if outliers[i]==1 and target[i]==id_2_target[ids[indices[i][1]]]:
            cnt_same_target+=1
        if outliers[i]==1 and target[i]!=id_2_target[ids[indices[i][1]]]:
            cnt_diff_target+=1

        if target[i]==1:
            bad_dis.append(distances[i][1])
        else:
            good_dis.append((distances[i][1]))

    MatplotUtils.show_multi_data_histogram([good_dis,bad_dis],["good","bad"])

    print(cntdiff)
    print(f"cnt_same_target {cnt_same_target} cnt_diff_target {cnt_diff_target}")
    # print(indices)
    # print(distances)

def explore_hardcode(raw_train_df,raw_test_df):
    print("hihi")
    prediction_df=pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/loc-submit-prediction.csv")
    assert len(raw_test_df) == len(prediction_df)
    abc=raw_test_df['id'].tolist()
    bcd=prediction_df['id'].tolist()
    for i in range(len(raw_test_df)):
        assert abc[i]==bcd[i]
    raw_test_df=raw_test_df.assign(pred=prediction_df['target'].tolist())

    all_df=pd.DataFrame(data={'id':raw_train_df['id'].tolist()+raw_test_df['id'].tolist(),'text':raw_train_df['text'].tolist()+raw_test_df['text'].tolist()})
    assert len(all_df)==len(raw_test_df)+len(raw_train_df)

    all_df['text'] = all_df['text'].apply(normalize_text)
    all_df = all_df.drop_duplicates(subset=['text'], keep='first')
    documents = (all_df['text'].tolist())
    vectorizer = CountVectorizer(min_df=1, stop_words=STOP_WORDS, ngram_range=(1, 2))
    vectorizer.fit(documents)
    X = vectorizer.transform(all_df['text'].tolist())
    nbrs = NearestNeighbors(n_neighbors=5, algorithm="brute", metric='cosine').fit(X)
    distances, indices = nbrs.kneighbors(X)
    ids = all_df['id'].tolist()

    id_2_text={}
    id_2_tar={}
    id_2_keyword={}
    raw_train_df['keyword']=raw_train_df['keyword'].fillna("")
    for idx in raw_train_df.index:
        id_2_text.update({raw_train_df['id'][idx]:raw_train_df['text'][idx]})
        id_2_tar.update({raw_train_df['id'][idx]:raw_train_df['target'][idx]})
        id_2_keyword.update({raw_train_df['id'][idx]:raw_train_df['keyword'][idx]})
    for idx in raw_test_df.index:
        id_2_text.update({raw_test_df['id'][idx]:raw_test_df['text'][idx]})

    id_2_pred = {}
    for idx in prediction_df.index:
        id_2_pred.update({prediction_df['id'][idx]:prediction_df['target'][idx]})

    cnt=0
    cnt_no_near=0
    cnt_diff_tar=0
    diff_list=[]
    for i in range(0,len(ids)):
        curid = ids[i]
        if curid not in id_2_pred:
            continue
        cnt+=1

        near_id =-1
        dist=-1
        for t in range(1,5):
            id = ids[indices[i][t]]
            if id not in id_2_pred:
                near_id=id
                dist=distances[i][t]
                break
        if near_id==-1:
            cnt_no_near+=1
            continue
        if id_2_tar[near_id]!=id_2_pred[curid] and dist<0.5:
            cnt_diff_tar+=1
            diff_list.append((curid,near_id,dist))

    print(f"processed {cnt} with no near id {cnt_no_near} cnt_diff_tar {cnt_diff_tar}")

    diff_list.sort(key=lambda x:x[2])

    for (id1,id2,dis) in diff_list:
        print(dis)
        print(f"test {id_2_text[id1]} .{id1}")
        print(f"tran {id_2_text[id2]} .{id_2_tar[id2]}.{id_2_keyword[id2]}.{id2}")
        print('')


def explore_knn_prediction(raw_test_df):
    print("hihi")
    prediction_df=pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/loc-submit-prediction.csv")
    assert len(raw_test_df) == len(prediction_df)
    abc=raw_test_df['id'].tolist()
    bcd=prediction_df['id'].tolist()
    for i in range(len(raw_test_df)):
        assert abc[i]==bcd[i]
    raw_test_df=raw_test_df.assign(pred=prediction_df['target'].tolist())

    all_df=pd.DataFrame(data={'id':raw_test_df['id'].tolist(),'text':raw_test_df['text'].tolist()})

    all_df['text'] = all_df['text'].apply(normalize_text)
    all_df = all_df.drop_duplicates(subset=['text'], keep='first')
    documents = (all_df['text'].tolist())
    vectorizer = CountVectorizer(min_df=1, stop_words=STOP_WORDS, ngram_range=(1, 2))
    vectorizer.fit(documents)
    X = vectorizer.transform(all_df['text'].tolist())
    NUM_NBRS=10
    nbrs = NearestNeighbors(n_neighbors=NUM_NBRS, algorithm="brute", metric='cosine').fit(X)
    distances, indices = nbrs.kneighbors(X)
    ids = all_df['id'].tolist()

    id_2_text={}
    id_2_tar={}
    id_2_keyword={}
    for idx in raw_test_df.index:
        id_2_text.update({raw_test_df['id'][idx]:raw_test_df['text'][idx]})
        id_2_keyword.update(({raw_test_df['id'][idx]:raw_test_df['keyword'][idx]}))

    id_2_pred = {}
    for idx in prediction_df.index:
        id_2_pred.update({prediction_df['id'][idx]:prediction_df['target'][idx]})

    diff_list=[]
    list_edge=[]
    for i in range(0,len(ids)):
        for j in range(1,NUM_NBRS):
            if distances[i][j]<0.5:
                list_edge.append((ids[i],ids[indices[i][j]]))


    print(f"found {len(list_edge)} edges")

    list_components=disjoin_set(ids,list_edge)

    diff_list.sort(key=lambda x:x[2])

    cnt=1
    for com in list_components:
        # if len(com)==1:
        #     continue
        cnt+=1
        print(cnt)
        for id in com:
            print(f"test {id_2_text[id]} .{id}.{id_2_pred[id]}.{id_2_keyword[id]}")
        print('')


def test_knn(df):
    bad_data_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/wrong-ids.csv")
    bad_data_df=bad_data_df[['id']]
    print(f"len of bad data = {len(bad_data_df)}")
    bad_data_df=bad_data_df.assign(outlier=[1]*len(bad_data_df))

    df =pd.merge(df,bad_data_df,how='left',on='id')
    df['outlier']=df['outlier'].fillna(0)
    df['outlier']=df['outlier'].astype(int)
    print(f"len outlier in df = {len(df[df['outlier']==1])}")
    print(f"len non-outlier in df = {len(df[df['outlier'] == 0])}")

    df['text'] = df['text'].apply(normalize_text)
    df = add_location_to_text(df)
    df = add_keyword_to_text(df)
    df['keyword'] = df['keyword'].fillna('')
    df['keyword'] = df['keyword'].apply(normalize_keyword)

    documents = (df['text'].tolist())
    vectorizer = CountVectorizer(min_df=2, stop_words=STOP_WORDS, ngram_range=(1, 2))
    vectorizer.fit(documents)
    scores=[]
    # for test in range(1,51):
    #
    #     df_train, df_test = split_data_by_keyword(df, 0.7, random_seed=test)
    #
    #     X_train = vectorizer.transform(df_train['text'].tolist())
    #     Y_train = df_train['outlier']
    #
    #
    #     X_test = vectorizer.transform(df_test['text'].tolist())
    #     Y_test = df_test['outlier']
    #
    #     # knn = KNeighborsClassifier(n_neighbors=1)
    #     # knn.fit(X_train,Y_train)
    #     # Y_pred=knn.predict(X_test)
    #
    #     model = LogisticRegression(max_iter=2000)
    #     model.fit(X_train, Y_train)
    #     Y_pred = model.predict(X_test)
    #
    #     scores.append(accuracy_score(Y_pred,Y_test))
    # MatplotUtils.show_histogram(scores)
    # print(f'mean acc = {np.mean(scores)}')


    # X = vectorizer.transform(df['text'].tolist())
    # nbrs = NearestNeighbors(n_neighbors=5,algorithm="brute",metric='cosine').fit(X)
    # distances, indices = nbrs.kneighbors(X)
    #
    # ids=df['id'].tolist()
    # outlier=df['outlier'].tolist()
    # good_data_dis=[]
    # bad_data_dis=[]
    # for i in range(0,len(ids)):
    #     if outlier[i]==1:
    #         bad_data_dis.append(distances[i][2])
    #     else:
    #         good_data_dis.append(distances[i][2])
    #
    # good_data_dis.sort(key=lambda x:x,reverse=True)
    # bad_data_dis.sort(key=lambda x: x, reverse=True)
    # print(good_data_dis)
    # print(bad_data_dis)
    #
    # print(np.mean(good_data_dis))
    # print(np.mean(bad_data_dis))
    # MatplotUtils.show_multi_data_histogram([good_data_dis,bad_data_dis],['good','bad'])

    print(f"len before drop dup {len(df)}")
    df=df.drop_duplicates(subset=['text'],keep='first')
    print(f"len after drop dup {len(df)}")
    for test in range(0,1):
        df_train, df_test = split_data_by_keyword(df, 0.8, random_seed=test)
        good_train = df_train[df_train['outlier']==0]
        bad_train = df_train[df_train['outlier'] == 1]
        good_test = df_test[df_test['outlier'] == 0]
        bad_test = df_test[df_test['outlier'] == 1]
        good_dis=[]

        good_train_vec=vectorizer.transform(good_train['text'].tolist()).toarray()

        good_test_vec = vectorizer.transform(good_test['text'].tolist()).toarray()
        bad_test_vec = vectorizer.transform(bad_test['text'].tolist()).toarray()

        cnt=0
        bad_dis=[]
        print(f"bad test len {len(bad_test_vec)}")
        loctes=[]
        for v1 in bad_test_vec[:200]:
            cnt+=1
            if cnt%20==0:
                print(cnt)

            res=distance.cdist([v1],good_train_vec,metric='cosine')
            if np.isnan(np.nanmin(res)):
                print("YES")
                continue
            bad_dis.append(np.nanmin(res))
        print(len(bad_dis))
        #mean distance bad test 0.79288388858008 mindf=1
        #mean distance bad test 0.6466350001873056 mindf=2
        print(f"mean distance bad test {np.mean(bad_dis)}")

        cnt = 0
        good_dis = []
        print(f"bad test len {len(good_test_vec)}")
        cnt=0
        for v1 in good_test_vec[:200]:
            cnt += 1
            if cnt % 20 == 0:
                print(cnt)
            res=distance.cdist([v1], good_train_vec, metric='cosine')
            if np.isnan(np.nanmin(res)):
                print("YES")
                continue

            good_dis.append(np.nanmin(res))
        print(len(good_dis))
        #mean distance good test 0.5971976633447953 mindf=1
        #mean distance good test 0.4575191335356463 mindf=2
        print(f"mean distance good test {np.mean(good_dis)}")
        MatplotUtils.show_multi_data_histogram([good_dis, bad_dis], ['good', 'bad'])
        print("a")


def collect_good_words_on_topic(train_df,test_df):
    select_keywords=['snowstorm']

    train_df['keyword']=train_df['keyword'].fillna('').apply(normalize_keyword)
    test_df['keyword'] = test_df['keyword'].fillna('').apply(normalize_keyword)

    def norm_text(text):
        text=text.lower()
        for p in PUNCTUATIONS_LIKE_SPACE:
            text = text.replace(p, SPACE)
        for c in LESS_COMMON_CHARACTERS:
            text = text.replace(c, '')
        return text
    train_df['text']=train_df['text'].apply(norm_text)
    test_df['text'] = test_df['text'].apply(norm_text)

    word_stats_on_topic = calc_word_rate_on_topic(train_df[train_df['keyword'].isin(select_keywords)],train_df)
    word_stats_on_target = calc_word_rate_on_target(train_df[train_df['keyword'].isin(select_keywords)])
    word_freq_overall = dict(calc_word_overall_rate(train_df))
    word_freq_test = calc_word_overall_rate(test_df)
    for i in range(0,len(word_freq_test)):
        v=word_freq_test[i][1]
        word_freq_test[i]=(word_freq_test[i][0],list(filter(lambda x:x[0] in select_keywords,v)))
    word_freq_test=dict(word_freq_test)
    assert len(word_stats_on_target) == len(word_stats_on_topic)
    word_stats_on_topic.sort(key=lambda x:x[0])
    word_stats_on_target.sort(key=lambda x: x[0])
    for i in range(0,len(word_stats_on_topic)):
        assert word_stats_on_topic[i][0]==word_stats_on_target[i][0]


    map_data={}
    map_data.update({'word':[x[0] for x in word_stats_on_target]})
    map_data.update({'tar':[x[1] for x in word_stats_on_target]})
    map_data.update({'tar-freq': [x[2] for x in word_stats_on_target]})
    map_data.update({'tar-rate': [x[3] for x in word_stats_on_target]})
    map_data.update({'topic-rate': [x[1] for x in word_stats_on_topic]})

    stat_df=pandas.DataFrame(map_data)
    stat_df['test-freq']=stat_df.apply(lambda row:(0 if row['word'] not in word_freq_test else sum(map(lambda x:x[1],word_freq_test.get(row['word'])))),axis=1)
    stat_df['topic-example'] = stat_df.apply(lambda row: word_freq_overall.get(row['word']), axis=1)
    stat_df=stat_df[(stat_df['tar-freq']>1) & (stat_df['tar-rate']>=0.7) & (stat_df['test-freq']>=1)]
    PdHelper.print_full(stat_df.sort_values(by=['topic-rate'],ascending=False))

    PdHelper.print_full(test_df[test_df['keyword'].isin(select_keywords)])

def init_data(raw_train_df, raw_test_df):
    arr_people_tag = []
    for text in raw_train_df['text'].tolist() + raw_test_df['text'].tolist():
        tmp = text
        punc = string.punctuation
        punc = punc.replace('@', '')  # exclude @
        for p in punc:
            tmp = tmp.replace(p, SPACE)
        words = tmp.split(SPACE)

        for w in words:
            if w.startswith('@') and not w.startswith(KEYWORD_TAG):
                arr_people_tag.append(w)

    people_tag_freq = Utils.group_list(arr_people_tag)
    global FREQ_PEOPLE_TAG_MAP
    FREQ_PEOPLE_TAG_MAP = dict(people_tag_freq)
    print(f"init FREQ_PEOPLE_TAG_MAP len ={len(FREQ_PEOPLE_TAG_MAP)}")

    arr_link = []
    for text in raw_train_df['text'].tolist() + raw_test_df['text'].tolist():
        tmp = text
        for p in ',\n':  # exclude @
            tmp = tmp.replace(p, SPACE)
        words = tmp.split(SPACE)

        for w in words:
            if 'https://' in w or 'http://' in w:
                arr_link.append(w)


    link_freq = Utils.group_list(arr_link)
    global FREQ_LINK_TAG_MAP
    FREQ_LINK_TAG_MAP = dict(link_freq)
    print(f"init FREQ_PEOPLE_LINK_MAP len ={len(FREQ_LINK_TAG_MAP)}")
    print(list(FREQ_LINK_TAG_MAP.items())[:10])

    global HARDCODE_KEYWORD_LEMMA
    HARDCODE_KEYWORD_LEMMA.append(("buildings-fire",["buildings-on-fire","buildings-burning","burning-buildings"]))
    HARDCODE_KEYWORD_LEMMA.append(("fire", ["flames"]))
    HARDCODE_KEYWORD_LEMMA.append(("sunk", ["sinking"]))
    HARDCODE_KEYWORD_LEMMA.append(("thunderstorm", ["thunder","lightning"])) # should remove thunder ?
    HARDCODE_KEYWORD_LEMMA.append(("trauma", ["traumatised"]))
    HARDCODE_KEYWORD_LEMMA.append(("storm", ["typhoon","violent-storm","hurricane"]))
    HARDCODE_KEYWORD_LEMMA.append(("tornado", ["whirlwind","windstorm"]))
    HARDCODE_KEYWORD_LEMMA.append(("wildfire", ["wild-fires"])) #95% target =1
    HARDCODE_KEYWORD_LEMMA.append(("forest-fire", ["bush-fires"])) #90% target =1
    HARDCODE_KEYWORD_LEMMA.append(("hail", ["hailstorm"]))

    # Detonation  FedEx Virgil One Direction
    HARDCODE_TARGET_TESTDATA.update({854:1,831:1,861:1,839:1,847:1,862:1,6161:1,6176:1,3854:1,3871:1,3872:1,529:0,532:0})
    #Biggest security update in history coming up: Google patches Android hijack bug Stagefright http://t.co/bFoZaptqCo .6142.1.hijack
    HARDCODE_TARGET_TESTDATA.update({6142:0,6149:0})
    # #3: Car Recorder ZeroEdgeå¨ Dual-lens Car Camera Vehicle Tacraffic/Driving History/Accident Camcorder  Large Re... http://t.co/kKFaSJv6Cj .106.1.accident
    #For Legal and Medical Referral Service @1800_Injured Call us at: 1-800-465-87332 #accident #slipandfall #dogbite .116.1.accident
    HARDCODE_TARGET_TESTDATA.update({106: 0,116:0})
    #Experts in France begin examining airplane debris found on Reunion Island: French air accident experts on Wedn...  http://t.co/KuBsM16OuD .239.1.airplane%20accident
    # HARDCODE_TARGET_TESTDATA.update({234: 0, 239: 0,236:0})   wrong

    #iBliz140: Breaking news! Unconfirmed! I just heard a loud bang nearby. in what appears to be a blast of wind from my neighbour's ass. .6805.0.loud%20bang
    # HARDCODE_TARGET_TESTDATA.update({6805: 1, 6826: 1,6812:1,6825:1,6816:1,6827:1,6818:1,6846:1,6822:1,6829:1})   wrong

    #Owner of Chicago-Area Gay Bar Admits to Arson Scheme: Frank Elliott pleaded guilty to hiring an arsonist to to... http://t.co/L82mrYxfNK .597.0.arsonist
    HARDCODE_TARGET_TESTDATA.update({591: 1, 597: 1,593:1})

    #Exploration takes seismic shift in Gabon to Somalia ÛÒ WorldOilåÊ(subscription) http://t.co/Hs6OhRFsA9 .8616.0.seismic
    HARDCODE_TARGET_TESTDATA.update({8616: 1})

    # ISIS # terrorism
    HARDCODE_TARGET_TESTDATA.update({9461: 1,9478:1,9487:1})

    #Manuscript suspiciously rejected by Earthquake Science Journal could rock your world https://t.co/6vBQEwsl1J viaYouTube .4370.1.earthquake
    HARDCODE_TARGET_TESTDATA.update({4370: 0})
    #test #SISMO ML 2.0 SICILY ITALY http://t.co/vsWivoDCkL .4394.0.earthquake
    HARDCODE_TARGET_TESTDATA.update({4394: 1})

    #test What scenes at Trent Bridge England could win the #Ashes today at this rate! #Pandemonium .7597.1.pandemonium
    HARDCODE_TARGET_TESTDATA.update({7597: 0})

    #@CrawliesWithCri @Rob0Sullivan What do they consider 'heat'? When I lived in London they thought a 'heat wave' was like 25C hahaha .6042.1.heat%20wave
    HARDCODE_TARGET_TESTDATA.update({6042: 0})

    #Little girl: Mommy is that a boy or a girl? Welp www .6129.1.hellfire
    HARDCODE_TARGET_TESTDATA.update({6129: 0})

    #Patience Jonathan On The Move To Hijack APC In Bayelsa State http://t.co/zzMwIebuci .6136.0.hijack
    HARDCODE_TARGET_TESTDATA.update({6144: 1,6136:1,6179:1})

    #hijack
    HARDCODE_TARGET_TESTDATA.update({6157: 1, 6158: 1,6168:1})

    #Pakistani Terrorist Was Captured By Villagers He Took Hostage - NDTV http://t.co/C5X10JAkGE .6289.0.hostage
    HARDCODE_TARGET_TESTDATA.update({6289: 1,6295:1,6319:1,6359:1,6367:1})

    #0.80355 above
    #I made him fall and hegot injured https://t.co/XX8RO94fBC .6460.0.injured
    HARDCODE_TARGET_TESTDATA.update({6460: 1,6468:1,6507:1,6509:1})

    #Mexican beaches inundated with algae http://t.co/Kcm4sgrceR .6595.0.inundated
    #MY SCHOOL IS INUNDATED and they wont let us go home til 4pm help .6623.0.inundated
    HARDCODE_TARGET_TESTDATA.update({6595: 1,6623:1})

    #landslide
    #WATCH: Likely landslide victims denied assistance again http://t.co/BxiohtiC5X .6666.0.landslide
    HARDCODE_TARGET_TESTDATA.update({6666: 1,6696:1})

    #test Southeast Regional is in a lightning delay. Buckle up folks this is gonna take awhile. .6768.0.lightning
    #test Two Films From Villanova University Students Selected As 2015 Student Academy Award Finalists: Lightning struckÛ_ http://t.co/4jyQfhHmXf .6782.1.lightning
    HARDCODE_TARGET_TESTDATA.update({6768: 1,6782:0,6791:1})

    #Asad is also a mass murderer.  https://t.co/dZOReZpxvR .6901.0.mass%20murderer
    HARDCODE_TARGET_TESTDATA.update({6901: 1})

    # Tajikistan #Mudslides #China aids to #Mudslide-hit #Tajiks http://t.co/BD546mtcpN .7151.0.mudslide
    HARDCODE_TARGET_TESTDATA.update({7151: 1})
    #test The Return of Planet-X Book by Jaysen Rand Wormwood Natural Disaster Survivors http://t.co/k8cgZiirps http://t.co/d5sXGvp2pI .7222.1.natural%20disaster
    HARDCODE_TARGET_TESTDATA.update({7222: 0})

    #Approaching rainstorm http://t.co/3YREo3SpNU .7919.0.rainstorm
    #Photo: Rainstorm On The Way To Brooks NearåÊDinosaur Provincial Park Alberta July 11 2015. http://t.co/Mp9EccqvLt .7927.0.rainstorm
    HARDCODE_TARGET_TESTDATA.update({7919: 1,7927:1,7951:1,7956:1})

    #Florida Firefighters Rescue Scared Meowing Kitten from Vent Shaft http://t.co/RVKffcxbvC #fire #firefighter .8063.0.rescue
    #kayaking about killed us so mom and grandma came to the rescue. .8078.0.rescue
    HARDCODE_TARGET_TESTDATA.update({8063: 1,8078:1})

    #http://t.co/OK9EBHurfl If your neighborhood school falls into a sinkhole State Sen. Steve Yarbrough may be close by... .8645.1.sinkhole
    HARDCODE_TARGET_TESTDATA.update({8645: 0})

    #Be careful out there. http://t.co/KoBavdEKWn .10510.0.wildfire
    HARDCODE_TARGET_TESTDATA.update({10510: 1})

    ##CityofCalgary has activated its Municipal Emergency Plan. #yycstorm .10875.0.nan
    HARDCODE_TARGET_TESTDATA.update({10875: 1})



    print(f"keyword mapping len = {len(KEYWORD_MAPPING)}")
    for k in KEYWORD_MAPPING.items():
        print(k)


    texts = raw_train_df['text'].tolist()+raw_test_df['text'].tolist()
    words=[]
    for t in texts:
        t=t.lower()
        for p in ' \n':
            t = t.replace(p, SPACE)
        words+= t.split(SPACE)

    global VOCABULARY_FREQ
    VOCABULARY_FREQ = dict(Utils.group_list(words))

    print(f"init VOCABULARY_FREQ len {len(VOCABULARY_FREQ)}")

raw_train_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/train.csv")
raw_test_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/test.csv")


init_data(raw_train_df, raw_test_df)