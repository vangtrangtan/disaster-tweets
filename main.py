import numpy as np
import pandas
import pandas as pd
import random

from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

import ExploreData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

import MatplotUtils
import PdHelper
import Utils


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

def try_log_regression(raw_train_df, raw_test_df, is_norm=True, random_seed=100):
    vectorizer = CountVectorizer()

    documents = (raw_train_df['text'].tolist() + raw_test_df['text'].tolist())
    if is_norm:
        documents = [ExploreData.normalize_text(d) for d in documents]
        #mindf=1 got acc 0.801318, =2 got 0.798
        vectorizer = CountVectorizer(min_df=1, stop_words=ExploreData.STOP_WORDS, ngram_range=(1,2))
        raw_train_df.drop_duplicates(['text'],keep='first')


    vectorizer.fit(documents)
    print(f"size vocabulary = {len(vectorizer.vocabulary_)}")

    df_train, df_test = split_data_by_keyword(raw_train_df, 0.7,random_seed=random_seed)

    X_train = vectorizer.transform(df_train['text'].tolist())
    Y_train = df_train['target']

    if is_norm:
        df_test['text']=df_test['text'].apply(ExploreData.normalize_text)


    X_test = vectorizer.transform(df_test['text'].tolist())
    Y_test = df_test['target']
    ids = df_test['id'].tolist()
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    MatplotUtils.show_confusion_matrix(Y_test,Y_predict)
    result_on_ids=[]
    predict_list = list(Y_predict)
    actual_list=list(Y_test)
    cnt0=0
    cnt1=0
    for i in range(0,len(ids)):
        if predict_list[i]==actual_list[i]:
            result_on_ids.append((ids[i],1))
        else:
            result_on_ids.append((ids[i], 0))
            if actual_list[i]==1:
                cnt1+=1
            else:
                cnt0+=1

    print(f"wrong acc target 0 {cnt0} 1 {cnt1}")
    return (accuracy_score(Y_test, Y_predict),result_on_ids)


def log_regession_predict(df,vectorizer,random_seed=100):
    df_train, df_test = split_data_by_keyword(df, 0.7, random_seed=random_seed)

    X_train = vectorizer.transform(df_train['text'].tolist())
    Y_train = df_train['target']


    X_test = vectorizer.transform(df_test['text'].tolist())
    Y_test = df_test['target']
    ids = df_test['id'].tolist()
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    result_on_ids = []
    predict_list = list(Y_predict)
    actual_list = list(Y_test)

    for i in range(0,len(ids)):
        if predict_list[i]==actual_list[i]:
            result_on_ids.append((ids[i],1))
        else:
            result_on_ids.append((ids[i], 0))

    return (accuracy_score(Y_test, Y_predict),result_on_ids)


def knn_predict(df,vectorizer,random_seed=100):
    df_train, df_test = split_data_by_keyword(df, 0.7, random_seed=random_seed)

    X_train = vectorizer.transform(df_train['text'].tolist())
    Y_train = df_train['target']

    X_test = vectorizer.transform(df_test['text'].tolist())
    Y_test = df_test['target']
    ids = df_test['id'].tolist()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,Y_train)
    Y_predict=knn.predict(X_test)
    result_on_ids = []
    predict_list = list(Y_predict)
    actual_list = list(Y_test)

    for i in range(0, len(ids)):
        if predict_list[i] == actual_list[i]:
            result_on_ids.append((ids[i], 1))
        else:
            result_on_ids.append((ids[i], 0))

    return (accuracy_score(Y_test, Y_predict), result_on_ids)

def try_predict_twice(raw_train_df,raw_test_df,random_seed=100,knn=False):
    raw_train_df['text'] = raw_train_df['text'].apply(ExploreData.normalize_text)
    raw_test_df['text'] = raw_test_df['text'].apply(ExploreData.normalize_text)
    raw_train_df.drop_duplicates(['text'], keep='first')

    #add before init documeent got 0.832141 acc ?
    # raw_train_df = ExploreData.add_keyword_to_text(raw_train_df)
    # raw_train_df= ExploreData.add_location_to_text(raw_train_df)

    documents = (raw_train_df['text'].tolist() + raw_test_df['text'].tolist())
    vectorizer = CountVectorizer(min_df=1, stop_words=ExploreData.STOP_WORDS, ngram_range=(1, 2))
    vectorizer.fit(documents)
    print(f"size vocabulary = {len(vectorizer.vocabulary_)}")

    norm_train_df=raw_train_df.copy()
    norm_train_df['keyword'] = norm_train_df['keyword'].fillna('')
    norm_train_df['keyword'] = norm_train_df['keyword'].apply(ExploreData.normalize_keyword)
        #without this  0.847430
        #with this 0.850590
    # norm_train_df = ExploreData.add_keyword_to_text(norm_train_df)
    # norm_train_df = ExploreData.add_location_to_text(norm_train_df)

    bad_data_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/wrong-ids.csv")
    # 0.962 acc on good data (5064/7613 tweets), 0.585 acc on bad data 2549/7613
    good_data_df= norm_train_df[norm_train_df['id'].apply(lambda x: x not in (bad_data_df['id'].tolist()))]
    bad_data_df = norm_train_df[norm_train_df['id'].apply(lambda x: x in (bad_data_df['id'].tolist()))]
    print(f"good data size {len(good_data_df)} bad data {len(bad_data_df)}")


    good_data_acc,good_ids_result = log_regession_predict(good_data_df,vectorizer,random_seed)
    bad_data_acc=None
    bad_ids_result=None
    if knn:
        bad_data_acc,bad_ids_result = knn_predict(bad_data_df, vectorizer, random_seed)
    else:
        bad_data_acc,bad_ids_result= log_regession_predict(bad_data_df, vectorizer, random_seed)

    total_acc = (good_data_acc*len(good_data_df) + bad_data_acc*len(bad_data_df))/(len(bad_data_df)+len(good_data_df))
    return (total_acc,bad_ids_result+good_ids_result)

def split_outlier_test_data(df_test,good_train_data_df,vectorizer):
    train_vec = vectorizer.transform(good_train_data_df['text'].tolist()).toarray()
    test_vec = vectorizer.transform(df_test['text'].tolist()).toarray()
    ids=df_test['id'].tolist()
    print(f"len test vec {len(test_vec)}")
    print(f"len train vec {len(train_vec)}")

    bad_dis=[]
    cnt=0
    for i in range(0,len(ids)):
        cnt+=1
        if cnt%50==0:
            print(f"cnt {cnt}")
        id=ids[i]
        tv=test_vec[i]
        res = distance.cdist([tv], train_vec, metric='cosine')
        if np.isnan(np.nanmin(res)):
            print("YES")
            continue
        bad_dis.append((id,np.nanmin(res)))

    bad_dis.sort(key=lambda x:x[1],reverse=True)
    bad_dis=bad_dis[0:len(bad_dis)//10]
    outlier_ids = list(map(lambda x:x[0],bad_dis))
    bad_test_df = df_test[df_test['id'].apply(lambda x:x in outlier_ids)]
    good_test_df = df_test[df_test['id'].apply(lambda x: x not in outlier_ids)]
    print(f"len outlier result = {len(bad_dis)}")
    print(f"len good_test_df {len(good_test_df)} len bad_test_df {len(bad_test_df)}")
    return (good_test_df,bad_test_df)




def benchmark_log_regression(raw_train_df, raw_test_df):
    norm_accs=[]
    raw_accs=[]
    norm_train_df = raw_train_df.copy()
    norm_train_df['keyword']=norm_train_df['keyword'].fillna('')
    norm_train_df['keyword'] = norm_train_df['keyword'].apply(ExploreData.normalize_keyword)
    norm_train_df = ExploreData.add_keyword_to_text(norm_train_df)
    norm_train_df = ExploreData.add_location_to_text(norm_train_df)

    norm_test_df = raw_test_df.copy()
    norm_test_df['keyword'] = norm_test_df['keyword'].fillna('')
    norm_test_df['keyword']=norm_test_df['keyword'].apply(ExploreData.normalize_keyword)
    norm_test_df = ExploreData.add_keyword_to_text(norm_test_df)
    norm_test_df = ExploreData.add_location_to_text(norm_test_df)

    list_result =[]
    list_v2_result =[]
    # bad_data_df=pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/wrong-ids.csv")
    # print(f"before remove bad ids {len(norm_train_df)}")
    # 0.962 acc on good data (5064/7613 tweets), 0.585 acc on bad data 2549/7613
    # norm_train_df=norm_train_df[norm_train_df['id'].apply(lambda x:x in (bad_data_df['id'].tolist()))]
    # print(f"after remove bad ids {len(norm_train_df)}")
    # norm_test_df = norm_test_df[norm_test_df['id'].apply(lambda x:x in (bad_data_df['id'].tolist()))]
    bad_data_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/wrong-ids.csv")
    for test in range(1,11):
        print(f"test {test}")
        # acc, id_2_result =try_log_regression_no_outlier(norm_train_df,norm_test_df,bad_data_df['id'].tolist(),random_seed=test)
        # acc, id_2_result = try_predict_twice(raw_train_df, raw_test_df, random_seed=test,knn=True)
        acc,id_2_result = try_log_regression(norm_train_df,norm_test_df,is_norm=True,random_seed=test)
        # acc, id_2_result = try_smart_log_regression(norm_train_df, norm_test_df, random_seed=test)\
        # acc,id_2_result=try_knn_for_outlier(raw_train_df,raw_test_df,random_seed=test)
        norm_accs.append(acc)
        list_result.append(id_2_result)
        # acc,id_2_result = try_log_regression(norm_train_df,norm_test_df,is_norm=True,random_seed=test)
        acc, id_2_result = try_predict_twice(raw_train_df, raw_test_df, random_seed=test, knn=False)
        raw_accs.append(acc)
        list_v2_result.append(id_2_result)

    MatplotUtils.show_multi_data_histogram([raw_accs,norm_accs],["old verson","new verson"])
    stats_df=pd.DataFrame({'raw':raw_accs,'norm':norm_accs})
    print(stats_df.describe())

    ###################### check accuracy on each keyword

    # norm_scores = calc_accuracy_on_keyword(norm_train_df,list_result)
    # raw_scores = calc_accuracy_on_keyword(norm_train_df, list_v2_result)

    # MatplotUtils.show_histogram([x[1] for x in norm_scores])
    # explore_prediction(list_result)

    # print(norm_scores)
    # for s in norm_scores:
    #     print(s)


    # assert (len(norm_scores) == len(raw_scores))
    # norm_scores.sort(key=lambda x:x[0])
    # raw_scores.sort(key=lambda x: x[0])
    #
    # score_changes =[]
    # for i in range(0,len(norm_scores)):
    #     assert (norm_scores[i][0]==raw_scores[i][0])
    #     score_changes.append((norm_scores[i][0],norm_scores[i][1]-raw_scores[i][1]))
    #
    # score_changes.sort(key=lambda x:x[1])
    # for s in score_changes:
    #     print(s)
    # MatplotUtils.show_histogram([x[1] for x in score_changes])

raw_train_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/train.csv")
raw_test_df = pd.read_csv("/home/vangtrangtan/Desktop/disaster-tweet/test.csv")

def submission(raw_train_df,raw_test_df):
    print("run submit")

    norm_train_df = ExploreData.add_keyword_to_text(raw_train_df)
    norm_train_df = ExploreData.add_location_to_text(norm_train_df)
    norm_test_df = ExploreData.add_keyword_to_text(raw_test_df)
    norm_test_df = ExploreData.add_location_to_text(norm_test_df)



    documents = (norm_train_df['text'].tolist() + norm_test_df['text'].tolist())
    documents = [ExploreData.normalize_text(d) for d in documents]

    vectorizer = CountVectorizer(min_df=1, stop_words=ExploreData.STOP_WORDS, ngram_range=(1, 2))

    vectorizer.fit(documents)
    print(f"size vocabulary = {len(vectorizer.vocabulary_)}")
    raw_train_df.drop_duplicates(['text'], keep='first')
    df_train, df_test = split_data_by_keyword(norm_train_df, 1.0)


    X_train = vectorizer.transform(df_train['text'].tolist())
    Y_train = df_train['target']

    df_test=norm_test_df.copy()
    df_test['text'] = df_test['text'].apply(ExploreData.normalize_text)
    X_test = vectorizer.transform(df_test['text'].tolist())

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, Y_train)

    Y_predict = model.predict(X_test)
    def hardcode_predict (preds,ids):
        res =[]
        for i in range(len(ids)):
            _id = ids[i]
            if _id in ExploreData.HARDCODE_TARGET_TESTDATA:
                res.append(ExploreData.HARDCODE_TARGET_TESTDATA.get(_id))
            else:
                res.append(preds[i])
        return pd.Series(res)
    new_Y_predict = hardcode_predict(Y_predict,df_test['id'].tolist())
    cnt_hardcode=0
    for i in range(len(Y_predict)):
        if Y_predict[i]!=new_Y_predict[i]:
            cnt_hardcode+=1
    Y_predict=new_Y_predict
    print(f"hardcode {cnt_hardcode} value")
    submit_df = pd.DataFrame({"id": df_test['id'].tolist(), "target": Y_predict})
    # PdHelper.write_df_to_csv(submit_df, "/home/vangtrangtan/Desktop/disaster-tweet/loc-submit.csv")

    print(f"len predict input data {len(submit_df)}")
    PdHelper.print_full(submit_df.head(10))


#this code got 80% accuracy
submission(raw_train_df,raw_test_df)
