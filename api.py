import numpy as np 
import pandas as pd 
import os
from fuzzywuzzy import fuzz
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
stopword=nltk.corpus.stopwords.words('english')
import string
from flask import Flask,jsonify,request
import requests
import pickle
def dupli(q1,q2):
    df=pd.read_csv('./input/data.csv')
    df.drop(df[df.isnull().any(axis=1)].index,inplace=True)
    fe=['diff_len','common_words','fuzz_qratio','fuzz_WRatio','fuzz_partial_ratio','fuzz_token_sort_ratio']
    X=df[fe]
    y=df['is_duplicate']
    filename = 'finalized_model.sav'
    clf=pickle.load(open(filename, 'rb'))
    '''q1=input("first question - ")
    q2=input("first question - ")'''
    data=[[q1,q2]]
    dat=pd.DataFrame(data,columns=['question1','question2'])
    dat['len_q1'] = dat.question1.apply(lambda x: len(str(x)))
    dat['len_q2'] = dat.question2.apply(lambda x: len(str(x)))
    dat['diff_len'] = dat.len_q1 - dat.len_q2
    dat['len_char_q1'] = dat.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    dat['len_char_q2'] = dat.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
    dat['len_word_q1'] = dat.question1.apply(lambda x: len(str(x).split()))
    dat['len_word_q2'] = dat.question2.apply(lambda x: len(str(x).split()))
    dat['common_words'] = dat.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
    dat['fuzz_qratio'] = dat.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    dat['fuzz_WRatio'] = dat.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    dat['fuzz_partial_ratio'] = dat.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    dat['fuzz_token_sort_ratio'] = dat.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    x=dat[fe]
    pred=clf.predict(x)
    if(pred[0]==1):
        return "they are similar"
    else:
        return "they are different"
app= Flask(__name__)
@app.route('/api',methods=['POST'])
def ana():
    data=request.get_json()
    p={}
    r=data['text']
    x = r.split("?")
    output=dupli(x[0],x[1])

    return jsonify(results=output)    

if __name__=='__main__':
    app.run(port=9000,debug=True)

