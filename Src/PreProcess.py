#This python code helps to preprocess the code for modelling stage
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
import spacy
from gensim.matutils import sparse2full
class PreProcess:
    def pre_process(self,x):
        #Normalization
        x=str(x).lower()
        #Removing Contraction
        x=x.replace("000,000","m").replace("000","k").replace("won't","will not").replace("can't","cannot").replace("shouldn't","should not")\
            .replace("didn't","did not").replace("doesn't","does not").replace("couldn't","could not").replace("ll","will").replace("%","percent")\
                .replace("$","dollar").replace("he's","he is").replace("he's","he is").replace("wouldn't","would not")
        #Removeing Unicode Characters
        x=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",x)
        #Removing Stopwords
        text=[word for word in x.split() if word  not in (stop)]
        #Stemming
        stemmer=PorterStemmer()
        text=" ".join([stemmer.stem(word) for word in text])
        #removing HTML tags
        bs4=BeautifulSoup(text)
        text=bs4.get_text()
        text=text.strip()
        return text
    def common_word(self,x,y):
        lst=[i for i in x.split(' ') if i in y.split(' ')]
        return len(lst)
        # Token Features

    def cwc_min(self,ws,val1,val2):
        c_len=min(val1,val2)
        return ws/c_len

    def cwc_max(self,ws,val1,val2):
        c_len=max(val1,val2)
        return ws/c_len

    def ctc_min(self,q1,q2):
        q1_list=q1.split()
        q2_list=q2.split()
        com_token=len(set(q1_list).intersection(set(q2_list)))
        min_tok=min(len(q1_list),len(q2_list))
        return com_token/min_tok

    def ctc_max(self,q1,q2):
        q1_list=q1.split()
        q2_list=q2.split()
        com_token=len(set(q1_list).intersection(set(q2_list)))
        max_tok=max(len(q1_list),len(q2_list))
        return com_token/max_tok

    def last_word_eq(self,q1,q2):
        q1_last_word=q1.split()[-1]
        q2_last_word=q2.split()[-1]
        if(q1_last_word==q2_last_word):
            return 1
        return 0

    def first_word_eq(self,q1,q2):
        q1_first_word=q1.split()[0]
        q2_first_word=q2.split()[0]
        if(q1_first_word==q2_first_word):
            return 1
        return 0
    # Length based features

    def longest_substr_ratio(self,val1,val2):
        m=list(distance.lcsubstrings(val1,val2))
        if(m==[]):
            ls=0
        else:
            ls=len(m[0].split())
        
        return ls/min(len(val1.split()),len(val2.split()))
    #Fuzzy based features

    def fuzzy_features(self,row):
        q1=row['question1']
        q2=row['question2']

        fuzz_feat=[0.0]*4

        #fuzzy ratio
        fuzz_feat[0]=fuzz.QRatio(q1,q2)

        #fuzzy partial ratio
        fuzz_feat[1]=fuzz.partial_ratio(q1,q2)

        #token sort ratio
        fuzz_feat[2]=fuzz.token_set_ratio(q1,q2)

        #token set ratio
        fuzz_feat[3]=fuzz.token_sort_ratio(q1,q2)

        return fuzz_feat
    def keep_token(self,t):
        return (t.is_alpha and not(t.is_space or t.is_punct or t.is_stop or t.like_num))
    def lemmatized(self,doc):
        return [ t.lemma_ for t in doc if self.keep_token(t)]

    def transformation(self,q_ar):
        #Basic preprocessing
        q_ar['question1']=q_ar['question1'].apply(self.pre_process)
        q_ar['question2']=q_ar['question2'].apply(self.pre_process)
        #Basic featurization
        q_ar['len_ques1']=q_ar['question1'].apply(len)
        q_ar['len_ques2']=q_ar['question2'].apply(len)
        q_ar['q1_wordcount']=q_ar['question1'].apply(lambda x:len(x.split(' ')))
        q_ar['q2_wordcount']=q_ar['question2'].apply(lambda x:len(x.split(' ')))
        q_ar['word_share']=q_ar[['question1','question2']].apply(lambda x:self.common_word(*x),axis=1)
        q_ar['cwc_min']=q_ar[['word_share','q1_wordcount','q2_wordcount']].apply(lambda x:self.cwc_min(*x),axis=1)
        q_ar['cwc_max']=q_ar[['word_share','q1_wordcount','q2_wordcount']].apply(lambda x:self.cwc_max(*x),axis=1)
        q_ar['ctc_min']=q_ar[['question1','question2']].apply(lambda x:self.ctc_min(*x),axis=1)
        q_ar['ctc_max']=q_ar[['question1','question2']].apply(lambda x:self.ctc_max(*x),axis=1)
        q_ar['last_word_eq']=q_ar[['question1','question2']].apply(lambda x:self.last_word_eq(*x),axis=1)
        q_ar['first_word_eq']=q_ar[['question1','question2']].apply(lambda x:self.first_word_eq(*x),axis=1)
        #using Length based features
        q_ar['mean_len']=q_ar[['q1_wordcount','q2_wordcount']].apply(lambda x:(x[0]+x[1])/2,axis=1)
        q_ar['abs_len_diff']=q_ar[['q1_wordcount','q2_wordcount']].apply(lambda x:abs(x[0]-x[1]),axis=1)
        q_ar['longest_substr_ratio']=q_ar[['question1','question2']].apply(lambda x:self.longest_substr_ratio(*x),axis=1)
        #using Fuzzy based features
        val=q_ar.apply(self.fuzzy_features,axis=1)
        q_ar['fuzzy_ratio']=list(map(lambda x:x[0],val))
        q_ar['fuzzy_partial_ratio']=list(map(lambda x:x[1],val))
        q_ar['token_set_ratio']=list(map(lambda x:x[2],val))
        q_ar['token_sort_ratio']=list(map(lambda x:x[3],val))

        ques_list2=list(q_ar['question1'])+list(q_ar['question2'])
        nlp=spacy.load("en_core_web_sm")

        doc=[self.lemmatized(nlp(que)) for que in ques_list2]
        docs_dict=Dictionary(doc)
        docs_dict.compactify()
        docs_corpus=[docs_dict.doc2bow(doc) for doc in doc]
        model_tfidf=TfidfModel(docs_corpus,id2word=docs_dict)
        docs_tfidf=model_tfidf[docs_corpus]
        docs_vec=np.vstack([sparse2full(c,len(docs_dict)) for c in docs_tfidf])
        tfidf_emb_vecs=np.vstack([nlp(docs_dict[i]).vector for i in range(len(docs_dict))])
        docs_emb=np.dot(docs_vec,tfidf_emb_vecs)
        q1,q2=np.vsplit(docs_emb,2)
        qu1=pd.DataFrame(q1,columns=range(0,96),index=q_ar.index)
        qu2=pd.DataFrame(q2,columns=range(96,192),index=q_ar.index)
        temp_data=q_ar[['len_ques1','len_ques2','q1_wordcount','q2_wordcount','word_share','cwc_min','cwc_max','ctc_min','ctc_max','last_word_eq','first_word_eq','mean_len','abs_len_diff','longest_substr_ratio','fuzzy_ratio','fuzzy_partial_ratio','token_set_ratio','token_sort_ratio']]
        tot_df=pd.concat([qu1,qu2],axis=1)
        df2=pd.concat([temp_data,tot_df],axis=1)


        return df2.to_numpy()






        

