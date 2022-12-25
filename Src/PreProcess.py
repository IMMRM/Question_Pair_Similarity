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
class PreProcess:
    def pre_process(x):
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