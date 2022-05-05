# By Suryaveer @IIT indore
# GITHUB: https://github.com/surya-veer
# Handle: ayrusreev

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import joblib
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


# give IS_TRAIN as True when traning is needed            
IS_TRAIN = False

class Split:
        
        """This class is for spliting data into individual words and lemmatistion of words."""
        
        def into_tokens(self,msg):
            return TextBlob(msg).words

        def into_lemmas(self,message):
            words = TextBlob(message).words
            # for each word, take its "base form" = lemma 
            return [word.lemma for word in words]

class Train:

        """Traning and testing our model."""
        
        def __init__(self):
            self.clf = MultinomialNB()
            
        def train(self,x,y):
            self.clf.fit(x,y)
            
        def score(self,x,y):
            return self.clf.score(x,y)
        
        def test(self,y):
            return self.clf.predict(y)
        
        def accuracy(self,x,y):
            acc = accuracy_score(x,y,normalize=False)
            print('accuracy', acc)
            return acc
        def probability(self,x):
            prob = self.clf.predict_proba(x)
            return prob

# Only for formating text in terminal 
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if(IS_TRAIN==True):
            
    df = pd.read_csv('./data/SpamCollection')
    encoder = LabelEncoder()
    df['labels'] = encoder.fit_transform(df['labels'])
    df['emails'] = df['emails'].apply(lambda x:x.lower())
    df['emails'] = df['emails'].apply(lambda x: x.replace('\n', ' '))
    df['emails'] = df['emails'].apply(lambda x: x.replace('\t', ' '))

    nltk.download('stopwords')
    ps = PorterStemmer()

    corpus = []
    for i in range(len(df)):
        ## Becuase of removed duplicated entries...
        review = re.sub('[^a-zA-Z]', ' ', df['emails'][i])
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
        
    f = open('data/corpus.pickle', 'wb')
    pickle.dump(corpus, f)
    f.close()
    
    cv = CountVectorizer(max_features = 2500)
    X = cv.fit_transform(corpus).toarray()
    y = df['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    CM = confusion_matrix(y_test, y_pred)

    #Save Model 
    f = open('data/spam_classifier.pickle', 'wb')
    pickle.dump(model, f)
    f.close()


print(bcolors.OKGREEN + 'Loading vectorizer and model...\n' +bcolors.ENDC)
clf = joblib.load('data/spam_classifier.pickle') 
vectorizer = joblib.load('data/corpus.pickle')

def parse(message):
    pat = '((.|\n)*)Inboxx((.|\n)*)\)to((.|\n)*)Reply ForwardClick(.|\n)*'   
    g = re.search(pat,message)
    print(message)
    print(g)
    try:
        return g[1],g[5]
    except:
        return "True","True"


def predict_in(message):

    # image_path = default_storage.save("tmp/test.txt", ContentFile(email_string.read()))
    # tmp_file = os.path.join(settings.MEDIA_ROOT, image_path)
    # f = open(tmp_file, 'r')
    # content = f.read() 
    df = pd.DataFrame({'emails': message}, index=[0])
    print(df)

    df['emails'] = df['emails'].apply(lambda x:x.lower())
    df['emails'] = df['emails'].apply(lambda x: x.replace('\n', ' '))
    df['emails'] = df['emails'].apply(lambda x: x.replace('\t', ' '))

    nltk.download('stopwords')

    ps = PorterStemmer()

    # Load Corpus
    f = open('data/corpus.pickle', 'rb')
    corpus = pickle.load(f)
    f.close()
    ## Becuase of removed duplicated entries...
    for i in range(len(df)):
        ## Becuase of removed duplicated entries...
        
        review = re.sub('[^a-zA-Z]', ' ', df['emails'][i])
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)

    #Load Model
    f = open('data/spam_classifier.pickle', 'rb')
    model = pickle.load(f)
    f.close()

    cv = CountVectorizer(max_features = 2500)
    X = cv.fit_transform(corpus).toarray()

    result = model.predict([X[-1]])
    labels = ["ham", "spam"]
    prob = model.predict_proba([X[-1]])
    # default_storage.delete(tmp_file)
    response = {"result": labels[int(result)]}

    return response, prob