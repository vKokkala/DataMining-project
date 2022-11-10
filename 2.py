import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def show_percenatges(df):
  for score in range(1, 6):
     helper2_df = df[df.Score==score]
     percentage = helper2_df.shape[0]/(df.shape[0])
     print("percentage of "+ str(score)+ " score is "+ str(percentage) +"% Οf the dataset")

CONTRACTIONS_DICT = {"can`t":"can not",
                     "won`t":"will not",
                     "don`t":"do not",
                     "aren`t":"are not",
                     "i`d":"i would",
                     "couldn`t": "could not",
                     "shouldn`t": "should not",
                     "wouldn`t": "would not",
                     "isn`t": "is not",
                     "it`s": "it is",
                     "didn`t": "did not",
                     "weren`t": "were not",
                     "mustn`t": "must not",
                    }

def prepare_data(df):
    
    df["text"] = df["Text"] \
              .apply(lambda x: replace_words(x,CONTRACTIONS_DICT)) \
              .apply(lambda x: clean_text(x))    
               
    return df.text.values, df.Score.values

def clean_text(
  
    string: str, 
    punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
    stop_words=['the', 'a', 'and', 'is', 'be', 'will',]) -> str:

    #Cleaning the urls
    string = re.sub(r'https?://\S+|www\.\S+', '', string)

    #Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)

    #Removing the punctuations
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 

    #Converting the text to lower
    string = string.lower()

    #Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])

    return string    

def replace_words(string:str, dictionary:dict):
    for shortcut, full_word in dictionary.items():
        string = string.replace(shortcut, full_word)
    return string

def get_sets(df):
    
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for score in range(1, 6):
     helper_df = df[df.Score==score] #filtro gia ka8e va8mologia
     train_size = int(helper_df.shape[0] * 0.8) 
     df_train_score = helper_df.loc[:train_size,:]
     df_test_score = helper_df.loc[train_size:,:] 
     train_df = pd.concat([train_df,df_train_score],axis=0)    
     test_df = pd.concat([test_df,df_test_score],axis=0) 
    
    print(train_df) 
    return train_df,test_df 

df = pd.read_csv('amazon.csv')

training_df, testing_df = get_sets(df)

#clean the text
train_reviews, train_scores= prepare_data(training_df)
test_reviews, test_scores = prepare_data(testing_df)


#mporouse na ginei ki egw h epejergasia twn texts alla gia 
#sugkekrimena xarakthristika
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_reviews)

#dunata modes: binary, count, tfidf,freq
train_tokenized = tokenizer.texts_to_matrix(
                             train_reviews,
                             mode='tfidf'
                             )
test_tokenized = tokenizer.texts_to_matrix(
                             test_reviews,
                             mode='tfidf'
                             )

forest_classifier = RandomForestClassifier(
                                n_estimators=200, 
                                oob_score=True,
                                min_samples_leaf=1,
                                n_jobs=-1,
                                )
forest_classifier.fit(train_tokenized,train_scores)

predictions = forest_classifier.predict(test_tokenized)


print(confusion_matrix(test_scores,predictions))
print(classification_report(test_scores,predictions))
print(accuracy_score(test_scores,predictions))