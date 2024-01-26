from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


##################################################
# 1. Text Preprocessing
##################################################
df = pd.read_excel("NLP/Case1-Amazon/amazon.xlsx")
df.head()


###############################
# Normalizing Case Folding
###############################

df['Review'] = df['Review'].str.lower()


###############################
# Punctuations
###############################

df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# regular expression


###############################
# Numbers
###############################

df['Review'] = df['Review'].str.replace('\d', '')


###############################
# Stopwords
###############################
import nltk
# nltk.download('stopwords')

sw = stopwords.words('english')

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


###############################
# Rarewords
###############################

drops = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))



###############################
# Lemmatization
###############################

df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



##################################################
# 2. Text Visualization
##################################################


###############################
# Terim Frekanslarının Hesaplanması
###############################

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

###############################
# Barplot
###############################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()


###############################
# Wordcloud
###############################

text = " ".join(i for i in df.Review)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Font,kelime sayısı,arka plan rengi vermek istersek aşağıdaki gibi detaylandırabiliriz.
wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



##################################################
# 3. Sentiment Analysis
##################################################

sia = SentimentIntensityAnalyzer()

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["Star"].mean()
# sentiment_label
# neg   3.40
# pos   4.59



train_X, test_X, train_y, test_y = train_test_split(df["Review"],
                                                    df["sentiment_label"],
                                                    random_state=42)

# TF-IDF Word Level
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_X)
X_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_X)
X_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_X)



###############################
# Logistic Regression
###############################

log_model_word = LogisticRegression().fit(X_train_tf_idf_word, train_y)
y_pred_word = log_model_word.predict(X_test_tf_idf_word)

print(classification_report(y_pred_word, test_y))

cross_val_score(log_model_word, X_test_tf_idf_word, test_y, cv=5).mean()
# 0.85


random_review = pd.Series(df["Review"].sample(1).values)

new_review = TfidfVectorizer().fit(train_X).transform(random_review)

pred = log_model_word.predict(new_review)

print(f'Review:  {random_review[0]} \n Prediction: {pred}')
# Review:  color brilliant shown
#  Prediction: ['pos']



###############################
# Random Forests
###############################

rf_model_word = RandomForestClassifier().fit(X_train_tf_idf_word, train_y)
print(classification_report(y_pred_word, test_y))

cross_val_score(rf_model_word, X_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()
# 0.89


