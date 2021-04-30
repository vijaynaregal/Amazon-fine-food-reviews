#!/usr/bin/env python
# coding: utf-8

# # Project :Sentimental Analysis of Amazon Fine food Review

# ###  Goal: 

# Given a review, determine whether the review is positive or negative review using sentimental Analysis techniques 

# ### Project Description:
# This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all ~600,000 reviews. Reviews include product and user information, ratings, and a plain text review. It also includes reviews from all other Amazon categories.
#  
# #### Data Contents:
# 
# * 568,454 reviews
# * 256,059 users
# * 74,258 products
# * 260 users with > 50 reviews

# ### Details about data
# 1. Product Id: Unique identifier for the product
# 2. User Id: unique identifier for the user
# 3. Profile Name: Profile name of the user
# 4. Helpfulness Numerator: Number of users who found the review helpful
# 5. Helpfulness Denominator: Number of users who indicated whether they found the review helpful or not
# 6. Score: Rating between 1 and 5
# 7. Time: Timestamp
# 8. Summary: Summary of the review
# 9. Text: Review

# ### Importing the Libraries

# ### Load the dataset 

# In[2]:


Reviews_df = pd.read_csv('C:/Users/billu/OneDrive/Desktop/5661 project/Reviews - Copy.csv')
Reviews_df


# ## Exploratory Data Analysis

# ###  1. Size of the dataset

# In[3]:


Reviews_df.shape


# ### 2.Converting the score values to positive or negative

# #### Distribution of Ratings

# In[4]:


plt.figure(figsize=(10,5))
sns.countplot(Reviews_df['Score'])
plt.title("Distribution of Ratings", fontweight='bold', fontsize=15)
plt.xlabel("Ratings of Reviews")
plt.ylabel("Total Number of reviews")
plt.show();


# On analysis,it shows that the dataset contains more positive reviews compared to negative reviews.
# Therefore,will assign all data points above rating 3 as positive and below as negative rating for the reviews

# In[5]:


#Give reviews with Score > 3 a 'Positive' tag, and reviews with a score < 3 a 'Negative' tag.
Reviews_df['Sentiment_Value'] = Reviews_df['Score'].apply(lambda x : 'Positive' if x > 3 else 'Negative')


#Creating the new column which stores "Positive" as 1 and "Negative " as 0
Reviews_df['Target'] = Reviews_df['Sentiment_Value'].apply(lambda x : 1 if x == 'Positive' else 0)


# In[6]:


# Distribution of Postive and Negative reviews in a bar graph
Reviews_df["Target"].value_counts().plot(kind='bar',color=['orange','blue'],title='Positive and Negative reviews distribution.',figsize=(5,5))


#  From the graph,it depicts that its an imbalanced dataset for classification.Therefore we need to choose AUC ROC for the accuracy as metric

# In[7]:


#Total datapoints from the loaded dataset.
print("Number of independent variables in our data", Reviews_df.shape[0])
print("Number of dependent variables in our data", Reviews_df.shape[1])


# ### 3. Handle Missing values

# In[8]:


Reviews_df.info()


# From the above result,it shows that the dataset don't have any null values.

#  ### 4. Data Cleaning: Deduplication

# * 1.The dataset contains many duplicate entries of reviews.So to overcome the unbaised results we are deduplicating the rows 
# based  on UserId','ProfileName','Time' and sorting the dataset based on productId before deduplication to reduce the redundancy
# * 2.The dataset contains the entries such that values in HelpfulnessNumerator is higher than the values in HelpfulnessDenominator as its practically not possible

# In[9]:


#Sorting data according to ProductId in ascending order
df_sa=Reviews_df.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Dropping the duplicates 
df_sa=df_sa.drop_duplicates(subset=['UserId','ProfileName','Time'],keep='first')
df_sa.tail(3)
#Removing the entries where HelpfulnessNumerator > HelpfulnessDenominator.
df_sa=df_sa[df_sa.HelpfulnessNumerator <= df_sa.HelpfulnessDenominator]


# ### Analysis of Dataset after datacleaning

# In[10]:


#Calculating the remaining data after datacleaning
print("\npercentage of remaining data")
retained_per = (df_sa['Sentiment_Value'].size*1.0)/(Reviews_df['Sentiment_Value'].size*1.0)*100
removed_per = 100 - retained_per
print("Percentage of redundant data removed : {}".format(removed_per))
print("Percentage of original data retained : {}".format(retained_per))

print('------------------------------------------------------------------------------------------\n')
# Size of Data Set after removing duplicate values
print('New data set size-',df_sa.shape)
print("\n\nThe shape of the dataset after deduplication : {}".format(df_sa.shape))
print("The number of positive and negative reviews after the deduplication.")
print(df_sa["Sentiment_Value"].value_counts())


# From the above results,the positive reviews are larger in number when compared to negative reviews,
# So to maintain the balance,we are dropping the 170k positive reviews for easy computation and processing of data

# In[11]:


#Data set sorting based on sentiment polarity and deleting positive values to balance Dataset
df_sort=df_sa.sort_values(by='Sentiment_Value')
df_dropped_last_n = df_sort.iloc[:-170000]
#df_dropped_last_n

#Display information about the dataset after the removal of postive data.
print("\nThe shape of the data matrix of balanced data".format(df_dropped_last_n.shape))
print("The number of positive and negative reviews after the removal of duplicate data.")
print(df_dropped_last_n["Sentiment_Value"].value_counts())


# Now,we can see that dataset is comparatively balance and also overcomes the baised results

# In[12]:


pos_1=Reviews_df["Sentiment_Value"].value_counts().get(key ='Positive')
pos_2=df_sa["Sentiment_Value"].value_counts().get(key ='Positive')
pos_3=df_dropped_last_n["Sentiment_Value"].value_counts().get(key ='Positive')
neg_1=Reviews_df["Sentiment_Value"].value_counts().get(key ='Negative')
neg_2=df_sa["Sentiment_Value"].value_counts().get(key ='Negative')
neg_3=df_dropped_last_n["Sentiment_Value"].value_counts().get(key ='Negative')

plotdata = pd.DataFrame({
    "positive":[pos_1, pos_2,pos_3],
    "negative":[neg_1, neg_2,neg_3],
 }, 
    index=["Original", "De-Duplication" ,"Balanced"]
)
plotdata.head()

plotdata.plot(kind='bar', stacked=True)

plt.title(" Total Distribution of reviews")
plt.xlabel("Datset")
plt.ylabel("Reveiw classification")


# From the above graph,we are analysis the three stages of data distribution of positive and negative reviews 
# after deduplication and dropping the rows.

# ## Data Preprocessing

# Need to preprocess the reviews text for analysis and to make prediction .
# * 1.It invovles removal of html tag,numbers associated with words,repeated characters,url,special characters etc which are 
#     defined under data_preprocess()
# * 2.Need to remove stop words
# * 3.Need to implement stemming to remove suffixes using Snowball Stemmimg technique as its is more aggressive 
# * 4.Storing all the positive and negative reviews in the different lists
# 
# 

# In[13]:


def data_preprocess(sentence):
    #Remove words with numbers 
    sentence = re.sub("\S*\d\S*", " ", sentence).strip()
    #Remove clean html tags from a sentence
    pattern = re.compile('<.*?>')
    sentence = re.sub(pattern,' ',sentence)
    #Remove words having three consecutive repeating characters.
    sentence  = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',sentence)
    #Remove URL from sentences.
    text = re.sub(r"http\S+", " ", sentence)
    sentence = re.sub(r"www.\S+", " ", text)
    #Keep only words containing letters A-Z and a-z and remove all punctuations, special characters etc
    sentence  = re.sub('[^a-zA-Z]',' ',sentence)
    return (sentence)


# ### Stemming and Removal of stopwords

# Stemming: The method of reducing a word to its word stem, which affixes to suffixes and prefixes or to the roots of words known as a lemma, is known as stemming.
# 
# Stop word: Stopwords are words that add little meaning to a sentence in any language. 
# 
# Here,we are defining custom stopwords which includes no,nor,not keywords so that classifier can learn and predict the sentiment of the review.

# In[14]:


#Stemming and stopwords removal
from nltk.stem.snowball import SnowballStemmer
sno = SnowballStemmer(language='english')

#Removing the word 'not' from stopwords
default_stopwords = set(stopwords.words('english'))
#print("default_stopwords:",default_stopwords)
remove_not = set(['no', 'nor', 'not'])
#print("--------------------------------------------------------------------------------------------------------------------")
defined_stopwords = default_stopwords - remove_not
print("defined_stopwords:\n",defined_stopwords)


# 
# ### Convert all the words to lower case and replace the contraction words

# To bulid good model, we are replacing the contraction words with meaning full word and 
# converting all the characters to lowercase

# In[15]:


import re

# Convert all the words to lower case and replace the contraction words
def replace_words(x):
    x = str(x).lower()
    x = x.replace(",000,000", " m").replace(",000", " k").replace("′", "'").replace("’", "'")                           .replace("won't", " will not").replace("cannot", " can not").replace("can't", " can not")                           .replace("n't", " not").replace("what's", " what is").replace("it's", " it is")                           .replace("finna"," fixing to").replace("gonna"," going to").replace("gimme"," give me").replace("gotta"," got to").replace("'d"," would")                           .replace("daresn't"," dare not").replace("dasn't"," dare not").replace("e'er"," ever").replace("everyone's"," everyone is")                           .replace("'cause'"," because")                           .replace("'ve", " have").replace("'m", " am").replace("'re", " are")                           .replace("he's", " he is").replace("she's", " she is").replace("'s", " own")                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                           .replace("€", " euro ").replace("'ll", " will").replace("how's"," how has").replace("y'all"," you all")                           .replace("o'clock"," of the clock").replace("ne'er"," never").replace("let's"," let us")                          
    #Converting 123000000 to 123m
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    #converting 123000 to 123k
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    return x


# ### Distribution of Stemmed word length

# Here we are printing a chart to showcase the distribution of each word in the review of size 1 to 15

# In[17]:


# Combining all the above data cleaning methodologies as discussed above.

#Processing review Texts
total_words = []
#List all the processed reviews
preprocessed_reviews = [] 
#list all the relevant words from Positive reviews
all_positive_words=[] 
#list all the relevant words from Negative reviews
all_negative_words=[] 
 
#Iterate through the list of reviews and check if a given review belongs to the positive or negative 
count=0  
string=' '    
stemed_word=' '

for review in tqdm(df_dropped_last_n['Text'].values):
    filtered_sentence=[]
    review = data_preprocess(review)
  
 
    
    for cleaned_words in review.split():   
        if((cleaned_words not in defined_stopwords) and (2<len(cleaned_words)<16)):
            stemed_word=(sno.stem(cleaned_words.lower()))
            total_words.append(stemed_word)
           
            filtered_sentence.append(stemed_word)
             #List of all the relevant words from Positive reviews
            if (df_dropped_last_n['Sentiment_Value'].values)[count] == 'Positive': 
                all_positive_words.append(stemed_word)
            #List of all the relevant words from Negative reviews
            if(df_dropped_last_n['Sentiment_Value'].values)[count] == 'Negative':
                all_negative_words.append(stemed_word) 
        else:
            continue
    
    review = " ".join(filtered_sentence) #Final string of cleaned words 
    #print("After review:",review)
    preprocessed_reviews.append(review.strip()) #Data corpus contaning cleaned reviews from the whole dataset
    #print("preprocessed_reviews:",preprocessed_reviews)
    count+=1

total_words = list(set(total_words))
#print("total_words:",total_words)   
#print(count(total_words))
#list all the length of words
dist = []
for i in tqdm(total_words):
    length = len(i)
    dist.append(length)


#print("------------------------------------------------------------------------------------------------------------------")
#print("length_word:",dist)


# In[18]:


# matplotlib histogram to see the distribution of the length of words
plt.figure(figsize=(10,5))
plt.hist(dist, color = 'orange', bins =90)
plt.title('Distribution of the length of Words across all reviews.')
plt.xlabel('Word Lengths')
plt.ylabel('Number of Words')


# From the graph,We can see that Words which has length range between 2 to 15 are more and words which has length greater than 15 are few so will consider the words whose length is greater than 2 and less than 16.

# ### Visualization of Positive words and negative words from the reviews after data processing using word cloud

# #### Positive review WordCloud

# In[19]:


from wordcloud import WordCloud
#plotting all the positive words
positive_string=(" ").join(all_positive_words)
wordcloud = WordCloud(width = 1000, height = 500).generate(positive_string)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis("off")

plt.show()


# #### Negative review WordCloud

# In[20]:


from wordcloud import WordCloud
#plotting all the negative words
negative_string=(" ").join(all_negative_words)
wordcloud = WordCloud(width = 1000, height = 500).generate(negative_string)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud)
plt.axis("off")

plt.show()


# ### Processing review Texts

# In[21]:


#Adding a column of CleanedText to the table final which stores the data_corpus after pre-processing the reviews 
df_dropped_last_n['CleanedText']=preprocessed_reviews 
  
print("The length of the data corpus is : {}".format(len(preprocessed_reviews)))
df_dropped_last_n.head(3)


# In[22]:


#Calculating the percentage of Positive and negative words
value_count=df_dropped_last_n['Target'].value_counts()
value_count
print("{}% data having positive reviews".format(round(value_count[1]/df_dropped_last_n.shape[0]*100,2)))
print("{}% data having negative reviews".format(round(value_count[0]/df_dropped_last_n.shape[0]*100,2)))#


# ### Split the dataset

# To train and test model,we are splitting the dataset to 70% for training and 30% for testing the dataset

# In[23]:


#splitting data to train.cv and test
# from sklearn.model_selection import train_test_split
# x = df_dropped_last_n['CleanedText']
# y = df_dropped_last_n['Target']
# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=100)


# In[3]:


python -m pip install -U notebook-as-pdf
pyppeteer-install


# ##  Feature Extraction: Bag of Words

#  Word Embedding is a technique to convert words to vectors.The Bag of Words model is the simplest form of text representing a sentence as a a string of numbers.It considers a sparse vector

# In[24]:


bow = CountVectorizer()
bow.fit(X_train)
X_train_bow = bow.transform(X_train)
X_test_bow = bow.transform(X_test)
bow_features = bow.get_feature_names()
print('shape of X_train_bow is {}'.format(X_train_bow.get_shape()))
print('shape of X_test_bow is {}'.format(X_test_bow.get_shape()))


# In[25]:


#  function to plot confusion matrix
# def plot_confusion_matrixes(model,x_train,y_train,x_test,y_test):
#     cm_train = confusion_matrix(y_train,model.predict(x_train))
#     cm_test =  confusion_matrix(y_test,model.predict(x_test))
#     class_label = ["negative", "positive"]
#     df_train = pd.DataFrame(cm_train, index = class_label, columns = class_label)
#     df_test = pd.DataFrame(cm_test, index = class_label, columns = class_label)
#     f, axes = plt.subplots(1, 2,figsize=(12,4))
#     #sns.heatmap(df, annot = True, fmt = "d",ax=axes[i])

#     for i in range(2):
#       df = df_train if i==0 else df_test
#       sns.heatmap(df, annot = True, fmt = "d",ax=axes[i])
#       axes[i].set_title(f"Confusion Matrix - {'Train' if i==0 else 'Test'}")
#       axes[i].set_xlabel("Predicted Label")
#       axes[i].set_ylabel("True Label")
#       plt.show()


# ### 1.XGBOOST On BagOfWords Features

# In[26]:


xg_best_est = XGBClassifier(n_estimators = 120,max_depth=15)
xg_best_est = xg_best_est.fit(X_train_bow,y_train)
y_predict = xg_best_est.predict(X_test_bow)
train_fpr_xg_bow, train_tpr_xg_bow, thresholds = roc_curve(y_train, xg_best_est.predict_proba(X_train_bow)[:,1])
test_fpr_xg_bow, test_tpr_xg_bow, thresholds = roc_curve(y_test, xg_best_est.predict_proba(X_test_bow)[:,1])


# ### Performance Metric:ROC , AUC ,Confusion Matrix

# In[27]:


plt.grid(True)
plt.plot(train_fpr_xg_bow, train_tpr_xg_bow, label="train AUC ="+str(auc(train_fpr_xg_bow, train_tpr_xg_bow)))
plt.plot(test_fpr_xg_bow, test_tpr_xg_bow, label="test AUC ="+str(auc(test_fpr_xg_bow, test_tpr_xg_bow)))
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE FOR OPTIMAL K")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr_xg_bow, train_tpr_xg_bow)))
print('Area under test roc {}'.format(auc(test_fpr_xg_bow, test_tpr_xg_bow)))
plot_confusion_matrixes(xg_best_est,X_train_bow,y_train,X_test_bow,y_test)


# From the graph,we can analyse that Xgboosting with the Bag of words model is slightly overfitting.So we tried training and testing on various classifiers

# # Top 20 Features

# In[28]:


top_words= xg_best_est.feature_importances_
features = bow_features
top_words = pd.DataFrame(top_words,columns=['coef'],index=features)
top = top_words.sort_values(by='coef',ascending=False).head(20)
print('Top 20 features are: \n {}'.format(top))


# ### Performace Metric:Cross-Validation

# In[29]:


from sklearn.model_selection import cross_val_score
accuracy= cross_val_score(xg_best_est, X_test_bow, y_test, cv=10, scoring='accuracy')
accuracy.mean()


# ### To print false predicted values

# In[30]:


results = pd.DataFrame()
results['actual'] = y_test 
results['prediction'] = y_predict
print(results[results['actual']!=results['prediction']])


# ## 2.SVM Classifier on Bag of Words feature

# In[31]:


from sklearn.linear_model import SGDClassifier
svm_opt = SGDClassifier(alpha=0.001) 
svm_opt.fit(X_train_bow,y_train)
best_est = CalibratedClassifierCV(base_estimator=svm_opt)
best_est = best_est.fit(X_train_bow,y_train)
train_fpr_svm_op, train_tpr_svm_op, thresholds = roc_curve(y_train, best_est.predict_proba(X_train_bow)[:,1])
test_fpr_svm_op, test_tpr_svm_op, thresholds = roc_curve(y_test, best_est.predict_proba(X_test_bow)[:,1])


# ### Performance Metric:ROC , AUC ,Confusion Matrix

# In[32]:



plt.grid(True)
plt.plot(train_fpr_svm_op, train_tpr_svm_op, label="train AUC ="+str(auc(train_fpr_svm_op, train_tpr_svm_op)))
plt.plot(test_fpr_svm_op, test_tpr_svm_op, label="test AUC ="+str(auc(test_fpr_svm_op, test_tpr_svm_op)))
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE FOR OPTIMAL K")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr_svm_op, train_tpr_svm_op)))
print('Area under test roc {}'.format(auc(test_fpr_svm_op, test_tpr_svm_op)))
plot_confusion_matrixes(svm_opt,X_train_bow,y_train,X_test_bow,y_test)


# From the graph,we can analyse that SVM with the Bag of words model performs better than xgboosting but still it is slightly overfitting.

# ### Performace Metric:Cross-Validation

# In[33]:


from sklearn.model_selection import cross_val_score
accuracy= cross_val_score(svm_opt, X_test_bow, y_test, cv=10, scoring='accuracy')
accuracy.mean()


# ## Feature Extraction:TF-IDF Vectorization

# TF-IDF also gives larger values for less frequent words and is high when both IDF and TF values are high 
# i.e the word is rare in all the documents combined but frequent in a single document.It considers as sparse vectors

# In[34]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
tfidf_features = vectorizer.get_feature_names()
# we use the fitted CountVectorizer to convert the text to vector
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ## 3. Navie Bayes on TFIDF Vectorization

# In[35]:


from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
optimal_alpha = 0.1
naive_opt = MultinomialNB(alpha=optimal_alpha)
naive_opt = naive_opt.fit(X_train_tfidf,y_train)
train_fpr_naive_opt, train_tpr_naive_opt, thresholds = roc_curve(y_train, naive_opt.predict_proba(X_train_tfidf)[:,1])
test_fpr_naive_opt, test_tpr_naive_opt, thresholds = roc_curve(y_test, naive_opt.predict_proba(X_test_tfidf)[:,1])


# ### Performance Metric:ROC , AUC ,Confusion Matrix

# In[36]:


plt.grid(True)
plt.plot(train_fpr_naive_opt, train_tpr_naive_opt, label="train AUC ="+str(auc(train_fpr_naive_opt, train_tpr_naive_opt)))
plt.plot(test_fpr_naive_opt, test_tpr_naive_opt, label="test AUC ="+str(auc(test_fpr_naive_opt, test_tpr_naive_opt)))
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE FOR OPTIMAL K")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr_naive_opt, train_tpr_naive_opt)))
print('Area under test roc {}'.format(auc(test_fpr_naive_opt, test_tpr_naive_opt)))

plot_confusion_matrixes(naive_opt,X_train_tfidf,y_train,X_test_tfidf,y_test)


# From the AUC ROC graph,the naive bayes model with tfidf vectorization also a slighlty overfitting  model  but performs better than Bag of words feature extraction

# ### Performace Metric:Cross-Validation

# In[37]:


from sklearn.model_selection import cross_val_score
accuracy= cross_val_score(naive_opt, X_test_tfidf, y_test, cv=10, scoring='accuracy')
accuracy.mean()


# ## Feature Extraction: Word2Vec 

# Its is the most powerful technique which takes sematic meaning into consideration.It is one of the type of NLP 
# which can detect synonyms and suggect additional words for partial sentences after training.It considers a dense vectors

# In[38]:


preprocessed_reviews = X_train.values
train_sentence = [rev.split() for rev in preprocessed_reviews]
# min_count = 5 considers only words that occured atleast 5 times
# size = length of vector
w2v_model_train = Word2Vec(train_sentence,min_count=5,vector_size=50, workers=4)
w2v_words = list(w2v_model_train.wv.key_to_index)


# ### Average Word2Vec

# In[39]:


#convert Train dataset to vectors
train_reviews = X_train.values
train_sentence = [rev.split() for rev in train_reviews]

sent_vectors_train = []
for sent in tqdm(train_sentence):
    sent_vec = np.zeros(50)
    cnt_words = 0
    for word in sent:
        if word in w2v_words:
            vector = w2v_model_train.wv[word]
            sent_vec += vector
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors_train.append(sent_vec)

print(len(sent_vectors_train))
print(len(sent_vectors_train[0]))

#convert Test dataset to vectors
test_reviews = X_test.values
test_sentence = [rev.split() for rev in test_reviews]

sent_vectors_test = []
for sent in tqdm(test_sentence):
    count = 0
    sent_vec = np.zeros(50)
    for word in sent:
        if word in w2v_words:
            vector = w2v_model_train.wv[word]
            sent_vec += vector
            count += 1
            
    if count != 0:
        sent_vec /= count
    sent_vectors_test.append(sent_vec)

print(len(sent_vectors_test))
print(len(sent_vectors_test[0]))


# In[40]:


X_train_avgw2v = np.array(sent_vectors_train)
X_test_avgw2v = np.array(sent_vectors_test)
print('shape of X_train_avgw2v is {}'.format(X_train_avgw2v.shape))
print('shape of X_test_avgw2v is {}'.format(X_test_avgw2v.shape))


# # SVM with Avgword2vec 

# In[41]:


from sklearn.linear_model import SGDClassifier
svm_opt_w2v = SGDClassifier(alpha=0.001) 
svm_opt_w2v.fit(X_train_avgw2v,y_train)
best_est_w2v = CalibratedClassifierCV(base_estimator=svm_opt_w2v)
best_est_w2v = best_est_w2v.fit(X_train_avgw2v,y_train)
train_fpr_svm_w2v, train_tpr_svm_w2v, thresholds = roc_curve(y_train, best_est_w2v.predict_proba(X_train_avgw2v)[:,1])
test_fpr_svm_w2v, test_tpr_svm_w2v, thresholds = roc_curve(y_test, best_est_w2v.predict_proba(X_test_avgw2v)[:,1])


# # Performance Metric:ROC , AUC ,Confusion Matrix

# In[42]:



plt.grid(True)
plt.plot(train_fpr_svm_w2v, train_tpr_svm_w2v, label="train AUC ="+str(auc(train_fpr_svm_w2v, train_tpr_svm_w2v)))
plt.plot(test_fpr_svm_w2v, test_tpr_svm_w2v, label="test AUC ="+str(auc(test_fpr_svm_w2v, test_tpr_svm_w2v)))
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE FOR OPTIMAL K")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr_svm_w2v, train_tpr_svm_w2v)))
print('Area under test roc {}'.format(auc(test_fpr_svm_w2v, test_tpr_svm_w2v)))
plot_confusion_matrixes(svm_opt_w2v,X_train_avgw2v,y_train,X_test_avgw2v,y_test)


# From the AUC ROC graph,the SVM model with avgword2vec vectorization produces slight overfitting  model and SVM classifier on Bag of words feature extraction performs equally good.

# ### Performace Metric:Cross-Validation

# In[43]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score
accuracy= cross_val_score(svm_opt_w2v, X_test_avgw2v, y_test, cv=10, scoring='accuracy')
accuracy.mean()


# ## 4.Random Forest on Tfidf feature

# In[44]:


best_est_rf = RandomForestClassifier(n_estimators = 120,max_depth=30)
best_est_rf = best_est.fit(X_train_tfidf,y_train)
train_fpr_est_rf, train_tpr_est_rf, thresholds = roc_curve(y_train, best_est_rf.predict_proba(X_train_tfidf)[:,1])
test_fpr_est_rf, test_tpr_est_rf, thresholds = roc_curve(y_test, best_est_rf.predict_proba(X_test_tfidf)[:,1])


# ### Performance Metric:ROC , AUC ,Confusion Matrix

# In[45]:


plt.grid(True)
plt.plot(train_fpr_est_rf, train_tpr_est_rf, label="train AUC ="+str(auc(train_fpr_est_rf, train_tpr_est_rf)))
plt.plot(test_fpr_est_rf, test_tpr_est_rf, label="test AUC ="+str(auc(test_fpr_est_rf, test_tpr_est_rf)))
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE FOR OPTIMAL K")
plt.show()

#Area under ROC curve
print('Area under train roc {}'.format(auc(train_fpr_est_rf, train_tpr_est_rf)))
print('Area under test roc {}'.format(auc(test_fpr_est_rf, test_tpr_est_rf)))
plot_confusion_matrixes(best_est_rf,X_train_tfidf,y_train,X_test_tfidf,y_test)


# From the AUC ROC graph,the Random Forest model with TF-idf vectorization produces good fit model and performs well 
# 

# ### Performace Metric:Cross-Validation

# In[46]:


from sklearn.model_selection import cross_val_score
accuracy= cross_val_score(best_est_rf, X_test_tfidf, y_test, cv=10, scoring='accuracy')
accuracy.mean()


# ## Conclusion

# In[47]:


from prettytable import PrettyTable
    
tb = PrettyTable()

tb.field_names = ["Vector","Algorithm","Train AUC", "Test AUC"]
tb.add_row(["bow","SVM",auc(train_fpr_svm_op, train_tpr_svm_op),auc(train_fpr_svm_op, train_tpr_svm_op)])
tb.add_row(["bow","xgboost",auc(train_fpr_xg_bow, train_tpr_xg_bow),auc(test_fpr_xg_bow, test_tpr_xg_bow)])
tb.add_row(["tfidf","Random_Forest",auc(train_fpr_est_rf, train_tpr_est_rf),auc(test_fpr_est_rf, test_tpr_est_rf)])
tb.add_row(["tfidf","NaiveBayes",auc(train_fpr_naive_opt, train_tpr_naive_opt), auc(test_fpr_naive_opt, test_tpr_naive_opt)])
tb.add_row(["Average_w2v","SVM",auc(train_fpr_svm_w2v, train_tpr_svm_w2v), auc(test_fpr_svm_w2v, test_tpr_svm_w2v)])
print(tb)


# After implementing several advanced machine learning classifier we can see that Random Forest on tfidf
# and SVM on bag of word features gives a more generalized model.
