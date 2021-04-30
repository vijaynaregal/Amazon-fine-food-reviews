import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
np.random.seed(7)

df_sa=pd.read_csv("abc1.csv")

#splitting data to train.cv and test
from sklearn.model_selection import train_test_split
x = df_sa['CleanedText']
y = df_sa['Label']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=100)

bow = CountVectorizer()
bow.fit(X_train)
X_train_bow = bow.transform(X_train)
X_test_bow = bow.transform(X_test)
bow_features = bow.get_feature_names()
print('shape of X_train_bow is {}'.format(X_train_bow.get_shape()))
print('shape of X_test_bow is {}'.format(X_test_bow.get_shape()))

from sklearn.linear_model import SGDClassifier
svm_opt = SGDClassifier(alpha=0.001) 
svm_opt.fit(X_train_bow,y_train)
best_est = CalibratedClassifierCV(base_estimator=svm_opt)
best_est = best_est.fit(X_train_bow,y_train)

# Saving model to disk
pickle.dump(best_est, open('model.pkl','wb'))
pickle.dump(bow, open('vector.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
review= 'Had dinner with girl friends. Menu is perfect, something for everyone. Service was awesome and Jason was very accommodating. Will be back definitely!'
review_vector = bow.transform([review]) 
print(model.predict(review_vector))