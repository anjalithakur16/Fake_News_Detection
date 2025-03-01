#Fake News Detection
#Importing the Libraries of Python



import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import streamlit as st
#Here we will Load the data
fake_data = pd.read_csv(r"C:\Users\Anjali\Downloads\Fake.csv\Fake.csv")
fake_data
true_data =pd.read_csv(r"C:\Users\Anjali\Downloads\True.csv\True.csv")
true_data
#Preprocessing ofthe data
true_data.head()
true_data.describe()
true_data.tail()
fake_data.head()
fake_data.tail()
#Find the shape ofthe given data that is no. of rows and columns
true_data.shape , fake_data.shape
#Here we will label the False and true data as 0 , 1
true_data['label'] = 1
fake_data['label'] = 0
true_data.head()
fake_data.head()
# here we will concatenate both the data 
#fnews = pd.concat([true_data , fake_data],axis=1) we will not merge by column cos it will + all cols
#fnews.head()
#concatenate by rows
fnews = pd.concat([true_data,fake_data],axis = 0) 
fnews.head()
fnews.tail()
#check the null values
fnews.isnull().sum()
#Dropping that columns are not required
fnews1 = fnews.drop(['date','subject','title'],axis=1)
fnews1.head()
#Reshuffling the data
#random.shuffle(fnews) recommended for list
fnews1 = fnews1.sample(frac= 1)
fnews1.head()
#Reset the index in a proper manner for better understanding
fnews1 . reset_index(inplace = True)
fnews1.head(10)
# Drop  shuffled index column for better understanding
fnews1.drop(['index'], axis=1 , inplace = True )
fnews1.head(7)
fnews1.columns
fnews1['text']        #text inthe text column
fnews1['text'][44894 ]
 
#creating data function to proceed the text (convert text into numerical form)
import re  #Regular expression
def wordopt(text):           #split the text into words 
#convert into lower case 
 text = text.lower()

# Remove URLs
 text = re.sub(r'https?://\S+|www\.S+' ,' ',text)

# Remove Html tags
 text = re.sub(r'<.*?> ' , ' ',text)

# Remove Punctuation
 text = re.sub(r'[^\w\s]' , ' ' ,text)

# Remove Digits
 text = re.sub(r'\d' ,' ' , text)

# Remove NewLine Characters 
 text = re.sub(r'\n' , ' ' ,text)

 return text
fnews1['text'] = fnews1['text'].apply(wordopt)
fnews1['text']
fnews1['text'][44894]
x =  fnews1['text']                                                     # X contains the features (independent variables or Predictor var ).

y =  fnews1['label']                                                     # y contains the target (dependent variable or Target var , label).
x
y
#Splitting the dataset into the Training set and Test set
from sklearn .model_selection import train_test_split
x_train , x_test ,y_train , y_test = train_test_split(x, y,test_size = 0.25,random_state = 0 )
x_train.shape
x_test.shape
#TF-IDF vectorization
# To convert text in to numerical vector
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_train
xv_test = vectorization .transform(x_test)
xv_test
#Training the Logistic Regression on the training set
from sklearn .linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(xv_train , y_train)
fnews1.columns
 
#Predicting the Test set results
pred_lr = lr.predict(xv_test)
lr.score(xv_test ,y_test)
# calculating accuracy score of logistic regression
log_reg_acc = accuracy_score(y_test , pred_lr)
log_reg_acc
from sklearn.metrics import classification_report
print(classification_report(y_test, pred_lr))
#Visualising of the Logistic Regression training set results
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Logistic Regression Label')
plt.show()
 
#Training the Decision Tree Classification model on the Training set
from sklearn .tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc = dtc. fit(xv_train , y_train)
#Predicting the Test set results
pred_dtc = dtc.predict(xv_test)
dtc.score(xv_test , y_test)
dec_tree_acc = accuracy_score(y_test , pred_dtc)
dec_tree_acc
from sklearn .metrics import classification_report
print(classification_report(y_test, pred_dtc))
#Visualising of the Decision Tree Classification model training set results
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, pred_dtc)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Decision Tree Label')
plt.show()
 
#Training the Random Forest Classifier model on the training set
from sklearn . ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(xv_train , y_train)
#Predicting the Test set results

pred_rfc = rfc . predict(xv_test)
rfc.score(xv_test , y_test)
rand_forest_acc = accuracy_score(y_test , pred_rfc)
rand_forest_acc
print(classification_report(y_test , pred_rfc))
#Visualising of the Random Forest Classifier model training set results
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Random Forest classifier Label')
plt.show()
#Training the Gradient Boosting Classifier onthe training set
from sklearn . ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state = 0)
gbc = gbc.fit(xv_train , y_train)
#Predicting the Test set results
pred_gbc = gbc . predict(xv_test)
gbc . score(xv_test,y_test)
grad_boost_acc = accuracy_score(y_test , pred_gbc)
grad_boost_acc 
print(classification_report(y_test,pred_gbc))
#Visualising of the Gradient Boosting Classifier training set results
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Gradient Boosting classifier Label')
plt.show()
#Combined Visualization of the algorithm on the basis of accuracy score
# Combine the accuracies in a list
accuracy_scores =  [log_reg_acc,dec_tree_acc, rand_forest_acc,grad_boost_acc]
# Create a bar chart to visualize the accuracies
model_names = ['Logistic Regression','Decision Tree','Random Forest','Gradient Boosting', ]
accuracy_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracy_scores})
# Plotting

plt.figure(figsize=(7, 3))
sns.barplot(x='Model', y='Accuracy', data=accuracy_df)
plt.title('Comparison of Model Accuracies')
plt.ylabel('Accuracy Score')
plt.xlabel('Machine Learning Model')

# Show the plot
plt.grid(True)
plt.show()
 
# Convert accuracy scores to a 2D array for heatmap
#accuracy_matrix = np.array([accuracy_scores]).reshape(1, -1)

# Plot the heatmap
#plt.figure(figsize=(8, 4))
#sns.heatmap(accuracy_matrix, annot=True, cmap='Blues', xticklabels=model_names, yticklabels=['Accuracy'])
#plt.title('Accuracy Heatmap of Different Algorithms', fontsize=16)
#plt.show() 
 
#Predicted Model

def manual_testing(fnews1):
    testing_news = {"text" : [fnews1]} #coorected syntax for defining dictionary
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test) # assuming 'vectorization'is a vectorizer object
    pred_lr = lr.predict(new_xv_test)
    pred_dtc = dtc .predict(new_xv_test)
    #pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    return "\n\nlr Prediction : {} \nrfc Prediction:{}".format(output_label(pred_lr[0]), output_label(pred_rfc[0]))
    
def output_label(n):
    if n==0:
        return "It is a fake news "
    elif n==1:
        return "It is a Genuine news "
news_article = str(input())
model =manual_testing(news_article)
