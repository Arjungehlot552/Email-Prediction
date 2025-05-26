import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

data = pd.read_csv('spam.csv', encoding='latin-1')
# print(data.head(5))  #print top 5 rows of the dataset
# print(data.shape)  #print the shape of the dataset number of rows and columns
data.drop_duplicates(inplace=True)  #remove duplicate rows
# print(data.shape)
# print(data.isnull().sum())  #check for null values in the dataset
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam' , 'Spam'])  #replace ham with Not Spam and spam with Spam
# print(data.head(5))

mess = data['Message']  #get the Message column
cat = data['Category']  #get the Category column

(mess_train , mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2, random_state=42)  #split the dataset into training and testing sets

cv = CountVectorizer(stop_words='english')  #create a CountVectorizer object
features = cv.fit_transform(mess_train)  #fit the CountVectorizer object to the training data and get features

#creating model
model = MultinomialNB()  #create a Multinomial Naive Bayes model
model.fit(features , cat_train)  #fit the model to the training data

#testing the model

features_test = cv.transform(mess_test)  #transform the test data using the CountVectorizer
# print(model.score(features_test, cat_test))  #print the accuracy of the model on the test data


# Example of predicting a new message

def predict_message(message):
    input_message = cv.transform([message]).toarray()  #transform the new message using the CountVectorizer 
    result = model.predict(input_message)  #predict the category of the new message
    return result

st.header('Spam Detection App')  #title of the app
st.subheader('Enter a message to check if it is spam or not')  #subtitle of the app

output = predict_message('Congratulations! You have won a lottery of $1000. Click here to claim your prize.')  #predict the category of the new message
# print(output)  #print the output of the prediction
user_message = st.text_input('Message', 'Type your message here')  #input box for the user to enter a message

if st.button('Validate'):
    output = predict_message(user_message)
    st.write(output[0])



# message = ['Congratulations! You have won a lottery of $1000. Click here to claim your prize.']
# features_new = cv.transform(message)  #transform the new message using the CountVectorizer
# print(model.predict(features_new))  #print the prediction of the model on the new message
