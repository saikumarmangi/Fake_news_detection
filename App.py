import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
a=r"c:\Users\Akshitha\Desktop\project\News _dataset\True.csv"
true_df = pd.read_csv(a)
true_df['label'] = 0  
b=r"c:\Users\Akshitha\Desktop\project\News _dataset\Fake.csv"
fake_df = pd.read_csv(b)
fake_df['label'] = 1  

df = pd.concat([true_df, fake_df], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)
# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)
# Display the result
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report_result)

d={
        "title": "Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing",
        "text": """Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and  the very dishonest fake news media.  The former reality show star had just one job to do and he couldn t do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year,  President Angry Pants tweeted.  2018 will be a great year for America! As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year. 2018 will be a great year for America!  Donald J. Trump (@realDonaldTrump) December 31, 2017Trump s tweet went down about as welll as you d expect.What kind of president sends a New Year s greeting like this despicable, petty, infantile gibberish? Only Trump! His lack of decency won t even allow him to rise above the gutter long enough to wish the American citizens a happy new year! ...""",
        "subject": "News",
        "date": "December 31, 2017"
    }
df2=pd.DataFrame([d])


# Take input from the user
# user_input = input("Enter a news article: ")

# Make a prediction
prediction = model.predict(df2['text'])

# Display the result
if prediction[0] == 0:
    print("The news is likely to be true.")
else:
    print("The news is likely to be fake.")
