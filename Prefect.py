import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn import metrics

from prefect import task, flow

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

@task
def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame inplace.
    
    """
    df.drop(columns=columns_to_drop, inplace=True)

@task
def remove_nan_values(df):
    """
    Check for NaN values in the DataFrame and remove all rows with NaN values inplace.
        
    """
    nan_values_count = df.isnull().sum().sum()
    if nan_values_count > 0:
        print("NaN values detected. Removing rows with NaN values.")
        # Remove all rows with NaN values
        df.dropna(inplace=True)
    else:
        print("No NaN values detected.")

@task
def create_sentiment_labels(df, threshold=4, column='Ratings'):
    """
    Create sentiment labels based on a threshold for positive/negative ratings inplace.
    
    """
    # Create sentiment labels
    df[column] = df[column].apply(lambda x: 'Positive' if x >= threshold else 'Negative')

@task
def split_features_target(dataframe, INPUTS, OUTPUT):
    """
    Split the DataFrame into input features (X) and target variable (y).
    
    """
    X = dataframe[INPUTS]
    y = dataframe[OUTPUT]
    return X, y

@task
def split_train_test(X, y, train_size=0.75, random_state=0):
    """
    Split the data into training and testing sets.
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

@task
def preprocess(text):
    """
    Preprocess text data by removing 'READ MORE', special characters, punctuation, stopwords,
    and converting text to lowercase, followed by lemmatization.
    
    """
    # Remove 'READ MORE'
    text = text.replace('READ MORE', '')
    
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    clean_text = ' '.join(tokens)
    
    return clean_text

@task
def feature_engineering(X_train_clean, X_test_clean, vectorizer_type, max_features):
    """
    Perform feature engineering using CountVectorizer or TfidfVectorizer.
    
    """
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(max_features=max_features)
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        raise ValueError("Invalid vectorizer_type. Please choose 'count' or 'tfidf'.")
    
    # Fit and transform the training data
    X_train_vectorized = vectorizer.fit_transform(X_train_clean)
    
    # Transform the test data
    X_test_vectorized = vectorizer.transform(X_test_clean)
    
    return X_train_vectorized, X_test_vectorized

@task
def train_model(X_train_vectorized, y_train, hyperparameters):
    """
    Train a machine learning model.
    
    """
    # Create the classifier with specified hyperparameters
    clf = SVC(**hyperparameters)
    
    # Train the model
    clf.fit(X_train_vectorized, y_train)
    
    return clf

@task
def evaluate_model(model, X_train_vectorized, y_train, X_test_vectorized, y_test):
    """
    Evaluate the trained model.
    
    """
    # Predictions on training and test sets
    y_train_pred = model.predict(X_train_vectorized)
    y_test_pred = model.predict(X_test_vectorized)

    # Calculate accuracy scores
    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score


@flow(name="Sentiment Analysis - SVC")
def workflow():
    DATA_PATH = 'reviews_badminton/data.csv'
    columns_to_drop=['Reviewer Name', 'Review Title', 'Place of Review', 'Up Votes', 'Down Votes', 'Month']
    INPUTS = 'Review text'
    OUTPUT = 'Ratings'
    max_features = 5000
    vectorizer_type = 'tfidf' #CountVectorizer == 'count'  TfidfVectorizer == 'tfidf'
    hyperparameters = {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'}
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Dropping columns
    drop_columns(df, columns_to_drop)
    
    # Removing NaN Values
    remove_nan_values(df)
    
    # Created sentiment labels
    create_sentiment_labels(df)

    # Identify Inputs and Output
    X, y = split_features_target(df, INPUTS, OUTPUT)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Preprocess the data
    X_train_clean = X_train.apply(preprocess)
    X_test_clean = X_test.apply(preprocess)

    # Apply feature engineering
    X_train_vectorized, X_test_vectorized = feature_engineering(X_train_clean, X_test_clean, vectorizer_type, max_features)
    
    # Build a model
    model = train_model(X_train_vectorized, y_train, hyperparameters)
    
    # Evaluate the model
    train_score, test_score = evaluate_model(model, X_train_vectorized, y_train, X_test_vectorized, y_test)

    # Print the evaluation scores
    print("Train Score:", train_score)
    print("Test Score:", test_score)
    

        
    
    if __name__ == "__main__":
        workflow.serve(
            name="my-first-deployment",
            cron="* * * * *"
    )
