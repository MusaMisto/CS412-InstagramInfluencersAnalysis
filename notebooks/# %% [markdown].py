# %% [markdown]
# # Import Dependencies

# %%
import numpy as np
import pandas as pd
import gzip
import json
import os
import gzip
import json
from pprint import pprint
import os
import pandas as pd

# Turkish StopWords

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
turkish_stopwords = stopwords.words('turkish')

# %% [markdown]
# # Influencer Category Classification
# 
# 
# 
# 1.   Read Data
# 2.   Preprocess Data
# 3.   Prepare Model
# 4.   Predict Test Data
# 4.   Save outputs
# 
# 

# %%
# Step 1: Define File Paths Dynamically
# Get the current notebook directory
current_notebook_dir = os.getcwd()

# Get the repo directory (assuming notebooks are inside the "notebooks" folder)
repo_dir = os.path.abspath(os.path.join(current_notebook_dir, '..'))

# Get the data directory
data_dir = os.path.join(repo_dir, 'data')

# Get the training directory
training_dir = os.path.join(data_dir, 'training')

# File path for 'train-classification.csv'
train_classification_path = os.path.join(training_dir, 'train-classification.csv')

# Step 2: Load Data Dynamically
train_classification_df = pd.read_csv(train_classification_path)
train_classification_df = train_classification_df.rename(columns={'Unnamed: 0': 'user_id', 'label': 'category'})

# Step 3: Unify Labels
train_classification_df["category"] = train_classification_df["category"].apply(str.lower)

# Step 4: Create User-to-Category Mapping
username2_category = train_classification_df.set_index("user_id").to_dict()["category"]

# Step 5: Verify Output
print("First few rows of the training classification DataFrame:")
train_classification_df.head()

# %%
print(username2_category["kod8net"] + "\n")

# stats about the labels
train_classification_df.groupby("category").count()

# %%
# Step 1: Define File Paths Dynamically
# Get the current notebook directory
current_notebook_dir = os.getcwd()

# Get the repo directory (assuming notebooks are inside the "notebooks" folder)
repo_dir = os.path.abspath(os.path.join(current_notebook_dir, '..'))

# Get the data directory
data_dir = os.path.join(repo_dir, 'data')

# Get the training directory
training_dir = os.path.join(data_dir, 'training')

# File path for 'training-dataset.jsonl.gz'
train_data_path = os.path.join(training_dir, 'training-dataset.jsonl.gz')

# Step 2: Initialize Dictionaries for Data
username2posts_train = dict()
username2profile_train = dict()

username2posts_test = dict()
username2profile_test = dict()

# Step 3: Process Data from 'training-dataset.jsonl.gz'
with gzip.open(train_data_path, "rt", encoding="utf-8") as fh:
    for line in fh:
        sample = json.loads(line)

        profile = sample["profile"]
        username = profile.get("username", "").strip()  # Handle missing or empty usernames
        if not username:
            continue  # Skip if username is missing or empty

        if username in username2_category:
            # Train data info
            username2posts_train[username] = sample["posts"]
            username2profile_train[username] = profile
        else:
            # Test data info
            username2posts_test[username] = sample["posts"]
            username2profile_test[username] = profile

# Step 4: Verify Output
print(f"Number of Training Users: {len(username2posts_train)}")
print(f"Number of Testing Users: {len(username2posts_test)}")

# %%
# Profile Dataframe
train_profile_df = pd.DataFrame(username2profile_train).T.reset_index(drop=True)
test_profile_df = pd.DataFrame(username2profile_test).T.reset_index(drop=True)

train_profile_df.head(2)

# %%
# Test Profie Dataframe
test_profile_df.head(2)

# %%
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Improved preprocessing function
def preprocess_text(text: str):
    # Lower-case the text (Turkish friendly)
    text = text.casefold()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Replace hashtags and mentions with tokens (optional but useful for some tasks)
    text = re.sub(r'#\w+', 'HASHTAG', text)
    text = re.sub(r'@\w+', 'MENTION', text)
    
    # Keep letters, digits, spaces, and emojis; remove unwanted punctuation
    text = re.sub(r'[^a-zçğıöşü0-9\s\U0001F600-\U0001F64F]+', '', text)
    
    # Remove all digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Initialize corpus and train usernames
corpus = []
train_usernames = []

# Preprocess training data
for username, posts in username2posts_train.items():
    train_usernames.append(username)
    
    cleaned_captions = []
    for post in posts:
        post_caption = post.get("caption", "")
        if post_caption is None:
            continue
        
        post_caption = preprocess_text(post_caption)
        
        if post_caption != "":
            cleaned_captions.append(post_caption)
    
    # Joining the posts of each user with a separator (helps retain user-specific context)
    user_post_captions = " <SEP> ".join(cleaned_captions)
    corpus.append(user_post_captions)

# Update TF-IDF parameters for improved feature extraction
vectorizer = TfidfVectorizer(
    stop_words=turkish_stopwords,
    max_features=10000,         # Increased feature count for richer representation
    ngram_range=(1, 2),         # Use unigrams and bigrams
    sublinear_tf=True,          # Use logarithmic scaling for term frequency
    max_df=0.7,                 # Ignore terms appearing in >70% of documents
    min_df=3                    # Ignore terms appearing in <3 documents
)

# Fit the vectorizer
vectorizer.fit(corpus)

# Transform training data
x_post_train = vectorizer.transform(corpus)

# Assign labels to training data
y_train = [username2_category.get(uname, "NA") for uname in train_usernames]

# Preprocess test data
test_usernames = []
test_corpus = []

for username, posts in username2posts_test.items():
    test_usernames.append(username)
    
    cleaned_captions = []
    for post in posts:
        post_caption = post.get("caption", "")
        if post_caption is None:
            continue
        
        post_caption = preprocess_text(post_caption)
        
        if post_caption != "":
            cleaned_captions.append(post_caption)
    
    user_post_captions = " <SEP> ".join(cleaned_captions)
    test_corpus.append(user_post_captions)

# Transform test data (do not fit again!)
x_post_test = vectorizer.transform(test_corpus)

# %%
# Making sure everything is fine
assert y_train.count("NA") == 0

# %%
feature_names = vectorizer.get_feature_names_out()
feature_names

# %%
df_tfidf = pd.DataFrame(x_post_train.toarray(), columns=feature_names)
df_tfidf.head(2)

# %%
print(df_tfidf.shape)

# %%
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(df_tfidf, y_train, test_size=0.2, stratify=y_train)

# %%
print(x_train.shape)

print(x_val.shape)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1.0, 10],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': ['balanced', None]
}

# Grid search for Logistic Regression
grid_search = GridSearchCV(
    LogisticRegression(max_iter=500),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(x_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# %% [markdown]
# # Naive Base Classifier
# 
# ### Now we can pass the numerical values to a classifier, Let's try Naive Base!
# 

# %%
from sklearn.linear_model import LogisticRegression

# Train a Logistic Regression model with balanced class weights
model = LogisticRegression(
    class_weight='balanced', 
    max_iter=500, 
    solver='liblinear', 
    penalty='l2', 
    C=10
)
model.fit(x_train, y_train)

# %%
# Validation Data
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


y_val_pred = model.predict(x_val)

print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))

# %%
# Step 1: Define File Paths Dynamically
# Get the current notebook directory
current_notebook_dir = os.getcwd()

# Get the repo directory (assuming notebooks are inside the "notebooks" folder)
repo_dir = os.path.abspath(os.path.join(current_notebook_dir, '..'))

# Get the data directory
data_dir = os.path.join(repo_dir, 'data')

# Get the testing directory
testing_dir = os.path.join(data_dir, 'testing')

# File path for 'test-classification-round1.dat'
test_data_path = os.path.join(testing_dir, 'test-classification-round1.dat')

# Step 2: Preview First 5 Lines of the Test File
with open(test_data_path, "rt", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        print(line.strip())
        if i == 4:  # Print only the first 5 lines
            break

print("*****")

# Step 3: Extract Usernames from Test Data
test_unames = []
with open(test_data_path, "rt", encoding="utf-8") as fh:
    for line in fh:
        test_unames.append(line.strip())

# Step 4: Verify Output
print(test_unames[:5])  # Display the first 5 usernames

# %%
x_test = []

for uname in test_unames:
  try:
    index = test_usernames.index(uname)
    x_test.append(x_post_test[index].toarray()[0])
  except Exception as e:
    try:
      index = train_usernames.index(uname)
      x_test.append(x_post_train[index].toarray()[0])
    except Exception as e:
      print(uname)


print(test_unames.remove("screenname"))

df_test = pd.DataFrame(np.array(x_test), columns=feature_names)
df_test.head(2)

# %%
test_pred = model.predict(df_test)

output = dict()
for index, uname in enumerate(test_unames):
  output[uname] = test_pred[index]

# %%
with open("output.json", "w") as of:
  json.dump(output, of, indent=4)

# %% [markdown]
# # Like Count Prediction
# 
# 
# Here, we use the average like_count of the user's previous posts to predict each post's like_count

# %%
def predict_like_count(username, current_post=None):
  def get_avg_like_count(posts:list):
    total = 0.
    for post in posts:
      if current_post is not None and post["id"] == current_post["id"]:
        continue

      like_count = post.get("like_count", 0)
      if like_count is None:
        like_count = 0
      total += like_count

    if len(posts) == 0:
      return 0.

    return total / len(posts)

  if username in username2posts_train:
    return get_avg_like_count(username2posts_train[username])
  elif username in username2posts_test:
    return get_avg_like_count(username2posts_test[username])
  else:
    print(f"No data available for {username}")
    return -1

# %%
def log_mse_like_counts(y_true, y_pred):
  """
  Calculate the Log Mean Squared Error (Log MSE) for like counts (log(like_count + 1)).

  Parameters:
  - y_true: array-like, actual like counts
  - y_pred: array-like, predicted like counts

  Returns:
  - log_mse: float, Log Mean Squared Error
  """
  # Ensure inputs are numpy arrays
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  # Log transformation: log(like_count + 1)
  log_y_true = np.log1p(y_true)
  log_y_pred = np.log1p(y_pred)

  # Compute squared errors
  squared_errors = (log_y_true - log_y_pred) ** 2

  # Return the mean of squared errors
  return np.mean(squared_errors)

# %%
# Train Dataset evaluation

y_like_count_train_true = []
y_like_count_train_pred = []
for uname, posts in username2posts_train.items():
  for post in posts:
    pred_val = predict_like_count(uname, post)
    true_val = post.get("like_count", 0)
    if true_val is None:
      true_val = 0

    y_like_count_train_true.append(true_val)
    y_like_count_train_pred.append(pred_val)

print(f"Log MSE Train= {log_mse_like_counts(y_like_count_train_true, y_like_count_train_pred)}")

# %%
# Step 1: Define File Paths Dynamically
# Get the current notebook directory
current_notebook_dir = os.getcwd()

# Get the repo directory (assuming notebooks are inside the "notebooks" folder)
repo_dir = os.path.abspath(os.path.join(current_notebook_dir, '..'))

# Get the data directory
data_dir = os.path.join(repo_dir, 'data')

# Get the testing directory
testing_dir = os.path.join(data_dir, 'testing')

# File path for 'test-regression-round1.jsonl'
test_dataset_path = os.path.join(testing_dir, 'test-regression-round1.jsonl')

# File path for output
output_dir = os.path.join(data_dir, 'output')
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
output_file_path = os.path.join(output_dir, 'test-regression-round1.jsonl')

# Step 2: Process the Test Dataset
to_predict_like_counts_usernames = []
output_list = []

with open(test_dataset_path, "rt", encoding="utf-8") as fh:
    for line in fh:
        sample = json.loads(line)

        # Perform prediction
        pred_val = predict_like_count(sample["username"])  # Ensure `predict_like_count` is defined
        sample["like_count"] = int(pred_val)
        output_list.append(sample)

# Step 3: Save the Output to a File
with open(output_file_path, "wt", encoding="utf-8") as of:
    json.dump(output_list, of)

# Step 4: Output Verification
print(f"Processed data saved to: {output_file_path}")

# %%
# output_list first 3 items
pprint(output_list[:3])

