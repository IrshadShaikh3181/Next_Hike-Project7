# %% [markdown]
# # Part 1: Data Exploration and Preparation
# 
# In this section, we will explore the structure and characteristics of the tweet dataset provided. The goal is to understand the data distribution, identify any missing values, and analyze patterns in disaster vs. non-disaster tweets.
# 

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv("twitter_disaster (1).csv")

# Display basic info
print("Shape of the dataset:", df.shape)
df.info()
df.head()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv("twitter_disaster (1).csv")

# Display basic info
print("Shape of the dataset:", df.shape)
df.info()
df.head()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv("twitter_disaster (1).csv")

# Display basic info
print("Shape of the dataset:", df.shape)
df.info()
df.head()


# %%
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])

import matplotlib.pyplot as plt



# %% [markdown]
# ### 🔍 Dataset Overview
# 
# - The dataset contains **10,000 tweets** classified into two categories:
#   - `target = 1`: Disaster-related tweets
#   - `target = 0`: Non-disaster tweets
# - Key columns:
#   - `id`: Unique identifier for the tweet
#   - `text`: The actual tweet content
#   - `location`, `keyword`: Optional metadata
#   - `target`: Classification label
# 
# We will now check for missing values and the distribution of the target variable.
# 

# %%
# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Count of disaster vs. non-disaster tweets
sns.countplot(x='target', data=df)
plt.title("Distribution of Tweets by Target Label")
plt.xlabel("Target (0 = Non-Disaster, 1 = Disaster)")
plt.ylabel("Tweet Count")
plt.show()


# %% [markdown]
# ### 📊 Data Quality and Class Distribution
# 
# - The `keyword` and `location` columns contain a significant number of missing values.
# - The `text` and `target` columns are complete and can be used directly.
# - Distribution:
#   - **Approximately 43%** of tweets are disaster-related.
#   - **Approximately 57%** are non-disaster tweets.
# 
# This class imbalance is moderate and should be considered during model training.
# 

# %%
# Word Cloud for Disaster Tweets
disaster_text = df[df['target'] == 1]['text'].dropna().str.cat(sep=' ')
wordcloud_disaster = WordCloud(width=800, height=400, background_color='black').generate(disaster_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_disaster, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Disaster Tweets", fontsize=16)
plt.show()


# %%
# Word Cloud for Non-Disaster Tweets
non_disaster_text = df[df['target'] == 0]['text'].dropna().str.cat(sep=' ')
wordcloud_non_disaster = WordCloud(width=800, height=400, background_color='white').generate(non_disaster_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_non_disaster, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Non-Disaster Tweets", fontsize=16)
plt.show()


# %% [markdown]
# ### ☁️ Word Cloud Analysis
# 
# - **Disaster Tweets**:
#   - Common words include "fire", "emergency", "earthquake", "killed", and "rescue".
#   - These indicate real-world events and crisis scenarios.
# 
# - **Non-Disaster Tweets**:
#   - Words such as "like", "love", "people", and "happy" dominate.
#   - These are more casual and conversational, indicating no emergency.
# 
# These insights confirm that the tweet content differs significantly between the two classes.
# 

# %% [markdown]
# # Part 2: Data Cleaning and Feature Engineering
# 
# In this section, we will clean the tweet text and convert it into numerical features that machine learning models can understand. This includes:
# - Removing unwanted characters (URLs, punctuation, numbers)
# - Lowercasing
# - Removing extra spaces
# - Vectorizing the cleaned text using TF-IDF
# 

# %%
import re

# Clean the text column
def clean_text(text):
    text = str(text).lower()  # Lowercase
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Preview cleaned text
df[['text', 'clean_text']].head()


# %% [markdown]
# ### 🧼 Text Cleaning Summary
# 
# We have cleaned the tweet text by:
# - Removing URLs and special characters
# - Converting all characters to lowercase
# - Stripping unnecessary white spaces
# 
# This makes the text consistent and easier to process in the next steps.
# 

# %% [markdown]
# ## Part 2: Feature Engineering and Model Selection
# ### Task: Feature Engineering
# 
# In this task, we will extract relevant features from the tweet data using the following methods:
# - TF-IDF vectorization
# - Additional manual features like tweet length, number of hashtags, and mentions
# - (Optional) Consideration of word embeddings like Word2Vec or GloVe
# 

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Re-initialize TF-IDF with more features if needed
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['clean_text'])

print("TF-IDF feature shape:", X_tfidf.shape)


# %%
# Extract additional features
df['tweet_length'] = df['text'].apply(len)
df['hashtag_count'] = df['text'].apply(lambda x: x.count('#'))
df['mention_count'] = df['text'].apply(lambda x: x.count('@'))

# Combine these with TF-IDF using sparse matrix
from scipy.sparse import hstack

X_additional = df[['tweet_length', 'hashtag_count', 'mention_count']]
X_combined = hstack([X_tfidf, X_additional])

# Target variable
y = df['target']

# Check final shape
print("Final feature matrix shape (TF-IDF + manual):", X_combined.shape)


# %% [markdown]
# ### 🧠 Feature Engineering Insights
# 
# - **TF-IDF** captured the relative importance of each word in the tweet.
# - We added **manual features** like:
#   - `tweet_length`: Longer tweets may carry more context.
#   - `hashtag_count`: Tweets with hashtags may be event-driven.
#   - `mention_count`: Mentions may suggest conversation rather than alerts.
# - The final feature matrix combines both automated and manually engineered features for improved model learning.
# 

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "gensim"])



# %% [markdown]
# ### 🌐 Word2Vec Embeddings (with Gensim)
# 
# Word2Vec helps to capture **semantic relationships** between words. Unlike TF-IDF, it learns the context of words and generates word vectors that reflect real-world meaning. We will:
# - Tokenize the cleaned tweets
# - Train a Word2Vec model using Gensim
# - Convert each tweet into a vector (mean of its word vectors)
# 

# %%
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Tokenize cleaned tweets
df['tokens'] = df['clean_text'].apply(word_tokenize)
df['tokens'].head()


# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade scipy"])



# %%



# %%



