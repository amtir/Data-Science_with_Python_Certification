
# 1. Import Libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Load datasets
df_ratings = pd.read_csv('BX-Book-Ratings.csv', encoding='ISO-8859-1')
df_books = pd.read_csv('BX-Books.csv', encoding='ISO-8859-1')
df_users = pd.read_csv('BX-Users.csv', encoding='ISO-8859-1')

# Check data
print("Ratings Data:\n", df_ratings.head())
print("Books Data:\n", df_books.head())
print("Users Data:\n", df_users.head())

# Check the null values 
print(df_ratings.isnull().sum(), '\n')
print(df_books.isnull().sum(),'\n')
print(df_users.isnull().sum(),'\n')


print(df_ratings.info(), '\n')
print(df_books.info(),'\n')
print(df_users.info(),'\n')

# Cleaning the Data:

# Replace missing Location values with 'Unknown Location'
df_users['Location'].fillna('Unknown Location', inplace=True)

# Replace missing Age values with the median age
df_users['Age'].fillna(df_users['Age'].median(), inplace=True)

# Replace missing values in df_books
df_books['book_author'].fillna('Unknown Author', inplace=True)
df_books['publisher'].fillna('Unknown Publisher', inplace=True)

# Verify updated null values
print("Updated Null Values in df_books:\n", df_books.isnull().sum(), '\n')
print("Updated Null Values in df_users:\n", df_users.isnull().sum(), '\n')


# The MemoryError occurs because the pivot operation generates a massive matrix 
# (61K rows × 128K columns), which consumes too much memory. 
# Focusing on a smaller subset of the data will address this.

# To reduce memory usage and focus on meaningful data, you can filter users 
# who have interacted with a significant number of books 
# (e.g., users who have rated or rented a certain threshold of books). 
# This helps limit the size of the user-book matrix while keeping high-quality data.

# Step 1: Filter ratings (remove 0 ratings)
df_ratings_filtered = df_ratings[df_ratings['rating'] > 0]

# Step 2: Count the number of books each user interacted with
user_book_counts = df_ratings_filtered['user_id'].value_counts()

# Step 3: Filter users who interacted with significant books (threshold = 5)
threshold = 20
users_significant = user_book_counts[user_book_counts >= threshold].index

# Filter the ratings data for these users
df_ratings_filtered = df_ratings_filtered[df_ratings_filtered['user_id'].isin(users_significant)]

# Step 4: Merge with books data
df_merged = pd.merge(df_ratings_filtered, df_books, on='isbn', how='left')

# Keep only necessary columns
df_merged = df_merged[['user_id', 'book_title']]

# Step 5: Build the User-Book Matrix
user_book_matrix = df_merged.pivot_table(index='user_id', 
                                         columns='book_title', 
                                         aggfunc='size', 
                                         fill_value=0)

# Convert to binary format (1 if user interacted with a book)
user_book_matrix = user_book_matrix.applymap(lambda x: 1 if x > 0 else 0)

print("User-Book Matrix (Filtered):\n", user_book_matrix.head())
print(f"Shape of the User-Book Matrix: {user_book_matrix.shape}")



# Apply Association Rule Mining
# We will use the Apriori algorithm to find frequently rented books and generate association rules.

import mlxtend
print(mlxtend.__version__)

# Step 1: Apply the Apriori algorithm
min_support = 0.01
frequent_itemsets = apriori(user_book_matrix, min_support=min_support, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 2: Generate association rules with a dummy 'num_itemsets'
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
#rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))

# Step 3: Sort and display the rules
rules = rules.sort_values(by="confidence", ascending=False)
print("\nTop Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())










