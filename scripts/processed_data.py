import os
import pandas as pd


FAKE_NEWS_RAW_PATH_FAKE = "data/fake_news/raw/gossipcop_fake.csv"
FAKE_NEWS__RAW_PATH_REAL = "data/fake_news/raw/gossipcop_real.csv"
NEWS_PROCESSED_PATH = "data/fake_news/processed/gossipcop_processed.csv"

if os.path.exists(NEWS_PROCESSED_PATH):
    print(f"Processed data already exists at {NEWS_PROCESSED_PATH}.")
else:

    print("Processing raw data files...")

if not os.path.exists(FAKE_NEWS_RAW_PATH_FAKE) or not os.path.exists(FAKE_NEWS__RAW_PATH_REAL):
    raise FileNotFoundError("One or more data files are missing.")


fake_news_df = pd.read_csv(FAKE_NEWS_RAW_PATH_FAKE)
real_news_df = pd.read_csv(FAKE_NEWS__RAW_PATH_REAL)

# get the id, title colums and the label for fake news + real news

fake_news_df = fake_news_df[['id', 'title']].copy()
fake_news_df['label'] = 0  # 0 for fake news
real_news_df = real_news_df[['id', 'title']].copy()
real_news_df['label'] = 1  # 1 for real news

# count the labels
fake_count = fake_news_df['label'].value_counts().get(0, 0)
real_count = real_news_df['label'].value_counts().get(1, 0)


print(f"Fake News Count: {fake_count}")
print(f"Real News Count: {real_count}")

# concat the two dataframes
news_df = pd.concat([fake_news_df, real_news_df], ignore_index=True)

# shuffle the dataframe
news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

# save the dataframe to csv
news_df.to_csv(NEWS_PROCESSED_PATH, index=False)

print(f"Processed data saved to {NEWS_PROCESSED_PATH}.")
print("Data processing complete.")