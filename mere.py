import pandas as pd

# Load CSVs
real_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

# Add labels: 0 for real, 1 for fake
real_df['label'] = 0
fake_df['label'] = 1

# Optionally combine title and text for better training
real_df['full_text'] = real_df['title'].fillna('') + ' ' + real_df['text'].fillna('')
fake_df['full_text'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')

# Combine and shuffle
merged_df = pd.concat([real_df, fake_df], ignore_index=True)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset for ML
merged_df[['full_text', 'label']].to_csv('data/processed_news.csv', index=False)
print('✅ Merged CSV saved as data/processed_news.csv')
import pandas as pd
df = pd.read_csv('data/processed_news.csv')
print(df['label'].value_counts())

