# MELD subset sampling script (e.g., balanced emotions)
# It will create a balanced train_labels_balanced.csv file (e.g., 80 samples per emotion, ~500 total)
# Or if you want full dataset → just use original train_labels.csv.


import pandas as pd

def sample_balanced(df_path, samples_per_emotion=80):
    df = pd.read_csv(df_path)
    balanced_df = df.groupby('Emotion').sample(n=samples_per_emotion, random_state=42)
    return balanced_df

if __name__ == "__main__":
    balanced_df = sample_balanced('train_labels.csv')
    balanced_df.to_csv('train_labels_balanced.csv', index=False)
    print("Saved balanced subset to train_labels_balanced.csv")
