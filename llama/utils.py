from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(file_path, test_size=0.2, seed=42):
    df = pd.read_csv(file_path)
    df = df.dropna()
    encoding_data = df['encoding'].tolist()
    train_data, validation_data = train_test_split(encoding_data, test_size=test_size, random_state=seed)
    train_dataset = Dataset.from_dict({'text': train_data})
    val_dataset = Dataset.from_dict({'text': validation_data})
    return {'train': train_dataset, 'validation': val_dataset}
