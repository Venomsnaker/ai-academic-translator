import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(file_path, test_size=0.2, seed=42):
    df = pd.read_csv(file_path)
    encoding_data = df['encoding'].tolist()
    train_data, validation_data = train_test_split(encoding_data, test_size=test_size, random_state=seed)
    return {'train': train_data, 'validation': validation_data}
