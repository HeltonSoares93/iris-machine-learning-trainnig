import pandas as pd
from pathlib import Path

def load_data(filepath: str = "../Iris.csv") -> tuple:
  df = pd.read_csv(filepath)
  df = df.drop(columns=['Id'])
  X = df.drop(columns=['Species'])
  y = df['Species']
  
  return X,y