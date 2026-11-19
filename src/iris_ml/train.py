import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from src.iris_ml.data import load_data

def train_model():
  # 1. Carregando os dados
  X,y = load_data(filepath="Iris.csv")
  
  # 2. Separando treino e teste
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
  
  # 3. Montando o Pipeline
  pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('knn',KNeighborsClassifier(n_neighbors=3))
    ])
  
  # 4. Treinando o modelo
  pipeline.fit(X_train, y_train)
  
  # 5. Salva o modelo treinado / Cria a pasta 'models' se ela não existir
  Path("models").mkdir(exist_ok=True)
  
  model_path = "models/modelo_iris.joblib"
  
# Dica: use a função joblib.dump(objeto, caminho) para salvar
  joblib.dump( pipeline,model_path)
    
  print(f"Modelo treinado e salvo com sucesso em: {model_path}")
  return pipeline
  
if __name__ == "__main__":
    train_model()