import joblib
import pandas as pd

def make_prediction(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
  # 1. Carregando o modelo salvo
    model_path = "models/modelo_iris.joblib"
    model = joblib.load(model_path)
    
    # 2. Preparando os novos dados
    # O modelo espera um DataFrame com os mesmos nomes de colunas que ele aprendeu
    new_data = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    })
    
    # 3. Fazendo a previsão
    prediction = model.predict(new_data)
    
    # 4. Retornando o resultado
    return prediction[0]
# Bloco para testar o script diretamente
if __name__ == "__main__":
    # Vamos testar com as medidas de uma flor hipotética (parecida com uma Setosa)
    resultado = make_prediction(
        sepal_length=6.5, 
        sepal_width=3.0, 
        petal_length=5.8, 
        petal_width=2.2
    )
    print(f"A espécie prevista para essas medidas é: {resultado}")