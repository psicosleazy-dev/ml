from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''import numpy as np

# Defina os dois pontos como arrays numpy
ponto1 = np.array([1, 2])
ponto2 = np.array([4, 6])

# Calcule a distância euclidiana
distancia = np.linalg.norm(ponto1 - ponto2)

print(f"A distância euclidiana entre os pontos é: {distancia}")
'''

# pensar bem em como dividir tudo
def calc_knn(X,y,k):
    # Exemplo de dados de treinamento (features) e rótulos (classes)
    # X = [[1, 2], [2, 3], [3, 4], [4, 5]]
    # y = ['A', 'B', 'B', 'A']

    # Dividir os dados em conjuntos de treinamento e teste
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0, random_state=0) nao e necessario

    # Criar o classificador k-NN
    # k = 3
    knn = KNeighborsClassifier(n_neighbors=k)

    # Treinar o modelo com os dados de treinamento
    knn.fit(X, y)

    # Fazer previsões com base nos dados de teste
    y_pred = knn.predict(X)

    # Calcular a precisão do modelo
    accuracy = accuracy_score(y, y_pred)
    print(f'Precisão/acurácia: {accuracy}')