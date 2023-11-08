from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Exemplo de dados de treinamento (features) e rótulos (classes)
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = ['A', 'B', 'B', 'A']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o classificador k-NN com k=2
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Treinar o modelo com os dados de treinamento
knn.fit(X_train, y_train)

# Fazer previsões com base nos dados de teste
y_pred = knn.predict(X_test)

# Calcular a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisão: {accuracy}')