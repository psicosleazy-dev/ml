from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# Carregar o conjunto de dados Iris como exemplo
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names  # nome de cada atributo
target_names = iris.target_names  # nome de cada classe

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo kNN com k=3
knn_model = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
knn_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn_model.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)


