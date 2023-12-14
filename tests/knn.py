from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Iris como exemplo
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names  # nome de cada atributo
target_names = iris.target_names  # nome de cada classe

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Inicializar o modelo kNN com k=3
knn_model = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
knn_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn_model.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)

print(f'Acurácia do modelo kNN: {accuracy:.2f}')

# print(f'X: {X}')
# print(f'y: {y}')

#print(f"Nomes dos atributos: {feature_names}\n")
#print(f"Nomes das classes: {target_names}")

'''import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
'''

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy)
print(y)
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_model.classes_)
disp.plot()
plt.show()