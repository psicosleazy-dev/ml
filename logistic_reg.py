from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

# Criar um conjunto de dados fictício para exemplo
data = {'feature_1': [1, 2, 3, 4, 5],
        'feature_2': [2, 3, 4, 5, 6],
        'target': [0, 0, 1, 1, 1]}

df = pd.DataFrame(data)

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df[['feature_1', 'feature_2']], df['target'], test_size=0.2, random_state=42)

# Inicializar e treinar o modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a performance do modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print(f'Acurácia: {accuracy}')
print(f'Precisão: {precision}')
print(f'Revocação: {recall}')
