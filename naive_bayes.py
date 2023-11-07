from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Suponha que você tenha um conjunto de dados com textos e rótulos
textos = ["Isso é um texto de exemplo.", "Outro exemplo de texto.", "Um terceiro exemplo."]
rotulos = ["Classe A", "Classe B", "Classe A"]

# Crie um vetor de características usando a representação CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

# Divida o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, rotulos, test_size=0.2, random_state=42)

# Crie uma instância do classificador Naive Bayes
clf = MultinomialNB()

# Treine o classificador com os dados de treinamento
clf.fit(X_train, y_train)

# Faça previsões com os dados de teste
y_pred = clf.predict(X_test)

# Calcule a acurácia do modelo
acuracia = accuracy_score(y_test, y_pred)
print("Acurácia:", acuracia)

# Exiba um relatório de classificação
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))