

# Additional Dataset Information
#This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them.  
#In particular, the Cleveland database is the only one that has been used by ML researchers to date. 
# The "goal" field refers to the presence of heart disease in the patient.  It is integer valued from 0 (no presence) to 4. 
#Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).  
#The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.
#One file has been "processed", that one containing the Cleveland database.  All four unprocessed files also exist in this directory.
#To see Test Costs (donated by Peter Turney), please see the folder "Costs" 
#Has Missing Values -> Yes  
# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split    



# Carregando o conjunto de dados de doenças cardíacas
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Obtendo informações sobre o conjunto de dados antes do tratamento
total_rows, total_columns = X.shape
percentage_missing = np.mean(np.isnan(X)) * 100
percentage_positive_cases = np.mean(y) * 100

print(f"The data has {total_rows} rows and {total_columns} columns.")
print(f"The percentage of missing values is: {percentage_missing:.1f}%")
print(f"Percentage of positive cases: {percentage_positive_cases:.1f}%\n")



# Visualizando informações sobre o target (y) no conjunto de dados
class_counts = y['num'].value_counts()

labels = class_counts.index
sizes = class_counts.values

colors = ['green', 'lightgreen', 'yellow', 'lightcoral', 'red']  # Cores para as classes

plt.figure(figsize=(10, 5))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, colors=colors)

# Adicionando uma legenda explicativa
legend_labels = ['No presence of heart disease',
                 'Presence (1): Mild heart disease',
                 'Presence (2): Moderate heart disease',
                 'Presence (3): Severe heart disease',
                 'Presence (4): Very severe heart disease']
plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.2, 1))

plt.title('Percentage of Presence and No Presence of Heart Disease')
plt.show()

# Inicializando os classificadores com imputação e padronização no pipeline
# kNN
knn_classifier = make_pipeline(
    SimpleImputer(strategy='mean'),  # Imputação de valores ausentes
    StandardScaler(),  # Padronização dos dados
    KNeighborsClassifier()  # Classificador kNN
)

# Árvore de decisão
dt_classifier = make_pipeline(
    SimpleImputer(strategy='mean'),  # Imputação de valores ausentes
    DecisionTreeClassifier()  # Classificador de árvore de decisão
)

# Naive Bayes
nb_classifier = make_pipeline(
    SimpleImputer(strategy='mean'),  # Imputação de valores ausentes
    GaussianNB()  # Classificador Naive Bayes
)

# Lista de classificadores
classifiers = [knn_classifier, dt_classifier, nb_classifier]
classifiers_names = ['kNN', 'Decision Tree', 'Naive Bayes']



# Definindo as métricas de desempenho
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

# Configurando a validação cruzada k-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Hiperparâmetros para o kNN
knn_param_grid = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Hiperparâmetros para a Decision Tree
dt_param_grid = {
    'decisiontreeclassifier__criterion': ['gini', 'entropy'],
    'decisiontreeclassifier__splitter': ['best', 'random'],
    'decisiontreeclassifier__max_depth': [None, 10, 20, 30]
}

# Otimização de hiperparâmetros para kNN
knn_opt = GridSearchCV(knn_classifier, knn_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
knn_opt.fit(X, y.values.ravel())
best_knn_params = knn_opt.best_params_

# Otimização de hiperparâmetros para Decision Tree
dt_opt = GridSearchCV(dt_classifier, dt_param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
dt_opt.fit(X, y.values.ravel())
best_dt_params = dt_opt.best_params_

# Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros para kNN:", best_knn_params)
print("Melhores hiperparâmetros para Decision Tree:", best_dt_params)

# Agora você pode usar os melhores hiperparâmetros encontrados nos classificadores
best_knn_classifier = knn_opt.best_estimator_
best_dt_classifier = dt_opt.best_estimator_


# Dicionário para armazenar os resultados
results = {}
best_classifiers = [best_knn_classifier, best_dt_classifier, nb_classifier]

# Loop para avaliar o desempenho de cada classificador
for classifier, name in zip(best_classifiers, classifiers_names):  # Considerando apenas os dois primeiros classificadores otimizados
    print(f"Validação cruzada {name}...")
    scores = cross_val_score(classifier, X, y.values.ravel(), cv=kf, scoring='accuracy', n_jobs=-1)
    results[name] = scores

# Exibindo os resultados da precisão
print("Resultados da precisão:")
for name in classifiers_names:
    mean_score = np.mean(results[name])
    std_score = np.std(results[name])
    print(f"{name}: Precisão média - {mean_score:.3f}, Desvio padrão - {std_score:.3f}")

# Visualizando os resultados usando boxplot
plt.boxplot([results[name] for name in classifiers_names], labels=classifiers_names)
plt.title("Comparação da precisão")
plt.show()

# Treinando e extraindo a importância das características para kNN e Decision Tree
best_knn_classifier.fit(X, y.values.ravel())
best_dt_classifier.fit(X, y.values.ravel())

# Obtendo as importâncias das características
dt_feature_importance = best_dt_classifier.named_steps['decisiontreeclassifier'].feature_importances_

# Exibindo a importância das características para Decision Tree
plt.figure(figsize=(12, 6))
plt.bar(range(len(dt_feature_importance)), dt_feature_importance, align='center')
plt.xticks(range(len(dt_feature_importance)), X.columns, rotation=45)  # Use X.columns para obter os nomes das características
plt.title('Feature Importance - Decision Tree')
plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)


# Fit the best classifiers on the training data
best_knn_classifier.fit(X_train, y_train)
best_dt_classifier.fit(X_train, y_train)
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = best_knn_classifier.predict(X_test)
y_pred_dt = best_dt_classifier.predict(X_test)
y_pred_nb = nb_classifier.predict(X_test)

# Obtain confusion matrices
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)

# Plotting confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot confusion matrix for kNN
class_labels = ['No disease', 'Mild', 'Moderate', 'Severe', 'Very severe']


disp = ConfusionMatrixDisplay.from_estimator(
    best_dt_classifier, X_test, y_test, display_labels=class_labels, normalize='true')
disp.ax_.set_title('Confusion Matrix - Decision Tree')


disp2 = ConfusionMatrixDisplay.from_estimator(
    best_knn_classifier, X_test, y_test, display_labels=class_labels, normalize='true')
disp2.ax_.set_title('Confusion Matrix - KNN')


disp3 = ConfusionMatrixDisplay.from_estimator(
    nb_classifier, X_test, y_test, display_labels=class_labels, normalize='true')
disp3.ax_.set_title('Confusion Matrix - Naive Bayes')

print(disp.confusion_matrix)
print(disp2.confusion_matrix)
print(disp3.confusion_matrix)

plt.show()