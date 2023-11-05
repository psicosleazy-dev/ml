import pandas as pd

def read(nome_arquivo):
    # Especifique o separador utilizado na tabela, por padrão é tabulação ('\t')
    separador = '\t'

    # Leia a tabela a partir do arquivo
    tabela = pd.read_csv('TrainingData_2F_Norm.txt', sep=separador)

    # Agora, você pode trabalhar com a tabela como um DataFrame
    # Por exemplo, para imprimir as primeiras 5 linhas:
    print(tabela.head())

    '''
    # Para acessar uma coluna específica, use o nome da coluna:
    coluna = tabela['Nome_da_Coluna']

    # Para acessar um valor específico, use a notação de índice:
    valor = tabela.at[0, 'Nome_da_Coluna']

    # E assim por diante...
'''
    y = tabela['Class']

    X = tabela.drop("Class", axis=1)

    return X,y