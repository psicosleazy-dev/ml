import read_data
import knn

if __name__ == "__main__":    
    print('2F')
    # primeiro para os originais
    # treino
    X_train,y_train = read_data.read('TrainingData_2F_Original.txt')
    
    # teste
    X_test,y_test = read_data.read('TestingData_2F_Original.txt')

    knn.calc_knn(X_train,X_test,y_train,y_test,1)
    knn.calc_knn(X_train,X_test,y_train,y_test,3)
    knn.calc_knn(X_train,X_test,y_train,y_test,5)
    knn.calc_knn(X_train,X_test,y_train,y_test,7)


    # depois para normalizados
    # treino
    X_train,y_train = read_data.read('TrainingData_2F_Norm.txt')
    
    # teste
    X_test,y_test = read_data.read('TestingData_2F_Norm.txt')

    knn.calc_knn(X_train,X_test,y_train,y_test,1)
    knn.calc_knn(X_train,X_test,y_train,y_test,3)
    knn.calc_knn(X_train,X_test,y_train,y_test,5)
    knn.calc_knn(X_train,X_test,y_train,y_test,7)

'''
    print('11F')
    # primeiro para os originais
    # treino
    X,y = read_data.read('TrainingData_2F_Original.txt')
    print('dados orig de treino:')
    knn.calc_knn(X,y,1)
    knn.calc_knn(X,y,3)
    knn.calc_knn(X,y,5)
    knn.calc_knn(X,y,7)

    # teste
    X,y = read_data.read('TestingData_2F_Original.txt')
    print('dados orig de teste:')
    knn.calc_knn(X,y,1)
    knn.calc_knn(X,y,3)
    knn.calc_knn(X,y,5)
    knn.calc_knn(X,y,7)


    # depois para normalizados
    X,y = read_data.read('TrainingData_11F_Norm.txt')
    print('dados norm de treino:')
    knn.calc_knn(X,y,1)
    knn.calc_knn(X,y,3)
    knn.calc_knn(X,y,5)
    knn.calc_knn(X,y,7)

    print('dados norm de teste:')
    X,y = read_data.read('TestingData_11F_Norm.txt')
    knn.calc_knn(X,y,1)
    knn.calc_knn(X,y,3)
    knn.calc_knn(X,y,5)
    '''