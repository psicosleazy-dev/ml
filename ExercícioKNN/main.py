import read_data
import knn

if __name__ == "__main__":
    # primeiro para normalizados
    X_train,y_train = read_data.read('TrainingData_2F_Norm.txt')
    X_test,y_test = read_data.read('TestingData_2F_Norm.txt')
    knn.calc_knn(X_train,X_test,y_train,y_test,1)
    knn.calc_knn(X_train,X_test,y_train,y_test,3)
    knn.calc_knn(X_train,X_test,y_train,y_test,5)
    knn.calc_knn(X_train,X_test,y_train,y_test,7)
    
    # depois para os originais
    X_train,y_train = read_data.read('TrainingData_2F_Original.txt')
    X_test,y_test = read_data.read('TestingData_2F_Original.txt')
    knn.calc_knn(X_train,X_test,y_train,y_test,1)
    knn.calc_knn(X_train,X_test,y_train,y_test,3)
    knn.calc_knn(X_train,X_test,y_train,y_test,5)
    knn.calc_knn(X_train,X_test,y_train,y_test,7)