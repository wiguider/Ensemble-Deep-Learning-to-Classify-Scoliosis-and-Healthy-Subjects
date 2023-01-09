from pandas import read_excel, DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import max, random, array, round
from ensemble import EnsembleNNClassifier


def load_scoliosis_data():
    dataset = read_excel('data/pone.0261511.s001.xlsx',
                         sheet_name='health-scoliotic patients')
    X = dataset.drop(['Patients','y'], axis=1)
    y = dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    print('Loading the data')
    X_train, y_train, X_test, y_test = load_scoliosis_data()
    
    print('Loading the model')
    loaded_clf = EnsembleNNClassifier.load('model')
    
    
    # feature_names = 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27'
    candidate_subject = [30.525,-12.75,-3,1.66,12.325,0.395,-0.000718327,-0.3780369,-0.672629766,-0.810873309,-0.909343173,67.085,26.475,41.875,32.18,17.385,3.95,-6.335,0.72,-6.335,7.055,-3.345,2.95,6.23,6.23,-3.33,9.56]
    # Making predictions for the first 10 rows in the test set
    # ensemble_predictions = [x[0] for x in loaded_clf.predict(X_test.iloc[:10])]
    # print('Making predictions using the loaded model',ensemble_predictions )

    print('Making predictions using the loaded model', loaded_clf.predict(candidate_subject))