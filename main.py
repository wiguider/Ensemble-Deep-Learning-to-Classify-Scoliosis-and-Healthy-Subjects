from pandas import read_excel, DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
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
    status_dict = {1: 'has Scoliosis', 0: 'is Healthy'}
    
    # feature_names = 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27'
    
    # Making predictions for the first 10 rows in the test set
    ensemble_predictions = [f'Subject {i} ' + status_dict[int(x[0])] for i, x in enumerate(loaded_clf.predict(X_test.iloc[:10]))]
    print()
    print('-------------------------------------------------------------------------' )
    print()
    print('Making predictions for the first 10 rows in the test set' )
    print(ensemble_predictions)
    # Making predictions for two candidate subjects
    candidate_subject = [30.525,-12.75,-3,1.66,12.325,0.395,-0.000718327,-0.3780369,-0.672629766,-0.810873309,-0.909343173,67.085,26.475,41.875,32.18,17.385,3.95,-6.335,0.72,-6.335,7.055,-3.345,2.95,6.23,6.23,-3.33,9.56]
    candidate_subject_1 = [8.37,-12,6,2.12,12.93,3.35,0.007217427,-0.329721479,-0.562601289,-0.787001093,-0.932781819,63.14,37.33,45.21,26.18,10.71,2.17,5.28,5.28,-1.32,6.61,0.91,1.58,-3.13,1.94,-3.13,5.07]
    prediction = loaded_clf.predict([candidate_subject])[0]
    status = status_dict[int(prediction)]
    print(f'The candidate subject {status}')
    prediction = loaded_clf.predict([candidate_subject_1])[0]
    status = status_dict[int(prediction)]
    print(f'The candidate subject {status}')
