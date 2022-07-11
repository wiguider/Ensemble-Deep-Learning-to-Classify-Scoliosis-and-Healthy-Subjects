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
    X_train, y_train, X_test, y_test = load_scoliosis_data()
    
    clf = EnsembleNNClassifier(input_shape=X_train.iloc[0].shape, layers_units=(64,64,), n_members=16)
    print('Start training...')
    clf.fit(X_train, y_train, 
                     epochs= 100, 
                     batch_size= 8)
    print('End training')
    acc = clf.evaluate(X_test, y_test)
    print(f'Balanced accuracy of the model: {round(acc,3)}')
    print('Saving the model')
    clf.save('model')
    print('Loading the model')
    loaded_clf = EnsembleNNClassifier.load('model')
    print('Evaluating the loaded model:', loaded_clf.evaluate(X_test, y_test))

    ensemble_predictions = [x[0] for x in loaded_clf.predict(X_test.iloc[:10])]
    
    print('Making predictions using the loaded model',ensemble_predictions )
