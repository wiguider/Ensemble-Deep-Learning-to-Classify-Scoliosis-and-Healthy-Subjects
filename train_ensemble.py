from pandas import read_excel
from sklearn.model_selection import train_test_split
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
    
    clf = EnsembleNNClassifier(input_shape=X_train.iloc[0].shape, layers_units=(64,64,64), n_members=5)
    print('Start training...')
    clf.fit(X_train, y_train, 
                     epochs= 25, 
                     batch_size= 8)
    print('End training')
    print('Balanced accuracy of the model:', clf.evaluate(X_test, y_test))
    print('Saving the model')
    clf.save('model')
    print('Loading the model')
    loaded_clf = EnsembleNNClassifier.load('model')
    print('Evaluating the loaded model:', loaded_clf.evaluate(X_test, y_test))
    print('Making predictions using the loaded model', loaded_clf.predict(X_test.iloc[:10]))
