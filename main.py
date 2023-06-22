
import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import neurolab as neuro
import matplotlib.pyplot as plt

names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
         "Age", "Outcome"]

def get_data():
    file = "/Users/dominikbednarz/Downloads/diabetes.csv"
    dataset = pandas.read_csv(file, names=names, header=0, delimiter=',')
    return dataset.values


def save_data_to_csv(data, file_name):
    df = pandas.DataFrame(data, columns=names)
    df.to_csv(file_name, index=False)


def check_zero(data):
    for i in range(data.shape[0]):
        if data[i, 2] < 0.1:
            print('znaleziona wartość w', i, 2, '->', data[i, 2])


def check_duplicates(input_data):
    data = np.copy(input_data)
    unique_rows = np.unique(data, axis=0)
    return unique_rows


def normalize_data(input_data):
    data = np.copy(input_data)
    result_norm = np.linalg.norm(data[:, -1])
    for i in range(data.shape[1]):
        data[:, i] = data[:, i] / np.linalg.norm(data[:, i])
    return data, result_norm


def split_data_sets(input_data):
    data = np.copy(input_data)
    training_data, testing_data = train_test_split(data, test_size=0.2, random_state=25)
    return training_data, testing_data


def split_data_to_args_and_result(input_data):
    data = np.copy(input_data)
    size = data.shape[0]
    args = data[:, :-1]
    result = data[:, -1]
    reshaped_result = np.reshape(result, (size, 1))
    return args, reshaped_result


def visualize_results(original_results, predicted_results):
    fig, ax = plt.subplots()
    ax.plot(original_results, label='Właściwy wynik')
    ax.plot(predicted_results, label='Przewidywany wynik')
    ax.legend()
    ax.set_title('Właściwy i przewidywany wynik')
    plt.show()


def train_model(training_data, testing_data):
    train_data_args, train_data_results = split_data_to_args_and_result(training_data)
    test_data_args, test_data_results = split_data_to_args_and_result(testing_data)
    neural = neuro.net.newff(
        [[np.min(train_data_args[:, i]), np.max(train_data_args[:, i])] for i in range(train_data_args.shape[1])],
        [10, 5, 1])
    err = neural.train(train_data_args, train_data_results, epochs=100, show=1, goal=0)
    predicted_results = neural.sim(test_data_args)
    func = neuro.error.SSE()
    test_error = func(test_data_results, predicted_results)

    print('Błędy nauczania: ', err)
    print('Błędy klasyfikacji: ', test_error)
    print('Dokładność: ', 100 - (test_error / test_data_results.shape[0] * 100))


def train_model_using_random_trees(training_data, testing_data, result_norm):
    train_data_args, train_data_results = split_data_to_args_and_result(training_data)
    test_data_args, test_data_results = split_data_to_args_and_result(testing_data)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(train_data_args, train_data_results.ravel())
    predicted_results = model.predict(test_data_args)
    test_error = mean_squared_error(test_data_results, predicted_results)
    print('Dokładność: ', 100 - (test_error / test_data_results.shape[0] * 100))

    visualize_results(test_data_results * result_norm, predicted_results * result_norm)


def run_multilayer():
    data = get_data()
    data, _ = normalize_data(data)
    data = check_duplicates(data)
    training_data, testing_data = split_data_sets(data)
    train_model(training_data, testing_data)


def run_tree():
    data = get_data()
    data, result_norm = normalize_data(data)
    data = check_duplicates(data)
    training_data, testing_data = split_data_sets(data)
    train_model_using_random_trees(training_data, testing_data, result_norm)


run_tree()
