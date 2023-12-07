from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import csv

#cria dataframe "tabela" que guarda a separação de X e y de treino e teste.
X_train = pd.DataFrame()
y_train = pd.DataFrame()
X_test = pd.DataFrame()
y_test = pd.DataFrame()
mse = 0

#importa dataset
col_names = ['ID', 'p_Sist', 'p_Diast','qualPres', 'pulso', 'resp', 'gravidade', 'clas_grav']
df = pd.read_csv('datasets\data_800vic\sinais_vitais_com_label.txt', names = col_names, index_col=False)
df = df[['ID', 'qualPres', 'pulso', 'resp', 'gravidade']]
X = df[['qualPres', 'pulso', 'resp']]
y = df[['gravidade']]

#tamanho do batch e número de folds
batch_Size = 200
k_num = 10

#cria o kfold com k_num folds e aleatorizando o dataset
kf = KFold(n_splits = k_num, shuffle = True)

#cria o multi layer perceptron(mlp), com os parametros: algoritmo 'sgd', tamanho do batch 'batch_Size',
#3 camadas com 3 neurônios cada, learning rate 0.0007, momentum 0.7, 
#número máximo de iterações(condição de parada) 100, warm_start(guarda os 'fits' anteriores) true.
mlp = MLPRegressor(solver='sgd',
                   batch_size = batch_Size,
                   hidden_layer_sizes=(4,4),
                   learning_rate_init = 0.0007,
                   momentum=0.9,
                   max_iter=700,
                   warm_start=True,
                   activation='logistic')

#separa o dataset df.
for train_index, test_index in kf.split(df):
    #zera o dataframa para que cada iteração não guarde os splits da iteração anterior
    X_train = X_train.iloc[0:0]
    y_train = y_train.iloc[0:0]
    X_test = X_test.iloc[0:0]
    y_test = y_test.iloc[0:0]

    #guarda os dados do split combinando oq tem em X('qualPres', 'pulso', 'resp') com oq tem em X_train e y_train
    #q no inicio estão vazios. Isso é feito pra separação de treino.
    for index in train_index:
        X_train = pd.concat([X_train, X.iloc[[index]]], ignore_index=True)
        y_train = pd.concat([y_train, y.iloc[[index]]], ignore_index=True)

    #mesma coisa q acima, mas para teste.
    for index in test_index:
        X_test = pd.concat([X_test, X.iloc[[index]]], ignore_index=True)
        y_test = pd.concat([y_test, y.iloc[[index]]], ignore_index=True)

    #fit = treino com os valores da separação do k-fold.
    mlp.fit(X_train, y_train.values.ravel())

    #predição dos valores de teste da separação do kfold.
    y_pred = mlp.predict(X_test)

    #erro médio quadrático.
    mse = mse + mean_squared_error(y_test, y_pred)

#print :)
print(mse / k_num)

with open('a.csv', 'a+', newline='') as f:
    writer_object = csv.writer(f)
    writer_object.writerows([[val] for val in y_pred])