#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# USP/ESALQ | MBA Data Science and Analytics
# Trabalho de Conclusão de Curso 
# Aluna: Gabriela Lima da Silva
# Análise da Base de Dados: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset


#%% [1] INSTALANDO OS PACOTES

!pip install pandas
!pip install numpy
!pip install seaborn
!pip install pingouin
!pip install matplotlib
!pip install statsmodels
!pip install scikit-learn
!pip install xgboost

#%% [2] IMPORTANDO OS PACOTES

import pandas as pd 
import numpy as np 
import seaborn as sns 
import pingouin as pg 
import matplotlib.cm as cm
import statsmodels.api as sm 
from matplotlib import pyplot as plt
from statstests.process import stepwise
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')  

#%% [3] ANÁLISE EXPLORATÓRIA DOS DADOS 

# Importando o banco de dados:
df_telecom = pd.read_csv('customer_churn.csv',delimiter=',') 
print(df_telecom)

# Analisando as primeiras linhas do banco de dados:
pd.set_option('display.max_columns', None)
df_telecom.head()

# Verificando a quantidade de linhas e colunas:
df_telecom.shape

# Verificando as características das variáveis:
df_telecom.info()

# Alterando o nome de todas as variáveis de inglês para português:
df_telecom.columns

df_telecom = df_telecom.rename(columns={'Call  Failure':'falhas_chamadas', 
                                        'Complains':'reclamacoes',
                                        'Subscription  Length':'duracao_contrato', 
                                        'Charge  Amount':'valor_fatura',
                                        'Seconds of Use':'chamadas_segundos',
                                        'Frequency of use':'frequencia_chamadas',
                                        'Frequency of SMS':'frequencia_sms', 
                                        'Distinct Called Numbers':'chamadas_distintas',
                                        'Age Group':'faixa_etaria', 
                                        'Tariff Plan':'tarifario',
                                        'Status':'estado', 'Age':'idade', 
                                        'Customer Value':'valor_cliente',
                                        'Churn':'churn'}
                               )

# Transformando os valores das variáveis 'tarifário' e 'estado' de '1' e '2' 
# para '0' e '1':
df_telecom['tarifario'] = df_telecom['tarifario'].replace({1: 0, 2: 1})
df_telecom['estado'] = df_telecom['estado'].replace({1: 0, 2: 1})

# Analisando os valores únicos:
df_telecom.nunique()

# Verificando se existem dados faltantes ou nulos:
df_telecom.isnull().sum()

# Verificando se existem dados duplicados:
df_telecom.duplicated().sum()

duplicated_rows = df_telecom[df_telecom.duplicated()] # Os duplicados foram mantidos
print(duplicated_rows) 

# Analisado as variáveis 'faixa etária' e 'idade':
df_selected = df_telecom[['faixa_etaria', 'idade']]

# Agrupando a coluna 'idade' em categorias definidas e adicionando na base de 
# dados a nova variável 'grupo etário':
age_categories = pd.cut(
    df_telecom['idade'],
    bins=[0, 30, 40, float('inf')],
    labels=['abaixo_30', '30_40', 'acima_40'],
    right=False)

df_telecom['grupo_etario'] = age_categories

# Exibindo as primeiras linhas do conjunto de dados atualizado:
df_telecom[['idade', 'grupo_etario']].head(20)
df_telecom.info()

# Removendo as variáveis redundantes 'faixa etária' e 'idade':
df_telecom.drop(columns=['faixa_etaria','idade'], inplace=True)
df_telecom.info()

# Reagrupando a coluna 'valor_fatura' em duas categorias, '0' (valores de 0-5) 
# e '1' (valores de 6-10), e adicionando na base de dados:
bins_new_var = [0, 5, 10]
labels_new_var = ['0', '1'] 

df_telecom['fatura'] = pd.cut(df_telecom['valor_fatura'], 
                                     bins=bins_new_var, labels=labels_new_var, 
                                     right=True, include_lowest=True
                                     )

# Removendo a variável 'valor_fatura':
df_telecom.drop(columns=['valor_fatura'], inplace=True)
df_telecom.info()    

# Transformando a variável Y e as variáveis preditoras para o tipo 'object':
df_telecom['reclamacoes'] = df_telecom['reclamacoes'].astype('object')
df_telecom['tarifario'] = df_telecom['tarifario'].astype('object')
df_telecom['estado'] = df_telecom['estado'].astype('object')
df_telecom['churn'] = df_telecom['churn'].astype('object')
df_telecom['grupo_etario'] = df_telecom['grupo_etario'].astype('object')
df_telecom['fatura'] = df_telecom['fatura'].astype('object')
df_telecom.info()

# Estatística descritiva univariada:
pd.set_option('display.max_columns', None)
df_telecom.describe()

# Tabela de frequências da variável Y e das variáveis categóricas:
df_telecom['reclamacoes'].value_counts().sort_index()
df_telecom['tarifario'].value_counts().sort_index()
df_telecom['estado'].value_counts().sort_index()
df_telecom['churn'].value_counts().sort_index()
df_telecom['grupo_etario'].value_counts().sort_index()
df_telecom['fatura'].value_counts().sort_index()

# Analisando a distribuição das variáveis quantitativas:
def analyze_var_quant(column, data, title):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data[column], kde=True, color='mediumblue')
    plt.title(f"Histograma de {title}", fontsize=16)
    plt.xlabel(title, fontsize=14)
    plt.ylabel('Frequência', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[column], color='mediumseagreen')
    plt.title(f"Boxplot de {title}", fontsize=16)
    plt.xlabel(title, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
variables_quanti = {'falhas_chamadas':'falhas de chamada', 
                    'duracao_contrato':'duração do contrato',
                    'chamadas_segundos':'chamadas em segundos',
                    'frequencia_chamadas':'frequência de chamadas',
                    'frequencia_sms':'frequência de SMS', 
                    'chamadas_distintas':'número de chamadas distintas', 
                    'valor_cliente':'valor de vida do cliente'
                    }
 
for col, custom_title in variables_quanti.items():
    analyze_var_quant(col, df_telecom, custom_title) # Os outliers foram mantidos
    
# Analisando a distribuição das variáveis categóricas:
def plot_percentages(column, data, title):
    plt.figure(figsize=(8, 6))
    
    ax = sns.countplot(x=column, data=data, palette='viridis')
    total = len(data)
    for p in ax.patches:
        percentage = f"{100 * p.get_height() / total:.1f}%"
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=12)
            
    plt.title(title, fontsize=16)
    plt.xlabel(column.lower(), fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

variables_quali = {'reclamacoes':'Distribuição das reclamações', 
                   'tarifario':'Distribuição do tarifário', 
                   'estado':'Distribuição do estado', 
                   'churn':'Distribuição do churn',
                   'grupo_etario': 'Distribuição da faixa etária', 
                   'fatura':'Distribuição do valor da fatura'
                   }

for col, custom_title in variables_quali.items():
    plot_percentages(col, df_telecom, custom_title)
    
# Gráfico de pizza para a variável 'churn':
churn_counts = df_telecom["churn"].value_counts()

labels = ["Non-Churn (0)", "Churn (1)"]
colors = ["royalblue", "mediumseagreen"]  

plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=labels, autopct="%1.1f%%", 
        colors=colors, startangle=90, shadow=True, explode=(0.1, 0))

plt.show()

# Realizando a análise bivariada das variáveis quantitativas, sendo o churn a variável alvo:
plt.figure(figsize=(20, 15))

plt.subplot(3, 3, 1)
sns.boxplot(x='churn', y='falhas_chamadas', hue='churn', palette='viridis', data=df_telecom)
plt.title('Boxplot: falhas de chamadas vs churn')

plt.subplot(3, 3, 2)
sns.boxplot(x='churn', y='duracao_contrato', hue='churn', palette='viridis', data=df_telecom)
plt.title('Boxplot: duração do contrato vs churn')

plt.subplot(3, 3, 3)
sns.boxplot(x='churn', y='chamadas_segundos', hue='churn', palette='viridis', data=df_telecom)
plt.title('Boxplot: chamadas em segundos vs churn')

plt.subplot(3, 3, 4)
sns.boxplot(x='churn', y='frequencia_chamadas', hue='churn', palette='viridis', data=df_telecom)
plt.title('Boxplot: frequência de chamadas vs churn')

plt.subplot(3, 3, 5)
sns.boxplot(x='churn', y='frequencia_sms', hue='churn', palette='viridis', data=df_telecom)
plt.title('Boxplot: frequência de SMS vs churn')

plt.subplot(3, 3, 6)
sns.boxplot(x='churn', y='chamadas_distintas', hue='churn', palette='viridis', data=df_telecom)
plt.title('Boxplot: número de chamadas distintas vs churn')

plt.subplot(3, 3, 7)
sns.boxplot(x='churn', y='valor_cliente', hue='churn', palette='viridis', data=df_telecom)
plt.title('Boxplot: valor de vida do cliente vs churn')
   
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()    
    
# Realizando a análise bivariada das variáveis categóricas, sendo o churn a variável alvo:
def add_percentage_by_group(ax, data, x, hue):
    counts = data.groupby([x, hue]).size().unstack(fill_value=0)
    
    percentages = counts.apply(lambda c: c / c.sum() * 100, axis=1)
    
    hue_order = ax.legend_.get_texts()
    hue_labels = [t.get_text() for t in hue_order]
    
    for i, c in enumerate(ax.containers):
        labels = []
        for j, v in enumerate(c):
            height = v.get_height()
            if height > 0:
                try:
                    percentage = percentages.iloc[j, percentages.columns.get_loc(hue_labels[i])]
                except KeyError:
                    percentage = percentages.iloc[j, i]
                labels.append(f'{percentage:.1f}%')
            else:
                labels.append('')
        ax.bar_label(c, labels=labels, label_type='edge', padding=2)

    plt.tight_layout()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)  

plt.figure(figsize=(20, 45))

plt.subplot(8, 2, 1)
plt.gca().set_title('Reclamações')
ax1 = sns.countplot(x='reclamacoes', hue='churn', palette='viridis', data=df_telecom)
ax1.set_ylabel('Contagem')
add_percentage_by_group(ax1, df_telecom, 'reclamacoes', 'churn')

plt.subplot(8, 2, 2)
plt.gca().set_title('Tipo de tarifário')
ax1 = sns.countplot(x='tarifario', hue='churn', palette='viridis', data=df_telecom)
ax1.set_ylabel('Contagem')
add_percentage_by_group(ax1, df_telecom, 'tarifario', 'churn')

plt.subplot(8, 2, 3)
plt.gca().set_title('Estado')
ax1 = sns.countplot(x='estado', hue='churn', palette='viridis', data=df_telecom)
ax1.set_ylabel('Contagem')
add_percentage_by_group(ax1, df_telecom, 'estado', 'churn')

plt.subplot(8, 2, 4)
plt.gca().set_title('Grupo etário')
ax1 = sns.countplot(x='grupo_etario', hue='churn', palette='viridis', data=df_telecom)
ax1.set_ylabel('Contagem')
add_percentage_by_group(ax1, df_telecom, 'grupo_etario', 'churn')

plt.subplot(8, 2, 5)
plt.gca().set_title('Fatura')
ax1 = sns.countplot(x='fatura', hue='churn', palette='viridis', data=df_telecom)
ax1.set_ylabel('Contagem')
add_percentage_by_group(ax1, df_telecom, 'fatura', 'churn')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Construindo a matriz de correlações:
correlation_matrix = pg.rcorr(df_telecom.iloc[:, [0, 2, 3, 4, 5, 6, 9]],
                              method='pearson',
                              upper='pval', decimals=6,
                              pval_stars={0.01: '***',
                                           0.05: '**',
                                           0.10: '*'})
print(correlation_matrix)

# Construindo mapa de calor com as correlações entre todas as variáveis quantitativas:
selected_data = df_telecom.iloc[:, [0, 2, 3, 4, 5, 6, 9]].apply(pd.to_numeric, errors='coerce')
correlation_matrix = selected_data.corr()
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, 
                      cmap=plt.cm.viridis_r, fmt=".4f", square=True, vmin=-1, vmax=1)

plt.show()

# Identificando pares de variáveis com alta correlação:
threshold = 0.8
high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_corr_pairs.append((
                correlation_matrix.index[i],  
                correlation_matrix.columns[j],  
                correlation_matrix.iloc[i, j]))

print(high_corr_pairs)

# Removendo as variáveis com alta correlação:
df_telecom2 = df_telecom.drop(columns=['chamadas_segundos', 'valor_cliente'])
df_telecom2.columns

#%% [4] APLICAÇÃO DOS MODELOS PREDITIVOS

# [4.1] MODELO LOGÍSTICO BINÁRIO

# Transformando a variável Y para o tipo 'int64' novamente:
df_telecom2['churn'] = df_telecom2['churn'].astype('int64')
df_telecom2.info()

# Dummizando as variáveis 'reclamacoes', 'tarifario','estado', 'grupo_etario', 
#'fatura':
df_telecom2_dummies = pd.get_dummies(df_telecom2,
                                       columns=['reclamacoes',
                                                'tarifario',
                                                'estado',
                                                'grupo_etario',
                                                'fatura'],
                                       dtype=int,
                                       drop_first=True)

df_telecom2_dummies.info()

# Definindo a fórmula a ser utilizada no modelo:
list_columns = list(df_telecom2_dummies.drop(columns=['churn']).columns)

formula_dummies_model = ' + '.join(list_columns)
formula_dummies_model = "churn ~ " + formula_dummies_model
print("Fórmula utilizada: ", formula_dummies_model)

# Definindo X (variáveis independentes) e y (variável dependente):
X = df_telecom2_dummies.drop(columns=['churn'])
y = df_telecom2_dummies['churn']

# Divisão em treino e teste (70% treino, 30% teste):
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=100)

# Adicionando a coluna de intercepto para o modelo de regressão logística:
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Rodando o modelo propriamente dito no conjunto de treino:
model_log_churn = sm.Logit(y_train, X_train).fit(maxiter=100)

# Parâmetros do modelo:
model_log_churn.summary()

# Estimando o modelo através do procedimento stepwise:
step_model_log_churn = stepwise(model_log_churn, pvalue_limit=0.05)

# Construindo uma função para definir a matriz de confusão:
def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm_log = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_log)
    disp.plot()
    plt.xlabel('Valor real')
    plt.ylabel('Valor previsto')
    plt.title('Regressão Logística')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade_log = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade_log = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia_log = accuracy_score(observado, predicao_binaria)
    precisao_log = precision_score(observado, predicao_binaria)
    f1_log = f1_score(observado, predicao_binaria)

    indicadores_log = pd.DataFrame({'Acurácia': [f"{acuracia_log*100:.1f}%"],
                                    'Sensitividade': [f"{sensitividade_log*100:.1f}%"],
                                    'Especificidade': [f"{especificidade_log*100:.1f}%"],
                                    'Precisão': [f"{precisao_log*100:.1f}%"],
                                    'F1-score': [f"{f1_log*100:.1f}%"]}
                                   )

    return indicadores_log

# Adicionando os valores previstos de probabilidade na base de dados de teste:
X_test['phat'] = step_model_log_churn.predict(X_test)

# Matriz de confusão (cutoff = 0.5) e indicadores:
indicadores_log = matriz_confusao(observado=y_test,
                predicts=X_test['phat'],
                cutoff=0.5)
print("\n=== Indicadores da Regressão Logística ===")
print(indicadores_log)

# Construindo a curva ROC:
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, X_test['phat'])
roc_auc_log = auc(fpr_log, tpr_log)

# Calculando o coeficiente de Gini:
gini_log = (roc_auc_log - 0.5)/(0.5)

# Plotando a curva ROC (base de teste):
plt.figure(figsize=(15,10), dpi=600)
plt.plot(fpr_log, tpr_log, color='dodgerblue', linewidth=4)
plt.plot(fpr_log, fpr_log, color='gray', linestyle='dashed')
plt.title('AUC-ROC Regressão Logística: %g' % round(roc_auc_log, 3) +
          ' | Coeficiente de Gini: %g' % round(gini_log, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%% [4.2] MODELO ÁRVORE DE DECISÃO (DECISION TREE)

# Separando as variáveis X e y:
X = df_telecom2_dummies.drop(columns=['churn'])
y = df_telecom2_dummies['churn']

# Separando as amostras de treino e teste (70% para treino e 30% para teste):
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=100)

# Gerando a árvore de decisão:
tree_telecom2 = DecisionTreeClassifier(criterion='gini', max_depth=3,
                                     min_samples_split=10,
                                     min_samples_leaf=5,
                                     random_state=100)
tree_telecom2.fit(X_train, y_train)

# Analisando a árvore de decisão: 
plt.figure(figsize=(20,10), dpi=600)
plot_tree(tree_telecom2,
          feature_names=X.columns.tolist(),
          class_names=['Non-churn','Churn'],
          proportion=False,
          filled=True,
          node_ids=True)
plt.show()

# Analisando os resultados dos splits:
tree_split_telecom2 = pd.DataFrame(tree_telecom2.cost_complexity_pruning_path(X_train, y_train))
tree_split_telecom2.sort_index(ascending=False, inplace=True)

print(tree_split_telecom2)

# Obtendo os valores preditos pela árvore na base de treino e na base de teste:
tree_pred_train_class = tree_telecom2.predict(X_train)
tree_pred_train_prob = tree_telecom2.predict_proba(X_train)

tree_pred_test_class = tree_telecom2.predict(X_test)
tree_pred_test_prob = tree_telecom2.predict_proba(X_test)

# Matriz de confusão e indicadores (base de treino):    
tree_cm_train = confusion_matrix(tree_pred_train_class, y_train)
cm_train_disp_tree = ConfusionMatrixDisplay(tree_cm_train)

plt.rcParams['figure.dpi'] = 600
cm_train_disp_tree.plot(colorbar=True, cmap='viridis_r')
plt.title('Árvore de Decisão: Treino')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_tree_train = accuracy_score(y_train, tree_pred_train_class)
sens_tree_train = recall_score(y_train, tree_pred_train_class, pos_label=1)
espec_tree_train = recall_score(y_train, tree_pred_train_class, pos_label=0)
prec_tree_train = precision_score(y_train, tree_pred_train_class)
f1_tree_train = f1_score(y_train, tree_pred_train_class)

print("\n=== Indicadores da Árvore (Base de Treino) ===")
print(f"Acurácia: {acc_tree_train:.1%}")
print(f"Sensitvidade: {sens_tree_train:.1%}")
print(f"Especificidade: {espec_tree_train:.1%}")
print(f"Precision: {prec_tree_train:.1%}")
print(f"F1-score: {f1_tree_train:.1%}")

# Matriz de confusão e indicadores (base de teste):
tree_cm_test = confusion_matrix(tree_pred_test_class, y_test)
cm_test_disp_tree = ConfusionMatrixDisplay(tree_cm_test)

plt.rcParams['figure.dpi'] = 600
cm_test_disp_tree.plot(colorbar=True, cmap='viridis')
plt.title('Árvore de Decisão')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_tree_test = accuracy_score(y_test, tree_pred_test_class)
sens_tree_test = recall_score(y_test, tree_pred_test_class, pos_label=1)
espec_tree_test = recall_score(y_test, tree_pred_test_class, pos_label=0)
prec_tree_test = precision_score(y_test, tree_pred_test_class)
f1_tree_test = f1_score(y_test, tree_pred_test_class)

print("\n=== Indicadores da Árvore (Base de Teste) ===")
print(f"Acurácia: {acc_tree_test:.1%}")
print(f"Sensitvidade: {sens_tree_test:.1%}")
print(f"Especificidade: {espec_tree_test:.1%}")
print(f"Precision: {prec_tree_test:.1%}")
print(f"F1-score: {f1_tree_test:.1%}")

# Parametrizando a função da curva ROC na base de teste (real vs. previsto):
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, tree_pred_test_prob[:,1])
roc_auc_tree = auc(fpr_tree, tpr_tree)

# Cálculando o coeficiente de Gini:
gini_tree = (roc_auc_tree - 0.5)/(0.5)

# Plotando a curva ROC (base de teste):
plt.figure(figsize=(15,10), dpi=600)
plt.plot(fpr_tree, tpr_tree, color='darkviolet', linewidth=4)
plt.plot(fpr_tree, fpr_tree, color='gray', linestyle='dashed')
plt.title('AUC-ROC Árvore de Decisão: %g' % round(roc_auc_tree, 3) +
          ' | Coeficiente de Gini: %g' % round(gini_tree, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitvidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%% [4.3] MODELO RANDOM FORREST

# Determinando a lista de hiperparâmetros desejados e seus valores (grid search):
param_grid_rf = {
    'n_estimators': [500, 1000],
    'max_depth': [5, 7],
    'max_features': [2, 5],
    'min_samples_split': [20, 50]
    }

# Identificando o algoritmo em uso:
rf_grid = RandomForestClassifier(random_state=100)

# Gerando o modelo de random forest (grid search):
rf_grid_model = GridSearchCV(estimator = rf_grid, 
                             param_grid = param_grid_rf,
                             scoring='accuracy', cv=None,
                             verbose=2)
                             
rf_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos:
rf_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros:
rf_best = rf_grid_model.best_estimator_

# Obtendo os valores preditos pelo RF na base de treino e na base de teste:
rf_grid_pred_train_class = rf_best.predict(X_train)
rf_grid_pred_train_prob = rf_best.predict_proba(X_train)

rf_grid_pred_test_class = rf_best.predict(X_test)
rf_grid_pred_test_prob = rf_best.predict_proba(X_test)

# Matriz de confusão e indicadores (base de treino):
rf_grid_cm_train = confusion_matrix(rf_grid_pred_train_class, y_train)
cm_rf_grid_train = ConfusionMatrixDisplay(rf_grid_cm_train)

plt.rcParams['figure.dpi'] = 600
cm_rf_grid_train.plot(colorbar=True, cmap='viridis_r')
plt.title('Random Forest: Treino')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_rf_grid_train = accuracy_score(y_train, rf_grid_pred_train_class)
sens_rf_grid_train = recall_score(y_train, rf_grid_pred_train_class, pos_label=1)
espec_rf_grid_train = recall_score(y_train, rf_grid_pred_train_class, pos_label=0)
prec_rf_grid_train = precision_score(y_train, rf_grid_pred_train_class)
f1_rf_grid_train = f1_score(y_train, rf_grid_pred_train_class)

print("\n=== Indicadores da RF (Base de Treino) ===")
print(f"Acurácia: {acc_rf_grid_train:.1%}")
print(f"Sensitvidade: {sens_rf_grid_train:.1%}")
print(f"Especificidade: {espec_rf_grid_train:.1%}")
print(f"Precision: {prec_rf_grid_train:.1%}")
print(f"F1-score: {f1_rf_grid_train:.1%}")

# Matriz de confusão e indicadores (base de teste):
rf_grid_cm_test = confusion_matrix(rf_grid_pred_test_class, y_test)
cm_rf_grid_test = ConfusionMatrixDisplay(rf_grid_cm_test)

plt.rcParams['figure.dpi'] = 600
cm_rf_grid_test.plot(colorbar=True, cmap='viridis')
plt.title('Random Forest')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_rf_grid_test = accuracy_score(y_test, rf_grid_pred_test_class)
sens_rf_grid_test = recall_score(y_test, rf_grid_pred_test_class, pos_label=1)
espec_rf_grid_test = recall_score(y_test, rf_grid_pred_test_class, pos_label=0)
prec_rf_grid_test = precision_score(y_test, rf_grid_pred_test_class)
f1_rf_grid_test = f1_score(y_test, rf_grid_pred_test_class)

print("\n=== Indicadores da RF (Base de Teste) ===")
print(f"Acurácia: {acc_rf_grid_test:.1%}")
print(f"Sensitvidade: {sens_rf_grid_test:.1%}")
print(f"Especificidade: {espec_rf_grid_test:.1%}")
print(f"Precision: {prec_rf_grid_test:.1%}")
print(f"F1-score: {f1_rf_grid_test:.1%}")

# Parametrizando a função da curva ROC da base de teste (real vs. previsto):
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_grid_pred_test_prob[:,1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Cálculando o coeficiente de Gini:
gini_rf = (roc_auc_rf - 0.5)/(0.5)

# Plotando a curva ROC (base de teste):
plt.figure(figsize=(15,10), dpi=600)
plt.plot(fpr_rf, tpr_rf, color='springgreen', linewidth=4)
plt.plot(fpr_rf, fpr_rf, color='gray', linestyle='dashed')
plt.title('AUC-ROC Random Forest: %g' % round(roc_auc_rf, 3) +
          ' | Coeficiente de Gini: %g' % round(gini_rf, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitvidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%% [4.4] MODELO XGBOOST

# Determinando a lista de hiperparâmetros desejados e seus valores (grid search):
param_grid_xgb = {
    'n_estimators': [100, 500],
    'max_depth': [3, 5],
    'colsample_bytree': [0.5, 1],
    'learning_rate': [0.01, 0.1]
    }

# Identificando o algoritmo em uso:
xgb_grid = XGBClassifier(random_state=100)

# Gerando o modelo de XGBoost (grid search):
xgb_grid_model = GridSearchCV(estimator = xgb_grid, 
                              param_grid = param_grid_xgb,
                              scoring='accuracy', cv=None,
                              verbose=2) 
                              
xgb_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos:
xgb_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros:
xgb_best = xgb_grid_model.best_estimator_

# Obtendo os valores preditos pelo XGBoost para a base de treino e para a 
# base de teste:
xgb_grid_pred_train_class = xgb_best.predict(X_train)
xgb_grid_pred_train_prob = xgb_best.predict_proba(X_train)

xgb_grid_pred_test_class = xgb_best.predict(X_test)
xgb_grid_pred_test_prob = xgb_best.predict_proba(X_test)

# Matriz de confusão (base de treino):
xgb_cm_train = confusion_matrix(xgb_grid_pred_train_class, y_train)
cm_xgb_train = ConfusionMatrixDisplay(xgb_cm_train)

plt.rcParams['figure.dpi'] = 600
cm_xgb_train.plot(colorbar=True, cmap='viridis_r')
plt.title('XGBoost: Treino')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_xgb_train = accuracy_score(y_train, xgb_grid_pred_train_class)
sens_xgb_train = recall_score(y_train, xgb_grid_pred_train_class, pos_label=1)
espec_xgb_train = recall_score(y_train, xgb_grid_pred_train_class, pos_label=0)
prec_xgb_train = precision_score(y_train, xgb_grid_pred_train_class)
f1_xgb_train = f1_score(y_train, xgb_grid_pred_train_class)

print("\n=== Indicadores do XGBoost (Base de Treino) ===")
print(f"Acurácia: {acc_xgb_train:.1%}")
print(f"Sensitvidade: {sens_xgb_train:.1%}")
print(f"Especificidade: {espec_xgb_train:.1%}")
print(f"Precision: {prec_xgb_train:.1%}")
print(f"F1-score: {f1_xgb_train:.1%}")

# Matriz de confusão (base de teste):
xgb_cm_test = confusion_matrix(xgb_grid_pred_test_class, y_test)
cm_xgb_test = ConfusionMatrixDisplay(xgb_cm_test)

plt.rcParams['figure.dpi'] = 600
cm_xgb_test.plot(colorbar=True, cmap='viridis')
plt.title('XGBoost')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_xgb_test = accuracy_score(y_test, xgb_grid_pred_test_class)
sens_xgb_test = recall_score(y_test, xgb_grid_pred_test_class, pos_label=1)
espec_xgb_test = recall_score(y_test, xgb_grid_pred_test_class, pos_label=0)
prec_xgb_test = precision_score(y_test, xgb_grid_pred_test_class)
f1_xgb_test = f1_score(y_test, xgb_grid_pred_test_class)

print("\n=== Indicadores do XGBoost (Base de Teste) ===")
print(f"Acurácia: {acc_xgb_test:.1%}")
print(f"Sensitvidade: {sens_xgb_test:.1%}")
print(f"Especificidade: {espec_xgb_test:.1%}")
print(f"Precision: {prec_xgb_test:.1%}")
print(f"F1-score: {f1_xgb_test:.1%}")

# Parametrizando a função da curva ROC na base de teste (real vs. previsto):
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, xgb_grid_pred_test_prob[:,1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Cálculando o coeficiente de Gini:
gini_xgb = (roc_auc_xgb - 0.5)/(0.5)

# Plotando a curva ROC (base de teste):
plt.figure(figsize=(15,10), dpi=600)
plt.plot(fpr_xgb, tpr_xgb, color='orange', linewidth=4)
plt.plot(fpr_xgb, fpr_xgb, color='gray', linestyle='dashed')
plt.title('AUC-ROC XGBoost: %g' % round(roc_auc_xgb, 3) +
          ' | Coeficiente de Gini: %g' % round(gini_xgb, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitvidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%% [4.5] MODELO LOGÍSTICO BINÁRIO COM K-FOLD (K=5)

# Separando as variáveis X e y:
X = df_telecom2_dummies.drop(columns=['churn'])
y = df_telecom2_dummies['churn']

# Gerando o modelo logístico:
log_reg_sklearn = LogisticRegression(max_iter=100, solver='liblinear')
log_reg_sklearn.fit(X, y)

# Coeficientes e importância das variáveis preditivas:
log_features_telecom2 = pd.DataFrame({
    'features': X.columns,
    'coeficiente': log_reg_sklearn.coef_[0],
    'importancia_absoluta': abs(log_reg_sklearn.coef_[0])
}).sort_values(by='importancia_absoluta', ascending=False)

print("\n=== Importância das Variáveis Preditivas ===")
print(log_features_telecom2)

# Validação cruzada K-Fold (K=5):
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_reg_sklearn, X, y, cv=kfold, scoring='accuracy')

print("\n=== Resultados da Validação Cruzada (K=5) ===")
print("Acurácia média:", round(cv_scores.mean(), 4))
print("Desvio padrão da acurácia:", round(cv_scores.std(), 4))

# Separando base treino/teste para matriz de confusão e ROC:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, random_state=42)

# Ajustando o modelo final e prevendo probabilidades:
log_reg_sklearn_final = LogisticRegression(max_iter=100, solver='liblinear')
log_reg_sklearn_final.fit(X_train, y_train)

# Criando uma cópia de X_test para adicionar 'phat' (sem alterar o original)
X_test_phat = X_test.copy()
X_test_phat['phat'] = log_reg_sklearn_final.predict_proba(X_test)[:, 1]

# Função de matriz de confusão e métricas:
def matriz_confusao(predicts, observado, cutoff):
    predicao_binaria = (predicts >= cutoff).astype(int)
    
    cm_log = confusion_matrix(observado, predicao_binaria)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_log)
    disp.plot()
    plt.xlabel('Valor real')
    plt.ylabel('Valor previsto')
    plt.title('Regressão Logística')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()

    sensitividade_log = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade_log = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia_log = accuracy_score(observado, predicao_binaria)
    precisao_log = precision_score(observado, predicao_binaria)
    f1_log = f1_score(observado, predicao_binaria)

    # Criando DataFrame com valores em percentagem
    df_metricas = pd.DataFrame({
        'Acurácia': [acuracia_log * 100],
        'Sensitividade': [sensitividade_log * 100],
        'Especificidade': [especificidade_log * 100],
        'Precisão': [precisao_log * 100],
        'F1-score': [f1_log * 100]
    })
    
    # Formatando para exibir com 2 casas decimais e o símbolo de %
    return df_metricas.round(1).astype(str) + ' %'

# Matriz de confusão com cutoff = 0.5:
indicadores_log = matriz_confusao(predicts=X_test_phat['phat'], observado=y_test, cutoff=0.5)
print("\n=== Indicadores da Regressão Logística (Base Teste) ===")
print(indicadores_log)

# Plotando a curva ROC (base de teste):
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, X_test_phat['phat'])
roc_auc_log = auc(fpr_log, tpr_log)
gini_log = 2 * roc_auc_log - 1

plt.figure(figsize=(10, 7))
plt.plot(fpr_log, tpr_log, color='dodgerblue', linewidth=2, label=f"AUC = {roc_auc_log:.3f}")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title(f'AUC-ROC Regressão Logística: {roc_auc_log:.3f} | Gini: {gini_log:.4f}')
plt.xlabel('1 - Especificidade')
plt.ylabel('Sensitividade')
plt.legend(loc='lower right')
plt.show()

#%% [4.6] MODELO ÁRVORES DE DECISÃO (DECISION TREE) COM K-FOLD (K=5)

# Separando as variáveis X e y:
X = df_telecom2_dummies.drop(columns=['churn'])
y = df_telecom2_dummies['churn']

# Gerando a árvore de decisão:
tree_telecom2 = DecisionTreeClassifier(max_depth=3,
                                     min_samples_split=10,
                                     min_samples_leaf=5,
                                     random_state=100)

# Aplicando validação cruzada K-fold (K = 5) para acurácia:
cv_scores_tree = cross_val_score(tree_telecom2, X_train, y_train, cv=5, scoring='accuracy')

# Exibindo os resultados da validação cruzada:
print("\n=== Resultados da Validação Cruzada (K=5) ===")
print(f"Acurácia média: {cv_scores_tree.mean():.4f}")
print(f"Desvio padrão da acurácia: {cv_scores_tree.std():.4f}")

# Treinando o modelo no conjunto de treino:
tree_telecom2.fit(X_train, y_train)

# Analisando a árvore de decisão: 
plt.figure(figsize=(20, 10), dpi=600)
plot_tree(tree_telecom2,
          feature_names=X.columns.tolist(),
          class_names=['Non-churn', 'Churn'],
          proportion=False,
          filled=True,
          node_ids=True)
plt.show()

# Analisando os resultados dos splits:
tree_split_telecom2 = pd.DataFrame(tree_telecom2.cost_complexity_pruning_path(X_train, y_train))
tree_split_telecom2.sort_index(ascending=False, inplace=True)

print(tree_split_telecom2)

# Importância das variáveis preditoras:
tree_features_telecom2 = pd.DataFrame({'features': X.columns.tolist(),
                                       'importance': tree_telecom2.feature_importances_})

print("\n=== Importância das Variáveis Preditivas ===")
print(tree_features_telecom2)

# Obtendo os valores preditos pela árvore na base de treino e na base de teste:
tree_pred_train_class = tree_telecom2.predict(X_train)
tree_pred_train_prob = tree_telecom2.predict_proba(X_train)

tree_pred_test_class = tree_telecom2.predict(X_test)
tree_pred_test_prob = tree_telecom2.predict_proba(X_test)

# Matriz de confusão (base de treino):
tree_cm_train = confusion_matrix(tree_pred_train_class, y_train)
cm_train_disp_tree = ConfusionMatrixDisplay(tree_cm_train)

plt.rcParams['figure.dpi'] = 600
cm_train_disp_tree.plot(colorbar=True, cmap='viridis_r')
plt.title('Árvore de Decisão: Treino')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_tree_train = accuracy_score(y_train, tree_pred_train_class)
sens_tree_train = recall_score(y_train, tree_pred_train_class, pos_label=1)
espec_tree_train = recall_score(y_train, tree_pred_train_class, pos_label=0)
prec_tree_train = precision_score(y_train, tree_pred_train_class)
f1_tree_train = f1_score(y_train, tree_pred_train_class)

print("\n=== Indicadores da Árvore (Base de Treino) ===")
print(f"Acurácia: {acc_tree_train:.1%}")
print(f"Sensitvidade: {sens_tree_train:.1%}")
print(f"Especificidade: {espec_tree_train:.1%}")
print(f"Precision: {prec_tree_train:.1%}")
print(f"F1-score: {f1_tree_train:.1%}")

# Matriz de confusão (base de teste):
tree_cm_test = confusion_matrix(tree_pred_test_class, y_test)
cm_test_disp_tree = ConfusionMatrixDisplay(tree_cm_test)

plt.rcParams['figure.dpi'] = 600
cm_test_disp_tree.plot(colorbar=True, cmap='viridis')
plt.title('Árvore de Decisão: Teste')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_tree_test = accuracy_score(y_test, tree_pred_test_class)
sens_tree_test = recall_score(y_test, tree_pred_test_class, pos_label=1)
espec_tree_test = recall_score(y_test, tree_pred_test_class, pos_label=0)
prec_tree_test = precision_score(y_test, tree_pred_test_class)
f1_tree_test = f1_score(y_test, tree_pred_test_class)

print("\n=== Indicadores da Árvore (Base de Teste) ===")
print(f"Acurácia: {acc_tree_test:.1%}")
print(f"Sensitvidade: {sens_tree_test:.1%}")
print(f"Especificidade: {espec_tree_test:.1%}")
print(f"Precision: {prec_tree_test:.1%}")
print(f"F1-score: {f1_tree_test:.1%}")

# Parametrizando a função da curva ROC na base de teste (real vs. previsto):
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_test, tree_pred_test_prob[:,1])
roc_auc_tree = auc(fpr_tree, tpr_tree)

# Plotando a curva ROC (base de teste):
plt.figure(figsize=(15,10), dpi=600)
plt.plot(fpr_tree, tpr_tree, color='darkviolet', linewidth=4)
plt.plot(fpr_tree, fpr_tree, color='gray', linestyle='dashed')
plt.title('AUC-ROC Árvore de Decisão: %g' % round(roc_auc_tree, 3), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitvidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%%# [4.7] MODELO RANDOM FOREST COM K-FOLD (K=5)

# Determinando a lista de hiperparâmetros desejados e seus valores:
param_grid_rf = {
    'n_estimators': [500, 1000],
    'max_depth': [5, 7],
    'max_features': [2, 5],
    'min_samples_split': [20, 50]
    }

# Identificando o algoritmo em uso:
rf_grid = RandomForestClassifier(random_state=100)

# Treinando os modelos para o grid search:
rf_grid_model = GridSearchCV(estimator=rf_grid, 
                             param_grid=param_grid_rf,
                             scoring='accuracy',
                             cv=5,
                             verbose=2)

rf_grid_model.fit(X_train, y_train)

# Exibindo os resultados da validação cruzada:
cv_results_rf = rf_grid_model.cv_results_
mean_accuracy_rf = rf_grid_model.best_score_
std_accuracy_rf = cv_results_rf['std_test_score'][rf_grid_model.best_index_]

print("\n=== Resultados da Validação Cruzada (K=5) ===")
print(f"Acurácia média: {mean_accuracy_rf:.4f}")
print(f"Desvio padrão da acurácia: {std_accuracy_rf:.4f}")

# Verificando os melhores parâmetros obtidos:
rf_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros:
rf_best = rf_grid_model.best_estimator_

# Obtendo os valores preditos pelo RF na base de treino e na base de teste:
rf_grid_pred_train_class = rf_best.predict(X_train)
rf_grid_pred_train_prob = rf_best.predict_proba(X_train)

rf_grid_pred_test_class = rf_best.predict(X_test)
rf_grid_pred_test_prob = rf_best.predict_proba(X_test)

# Matriz de confusão (base de treino):
rf_grid_cm_train = confusion_matrix(rf_grid_pred_train_class, y_train)
cm_rf_grid_train = ConfusionMatrixDisplay(rf_grid_cm_train)

plt.rcParams['figure.dpi'] = 600
cm_rf_grid_train.plot(colorbar=True, cmap='viridis_r')
plt.title('Random Forest: Treino')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_rf_grid_train = accuracy_score(y_train, rf_grid_pred_train_class)
sens_rf_grid_train = recall_score(y_train, rf_grid_pred_train_class, pos_label=1)
espec_rf_grid_train = recall_score(y_train, rf_grid_pred_train_class, pos_label=0)
prec_rf_grid_train = precision_score(y_train, rf_grid_pred_train_class)
f1_rf_grid_train = f1_score(y_train, rf_grid_pred_train_class)

print("\n=== Indicadores da RF (Base de Treino) ===")
print(f"Acurácia: {acc_rf_grid_train:.1%}")
print(f"Sensitvidade: {sens_rf_grid_train:.1%}")
print(f"Especificidade: {espec_rf_grid_train:.1%}")
print(f"Precision: {prec_rf_grid_train:.1%}")
print(f"F1-score: {f1_rf_grid_train:.1%}")

# Matriz de confusão e indicadores (base de teste):
rf_grid_cm_test = confusion_matrix(rf_grid_pred_test_class, y_test)
cm_rf_grid_test = ConfusionMatrixDisplay(rf_grid_cm_test)

plt.rcParams['figure.dpi'] = 600
cm_rf_grid_test.plot(colorbar=True, cmap='viridis')
plt.title('Random Forest: Teste')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_rf_grid_test = accuracy_score(y_test, rf_grid_pred_test_class)
sens_rf_grid_test = recall_score(y_test, rf_grid_pred_test_class, pos_label=1)
espec_rf_grid_test = recall_score(y_test, rf_grid_pred_test_class, pos_label=0)
prec_rf_grid_test = precision_score(y_test, rf_grid_pred_test_class)
f1_rf_grid_test = f1_score(y_test, rf_grid_pred_test_class)

print("\n=== Indicadores da RF (Base de Teste) ===")
print(f"Acurácia: {acc_rf_grid_test:.1%}")
print(f"Sensitvidade: {sens_rf_grid_test:.1%}")
print(f"Especificidade: {espec_rf_grid_test:.1%}")
print(f"Precision: {prec_rf_grid_test:.1%}")
print(f"F1-score: {f1_rf_grid_test:.1%}")

# Importância das variáveis preditoras:

rf_features = pd.DataFrame({'features':X.columns.tolist(),
                            'importance':rf_best.feature_importances_})

print("\n=== Importância das Variáveis Preditivas ===")
print(rf_features)

# Parametrizando a função da curva ROC (real vs. previsto):
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_grid_pred_test_prob[:,1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plotando a curva ROC (base de teste):
plt.figure(figsize=(15,10), dpi=600)
plt.plot(fpr_rf, tpr_rf, color='springgreen', linewidth=4)
plt.plot(fpr_rf, fpr_rf, color='gray', linestyle='dashed')
plt.title('AUC-ROC Random Forest: %g' % round(roc_auc_rf, 3), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitvidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%% [4.8] MODELO XGBOOST COM K-FOLD (K=5)

# Determinando a lista de hiperparâmetros desejados e seus valores:
param_grid_xgb = {
    'n_estimators': [100, 500],
    'max_depth': [3, 5],
    'colsample_bytree': [0.5, 1],
    'learning_rate': [0.01, 0.1]
    }

# Identificar o algoritmo em uso:
xgb_grid = XGBClassifier(random_state=100, use_label_encoder=False, eval_metric='logloss')

# Treinar os modelos para o grid search:
xgb_grid_model = GridSearchCV(estimator=xgb_grid, 
                              param_grid=param_grid_xgb,
                              scoring='accuracy', 
                              cv=5,  
                              verbose=2,
                              return_train_score=True)

xgb_grid_model.fit(X_train, y_train)

# Exibindo os resultados da validação cruzada:
cv_results_xgb = xgb_grid_model.cv_results_
mean_accuracy_xgb = xgb_grid_model.best_score_
std_accuracy_xgb = cv_results_xgb['std_test_score'][xgb_grid_model.best_index_]

print("\n=== Resultados da Validação Cruzada (K=5) ===")
print(f"Acurácia média: {mean_accuracy_xgb:.4f}")
print(f"Desvio padrão da acurácia: {std_accuracy_xgb:.4f}")

# Verificando os melhores parâmetros obtidos:
xgb_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros:
xgb_best = xgb_grid_model.best_estimator_

# Importância das variáveis preditoras:
xgb_features_telecom2 = pd.DataFrame({'features': X.columns.tolist(),
                                      'importance': xgb_best.feature_importances_})

print("\n=== Importância das Variáveis Preditivas ===")
print(xgb_features_telecom2)

# Obtendo os valores preditos pelo XGBoost na base de treino e na base de teste:
xgb_grid_pred_train_class = xgb_best.predict(X_train)
xgb_grid_pred_train_prob = xgb_best.predict_proba(X_train)

xgb_grid_pred_test_class = xgb_best.predict(X_test)
xgb_grid_pred_test_prob = xgb_best.predict_proba(X_test)

# Matriz de confusão (base de treino):
xgb_cm_train = confusion_matrix(xgb_grid_pred_train_class, y_train)
cm_xgb_train = ConfusionMatrixDisplay(xgb_cm_train)

plt.rcParams['figure.dpi'] = 600
cm_xgb_train.plot(colorbar=True, cmap='viridis_r')
plt.title('XGBoost: Treino')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_xgb_train = accuracy_score(y_train, xgb_grid_pred_train_class)
sens_xgb_train = recall_score(y_train, xgb_grid_pred_train_class, pos_label=1)
espec_xgb_train = recall_score(y_train, xgb_grid_pred_train_class, pos_label=0)
prec_xgb_train = precision_score(y_train, xgb_grid_pred_train_class)

print("\n=== Indicadores da XGBoost (Base de Treino) ===")
print(f"Acurácia: {acc_xgb_train:.1%}")
print(f"Sensitvidade: {sens_xgb_train:.1%}")
print(f"Especificidade: {espec_xgb_train:.1%}")
print(f"Precision: {prec_xgb_train:.1%}")

# Matriz de confusão e indicadores (base de teste):
xgb_cm_test = confusion_matrix(xgb_grid_pred_test_class, y_test)
cm_xgb_test = ConfusionMatrixDisplay(xgb_cm_test)

plt.rcParams['figure.dpi'] = 600
cm_xgb_test.plot(colorbar=True, cmap='viridis')
plt.title('XGBoost: Teste')
plt.xlabel('Valor real')
plt.ylabel('Valor previsto')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()

acc_xgb_test = accuracy_score(y_test, xgb_grid_pred_test_class)
sens_xgb_test = recall_score(y_test, xgb_grid_pred_test_class, pos_label=1)
espec_xgb_test = recall_score(y_test, xgb_grid_pred_test_class, pos_label=0)
prec_xgb_test = precision_score(y_test, xgb_grid_pred_test_class)
f1_xgb_test = f1_score(y_test, xgb_grid_pred_test_class)

print("\n=== Indicadores da XGBoost (Base de Teste) ===")
print(f"Acurácia: {acc_xgb_test:.1%}")
print(f"Sensitvidade: {sens_xgb_test:.1%}")
print(f"Especificidade: {espec_xgb_test:.1%}")
print(f"Precision: {prec_xgb_test:.1%}")
print(f"F1-score: {f1_xgb_test:.1%}")

# Parametrizando a função da curva ROC na base de teste (real vs. previsto):
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, xgb_grid_pred_test_prob[:, 1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plotando a curva ROC (base de teste):
plt.figure(figsize=(15, 10), dpi=600)
plt.plot(fpr_xgb, tpr_xgb, color='orange', linewidth=4)
plt.plot(fpr_xgb, fpr_xgb, color='gray', linestyle='dashed')
plt.title('AUC-ROC XGBoost: %g' % round(roc_auc_xgb, 3), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitvidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

#%% [4.9] PLOTANDO AS CURVAS ROC PARA OS MODELOS

plt.figure(figsize=(15, 10))

plt.plot(fpr_log, tpr_log, color='dodgerblue', markersize=10, linewidth=3,
         label=f'Regressão Logística (AUC = {roc_auc_log:.1%})')
plt.plot(fpr_tree, tpr_tree, color='darkviolet', markersize=10, linewidth=3,
         label=f'Árvore de Decisão (AUC = {roc_auc_tree:.1%})')
plt.plot(fpr_rf, tpr_rf, color='springgreen', markersize=10, linewidth=3,
         label=f'Random Forest (AUC = {roc_auc_rf:.1%})')
plt.plot(fpr_xgb, tpr_xgb, color='orange', markersize=10, linewidth=3,
         label=f'XGBoost (AUC = {roc_auc_xgb:.1%})')

plt.plot(fpr_log, fpr_log, color='gray', linestyle='dashed')
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(fontsize = 16, loc='lower right')
plt.show()

#%% [4.10] IMPORTÂNCIA DAS VARIÁVEIS EXPLICATIVAS PARA CADA MODELO:

# Criando DataFrames com a importância das variáveis para cada modelo
log_features_telecom2 = pd.DataFrame({
    'feature': X.columns.tolist(),
    'coeficiente': log_reg_sklearn.coef_[0],
    'importance': abs(log_reg_sklearn.coef_[0])
}).sort_values(by='importance', ascending=False)

tree_features_telecom2 = pd.DataFrame({
    'feature': X.columns.tolist(),
    'importance': tree_telecom2.feature_importances_
}).sort_values(by='importance', ascending=False)

rf_features_telecom2 = pd.DataFrame({
    'feature': X.columns.tolist(),
    'importance': rf_best.feature_importances_
}).sort_values(by='importance', ascending=False)

xgb_features_telecom2 = pd.DataFrame({
    'feature': X.columns.tolist(),
    'importance': xgb_best.feature_importances_
}).sort_values(by='importance', ascending=False)

# Configuração dos gráficos    
plt.figure(figsize=(11, 8))

def add_value_labels(ax, df, spacing=5):
    max_value = df['importance'].max()
    ax.set_xlim(0, max_value * 1.25)
    
    for rect in ax.patches:
        x = rect.get_width()
        y = rect.get_y() + rect.get_height() / 2
        value = round(x, 4)
        ax.annotate(
            value,
            (x, y),
            xytext=(spacing, 0),
            textcoords="offset points",
            ha='left',
            va='center',
            fontsize=10
        )

# Gráfico para Regressão Logística
plt.subplot(2, 2, 1)
ax1 = sns.barplot(x='importance', y='feature', 
                 data=log_features_telecom2.head(11),
                 palette='viridis')
plt.title('Regressão Logística', fontsize=12)
plt.xlabel('Importância')
plt.ylabel('Variáveis')
add_value_labels(ax1, log_features_telecom2.head(11))

# Gráfico para Árvore de Decisão
plt.subplot(2, 2, 2)
ax2 = sns.barplot(x='importance', y='feature', 
                 data=tree_features_telecom2.head(11),
                 palette='viridis')
plt.title('Árvore de Decisão', fontsize=12)
plt.xlabel('Importância')
plt.ylabel('Variáveis')
add_value_labels(ax2, tree_features_telecom2.head(11))

# Gráfico para Random Forest
plt.subplot(2, 2, 3)
ax3 = sns.barplot(x='importance', y='feature', 
                 data=rf_features_telecom2.head(11),
                 palette='viridis')
plt.title('Random Forest', fontsize=12)
plt.xlabel('Importância')
plt.ylabel('Variáveis')
add_value_labels(ax3, rf_features_telecom2.head(11))

# Gráfico para XGBoost
plt.subplot(2, 2, 4)
ax4 = sns.barplot(x='importance', y='feature', 
                 data=xgb_features_telecom2.head(11),
                 palette='viridis')
plt.title('XGBoost', fontsize=12)
plt.xlabel('Importância')
plt.ylabel('Variáveis')
add_value_labels(ax4, xgb_features_telecom2.head(11))

plt.tight_layout()
plt.show()

#%% ###### FIM ######