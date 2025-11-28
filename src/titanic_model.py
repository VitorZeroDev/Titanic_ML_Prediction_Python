# ======================================================
# 1. Importação de bibliotecas
# ======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, median_absolute_error

# ======================================================
# 2. Leitura dos dados
# ======================================================
url = 'data/titanic_data.xlsx'
titanic = pd.read_excel(url)

# Visualizar os primeiros registros
display(titanic.head())

# ======================================================
# 3. Limpeza da base de dados
# ======================================================
# Remover colunas irrelevantes
titanic.drop(['Fare', 'index', 'PassengerId'], axis=1, inplace=True)

# Substituir a coluna 'Sex' por valores binários (0 = female, 1 = male)
titanic['Sex'] = titanic['Sex'].replace({'female': 0, 'male': 1})

# Visualizar a base limpa
display(titanic.head())

# ======================================================
# 4. Separação das variáveis independentes (X) e dependente (y)
# ======================================================
X = titanic.drop(['Survived_y', 'Name'], axis=1)
y = titanic['Survived_y']

# ======================================================
# 5. Separação dos dados em treino e teste
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# ======================================================
# 6. Criação e treinamento do modelo
# ======================================================
modelo = LinearSVC()
modelo.fit(X_train, y_train)

# ======================================================
# 7. Avaliação do modelo
# ======================================================
# Previsões
y_pred = modelo.predict(X_test)

# Erro médio absoluto
erro_medio = median_absolute_error(y_test, y_pred)
print(f"Erro médio absoluto: {erro_medio}")

# Acurácia (%)
acuracia = accuracy_score(y_test, y_pred) * 100
print(f"Acurácia: {acuracia:.2f}%")

# ======================================================
# 8. Informações gerais da base
# ======================================================
titanic.info()
display(titanic.head())

# ======================================================
# 1. Carregamento do modelo treinado
# ======================================================
import joblib
import pandas as pd
import numpy as np


joblib.dump(modelo, 'modelo.pkl')
print("Modelo salvo com sucesso!")

# ======================================================
# 2. Funções auxiliares para entrada do usuário
# ======================================================

def entrada_numerica(pergunta, tipo=float):
    """
    Função para garantir que a entrada seja um número positivo.
    """
    while True:
        try:
            valor = tipo(input(pergunta))
            if valor < 0:
                print("Por favor, insira um número positivo.")
            else:
                return valor
        except ValueError:
            print("Entrada inválida. Por favor, insira um valor numérico.")

def entrada_binaria(pergunta, opcoes=(0, 1)):
    """
    Função para garantir que a entrada seja binária (0 ou 1).
    """
    while True:
        try:
            valor = int(input(pergunta))
            if valor in opcoes:
                return valor
            else:
                print(f"Valor inválido. Digite {opcoes[0]} ou {opcoes[1]}.")
        except ValueError:
            print("Entrada inválida. Digite 0 ou 1.")

# ======================================================
# 3. Coleta de dados do usuário
# ======================================================
pclass = entrada_numerica(
    "Pclass (1 = Primeira Classe, 2 = Segunda Classe, 3 = Terceira Classe): ", tipo=int)
sex = entrada_binaria("Sexo (0 = Feminino, 1 = Masculino): ")
age = entrada_numerica("Idade: ", tipo=float)
sibsp = entrada_numerica("Número de irmãos/cônjuges a bordo (SibSp): ", tipo=int)
parch = entrada_numerica("Número de pais/filhos a bordo (Parch): ", tipo=int)

# Criando DataFrame com os dados do usuário
df_usuario = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch]
})

# ======================================================
# 4. Previsão pelo modelo
# ======================================================
prev = modelo_treinado.predict(df_usuario)

# ======================================================
# 5. Interpretação do resultado
# ======================================================
if prev[0] == 1:
    mensagem = "O passageiro provavelmente VAI sobreviver. (1)"
else:
    mensagem = "O passageiro provavelmente NÃO VAI sobreviver.(0)"

print(mensagem)

