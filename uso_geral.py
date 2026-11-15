import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    f1_score
)



# 1. CARREGAMENTO DOS DADOS
df = pd.read_csv(
    'acidentes2023_todas_causas_tipos.csv',
    sep=';',
    encoding='latin1'
)

# Remover espaços extras dos nomes das colunas
df.columns = df.columns.str.strip()

linhas_iniciais = len(df)
print(f"Total de registros na planilha original: {linhas_iniciais}")


# 2. PRÉ-PROCESSAMENTO E LIMPEZA DOS DADOS


# 2.1 Padronizar nomes alternativos da condição meteorológica
df = df.rename(columns={
    'condicao_metereologic': 'condicao_meteorologica',
    'condicao_metereologica': 'condicao_meteorologica'
})

# Colunas de entrada ativas para análise
colunas_texto = [
    'dia_semana',
    'tipo_veiculo',
    'tipo_pista',
    'condicao_meteorologica',
    'fase_dia',
    'causa_acidente',
    'tipo_acidente',
    'tracado_via',
    'uso_solo',
    'tipo_envolvido', 
    'sexo'
]

colunas_vitimas = [
    'ilesos',
    'feridos_leves',
    'feridos_graves',
    'mortos'
]

colunas_analise = ['idade'] + colunas_texto


# 2.2 Padronizar os textos das colunas
for col in colunas_texto:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# 2.2.1 Otimização de colunas com alta cardinalidade (Agrupamento)
# Mantém as 10 categorias mais frequentes de cada coluna e agrupa o restante como 'outros'

if 'tracado_via' in df.columns:
    top_tracados = df['tracado_via'].value_counts().nlargest(10).index
    df['tracado_via'] = np.where(df['tracado_via'].isin(top_tracados), df['tracado_via'], 'outros')

if 'causa_acidente' in df.columns:
    top_causas = df['causa_acidente'].value_counts().nlargest(10).index
    df['causa_acidente'] = np.where(df['causa_acidente'].isin(top_causas), df['causa_acidente'], 'outros')       

# Traduzindo o padrão da PRF para Urbano e Rural (FORA DO LAÇO FOR)
if 'uso_solo' in df.columns:
    df['uso_solo'] = df['uso_solo'].replace({'sim': 'urbano', 'não': 'rural', 'nao': 'rural'})

# 2.3 Substituir valores inválidos por NaN
valores_invalidos = [
    '', ' ', 'nan', 'null', 'none', '(null)',
    'ignorado', 'não informado', 'nao informado',
    'sem informação', 'sem informacao'
]

df = df.replace(valores_invalidos, np.nan)


# 2.4 Converter idade para número e remover idades impossíveis
if 'idade' in df.columns:
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
    df.loc[(df['idade'] < 0) | (df['idade'] > 120), 'idade'] = np.nan


# 2.5 Converter colunas de vítimas para número
colunas_vitimas_existentes = [
    col for col in colunas_vitimas
    if col in df.columns
]

for col in colunas_vitimas_existentes:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# 2.6 Selecionar apenas as colunas efetivamente existentes
colunas_existentes = [
    col for col in colunas_analise
    if col in df.columns
]

todas_colunas_limpeza = colunas_existentes + colunas_vitimas_existentes

print(f"Colunas ativas utilizadas na análise: {colunas_existentes}\n")


# 2.7 Remover linhas com dados ausentes nas colunas analisadas
df_limpo = df.dropna(subset=todas_colunas_limpeza).copy()


# 2.8 Converter idade, criar Faixa Etária e converter vítimas para inteiro
#APENAS PARA VISUALIZAÇAO HUMANA
if 'idade' in df_limpo.columns:
    df_limpo['idade'] = df_limpo['idade'].astype(int)
    
    # Criar a coluna de Faixa Etária com rótulos amigáveis
    df_limpo['faixa_etaria'] = pd.cut(
        df_limpo['idade'],
        bins=[-1, 17, 29, 39, 49, 59, 120],
        labels=['0 a 17 anos', '18 a 29 anos', '30 a 39 anos', '40 a 49 anos', '50 a 59 anos', '60+ anos']
    )
    
    # Trocar 'idade' por 'faixa_etaria' na lista de colunas que vão gerar gráficos
    if 'idade' in colunas_existentes:
        colunas_existentes.remove('idade')
        colunas_existentes.insert(0, 'faixa_etaria')

for col in colunas_vitimas_existentes:
    df_limpo[col] = df_limpo[col].astype(int)


# 2.9 Consolidar a Variável-Alvo
df_limpo = df_limpo[
    (df_limpo['ilesos'] > 0) |
    (df_limpo['feridos_leves'] > 0) |
    (df_limpo['feridos_graves'] > 0) |
    (df_limpo['mortos'] > 0)
].copy()

# Variável original de 4 classes (para os gráficos descritivos)
condicoes_4 = [
    df_limpo['mortos'] > 0,
    df_limpo['feridos_graves'] > 0,
    df_limpo['feridos_leves'] > 0,
    df_limpo['ilesos'] > 0
]
classes_alvo_4 = ['Mortos', 'Feridos Graves', 'Feridos Leves', 'Ilesos']
df_limpo['condicao_final'] = np.select(condicoes_4, classes_alvo_4, default='Ilesos')

# Classificação Binária (Leve/Ileso vs Grave/Fatal)
condicoes_bin = [
    (df_limpo['mortos'] > 0) | (df_limpo['feridos_graves'] > 0)
]
classes_alvo_bin = ['Grave/Fatal']
df_limpo['gravidade_binaria'] = np.select(condicoes_bin, classes_alvo_bin, default='Leve/Ileso')


# 2.10 Exibir resumo da limpeza
linhas_finais = len(df_limpo)
print(f"Total de registros limpos para análise: {linhas_finais}")
print(f"Registros removidos (nulos/inválidos): {linhas_iniciais - linhas_finais}\n")

# 6. PREPARAÇÃO PARA MACHINE LEARNING

# Criamos uma lista específica para o ML juntando as colunas atuais com a idade exata
colunas_para_ml = colunas_existentes + ['idade']

# Descobrir quantas categorias únicas cada coluna tem
print("\nVerificando a quantidade de categorias únicas por coluna:")
print(df_limpo[colunas_para_ml].nunique())

# One-Hot Encoding converte os textos e faixas, mantendo a idade numérica intacta
X = pd.get_dummies(df_limpo[colunas_para_ml], drop_first=True, dtype=int)
y = df_limpo['gravidade_binaria']
ordem_classes = ['Leve/Ileso', 'Grave/Fatal']   #APENAS PARA VISUALIZAÇÃO FUTURAS

# Divisão Treino e Teste
# X_train é para treinar a ia E y_train é o gabarito do treino
# X_test é a validaçao final e o y_test o gabarito dessa validaçao final
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"Total no Treino: {len(X_train):,}".replace(',', '.'))
print(f"Total no Teste:  {len(X_test):,}\n".replace(',', '.'))

# --- MAPEAMENTO DA VARIÁVEL ALVO ---
# Para calcular métricas avançadas (como AUROC, Sensibilidade e Especificidade) de forma segura,
# mapeamos a classe de interesse (Grave/Fatal) como 1 e (Leve/Ileso) como 0.
y_train_num = y_train.map({'Leve/Ileso': 0, 'Grave/Fatal': 1})
y_test_num = y_test.map({'Leve/Ileso': 0, 'Grave/Fatal': 1})

# 7. MODELAGEM COM DECISION TREE E GRID SEARCH

# 7.1 Definir a grade de hiperparâmetros (Grid Search)
# Isso testa diferentes níveis de poda para evitar overfitting.
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

# 7.2 Configurar o GridSearchCV com Validação Cruzada (Cross-Validation)
grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_dt,
    scoring='f1', # Otimizando para o F1-Score devido ao possível desbalanceamento
    cv=5,         # Validação cruzada com 5 folds
    n_jobs=-1
)

# 7.3 Treinar o modelo executando a busca exaustiva (apenas nos dados de treino)
print("Iniciando o treinamento e busca de hiperparâmetros (Grid Search)...")
grid_dt.fit(X_train, y_train_num)

# Extrair o melhor modelo encontrado
best_dt = grid_dt.best_estimator_
print(f"\nMelhores hiperparâmetros encontrados:\n{grid_dt.best_params_}")

# 7.4 Fazer previsões utilizando os dados de teste (separados e intocados)
y_pred_dt = best_dt.predict(X_test)
y_prob_dt = best_dt.predict_proba(X_test)[:, 1] # Extrai as probabilidades da classe 1 para o AUROC

# 7.5 Avaliação do Modelo (Métricas exigidas pela Diretriz)
print("\n--- RESULTADOS DA AVALIAÇÃO (DECISION TREE) ---")

# Cálculo da Matriz de Confusão para extrair TN, FP, FN, TP
cm = confusion_matrix(y_test_num, y_pred_dt)
tn, fp, fn, tp = cm.ravel()

acuracia = accuracy_score(y_test_num, y_pred_dt)
sensibilidade = tp / (tp + fn) # Recall da classe positiva
especificidade = tn / (tn + fp) # Recall da classe negativa
precisao = tp / (tp + fp)
f1 = f1_score(y_test_num, y_pred_dt)
auroc = roc_auc_score(y_test_num, y_prob_dt)

print(f"Acurácia: {acuracia:.4f}")
print(f"Sensibilidade (Recall): {sensibilidade:.4f}")
print(f"Especificidade: {especificidade:.4f}")
print(f"Precisão: {precisao:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")

# 7.6 Visualizar a Matriz de Confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Leve/Ileso', 'Grave/Fatal'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão - Decision Tree')
plt.show()

# 7.7 Visualizar a Árvore de Decisão
plt.figure(figsize=(20, 10))
plot_tree(
    best_dt,
    feature_names=X.columns,
    class_names=['Leve/Ileso', 'Grave/Fatal'],
    filled=True,
    rounded=True,
    max_depth=3, # Limite visual para a árvore não ficar ilegível na tela
    fontsize=10
)
plt.title("Visualização da Árvore de Decisão (Profundidade reduzida para exibição)")
plt.show()