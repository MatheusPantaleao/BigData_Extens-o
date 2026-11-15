import pandas as pd
import numpy as np

print("Arquivo carregado com sucesso!")

df = pd.read_csv(
    'acidentes2023_todas_causas_tipos.csv',
    sep=';',
    encoding='latin1'
)

print("Quantidade de linhas e colunas:", df.shape)

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score
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

# One-Hot Encoding converte os textos e faixas, mantendo a idade numérica intacta
X = pd.get_dummies(df_limpo[colunas_para_ml], drop_first=True, dtype=int)
y = df_limpo['gravidade_binaria']
ordem_classes = ['Leve/Ileso', 'Grave/Fatal'] #APENAS PARA VISUALIZAÇÃO FUTURAS

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
print(f"Total no Teste: {len(X_test):,}\n".replace(',', '.'))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Acurácia:", accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=['Leve/Ileso', 'Grave/Fatal']
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Leve/Ileso', 'Grave/Fatal']
)

disp.plot(cmap='Blues')
plt.title('Matriz de Confusão - Random Forest')
plt.show()

print(cm)

indice_grave = list(rf.classes_).index('Grave/Fatal')

probabilidades = rf.predict_proba(X_test)[:, indice_grave]

auc = roc_auc_score(
    (y_test == 'Grave/Fatal').astype(int),
    probabilidades
)

print(f"AUC: {auc:.4f}")