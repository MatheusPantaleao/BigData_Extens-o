import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys 

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_predict,  GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
    accuracy_score,
    make_scorer,
    f1_score
)

# 1. CARREGAMENTO DOS DADOS
print("Carregando o arquivo bruto...")
df = pd.read_csv('acidentes2023_todas_causas_tipos.csv', sep=';', encoding='latin1')
df.columns = df.columns.str.strip()

linhas_iniciais = len(df)
print(f"Total de registros na planilha original: {linhas_iniciais}\n")

colunas_para_verificar = [
    'dia_semana', 'tipo_veiculo', 'tipo_pista', 'condicao_meteorologica',
    'fase_dia', 'causa_acidente', 'tipo_acidente', 'tracado_via',
    'uso_solo', 'tipo_envolvido', 'sexo'
]

# 2. VISUALIZAÇÃO DOS DADOS CRUS
print("="*50)
print(" ***** ANTES DO PROCESSAMENTO (DADOS CRUS) ***** ")
print("="*50)

for col in colunas_para_verificar:
    if col in df.columns:
        print(f"\n--- COLUNA: {col.upper()} ---")
        contagem = df[col].value_counts(dropna=False)
        print(contagem)
        print(f"-> TOTAL DE CATEGORIAS DIFERENTES: {len(contagem)}")


# 3. PRÉ-PROCESSAMENTO E LIMPEZA DOS DADOS
print("\nIniciando a faxina dos dados...\n")

df = df.rename(columns={
    'condicao_metereologic': 'condicao_meteorologica',
    'condicao_metereologica': 'condicao_meteorologica'
})

colunas_vitimas = ['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']
colunas_analise = ['idade'] + colunas_para_verificar

# 3.1 Padronizar textos (letras minúsculas e sem espaços)
for col in colunas_para_verificar:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# 3.2 Otimização de colunas com muitas variaveis, Top 10 e outros
if 'tracado_via' in df.columns:
    top_tracados = df['tracado_via'].value_counts().nlargest(10).index
    df['tracado_via'] = np.where(df['tracado_via'].isin(top_tracados), df['tracado_via'], 'outros')

if 'causa_acidente' in df.columns:
    top_causas = df['causa_acidente'].value_counts().nlargest(10).index
    df['causa_acidente'] = np.where(df['causa_acidente'].isin(top_causas), df['causa_acidente'], 'outros')       

# 3.3 Traduzindo o padrão da PRF para Urbano e Rural
if 'uso_solo' in df.columns:
    df['uso_solo'] = df['uso_solo'].replace({'sim': 'urbano', 'não': 'rural', 'nao': 'rural'})

# 3.4 Apagando lixo e nulos
valores_invalidos = [
    '', ' ', 'nan', 'null', 'none', '(null)', 'ignorado', 
    'não informado', 'nao informado', 'sem informação', 'sem informacao'
]
df = df.replace(valores_invalidos, np.nan)

if 'idade' in df.columns:
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
    df.loc[(df['idade'] < 0) | (df['idade'] > 120), 'idade'] = np.nan

colunas_vitimas_existentes = [col for col in colunas_vitimas if col in df.columns]
for col in colunas_vitimas_existentes:
    df[col] = pd.to_numeric(df[col], errors='coerce')

colunas_existentes = [col for col in colunas_analise if col in df.columns]
todas_colunas_limpeza = colunas_existentes + colunas_vitimas_existentes

df_limpo = df.dropna(subset=todas_colunas_limpeza).copy()

if 'idade' in df_limpo.columns:
    df_limpo['idade'] = df_limpo['idade'].astype(int)

for col in colunas_vitimas_existentes:
    df_limpo[col] = df_limpo[col].astype(int)

# 3.5 Criando a Variável Alvo
df_limpo = df_limpo[
    (df_limpo['ilesos'] > 0) | (df_limpo['feridos_leves'] > 0) | 
    (df_limpo['feridos_graves'] > 0) | (df_limpo['mortos'] > 0)
].copy()

condicoes_bin = [(df_limpo['mortos'] > 0) | (df_limpo['feridos_graves'] > 0)]
df_limpo['gravidade_binaria'] = np.select(condicoes_bin, ['Grave/Fatal'], default='Leve/Ileso')


# 4. O "DEPOIS" - VISUALIZAÇÃO DOS DADOS LIMPOS
print("="*50)
print(" ***** DEPOIS DO PROCESSAMENTO (DADOS LIMPOS) ***** ")
print("="*50)

for col in colunas_para_verificar:
    if col in df_limpo.columns:
        print(f"\n--- COLUNA: {col.upper()} ---")
        contagem = df_limpo[col].value_counts(dropna=False)
        print(contagem)
        print(f"-> TOTAL DE CATEGORIAS DIFERENTES: {len(contagem)}")

linhas_finais = len(df_limpo)
print(f"\nTotal de registros limpos para análise: {linhas_finais}")
print(f"Registros removidos (nulos/inválidos): {linhas_iniciais - linhas_finais}\n")

# 4.1 GRÁFICOS
sns.set_theme(style="whitegrid")
ordem_dias = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'domingo']

print("\nGerando Gráficos Exploratórios... (Feche as janelas para continuar)")

for coluna in colunas_para_verificar:
    if coluna in df_limpo.columns:
        if coluna == 'dia_semana':
            top5 = df_limpo[coluna].value_counts().reindex(ordem_dias).dropna()
        else:
            top5 = df_limpo[coluna].value_counts().head(5)

        rotulos = [str(x).title() for x in top5.index]

        plt.figure(figsize=(10, 5))
        ax = sns.barplot(x=rotulos, y=top5.values, hue=rotulos, palette="mako", legend=False)

        nome_titulo = coluna.replace('_', ' ').title()
        titulo = 'Acidentes por Dia da Semana' if coluna == 'dia_semana' else f'Top 5 Mais Frequentes: {nome_titulo}'

        plt.title(titulo, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(nome_titulo, fontsize=11, fontweight='bold')
        plt.ylabel('Quantidade de Ocorrências', fontsize=11, fontweight='bold')

        for p in ax.patches:
            altura = int(p.get_height())
            ax.annotate(f'{altura:,}'.replace(',', '.'), 
                        (p.get_x() + p.get_width() / 2., altura),
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color='black', xytext=(0, 5), textcoords='offset points')
        ax.margins(y=0.15)
        plt.tight_layout()
        plt.show(block=True)

ordem_classes = ['Leve/Ileso', 'Grave/Fatal']
contagem_vitimas = df_limpo['gravidade_binaria'].value_counts().reindex(ordem_classes)

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=contagem_vitimas.index, y=contagem_vitimas.values, hue=contagem_vitimas.index, palette="rocket", legend=False)
plt.title('Distribuição da Variável-Alvo', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Gravidade do Acidente', fontsize=11, fontweight='bold')
plt.ylabel('Total de Vítimas', fontsize=11, fontweight='bold')

for p in ax.patches:
    altura = int(p.get_height())
    ax.annotate(f'{altura:,}'.replace(',', '.'), 
                (p.get_x() + p.get_width() / 2., altura),
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='black', xytext=(0, 5), textcoords='offset points')
ax.margins(y=0.15)
plt.tight_layout()
plt.show(block=True)


# 5. TRAVA DE SEGURANÇA e PREPARAÇÃO PARA MACHINE LEARNING
print("="*50)
resposta = input("Deseja prosseguir para o treinamento da Árvore de Decisão? (s/n): ")

if resposta.lower() != 's':
    print("Processamento encerrado pelo usuário. Ajuste o que for necessário!")
    sys.exit() 

colunas_para_ml = colunas_existentes

print("\nTraduzindo textos para números (One-Hot Encoding)...")
X = pd.get_dummies(df_limpo[colunas_para_ml], drop_first=True, dtype=int)
y = df_limpo['gravidade_binaria']
ordem_classes = ['Leve/Ileso', 'Grave/Fatal']


# 
# 6. BUSCA DE HIPERPARÂMETROS e K-FOLD
# 
print("\nIniciando a Busca pelos Melhores Hiperparâmetros (Grid Search)...")
print("A IA vai testar 90 combinações diferentes. Pode demorar alguns minutos!")

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

meu_f1 = make_scorer(f1_score, pos_label='Grave/Fatal')

busca = GridSearchCV(
    estimator=DecisionTreeClassifier(class_weight='balanced', random_state=42),
    param_grid=param_grid,
    scoring=meu_f1, 
    cv=3,         
    n_jobs=-1     
)

busca.fit(X, y)
arvore_otimizada = busca.best_estimator_

print(f"\nmelhor combinação:")
print(busca.best_params_)

print("\nAplicando K-Fold = 10 na Árvore")
cv_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

y_pred_kfold = cross_val_predict(arvore_otimizada, X, y, cv=cv_10, n_jobs=-1)

acc = accuracy_score(y, y_pred_kfold)
mcc = matthews_corrcoef(y, y_pred_kfold)

print(f"\nRESULTADO FINAL DA ARVORE DE DECISÃO")
print(f"Acurácia: {acc * 100:.2f}%")
print(f"MCC: {mcc:.4f}")


# 7. RELATÓRIO FINAL E GRÁFICOS
print(f"\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y, y_pred_kfold, labels=ordem_classes))

cm = confusion_matrix(y, y_pred_kfold, labels=ordem_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordem_classes)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title(f'Matriz de Confusão k-Fold - 10 da arvore de decisão')
plt.grid(False)
plt.tight_layout()
plt.show()

print("Extraindo a inteligência da arvore para o desenho...")
arvore_otimizada.fit(X, y)

plt.figure(figsize=(20, 10))
plot_tree(
    arvore_otimizada,
    feature_names=X.columns,
    class_names=ordem_classes,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=10
)
plt.title("arvore de decisão otimizada visão Resumida")
plt.show()