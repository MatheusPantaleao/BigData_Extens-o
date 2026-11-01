import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score
)
from lightgbm import LGBMClassifier


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
    'tipo_acidente'
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

# Classificação Binária (Leve/Ileso vs Grave/Fatal - Alta Acurácia)
condicoes_bin = [
    (df_limpo['mortos'] > 0) | (df_limpo['feridos_graves'] > 0)
]
classes_alvo_bin = ['Grave/Fatal']
df_limpo['gravidade_binaria'] = np.select(condicoes_bin, classes_alvo_bin, default='Leve/Ileso')


# 2.10 Exibir resumo da limpeza
linhas_finais = len(df_limpo)
print(f"Total de registros limpos para análise: {linhas_finais}")
print(f"Registros removidos (nulos/inválidos): {linhas_iniciais - linhas_finais}\n")


# 3. CONFIGURAÇÃO DOS GRÁFICOS INICIAIS
sns.set_theme(style="whitegrid")

ordem_dias = [
    'segunda-feira',
    'terça-feira',
    'quarta-feira',
    'quinta-feira',
    'sexta-feira',
    'sábado',
    'domingo'
]

ordem_vitimas = [
    'Ilesos',
    'Feridos Leves',
    'Feridos Graves',
    'Mortos'
]


# 4. GERAR GRÁFICOS POR CATEGORIA (ORIGINAIS)

for coluna in colunas_existentes:

    if coluna == 'dia_semana':
        contagem = df_limpo[coluna].value_counts()
        top5 = contagem.reindex(ordem_dias).dropna()
    else:
        top5 = df_limpo[coluna].value_counts().head(5)

    # Formatar rótulos dependendo do tipo da coluna
    if coluna == 'faixa_etaria':
        rotulos = [str(x) for x in top5.index]
    elif coluna in colunas_texto:
        rotulos = [str(x).title() for x in top5.index]
    else:
        rotulos = [f"{int(x)} anos" for x in top5.index]

    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        x=rotulos,
        y=top5.values,
        hue=rotulos,
        palette="mako",
        legend=False
    )

    nome_titulo = coluna.replace('_', ' ').title()

    if coluna == 'dia_semana':
        titulo = 'Acidentes por Dia da Semana'
    else:
        titulo = f'Top 5 Mais Frequentes: {nome_titulo}'

    plt.title(titulo, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(nome_titulo, fontsize=11, fontweight='bold')
    plt.ylabel('Quantidade de Ocorrências', fontsize=11, fontweight='bold')

    for p in ax.patches:
        altura = int(p.get_height())
        ax.annotate(
            f'{altura:,}'.replace(',', '.'),
            (p.get_x() + p.get_width() / 2., altura),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
        )

    ax.margins(y=0.15)
    plt.tight_layout()
    plt.show()


# 5. GRÁFICO DA VARIÁVEL-ALVO ORIGINAL
contagem_vitimas = df_limpo['condicao_final'].value_counts().reindex(ordem_vitimas)

plt.figure(figsize=(10, 5))

ax = sns.barplot(
    x=contagem_vitimas.index,
    y=contagem_vitimas.values,
    hue=contagem_vitimas.index,
    palette="mako",
    legend=False
)

plt.title(
    'Distribuição da Variável-Alvo: Condição Final das Pessoas',
    fontsize=14,
    fontweight='bold',
    pad=15
)

plt.xlabel('Condição Final (Classe Alvo)', fontsize=11, fontweight='bold')
plt.ylabel('Total de Pessoas', fontsize=11, fontweight='bold')

for p in ax.patches:
    altura = int(p.get_height())
    ax.annotate(
        f'{altura:,}'.replace(',', '.'),
        (p.get_x() + p.get_width() / 2., altura),
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='black',
        xytext=(0, 5),
        textcoords='offset points'
    )

ax.margins(y=0.15)
plt.tight_layout()
plt.show()

# 6. PREPARAÇÃO PARA MACHINE LEARNING

# Criamos uma lista específica para o ML juntando as colunas atuais com a idade exata
colunas_para_ml = colunas_existentes + ['idade']

# One-Hot Encoding converte os textos e faixas, mantendo a idade numérica intacta
X = pd.get_dummies(df_limpo[colunas_para_ml], drop_first=True, dtype=int)
y = df_limpo['gravidade_binaria']
ordem_classes = ['Leve/Ileso', 'Grave/Fatal']

# Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(f"Total no Treino: {len(X_train):,}".replace(',', '.'))
print(f"Total no Teste:  {len(X_test):,}\n".replace(',', '.'))

# 7. ÁRVORE DE DECISÃO DIDÁTICA (GRÁFICO PARA O PROFESSOR)
print("A treinar Árvore de Decisão Didática para visualização...")

arvore_didatica = DecisionTreeClassifier(
    max_depth=3,
    class_weight='balanced',
    random_state=42
)
arvore_didatica.fit(X_train, y_train)

plt.figure(figsize=(22, 9))
plot_tree(
    arvore_didatica,
    feature_names=list(X.columns),
    class_names=list(arvore_didatica.classes_),
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True,
    impurity=False
)

plt.title("Árvore de Decisão Didática (Top 3 Níveis - Principais Regras)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("arvore_decisao_resumida.png", dpi=300, bbox_inches="tight")
plt.show()

print("Imagem 'arvore_decisao_resumida.png' gerada e salva com sucesso!\n")

# 8. LIGHTGBM OTIMIZADO (ALTA ACCURACY ~73.47%)
print("Otimizando hiperparâmetros do LightGBM com RandomizedSearchCV...")

param_dist = {
    'n_estimators': [150, 300, 450],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [6, 8, 12, -1],
    'num_leaves': [31, 63, 127],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'min_child_samples': [20, 50, 100]
}

base_lgb = LGBMClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

search = RandomizedSearchCV(
    estimator=base_lgb,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
modelo_otimizado = search.best_estimator_

print("\n=== MELHORES HIPERPARÂMETROS ENCONTRADOS NO LIGHTGBM ===")
print(search.best_params_)
print()


# 9. AVALIAÇÃO DO MODELO E MATRIZ DE CONFUSÃO
y_pred = modelo_otimizado.predict(X_test)
y_prob = modelo_otimizado.predict_proba(X_test)

acuracia = accuracy_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_prob[:, 1])

print("\n==========================================")
print(f"Acurácia Geral no Teste (LightGBM): {acuracia * 100:.2f}%")
print(f"AUROC: {auroc:.4f}")
print("==========================================\n")

print("=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred, labels=ordem_classes))

# 9.1 Matriz de Confusão
cm = confusion_matrix(y_test, y_pred, labels=ordem_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordem_classes)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Matriz de Confusão - LightGBM Otimizado', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Previsão do Modelo', fontweight='bold')
plt.ylabel('Verdadeira Classe (Realidade)', fontweight='bold')
plt.grid(False)
plt.tight_layout()
plt.show()

# 9.2 Top 10 Atributos Mais Relevantes
importancias = pd.Series(
    modelo_otimizado.feature_importances_, 
    index=X.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
ax = sns.barplot(
    x=importancias.values,
    y=importancias.index,
    hue=importancias.index,
    palette="mako",
    legend=False
)
plt.title("Top 10 Atributos de Maior Impacto na Previsão", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Importância Relativa (Feature Importance)", fontsize=11, fontweight='bold')
plt.ylabel("Atributos", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# 10. EXPORTAÇÃO PARA A EQUIPE DA INTERFACE (STREAMLIT)
import joblib 

print("\n=== EXPORTANDO ARTEFATOS PARA O GITHUB ===")

df_limpo.to_csv("dados_processados.csv", index=False, sep=";")
print("- Base limpa salva como 'dados_processados.csv'")

joblib.dump(modelo_otimizado, "modelo_lightgbm_otimizado.pkl")
print("- Modelo de Machine Learning salvo como 'modelo_lightgbm_otimizado.pkl'")

joblib.dump(list(X.columns), "colunas_treino.pkl")
print("- Estrutura de colunas salva como 'colunas_treino.pkl'\n")