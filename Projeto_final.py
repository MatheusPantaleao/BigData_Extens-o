import inspect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
    accuracy_score,
    make_scorer,
    f1_score
)


# True  = pula gráficos exploratórios e usa 10 mil linhas e 3 folds
# False = roda a base completa com todos os gráficos e 10 folds
MODO_TESTE = True

# 1 carregamento e pre processamento
print("Carregando o arquivo bruto")
df = pd.read_csv('acidentes2023_todas_causas_tipos.csv', sep=';', encoding='latin1')
df.columns = df.columns.str.strip()

linhas_iniciais = len(df)
print(f"Total de registros na planilha original: {linhas_iniciais}\n")

colunas_para_verificar = [
    'dia_semana', 'tipo_veiculo', 'tipo_pista', 'condicao_meteorologica',
    'fase_dia', 'causa_acidente', 'tipo_acidente', 'tracado_via',
    'uso_solo', 'tipo_envolvido', 'sexo'
]

# dados crus
print(" ***** ANTES DO PROCESSAMENTO DADOS CRUS ***** ")

for col in colunas_para_verificar:
    if col in df.columns:
        print(f"\n--- COLUNA: {col.upper()} ---")
        contagem = df[col].value_counts(dropna=False)
        print(contagem)
        print(f"-> TOTAL DE CATEGORIAS DIFERENTES: {len(contagem)}")

df = df.rename(columns={
    'condicao_metereologic': 'condicao_meteorologica',
    'condicao_metereologica': 'condicao_meteorologica'
})

colunas_vitimas = ['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']
colunas_analise = ['idade'] + colunas_para_verificar

# padronizar textos
for col in colunas_para_verificar:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# Otimização de colunas com muitas variaveis, Top 10 e outros
if 'tracado_via' in df.columns:
    top_tracados = df['tracado_via'].value_counts().nlargest(10).index
    df['tracado_via'] = np.where(df['tracado_via'].isin(top_tracados), df['tracado_via'], 'outros')

if 'causa_acidente' in df.columns:
    top_causas = df['causa_acidente'].value_counts().nlargest(10).index
    df['causa_acidente'] = np.where(df['causa_acidente'].isin(top_causas), df['causa_acidente'], 'outros')

# padronizando urbano e rural
if 'uso_solo' in df.columns:
    df['uso_solo'] = df['uso_solo'].replace({'sim': 'urbano', 'não': 'rural', 'nao': 'rural'})

# apagando lixos e nulos
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

# variavel alvo
df_limpo = df_limpo[
    (df_limpo['ilesos'] > 0) | (df_limpo['feridos_leves'] > 0) |
    (df_limpo['feridos_graves'] > 0) | (df_limpo['mortos'] > 0)
].copy()

condicoes_bin = [(df_limpo['mortos'] > 0) | (df_limpo['feridos_graves'] > 0)]
df_limpo['gravidade_binaria'] = np.select(condicoes_bin, ['Grave/Fatal'], default='Leve/Ileso')

# dados limpos
print("\n ***** DADOS LIMPOS ***** \n")

for col in colunas_para_verificar:
    if col in df_limpo.columns:
        print(f"\n--- COLUNA: {col.upper()} ---")
        contagem = df_limpo[col].value_counts(dropna=False)
        print(contagem)
        print(f"-> TOTAL DE CATEGORIAS DIFERENTES: {len(contagem)}")

linhas_finais = len(df_limpo)
print(f"\nTotal de registros limpos para análise: {linhas_finais}")
print(f"Registros removidos nulos/inválidos: {linhas_iniciais - linhas_finais}\n")

# 2 graficos
ordem_classes = ['Leve/Ileso', 'Grave/Fatal']

if not MODO_TESTE:
    sns.set_theme(style="whitegrid")
    ordem_dias = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'domingo']

    print("\nGerando Gráficos")

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
else:
    print("\nMODO TESTE ATIVADO graficos exploratorios pulados.")


# 3 preparação machine learning
colunas_para_ml = colunas_existentes

if MODO_TESTE:
    print("[MODO TESTE ATIVADO] Usando amostra de 10.000 registros e K-Fold = 3.")
    df_ml = df_limpo.sample(n=10000, random_state=42)
    n_folds_final = 3
else:
    df_ml = df_limpo
    n_folds_final = 10

print("\nGerando variáveis One-Hot Encoding...")
X = pd.get_dummies(df_ml[colunas_para_ml], drop_first=True, dtype=int)
y = df_ml['gravidade_binaria']

meu_f1 = make_scorer(f1_score, pos_label='Grave/Fatal')
cv_final = StratifiedKFold(n_splits=n_folds_final, shuffle=True, random_state=42)

# 4 menu de escolhas do modelo
while True:
    print("\n" + "=" * 50)
    print("ESCOLHA O MODELO PARA TREINAMENTO:")
    print("1 - Decision Tree")
    print("2 - Random Forest")
    print("3 - AdaBoost")
    print("0 - Sair")
    print("=" * 50)

    opcao = input("Digite a opção desejada: ").strip()

    match opcao:
        case '1':
            nome_modelo = "Árvore de Decisão"
            estimador = DecisionTreeClassifier(class_weight='balanced', random_state=42)
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 3, 5]
            }

        case '2':
            nome_modelo = "Random Forest"
            estimador = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [3, 5]
            }

        case '3':
            nome_modelo = "AdaBoost"
            estimador_base = DecisionTreeClassifier(class_weight='balanced', random_state=42)
            parametro_estimador = (
                'estimator'
                if 'estimator' in inspect.signature(AdaBoostClassifier).parameters
                else 'base_estimator'
            )
            estimador = AdaBoostClassifier(**{parametro_estimador: estimador_base, 'random_state': 42})
            param_grid = {
                f'{parametro_estimador}__max_depth': [2, 3],
                'n_estimators': [100, 150],
                'learning_rate': [0.5, 1.0]
            }

        case '0':
            print("Código encerrado.")
            sys.exit()

        case _:
            print("Opção inválida! Digite 1, 2, 3 ou 0.")
            continue

    # ==========================================================
    # 5. TREINAMENTO, VALIDAÇÃO E RESULTADOS DO MODELO ESCOLHIDO
    # ==========================================================
    print(f"\nBusca pelos melhores Hiperparâmetros ({nome_modelo})...")
    busca = GridSearchCV(
        estimator=estimador,
        param_grid=param_grid,
        scoring=meu_f1,
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    busca.fit(X, y)
    modelo_otimizado = busca.best_estimator_

    print("\nMelhor combinação:")
    print(busca.best_params_)
    print(f"Melhor F1 (média dos 3 folds): {busca.best_score_:.4f}")

    print(f"\nAplicando K-Fold = {n_folds_final} no modelo {nome_modelo}...")
    y_pred_kfold = cross_val_predict(modelo_otimizado, X, y, cv=cv_final, n_jobs=-1)

    acc = accuracy_score(y, y_pred_kfold)
    mcc = matthews_corrcoef(y, y_pred_kfold)

    print(f"\nRESULTADO FINAL - {nome_modelo.upper()}")
    print(f"Acurácia: {acc * 100:.2f}%")
    print(f"MCC: {mcc:.4f}")

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y, y_pred_kfold, labels=ordem_classes))

    cm = confusion_matrix(y, y_pred_kfold, labels=ordem_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordem_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    plt.title(f'Matriz de Confusão k-Fold - {n_folds_final} ({nome_modelo})')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Treina na base para gerar o gráfico final específico do modelo
    modelo_otimizado.fit(X, y)

    if opcao == '1':
        plt.figure(figsize=(20, 10))
        plot_tree(
            modelo_otimizado,
            feature_names=X.columns,
            class_names=ordem_classes,
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=10
        )
        plt.title("Árvore de decisão otimizada resumida")
        plt.show()
    else:
        importancias = pd.Series(modelo_otimizado.feature_importances_, index=X.columns)
        top15 = importancias.sort_values(ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top15.values, y=top15.index, color='steelblue')
        plt.title(f'Top 15 Variáveis Mais Importantes - {nome_modelo}')
        plt.xlabel('Importância')
        plt.ylabel('Variável')
        plt.tight_layout()
        plt.show()