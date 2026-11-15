import pandas as pd

# 1. Carrega o banco de dados original (sem alterar nada)
print("Carregando o arquivo bruto...")
df_bruto = pd.read_csv('acidentes2023_todas_causas_tipos.csv', sep=';', encoding='latin1')

# 2. Define quais colunas você quer investigar
colunas_para_verificar = [
    'dia_semana', 'tipo_veiculo', 'tipo_pista', 'condicao_meteorologica',
    'fase_dia', 'causa_acidente', 'tipo_acidente', 'tracado_via',
    'uso_solo', 'tipo_envolvido', 'sexo'
]


# 3. Imprime o que está escrito dentro de cada coluna no estado BRUTO
for col in colunas_para_verificar:
    if col in df_bruto.columns:
        print(f"\n========================================")
        print(f"COLUNA: {col.upper()} (DADOS CRUS)")
        print(f"========================================")
        
        # value_counts(dropna=False) mostra os textos exatos e também os valores nulos
        contagem = df_bruto[col].value_counts(dropna=False)
        
        print(contagem)
        print(f"\n-> TOTAL DE CATEGORIAS DIFERENTES NESTA COLUNA: {len(contagem)}")
