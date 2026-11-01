import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURAÇÃO DE TEMA DO DASHBOARD
# ==========================================
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#121212",       
    "figure.facecolor": "#121212",
    "grid.color": "#282828",           
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,         
    "axes.spines.bottom": False,       
    "text.color": "#FFFFFF",
    "axes.labelcolor": "#A0A0A5",
    "xtick.color": "#A0A0A5",
    "ytick.color": "#A0A0A5"
})

# Leitura e preparação dos dados
df = pd.read_csv('acidentes2023_todas_causas_tipos.csv', sep=';', encoding='latin1')
colunas = ['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']
df_vitimas = df[colunas]
totais = df_vitimas.sum()

cores = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C'] 
nomes_labels = ['Ilesos', 'Feridos Leves', 'Feridos Graves', 'Mortos']

# ==========================================
# 2. ESTRUTURA DO DASHBOARD (1 Linha, 2 Colunas)
# ==========================================
# Cria uma figura mais larga e menos alta (16x7)
fig, (ax_bar, ax_donut) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'wspace': 0.1})

# Título Geral do Dashboard adaptado
fig.suptitle('Panorama de Vítimas de Trânsito - PRF 2023', fontsize=24, fontweight='bold', color='#FFFFFF', y=1.02)
fig.text(0.5, 0.94, 'Análise de Volume e Proporção por Categoria', 
         ha='center', fontsize=14, color='#A0A0A5')

# ==========================================
# GRÁFICO 1: BARRAS (ax_bar)
# ==========================================
sns.barplot(x=nomes_labels, y=totais.values, palette=cores, ax=ax_bar, alpha=0.9)

ax_bar.set_title('Volume Total por Categoria', fontsize=14, fontweight='bold', pad=15)
ax_bar.set_ylabel('')
ax_bar.set_yticks([]) # Oculta os números do eixo Y

for p in ax_bar.patches:
    ax_bar.annotate(f'{int(p.get_height()):,}'.replace(',', '.'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 12), 
                    textcoords='offset points', fontsize=12, fontweight='bold', color='#FFFFFF')

# ==========================================
# GRÁFICO 2: ROSCA (ax_donut)
# ==========================================
wedges, texts, autotexts = ax_donut.pie(
    totais.values, labels=nomes_labels, autopct='%1.1f%%', startangle=140, 
    colors=cores, pctdistance=0.75,
    wedgeprops={'edgecolor': '#121212', 'linewidth': 4}
)

# Estilização das fontes dentro da rosca
for autotext in autotexts:
    autotext.set_color('#121212')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)
for text in texts:
    text.set_color('#A0A0A5')
    text.set_fontsize(11)

centro = plt.Circle((0, 0), 0.60, fc='#121212')
ax_donut.add_artist(centro)
ax_donut.set_title('Proporção de Severidade', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.show()