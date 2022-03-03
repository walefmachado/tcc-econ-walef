# Martiz G de Getis Ord

## Bibliotecas

#!pip install geopandas==0.8.1
#!pip install --upgrade pyshp
#!pip install shapely==1.7.0
#!pip install --upgrade descartes
#!pip install mapclassify==2.3.0 libpysal==4.3.0 splot==1.1.3
#!pip install jenkspy
#!pip install pyshp

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import scipy.stats as stats
import jenkspy

# para a análise de dados espaciais
import geopandas
import splot
import mapclassify as mc
from libpysal.weights import Queen
from libpysal import weights
from esda import Moran, Moran_Local, G_Local
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster, plot_local_autocorrelation

# para gráficos
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from matplotlib.lines import Line2D
from matplotlib.collections import EventCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection  
import matplotlib.image as mpimg
import seaborn as sns
# %matplotlib inline

# para agrupamento - K-médias
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cut_tree
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

import shapefile
from google.colab import drive, files

drive.mount("/content/drive")

# Dados

link = 'https://raw.githubusercontent.com/walefmachado/spreg_rural_insurance/main/dados/'

### Dados de seguro rural

dados_br = pd.read_csv(link+'/dados_06_19.csv')

### Dados para espacial 

cod = pd.read_csv(link+'/codigos-mun.csv')

br = geopandas.read_file(link+'/br.json')

br = br.rename(columns={'CD_GEOCMU': 'mun'})
br.mun = br.mun.astype(int)
br2 = br.drop('NM_MUNICIP', axis=1)

cod_dados = cod.merge(br2, how='left')
cod_dados = geopandas.GeoDataFrame(cod_dados) # Ate aqui junta geometry com todos os códigos

dados_br = cod_dados.merge(dados_br, on='mun', how='left')

dados_br = dados_br.fillna(0)
#dados_br = dados_br.drop([1525, 3499]) # retira F. Noronha e Ilhabela
dados_br = dados_br.drop(['rm'], axis = 1)

dados_br.rename(columns = {'nome_mun_x': 'nome_mun', 'nome_meso_x':'nome_meso'}, inplace = True)

retirar = ['nome_mun_y', 'nome_meso_y']
dados_br = dados_br.drop(retirar, axis = 1)

img=mpimg.imread('/content/drive/My Drive/Mestrado/Imagens/rosa_dos_ventos_3.png')
img2=mpimg.imread('/content/drive/My Drive/Mestrado/Imagens/rosa_dos_ventos_p.png')

#Regioes geograficas
sf = shapefile.Reader('/content/drive/My Drive/Mestrado/Dados/estados/estados_2010.shp')
shapes = sf.shapes()
Nshp = len(shapes)

ptchs   = []
for nshp in range(Nshp):
    pts     = np.array(shapes[nshp].points)
    prt     = shapes[nshp].parts
    par     = list(prt) + [pts.shape[0]]

    for pij in range(len(prt)):
       ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))

#UF
sf_uf = shapefile.Reader('/content/drive/My Drive/Mestrado/Dados/estados/estados_2010.shp')
shapes = sf.shapes()
shapes_uf = sf_uf.shapes()
Nshp_uf = len(shapes_uf)

ptchs_uf   = []
for nshp_uf in range(Nshp_uf):
    pts_uf     = np.array(shapes_uf[nshp_uf].points)
    prt_uf     = shapes_uf[nshp_uf].parts
    par_uf     = list(prt_uf) + [pts_uf.shape[0]]

    for pij_uf in range(len(prt_uf)):
       ptchs_uf.append(Polygon(pts_uf[par_uf[pij_uf]:par_uf[pij_uf+1]]))


fig     = plt.figure(figsize = (9,9))
ax      = fig.add_subplot(111)

ax.add_collection(PatchCollection(ptchs,facecolor='0.75', edgecolor='w', linewidths=0))
ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))
ax.axis('auto'); ax.axis('off')
plt.show()
# Fonte: http://www.phme.it/wilt/2017/05/06/dynamic-mapping-with-shapefiles-and-python/


f, ax = plt.subplots(figsize=(16,16)) # 
dados_br[dados_br['ano']==2019].plot(column='apolices_contratadas', ax=ax, cmap='viridis', scheme='quantiles', legend=True)
ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.4))
ax.set_axis_off()
plt.figimage(img2, 720, 25, zorder=1)
plt.show();

variaveis = ['apolices_contratadas', 'total_segurado_mil', 'soma_premio_total_mil',
             'total_subvencao_mil', 'valor_indenizacoes_pagas_mil',
             'sinistralidade_media', 'taxa_media', 'apolices_indenizadas'] # 
anos = dados_br.ano.unique()

dados_br = dados_br.rename({'apolices_contratadas':'TAC',
                            'total_segurado_mil':'SIS',
                            'soma_premio_total_mil':'SPR',
                            'total_subvencao_mil':'TSB',
                            'valor_indenizacoes_pagas_mil':'SIP',
                            'taxa_media':'TMA',
                            'apolices_indenizadas':'NAI'}, axis=1)
variaveis = ['TAC', 'SIS', 'SPR', 'TSB', 'SIP', 'TMA', 'NAI']

dados_19 = dados_br[dados_br['ano']==2019]
dados_19.drop('ano', axis=1, inplace=True)

f, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 20)) 
db = geopandas.GeoDataFrame(dados_19)
axs = axs.flatten()
for i, variavel in enumerate(variaveis):
    ax = axs[i]
    #y = dados_19[variavel].values 
    #g_loc_br = G_Local(y, w)
    #g_map(g_loc_br, dados_19, ax=ax)
    dados_19.plot(column=variavel, ax=ax, cmap='viridis', scheme='quantiles', legend=True)
    ax.set_axis_off()
    ax.set_title(variavel, fontsize=25)
    f.tight_layout() 
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
axs[7].set_axis_off()
axs[8].set_axis_off()
axs[7].imshow(img)
plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.0005, hspace=0.001)
plt.show();

## Matriz de G de Getis-Ord

### Criando a matriz

### G de Getis-Ord local para variáveis

dados_19.drop(index=dados_19[dados_19['mun'] == 2605459].index, inplace=True) # retira F. Noronha e Ilhabela
dados_19.drop(index=dados_19[dados_19['mun'] == 3520400].index, inplace=True)

getis_matrix = dados_19

retirar = ['uf', 'nome_uf', 'mun', 'meso', 'nome_meso', 
           'micro', 'nome_micro', 'codmun6', 'regiao', 
           'nome_regiao', 'cod_uf','sinistralidade_media'] # , 'geometry'
getis_matrix = getis_matrix.drop(retirar, axis = 1)

# demora um pouco
w = Queen.from_dataframe(dados_19)
w.transform = 'r'

for variavel in getis_matrix.drop(['nome_mun', 'geometry'], axis = 1).columns:
    getis_matrix[variavel] = G_Local(dados_19[variavel].values, w).Zs

getis_matrix = getis_matrix.set_index('nome_mun')

getis_matrix.describe().round(2)

### Mapa G de Getis-Ord

def g_map(g, geog, ax):

    ec = '0.8'
    
    # Break observations into significant or not
    sig = g.p_sim < 0.05

    # Plot non-significant clusters
    ns = db.loc[sig==False, 'geometry']
    ns.plot(ax=ax, color='lightgrey', edgecolor=ec, linewidth=0.1)
    # Plot HH clusters
    hh = db.loc[(g.Zs > 0) & (sig==True), 'geometry']
    hh.plot(ax=ax, color='#fde725', edgecolor=ec, linewidth=0.1)
    # Plot LL clusters
    ll = db.loc[(g.Zs < 0) & (sig==True), 'geometry']
    ll.plot(ax=ax, color='#3b528b', edgecolor=ec, linewidth=0.1)
    # Style and draw

    st = ''
    if g.star:
        st = '*'
    #ax.set_title(f'G{st} statistic for Pct of Leave votes', size=15)
    ax.set_axis_off()
    return ax

# cria a legenda G loca
labels_g = ['Hot spot', 'Cool spot', 'não significativo']
color_list_g = ["#fde725", "#3b528b", "lightgrey"]
hmap_g = colors.ListedColormap("", color_list_g)
lines_g= [Line2D([0], [0], color=c, marker='o', markersize=25, linestyle='') for c in color_list_g] # linewidth=20, linestyle='dotted'


f, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 20)) 
db = geopandas.GeoDataFrame(dados_19)
axs = axs.flatten()
for i, variavel in enumerate(variaveis):
    ax = axs[i]
    y = dados_19[variavel].values 
    g_loc_br = G_Local(y, w)
    g_map(g_loc_br, dados_19, ax=ax)
    #moran_matrix.plot(column=variavel, ax=ax, legend=True, scheme='quantiles', cmap = 'viridis'); # , cmap='OrRd'  scheme='quantiles',
    ax.set_axis_off()
    ax.set_title(variavel, fontsize=25)
    f.tight_layout() 
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual
axs[7].legend(lines_g, labels_g, loc='botton left', bbox_to_anchor=(1.7, 0.75), frameon=False,  prop={'size': 20})
axs[7].set_axis_off()
axs[8].set_axis_off()
axs[7].imshow(img)
plt.subplots_adjust(left=7, bottom=7, right=8, top=8, wspace=0.0005, hspace=0.001)
plt.show();

f, axs = plt.subplots(nrows=2, ncols=4, figsize=(30, 15)) 
axs = axs.flatten()
for i, variavel in enumerate(variaveis):
    ax = axs[i] 
    getis_matrix.plot(column=variavel, ax=ax, legend=True, scheme='quantiles', cmap = 'viridis'); # , cmap='OrRd'  scheme='quantiles',
    ax.set_axis_off()
    ax.set_title(variavel, fontsize=20)
    f.tight_layout() 
    ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.5))   # adiciona a divisão estadual

f, ax = plt.subplots(figsize=(16,16)) # 
getis_matrix.plot(column='apolices_contratadas', ax=ax,  figsize=(10, 10), legend=True, scheme='quantiles', cmap = 'viridis'); # , cmap='OrRd'  scheme='quantiles',
ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.3))
ax.set_axis_off()
plt.figimage(img2, 720, 25, zorder=1)
plt.show();

## Análise de agrupamento - G de Getis-Ord

# subselecionar variáveis
X = getis_matrix.drop('geometry', axis=1)
# transformar em matriz (necessário para gerar o gráfico)
XX = X.values
# mudar o tipo dos dados
XX = np.asarray(XX, dtype=float)
n = XX.shape[0]
p = XX.shape[1]
# vetor de médias
Xb = np.mean(XX, axis=0)
# matriz de covariâncias
S = np.cov(XX.T)
# matriz de somas de quadrados e produtos
W = (n - 1) * S

Z = linkage(X, method='ward')

max_d = 0
grupos = cut_tree(Z, height=max_d)

import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(3000)

fig, ax = plt.subplots(figsize=(10, 7))
ax = dendrogram(
    Z,
    truncate_mode='lastp',  # mostrar apenas os p últimos grupos formados
    p=5,  # quantos passos mostrar
    show_leaf_counts=True,  # mostrar quantas observações há em cada grupo entre parênteses
    leaf_rotation=90., # rotação
    leaf_font_size=10., # tamanho da fonte
    labels=dados_19.index, # rótulos do eixo x
    show_contracted=True,  # to get a distribution impression in truncated branches,
    above_threshold_color='black',
    color_threshold=0.1, # para que todas as linhas sejam da mesma cor
    # color_threshold=max_d, # para que os grupos fiquem com cores diferentes
)
plt.axhline(y=max_d, c='grey', lw=1, linestyle='dashed')
plt.xlabel('município', fontsize = 20)
plt.ylabel('distância', fontsize = 20);

## Método escolhido:

# definir a distância de corte baseando no dendrograma
max_d = 75
grupos = cut_tree(Z, height=max_d)


# incluir no dataframe de dados as informações sobre a qual grupo cada observação pertence
getis_matrix['grupo'] = grupos
# moran_matrix.head(2)

# contagem de observações em cada grupo
getis_matrix.grupo.value_counts()

# média dos grupos - todas as variáveis
# inclusive as não utilizadas para agrupar
getis_matrix.groupby('grupo').mean().round(2).to_latex()

# mediana das variáveis para cada grupo
getis_matrix.groupby('grupo').median()

### As observações de cada grupo

#grupo0 = dados.query('grupo == 0').index
#list(grupo0)

### Mapa Grupos G de Getis Ord

getis_matrix['grupo'] = getis_matrix['grupo'].astype('category')

f, ax = plt.subplots(figsize=(16,16)) # 
getis_matrix.plot(column='grupo', figsize=(10, 10), legend=True, ax=ax, cmap='viridis'); # , cmap='OrRd'ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.3))
ax.set_axis_off()
ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.3))
plt.figimage(img2, 720, 25, zorder=1)
plt.show();

## K-Médias - G de Getis-Ord

# SQDG
SQDG = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(getis_matrix.drop('geometry', axis=1))
    getis_matrix['grupos'] = kmeans.labels_
    SQDG[k] = kmeans.inertia_ # Inertia: soma das distâncias das obs. ao centro mais próximo
plt.figure(figsize=(10, 7))
plt.plot(list(SQDG.keys()), list(SQDG.values()), linewidth=4)
plt.xlabel('Número de grupos (k)', fontsize = 20)
plt.ylabel('SQDG', fontsize = 20);

k = 2

kmeans = KMeans(n_clusters=k, random_state=10).fit(X)


# com o método das k-médias
getis_matrix['grupo'] = kmeans.labels_

### Métricas

# métricas
print(
  round(metrics.calinski_harabasz_score(dados_metrics.values, kmeans.labels_),2),
  round(davies_bouldin_score(dados_metrics.values, kmeans.labels_),2),
  round(metrics.silhouette_score(dados_metrics.values, kmeans.labels_, metric='euclidean'),2)  
)

# contagens
getis_matrix.grupo.value_counts()

# incluir no dataframe de dados as informações sobre a qual grupo cada município pertence
# dados['grupo'] = kmeans.labels_

# média dos grupos - todas as variáveis
# inclusive as não utilizadas para agrupar
getis_matrix.groupby('grupo').mean()

pd.options.display.float_format = '{:20,.2f}'.format
resumo_estatistico = getis_matrix.groupby('grupo').mean().drop(['apolices_contratadas', 'total_segurado_mil',
       'soma_premio_total_mil', 'total_subvencao_mil','grupos'],axis=1)
print(resumo_estatistico.to_latex(index=True))

pd.options.display.float_format = '{:20,.2f}'.format
resumo_estatistico = getis_matrix.groupby('grupo').mean().drop(['grupos','valor_indenizacoes_pagas_mil', 'sinistralidade_media', 'taxa_media',
       'apolices_indenizadas'],axis=1)
print(resumo_estatistico.to_latex(index=True))

# mediana das variáveis para cada grupo
getis_matrix.groupby('grupo').median()

#grupo0 = mg.query('grupo == 0').index
#list(grupo0)

### Mapa Grupos com G de Getis Ord 

getis_matrix['grupo'] = getis_matrix['grupo'].astype('category')

f, ax = plt.subplots(figsize=(16,16)) # 
getis_matrix.plot(column='grupo', figsize=(10, 10), legend=True, ax=ax, cmap='viridis'); # , cmap='OrRd'ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.3))
ax.set_axis_off()
ax.add_collection(PatchCollection(ptchs_uf,color='none', edgecolor='w', linewidths=0.3))
plt.figimage(img2, 720, 25, zorder=1)
plt.show();