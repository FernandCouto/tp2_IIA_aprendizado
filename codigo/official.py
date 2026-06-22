import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


def carregaDadosKNN(arq_treino, arq_teste):
    treino = pd.read_csv(arq_treino)
    teste = pd.read_csv(arq_teste)
    
    y_treino = treino['TARGET_5Yrs'].to_numpy()
    X_treino = treino.drop(columns=['TARGET_5Yrs']).to_numpy()
    
    y_teste = teste['TARGET_5Yrs'].to_numpy()
    X_teste = teste.drop(columns=['TARGET_5Yrs']).to_numpy()
    
    return X_treino, y_treino, X_teste, y_teste

def carregaDadosKMeans(arq_treino, arq_teste):
    treino = pd.read_csv(arq_treino)
    teste = pd.read_csv(arq_teste)
    
    df_completo = pd.concat([treino, teste], ignore_index=True)
    
    y = df_completo['TARGET_5Yrs'].to_numpy()
    X = df_completo.drop(columns=['TARGET_5Yrs']).to_numpy()
    
    return X, y


def rodaKNNSkLearn(X_treino, y_treino, X_teste, y_teste, k):
    print(f"\n--- k-NN (Scikit-Learn) | k={k} ---")
    
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)
    
    cm = confusion_matrix(y_teste, previsoes)
    acc = accuracy_score(y_teste, previsoes)
    rec = recall_score(y_teste, previsoes, zero_division=0)
    prec = precision_score(y_teste, previsoes, zero_division=0)
    f1 = f1_score(y_teste, previsoes, zero_division=0)
    
    (vn, fp, fn, vp) = cm.ravel()
    
    print("-> Matriz de Confusão")
    print("Verdadeiros positivos | Falsos negativos")
    print("Falsos positivos      | Verdadeiros negativos\n")
    print(f"{vp:<21} | {fn}")
    print(f"{fp:<21} | {vn}\n")
    
    print(f"Acurácia  = {acc:.4f}")
    print(f"Revocação = {rec:.4f}")
    print(f"Precisão  = {prec:.4f}")
    print(f"F1        = {f1:.4f}")

def rodaKMeansSkLearn(X, y, k):
    print(f"\n--- k-Means (Scikit-Learn) | k={k} ---")
    
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos = modelo.fit_predict(X)
    centros = modelo.cluster_centers_

    print(f"-> Convergiu na iteração {modelo.n_iter_}!")
    
    print("\n-> Valores finais dos Centróides:")
    for (i, centro) in enumerate(centros):
        print(f"   Centróide {i}:\n   {np.round(centro, 4)}")
        
    print(f"\n-> Análise de Relação com os Rótulos Originais (k={k}):")
    for i in range(k):
        total_grupo = np.sum(grupos == i)
        if total_grupo > 0:
            longos = np.sum((grupos == i) & (y == 1))
            curtos = np.sum((grupos == i) & (y == 0))
            pct_longos = (longos / total_grupo) * 100
            pct_curtos = (curtos / total_grupo) * 100
            
            print(f"\n   • Cluster {i}: Total de {total_grupo} jogadores")
            print(f"     > Com carreira >= 5 anos (TARGET=1): {longos} ({pct_longos:.2f}%)")
            print(f"     > Com carreira < 5 anos  (TARGET=0): {curtos} ({pct_curtos:.2f}%)")


if __name__ == "__main__":
    arq_treino = 'dados/nba_treino.csv'
    arq_teste = 'dados/nba_teste.csv'
    
    try:
        # SUPERVISIONADO
        print("="*50)
        print(" APRENDIZADO SUPERVISIONADO (SCIKIT-LEARN)")
        print("="*50)
        (X_treino, y_treino, X_teste, y_teste) = carregaDadosKNN(arq_treino, arq_teste)
        
        for k in [2, 5, 10, 50]:
            rodaKNNSkLearn(X_treino, y_treino, X_teste, y_teste, k)
            
        # NÃO-SUPERVISIONADO
        print("\n" + "="*50)
        print(" APRENDIZADO NÃO-SUPERVISIONADO (SCIKIT-LEARN)")
        print("="*50)
        (X_kmeans, y_kmeans) = carregaDadosKMeans(arq_treino, arq_teste)
        
        for k in [2, 3]:
            rodaKMeansSkLearn(X_kmeans, y_kmeans, k)
            
    except FileNotFoundError:
        print("\nERRO: Arquivos CSV não encontrados.")