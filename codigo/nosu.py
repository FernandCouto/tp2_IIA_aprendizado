import pandas as pd
import numpy as np


def DadosKMeans(arquivo_treino, arquivo_teste): #lê e une os arquivos (test e treino)
   
    treino = pd.read_csv(arquivo_treino) #carrega os dados com o Pandas, ele é inteligente
    teste = pd.read_csv(arquivo_teste)
    
    df_completo = pd.concat([treino, teste], ignore_index=True) #concatena os dois dataframes
    
    gabarito = df_completo['TARGET_5Yrs'].to_numpy() #salva o gabarit
    
    jogadores = df_completo.drop(columns=['TARGET_5Yrs']).to_numpy() #apaga gabarito
    
    return jogadores, gabarito

def InicializarCentros(X, k):

    np.random.seed(42) #segredo da vida tempo universo, tudo, para garantir que os resultados sejam reproduzíveis
    indices = np.random.choice(X.shape[0], k, replace=False) #escolhe k pontos diferentes de partida para serem os clusters
    return X[indices]

def Ajustes(X, k, max_iter=300): #vai repetindo e ajustando a posição do cluster

    centros = InicializarCentros(X, k)
    indice_clusters = np.zeros(X.shape[0]) #vetor que armazena onde cada jogador esta | X.shape[0] = numero de jogadores
    
    print(f"Iniciando k-Means com k={k}...")
    
    for iteracao in range(max_iter):
        
        distancias = np.linalg.norm(X[:, np.newaxis] - centros, axis=2) #distancia de cada ponto para os centro atuais | np.linalg.norm = calcula a norma (distancia euclideana) | X[:, np.newaxis] = cruzar os dados pelos centros
        clusters_novos = np.argmin(distancias, axis=1) #pega a distacia para o centro mais proximo
        
        #recalcula as coordenadas dos centros pela a média dos pontos em cada cluster
        novos_centros = np.zeros((k, X.shape[1])) # X.shape[1] = numero de atributos
        for i in range(k): #pega cada cluster
            pontos_do_grupo = X[clusters_novos == i] #pega só os pontos que estão naquele cluster
            
            if len(pontos_do_grupo) > 0: #confere se tem alguem no cluster
                novos_centros[i] = np.mean(pontos_do_grupo, axis=0) #calcula a média dos pontos daquele cluster e atualiza o centro
            else:
                novos_centros[i] = centros[i] #problema da Má Inicialização 
                
        if np.array_equal(indice_clusters, clusters_novos): #convergencia - clusters nao mudaram
            print(f"-> Convergiu na iteração {iteracao + 1}!")
            break
            
        centros = novos_centros #atualiza os centros
        indice_clusters = clusters_novos #atualiza o cluster de cada jogador
        
    return centros, indice_clusters


def AvaliarClusters(grupos_kmeans, gabarito_real):
    
    #v = 1 (longa carreira) e f = 0 (curta carreira)
    total = len(gabarito_real)
    
    acertos_diretos = np.sum(grupos_kmeans == gabarito_real) #teste para saber qual cluster faz mais sentido sem o V ou F
    acertos_invertidos = np.sum((1 - grupos_kmeans) == gabarito_real) 
    
    if acertos_diretos >= acertos_invertidos: #se o cluster 1 tiver mais acertos, mantemos ele como 1, caso contrário, invertemos os clusters
        grupos_finais = grupos_kmeans
    else:
        grupos_finais = 1 - grupos_kmeans
        
    reais_longos = np.sum(gabarito_real == 1) #quantos jogadores tiveram v
    reais_curtos = np.sum(gabarito_real == 0) #quantos jogadores tiveram f
    
    algo_longos = np.sum(grupos_finais == 1) #extimativa do algoritmo para v
    algo_curtos = np.sum(grupos_finais == 0) #extimativa do algoritmo para f

    #matriz de confusão

    vp = np.sum((grupos_finais == 1) & (gabarito_real == 1)) #realidade = 1 | algoritmo = 1
    
    vn = np.sum((grupos_finais == 0) & (gabarito_real == 0)) #realidade = 0 | algoritmo = 0
    
    fp = np.sum((grupos_finais == 1) & (gabarito_real == 0)) #realidade = 0 | algoritmo = 1

    fn = np.sum((grupos_finais == 0) & (gabarito_real == 1)) #realidade = 1 | algoritmo = 0
    
    acuracia = (vp + vn) / total #quantos acertos o algoritmo teve no geral
    revocacao = vp / (vp + fn) if (vp + fn) > 0 else 0.0 #quantos dos jogadores que tiveram v o algoritmo conseguiu acertar
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0.0 #quantos dos jogadores que o algoritmo disse que tiveram v realmente tiveram v
    
    print("\n-> Na Realidade:")
    print(f"   Tiveram >5 anos de carreira: {reais_longos}")
    print(f"   Tiveram <5 anos de carreira: {reais_curtos}\n")
    
    print("-> No algoritmo:")
    print(f"   Tiveram >5 anos de carreira: {algo_longos}")
    print(f"   Tiveram <5 anos de carreira: {algo_curtos}\n")
    
    print("-> matriz de confusão")
    print("Verdadeiros positivos | Falsos negativos")
    print("Falsos positivos      | Verdadeiros negativos\n")
    print(f"{vp:<21} | {fn}")
    print(f"{fp:<21} | {vn}\n")
    
    print(f"Acuracia = {acuracia:.4f}")
    print(f"Revocação = {revocacao:.4f}")
    print(f"Precisão = {precisao:.4f}")
    
    return

#teste
 
if __name__ == "__main__":
    # caminho dos dados
    arq_treino = 'dados/nba_treino.csv'
    arq_teste = 'dados/nba_teste.csv'
    
    try:
        
        Matriz_X, Vetor_Y = DadosKMeans(arq_treino, arq_teste) #prepara os dados
        
        #k=2
        print("\n--- Experimento k=2 ---")
        centros_k2, grupos_k2 = Ajustes(Matriz_X, k=2)
        AvaliarClusters(grupos_k2, Vetor_Y)
        
        k=3 
        print("\n--- Experimento k=3 ---")
        centros_k2, grupos_k2 = Ajustes(Matriz_X, k=3)
        
        
    except FileNotFoundError:
        print("\nERRO: Os arquivos 'nba_treino.csv' e 'nba_teste.csv' não foram encontrados.")
        print("Certifique-se de que eles estão na mesma pasta que este script!")
