import pandas as pd
import numpy as np

def DadosKNN(arquivo_treino, arquivo_teste): #separa os gabaritos dos dados
   
    treino = pd.read_csv(arquivo_treino) #carrega dados com o Pandas, ele é inteligente
    teste = pd.read_csv(arquivo_teste)
    
    gabarito_treino = treino['TARGET_5Yrs'].to_numpy() #salva o gabarito do treino
    jogadores_treino = treino.drop(columns=['TARGET_5Yrs']).to_numpy() #apaga o gabarito do treino
    
    gabarito_teste = teste['TARGET_5Yrs'].to_numpy() #salva o gabarito do teste
    jogadores_teste = teste.drop(columns=['TARGET_5Yrs']).to_numpy() #apaga o gabarito do teste
    
    return jogadores_treino, gabarito_treino, jogadores_teste, gabarito_teste


def PreverKNN(jogadores_treino, gabarito_treino, jogadores_teste, k): #faz a votação com os k vizinhos mais próximos
    
    print(f"previsões com k-NN (k={k})")
    previsoes = []
    
    for ponto_teste in jogadores_teste: # passa por cada jogador novato
        
        distancias = np.linalg.norm(jogadores_treino - ponto_teste, axis=1) #mede a distancia desse novato para TODOS os veteranos
        
        k_vizinhos_indices = np.argsort(distancias)[:k] #ordena as distâncias do menor pro maior e pega as posições dos 'k' primeiros
        
        k_etiquetas = gabarito_treino[k_vizinhos_indices]  #olha o gabarito de treino dos 'k' vizinhos mais próximos
        
        valores, contagens = np.unique(k_etiquetas, return_counts=True) #conta os votos | return_counts=True quantos cada candidato recebeu
        if len(contagens) > 1 and contagens[0] == contagens[1]: #empate
            voto_vencedor = k_vizinhos_indices[0]
        else:
            voto_vencedor = valores[np.argmax(contagens)] #pega o candidato com mais votos
        
        previsoes.append(voto_vencedor)
        
    return np.array(previsoes)


def AvaliarKNN(previsoes, gabarito_real): # matriz de confusão direta, sem precisar testar alinhamento
    
    # v = 1 (longa carreira) e f = 0 (curta carreira)
    total = len(gabarito_real)
    
    # matriz de confusão
    vp = np.sum((previsoes == 1) & (gabarito_real == 1)) # realidade = 1 | algoritmo = 1
    
    vn = np.sum((previsoes == 0) & (gabarito_real == 0)) # realidade = 0 | algoritmo = 0
    
    fp = np.sum((previsoes == 1) & (gabarito_real == 0)) # realidade = 0 | algoritmo = 1

    fn = np.sum((previsoes == 0) & (gabarito_real == 1)) # realidade = 1 | algoritmo = 0
    
    acuracia = (vp + vn) / total # quantos acertos o algoritmo teve no geral
    revocacao = vp / (vp + fn) if (vp + fn) > 0 else 0.0 # quantos dos jogadores que tiveram v o algoritmo acertou
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0.0 # quantos dos que o algoritmo disse que tiveram v realmente tiveram v
    f1 = 2 *( (precisao*revocacao) / (precisao+revocacao))

    print("\n-> matriz de confusão")
    print("Verdadeiros positivos | Falsos negativos")
    print("Falsos positivos      | Verdadeiros negativos\n")
    print(f"{vp:<21} | {fn}")
    print(f"{fp:<21} | {vn}\n")
    
    print(f"Acuracia = {acuracia:.4f}")
    print(f"Revocação = {revocacao:.4f}")
    print(f"Precisão = {precisao:.4f}")
    print(f"F1 = {f1:.4f}")
    
    return


# teste
if __name__ == "__main__":
    # caminho dos dados
    arq_treino = 'dados/nba_treino.csv'
    arq_teste = 'dados/nba_teste.csv'
    
    try:

        jogadores_treino, gabarito_treino, jogadores_teste, gabarito_teste = DadosKNN(arq_treino, arq_teste) # prepara os dados
        
        # k=2
        print("\n--- Experimento k=2 ---")
        chutes_k2 = PreverKNN(jogadores_treino, gabarito_treino, jogadores_teste, k=2)
        AvaliarKNN(chutes_k2, gabarito_teste)

        # k=5
        print("\n--- Experimento k=5 ---")
        chutes_k5 = PreverKNN(jogadores_treino, gabarito_treino, jogadores_teste, k=5)
        AvaliarKNN(chutes_k5, gabarito_teste)

        # k=10
        print("\n--- Experimento k=10 ---")
        chutes_k10 = PreverKNN(jogadores_treino, gabarito_treino, jogadores_teste, k=10)
        AvaliarKNN(chutes_k10, gabarito_teste)

        # k=50
        print("\n--- Experimento k=50 ---")
        chutes_k50 = PreverKNN(jogadores_treino, gabarito_treino, jogadores_teste, k=50)
        AvaliarKNN(chutes_k50, gabarito_teste)

        
    except FileNotFoundError:
        print("\nERRO: Os arquivos 'nba_treino.csv' e 'nba_teste.csv' não foram encontrados.")
        print("Certifique-se de que eles estão na mesma pasta que este script!")