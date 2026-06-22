import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

def plotar_knn_desempenho():
    """
    Gráfico 1: Comparação de Acurácia e F1 Score entre as implementações do k-NN.
    Mostra como a escolha do k afeta o desempenho e evidencia a divergência no k=2.
    """
    k_valores = ['k=2', 'k=5', 'k=10', 'k=50']
    
    acc_custom = [0.8172, 0.6903, 0.7052, 0.6791]
    acc_skl = [0.6381, 0.6567, 0.6567, 0.6978]
    
    f1_custom = [0.8535, 0.7674, 0.7706, 0.7701]
    f1_skl = [0.5679, 0.7374, 0.7548, 0.7947]

    x = np.arange(len(k_valores))
    largura = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Acurácia
    ax1.bar(x - largura/2, acc_custom, largura, label='Implementação Própria', color='#1f77b4')
    ax1.bar(x + largura/2, acc_skl, largura, label='Scikit-Learn', color='#ff7f0e')
    ax1.set_ylabel('Acurácia')
    ax1.set_title('Acurácia Geral por Valor de k')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_valores)
    ax1.legend()
    ax1.set_ylim(0, 1.0)

    # Subplot 2: F1 Score
    ax2.bar(x - largura/2, f1_custom, largura, label='Implementação Própria', color='#1f77b4')
    ax2.bar(x + largura/2, f1_skl, largura, label='Scikit-Learn', color='#ff7f0e')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Métrica F1 por Valor de k')
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_valores)
    ax2.legend()
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

def plotar_kmeans_convergencia():
    """
    Gráfico 2: Velocidade de convergência (número de iterações).
    Ilustra a eficiência do k-means++ (Scikit-Learn) contra a inicialização aleatória.
    """
    cenarios = ['k=2', 'k=3']
    
    iter_custom = [14, 8]
    iter_skl = [6, 10]
    
    x = np.arange(len(cenarios))
    largura = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    barras1 = ax.bar(x - largura/2, iter_custom, largura, label='Implementação Própria (Aleatória)', color='#2ca02c')
    barras2 = ax.bar(x + largura/2, iter_skl, largura, label='Scikit-Learn (k-means++)', color='#d62728')
    
    ax.set_ylabel('Número de Iterações')
    ax.set_title('Esforço de Convergência do k-Means')
    ax.set_xticks(x)
    ax.set_xticklabels(cenarios)
    ax.legend()
    
    ax.bar_label(barras1, padding=3)
    ax.bar_label(barras2, padding=3)
    
    plt.tight_layout()
    plt.show()

def plotar_kmeans_distribuicao_k3():
    """
    Gráfico 3: Distribuição das classes reais dentro dos 3 clusters gerados pelo Scikit-Learn.
    Responde visualmente à pergunta do enunciado: "O que acontece para k=3?"
    """
    clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    
    # Dados de porcentagem do k=3 do Scikit-Learn (saídas.txt)
    # Cluster 0: Longos: 36.52%, Curtos: 63.48%
    # Cluster 1: Longos: 68.64%, Curtos: 31.36%
    # Cluster 2: Longos: 75.83%, Curtos: 24.17%
    
    pct_longos = [36.52, 68.64, 75.83]
    pct_curtos = [63.48, 31.36, 24.17]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.bar(clusters, pct_longos, label='Carreira >= 5 Anos (TARGET 1)', color='#9467bd')
    ax.bar(clusters, pct_curtos, bottom=pct_longos, label='Carreira < 5 Anos (TARGET 0)', color='#8c564b')
    
    ax.set_ylabel('Composição do Cluster (%)')
    ax.set_title('Composição Interna dos Grupos para k=3 (Scikit-Learn)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))
    
    # Linha de 50% para facilitar a visualização de quem domina o cluster
    ax.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plotar_knn_desempenho()
    plotar_kmeans_convergencia()
    plotar_kmeans_distribuicao_k3()