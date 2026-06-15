<div align="center">
  <h1>🏀 Previsão de Carreiras na NBA (TP2 - IIA)</h1>
  <p><em>Implementação do zero dos algoritmos de Machine Learning k-Means e k-NN.</em></p>

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/UFMG-CC0000?style=for-the-badge&logo=academic&logoColor=white" />
</div>

<br>

## 📌 Sobre o Projeto

Este projeto foi desenvolvido como Trabalho Prático 2 para a disciplina de **Introdução à Inteligência Artificial**. O objetivo é prever se jogadores novatos da NBA terão uma carreira superior ou inferior a 5 anos na liga, baseando-se em suas estatísticas de jogo (minutos, pontos, rebotes, etc.).

A arquitetura foi construída **do zero** (sem o uso do scikit-learn na etapa inicial) para demonstrar o domínio sobre o motor matemático por trás do aprendizado de máquina supervisionado e não-supervisionado.

---

## ⚙️ O que tem debaixo do capô?

<details>
<summary><b>1. Aprendizado Não-Supervisionado (k-Means)</b> <i>(Clique para expandir)</i></summary>

Implementação autoral do k-Means (`meu_kmeans.py`) focada em agrupamento de dados às cegas:
* **Inicialização:** Sorteio seguro de centróides (sem reposição) para evitar mínimos locais.
* **Métrica:** Cálculo da Distância Euclidiana em lote usando *broadcasting* do NumPy.
* **Convergência:** Loop iterativo com atualização de centróides pelas médias aritméticas dos grupos.
* **Alinhamento:** Teste de cenários para descobrir qual agrupamento corresponde às carreiras longas/curtas.
</details>

<details>
<summary><b>2. Aprendizado Supervisionado (k-NN)</b> <i>(Clique para expandir)</i></summary>

Implementação autoral do k-Nearest Neighbors (`meu_knn.py`) focada em classificação baseada em memória:
* **Isolamento de Dados:** Separação estrita entre bases de Treino (`X_treino`, `y_treino`) e Teste (`X_teste`, `y_teste`) para evitar vazamento de dados.
* **Votação Majoritária:** Apuração da moda estatística dos $k$ vizinhos mais próximos.
* **Critério de Desempate Ponderado:** Em caso de empates em valores de $k$ pares, a decisão é tomada calculando a menor distância acumulada dos vizinhos.
</details>

<details>
<summary><b>3. Avaliação de Desempenho</b> <i>(Clique para expandir)</i></summary>

Ambos os algoritmos contam com geradores de **Matriz de Confusão** customizados, extraindo métricas valiosas como:
* Acurácia Geral
* Revocação (Recall)
* Precisão
</details>

---

## 📂 Estrutura do Repositório


## 📂 Estrutura do Repositório

```text
📦 tp2        
 ┣ 📂 codigo
 ┃ ┣ 📜 comp.py       # Algoritmo supervisionado (k-NN) e sistema de votação
 ┃ ┗ 📜 nosu.py       # Lógica matemática e execução do agrupamento (k-Means)
 ┣ 📂 dados
 ┃ ┣ 📜 nba_teste.csv # Base de "desafio"
 ┃ ┗ 📜 nba_treino.csv# Base de "memória" e gabarito (1340 jogadores)
