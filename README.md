# Comparativo Modelos Embedding
Comparação de modelos para geração de embedding utilzando o dataset [Msmarco](https://huggingface.co/datasets/microsoft/ms_marco).

Utilizamos a base disponível no Hugginface para nossa análise, traduzida para portuguës pelo Google - [link](https://huggingface.co/datasets/unicamp-dl/mmarco/tree/main/data/google).

Criei um dataset no Kaggle somente com os arquivos utilizados em nossa análise, sendo um subconjunto do material acima citado - [link](https://www.kaggle.com/datasets/gustavodutramartins/msmarco) 

## Objetivo
Analisar a performance do Amazon Titan, disponível na infraestrutura de foundation models do AWS Bedrock, frente a modelos do Huggingfaces, na língua portuguesa (pt_br). O corpus utilzado pela AWS, bem como o suporte a outras línguas, no caso o português, não estava claro na documentação. 
Modelos utilizados no teste:
- Amazon Titan - projetado para uma ampla gama de aplicações de IA, incluindo busca semântica, sistemas de recomendação e chatbots.
- BAAI/bge-m3 - modelo de embedding multilíngue, conhecido por seu desempenho de ponta em benchmarks de recuperação de informações. Projetado para ser eficiente em termos de velocidade e uso de memória, tornando-o adequado para aplicações de grande escala.  
- nonola/portuguese-bge-m3 - modelo treinado a partir do modelo BGE-M3 para fornecer embeddings de alta qualidade para textos em português. 
- sentence-transformers/all-MiniLM-L6-v2 - conhecido por sua eficiência em gerar embeddings, com um tamanho de modelo relativamente pequeno, projetada para ser rápido. Conseguimos executá-lo em um CPU sem problemas.

## Dataset
Em nosso estudo foi utilizado o dataset Msmarco (Microsoft MAchine Reading Comprehension), conjunto de dados de grande escala criado pela Microsoft Research. Ele foi projetado para impulsionar a pesquisa em compreensão de leitura de máquina e resposta a perguntas. O conjunto de dados consiste em mais de 1 milhão de consultas de pesquisa reais do Bing, juntamente com documentos relevantes e respostas geradas por humanos. Ele abrange uma ampla gama de tópicos e complexidades, tornando-o um recurso valioso para treinar e avaliar modelos de processamento de linguagem natural.
Motivos de sua escolha:
- O conjunto de dados é construído a partir de consultas de pesquisa reais do Bing, o que garante que os testes reflitam cenários de pesquisa do mundo real.
- O MSMARCO exige que os modelos compreendam o significado e o contexto das consultas, em vez de apenas corresponder palavras-chave. Isso é fundamental para a pesquisa semântica.
- A ampla gama de tópicos e a complexidade das consultas no MSMARCO ajudam a garantir que os modelos sejam testados em uma variedade de cenários de pesquisa.
- o conjunto de dados msmarco pode ser usado para treinar e avaliar modelos de "embeddings de texto", que são utilizados em pesquisas de similaridade vetorial.

### Estudo do Dataset
O dataset original Msmarco é um pouco diferente do utilizado nesse estudo, o material dispoonibilizado no Huggingface tem separações diferentes, as quais explicaremos nos próximos tópicos.  
<br>
#### Coleção de Respostas em Português  
---
- portuguese_collection.tsv - 8.841.823 registros com 2 colunas, identificador do documento (doc_id) e texto (document)

| doc_id | document | 
| :------: | -------- | 
|  0       |  A presença de comunicação entre mentes científicas foi tão importante para o sucesso do Projeto Manhattan quanto o intelecto científico. A única nuvem que paira sobre a impressionante ...   | 
| 94       | Steve Wheeler foi recentemente nomeado Diretor de Crédito da Lendmark Financial Services, LLC. Clique para saber mais sobre Steve Wheeler.   | 
| 125      | O District Griffin é um dos primeiros edifícios de super condomínio a ser concluído na área de Griffintown. Apropriadamente nomeado para sua localização na parte inferior da rua Peel, District Griffin ...  |
  
<br>

#### Coleção de Perguntas  
--- 
Três arquivos com o mesmo layout: _query_id_ - identificador da pergunta e _query_ - texto da pergunta

- portuguese_queries.train.tsv -  808.731 registros (dados de treino, não incluso no gabarito)  
- portuguese_queries.dev.tsv - 101.093 registros  
- portuguese_queries.dev.small.tsv - 6.980 registros ( está incluído em portuguese_queries.dev.tst)  
  
<br>

#### Gabarito - Perguntas com respostas   
--- 
- qrels.dev.tsv = 59.273 registros
- Existem repetições nas respostas (demonstração abaixo)
- Colunas :  
  - query_id - identificador da pergunta  
  - coluna com o valor 0  
  - doc_id - resposta correta    
  - relevance - 1 (sempre 1)    

Algumas perguntas possuem mais de uma resposta e isso deve ser tratado na avaliação da resposta encontrada pelo modelo

```
df_qrels = pd.read_csv(caminho_qrels, sep='\t', header=None, names=['query_id', '0', 'doc_id', 'relevance'])
contagem = df_qrels['query_id'].value_counts()
qtd_2_ocorrencias = len(contagem[contagem == 2])
qtd_3_ocorrencias = len(contagem[contagem == 3])
qtd_4_ocorrencias = len(contagem[contagem == 4])
qtd_5_ocorrencias = len(contagem[contagem == 5])

print(f"Número de perguntas com mais de uma resposta")
print(f"2 ocorrências: {qtd_2_ocorrencias}")
print(f"3 ocorrências: {qtd_3_ocorrencias}")
print(f"4 ocorrências: {qtd_4_ocorrencias}")
print(f"5 ocorrências: {qtd_5_ocorrencias}")
```  

> Número de perguntas com mais de uma resposta  
>  - 2 ocorrências: 2690  
>  - 3 ocorrências: 355  
>  - 4 ocorrências: 72  
>  - 5 ocorrências: 16   

<br>

#### Arquivo complementar
---  
- run.bm25_portuguese-msmarco.txt - Pelo que compreendi no link do Huggingface, esse arquivo é resultado da execução de um modelo T5 (Text-to-Text Transfer Transformer).
- Colunas
  - query_id - identificador da query  
  - doc_id - identificador do documento que responde a pergunta
  - rank - ranking da resposta. Para cada pergunta o ranking vai até 1.000.
- 6.975.268 registros
- Esse arquivo não é o gabarito, mas para tornar o teste interessante optamos por só utilizar queries que estão nesse arquivo, adicionando os 5 primeiros itens do ranking a nossa base. Teremos então para cada pergunta 5 respostas com semântica parecida, segundo a execução do modelo, para dificultar a seleção da resposta correta. **Importante ressaltar que esse arquivo não é um gabarito, mas uma fonte de documentos próximos semanticamente segundo as conclusões de um modelo**.  
<br><br>

## Execução
Como nosso objetivo é comparar modelos então os passos executados são similares para cada um dos participantes, ajustando somente a chamada ao modelo para fazer o embedding do documento e da pergunta. O Amazon Titan teve um tratamento diferente porque sua invocação é por serviço, mas os detalhes serão tratados em um subtópico.   
Os passos, em alto nível, são os seguintes:   
1 - Montagem do Dataset. - Escolha das perguntas, respectivas respostas gabarito e respostas com semantica parecida.   
2 - Montagem da representação vetorial de todas as respostas (embedding), sejam elas corretas ou não.   
3 - Armazenamento dos vetores multidimencionais na biblioteca FAISS(armazenamento e indexão vetorial para pesquisas semânticas)  
4 - Percorrer todas as perguntas, calculando sua representação vetorial (embedding) e pesquisando no indice faiss as 5 mais próximas.  
5 - Confrontar a acurácia da pesquisa contra as respostas gabarito.

<br>

**1 - Montagem do Dataset**  
Mil perguntas da base de perguntas [qrels.dev.tsv](https://www.kaggle.com/datasets/gustavodutramartins/msmarco?select=qrels.dev.tsv) foram escolhidas aleatoriamente para a análise. 
Para dificultar a tarefa do modelo utilizamos um outro arquivo [bm25_portuguese-msmarco.txt](https://www.kaggle.com/datasets/gustavodutramartins/msmarco?select=run.bm25_portuguese-msmarco.txt) que contém perguntas e respostas da base msmarco, ranqueadas segundo a análise de um modelo. Creio que seja o modelo [mt5-base Reranker finetuned on mMARCO](https://huggingface.co/unicamp-dl/mt5-base-en-pt-msmarco-v2). O arquivo não é um gabarito, ou seja, não tinha todas as respostas corretas, no entanto se mostrou bem interessante para o teste porque mostrava cerca de 10 respostas para um subconjunto de perguntas, ranqueadas segundo sua análise.... Perfeito para nosso teste, dificultando a solução do modelo e trazendo mais realidade para nossa análise.

Foram incluídas as respostas corretas nos casos em que o arquivo *bm25_portuguese-msmarco.txt* estava incompleto, ou seja, o modelo não tinha acertado dentro do seu ranking.  

O resultado final desse passo foi um conjunto de respostas com texto e identificador(doc_id) e outro conjunto com respostas e suas identificações (query_id)  

**2 - Representação Vetorial - Embedding**  
Para cada uma das respostas foi criado o respectivo embedding, ou seja, sua representação em um vetor multidimensional. Importante ressaltar que cada modelo tem um tamanho de vetor diferente para representação multidimensional.

**3 - Armazenamento**  
Para armazenamento dos vetores utilizamos a biblioteca FAISS (Facebook AI Similarity Search) é uma biblioteca eficiente para pesquisa de similaridade e agrupamento de vetores densos. Ela permite buscar vetores de qualquer tamanho.
Cada resposta foi adicionada ao FAISS que possui índices para acelerar a pesquisa.

**4 - Avaliar Modelo**   
Percorrer cada pergunta do dataset selecionado, gerando sua representação vetorial (embedding) e procurando os 5 vetores semanticamene mais próximos.

**5 - Calcular Métricas** 
Avaliar as respostas, identificando quando o acerto é completo (primeiro item da lista) , parcial (entre o segundo e quinto) ou falha (a resposta correta não estava entre os 5 retornados).  
Foram também utilizada as métricas MRR, Recall e NDCG@10. Serão detalhadas em tópico posterior.

**Observações**
Tivemos que criar dois programas distintos, já que os modelos fornecidos pelo Huggingfaces podem invocados diretamente, ou seja, executam junto com o código. [link](Analise_Comparativa_Modelos_Embedding_HF.ipynb).    
O Amazon Titan é chamado por serviço, necessitando de um tratamento diferenciado [link](analise_embedding_aws.py)

## Arquitetura

Para os modelos do Huggingface foi criado um jupyter notebook, sendo executado no Google Colab com uma GPU T4 de 16gb. Para as chamadas na Amazon o programa foi executado no meu notebook, sem GPU.

Para pesquisa multidimensional dos embeddings foi utilizada a biblioteca FAISS. 

## Sobre as Métricas
As seguints métricas foram utilizadas:
- MRR (Mean Reciprocal Rank): Mede a posição do primeiro documento correto.
- NDCG (Normalized Discounted Cumulative Gain): Avalia a relevância dos documentos recuperados.
- Recall@k: Mede quantos documentos relevantes aparecem no top-k resultados.



Uma segunda análise foi realizada, gerando arquivos com exemplos, da seguinte forma:
- Acertos completos - o primeiro vetor retornado, ou seja, o semanticamente mais próximo está entre as resposatas do gabarito (existem casos onde existem 5 respostas corretas).
> Exemplo  
> Pergunta: sintomas da gripe a e b em crianças  
> Resposta: R: Os sintomas da gripe em crianças incluem febre alta...  
> Pergunta: que tipos de vulcões  
> Resposta: Os quatro tipos de vulcões. Os vulcões são agrupados em cones ...

- Acertos parciais - o primeiro vetor retornado está na segunda, terceira, quarta ou quinta posição. Fizemos um arquivo para os casos de segunda e terceira posição, respectivamente, já o quarto e quinto ficou agrupado em um único arquivo.  
> Pergunta: custo de correio usps  
> Resposta: O custo médio para repintar armários de cozinha é de cerca de US $ 1.000, dependendo ...   
> Resposta: -->Atualização das taxas de correio certificado de 2015 em 31 de maio de 2015 o serviço postal ... 
> Pergunta: qual a largura da bancada padrão  
> Resposta: A direção da força de atrito é oposta à direção do movimento.  
> Resposta: Filhotes de qualquer idade gostam de se sujar. Mas não é aconselhável dar banho em seu filhote ...  
> Resposta: -->A altura padrão da bancada é de 36 polegadas (92 cm). Há um pouco mais de discussão ... 

- Erros - a(s) resposta(s) correta(s) não estão na lista de vetores retornados na pesquisa.
> Pergunta: como beber água quente com limão  
> Resposta: Encha um copo alto com água morna (tão quente quanto você pode....  
> Resposta: Beber mais de 100 onças de água pode parecer impossível no início... 
> Resposta: Visão geral. A bexiga, um saco oco localizado atrás do osso púbico, ... 
> Resposta: A garça-real-azul é encontrada em habitats de água doce e salgada ... 
> Resposta: A direção da força de atrito é oposta à direção do movimento.  
> Resposta: -->Ferva a agua. Despeje em um copo ou 2 xícaras de água em temperatura...


## Sobre o Resultado Gerado
- diponível na aba [resultado](resultado/)
- 
