# Comparativo Modelos Embedding
Comparação de modelos para geração de embedding utilzando o dataset [Msmarco](https://huggingface.co/datasets/microsoft/ms_marco).

Utilizamos a base disponível no Hugginface para nossa análise, traduzida para portuguës pelo Google - [link](https://huggingface.co/datasets/unicamp-dl/mmarco/tree/main/data/google).

Criei um dataset no Kaggle somente com os arquivos utilizados em nossa análise, sendo um subconjunto do material acima citado - [link](https://www.kaggle.com/datasets/gustavodutramartins/msmarco) 

## Objetivo
Analisar a performance do Amazon Titan, disponível na infraestrutura de foundation models do AWS Bedrock, frente a modelos do Huggingfaces, no portuguës brasil (pt_br). O corpus utilzado pela AWS, bem como o suporte a outras línguas, no caso o português, não ficava claro. Decidimos então analisar sua performance.   
Modelos utilizados no teste:
- Amazon Titan - projetados para uma ampla gama de aplicações de IA, incluindo busca semântica, sistemas de recomendação e chatbots.
- BAAI/bge-m3 - modelo de embedding multilíngue, conhecido por seu desempenho de ponta em benchmarks de recuperação de informações. Projetado para ser eficiente em termos de velocidade e uso de memória, tornando-o adequado para aplicações de grande escala.
- nonola/portuguese-bge-m3 - modelo treinado a partir do modelo BGE-M3 para fornecer embeddings de alta qualidade para textos em português. 
- sentence-transformers/all-MiniLM-L6-v2 - conhecido por sua eficiência em gerar embeddings, com um tamanho de modelo relativamente pequeno, projetada para ser rápido. Conseguimos executá-lo em um CPU sem problemas.

## Dataset
Em nosso estudo foi utilizado o dataset Msmarco (Microsoft MAchine Reading COmprehension) é um conjunto de dados de grande escala criado pela Microsoft Research. Ele foi projetado para impulsionar a pesquisa em compreensão de leitura de máquina e resposta a perguntas. O conjunto de dados consiste em mais de 1 milhão de consultas de pesquisa reais do Bing, juntamente com documentos relevantes e respostas geradas por humanos. Ele abrange uma ampla gama de tópicos e complexidades, tornando-o um recurso valioso para treinar e avaliar modelos de processamento de linguagem natural.
Motivos de sua escolha:
- O conjunto de dados é construído a partir de consultas de pesquisa reais do Bing, o que garante que os testes reflitam cenários de pesquisa do mundo real.
- O MSMARCO exige que os modelos compreendam o significado e o contexto das consultas, em vez de apenas corresponder palavras-chave. Isso é fundamental para a pesquisa semântica.
- A ampla gama de tópicos e a complexidade das consultas no MSMARCO ajudam a garantir que os modelos sejam testados em uma variedade de cenários de pesquisa.
- o conjunto de dados msmarco pode ser usado para treinar e avaliar modelos de "embeddings de texto", que são utilizados em pesquisas de similaridade vetorial.


## Execução

Passos realizados para cada modelo em alto nível:  
1 - Montagem do Dataset. - Escolha das perguntas, respostas gabarito e respostas com semantica parecida.  
2 - Montagem da representação vetorial de todas as respostas (embedding), corretas e incorretas.  
3 - Armazenamento dos vetores multidimencionais na biblioteca FAISS(armazenamento e indexão vetorial para pesquisas semânticas)
4 - Percorrer todas as perguntas, realizando sua representação vetorial (embedding) e pesquisando no indice faiss as 5 mais próximas.
5 - Calcular métricas de acurácia  

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
- Acertos completos - a pesquisa semântica encontrou o resultado


## Sobre o Resultado Gerado
- diponível na aba [resultado](resultado/)
- 