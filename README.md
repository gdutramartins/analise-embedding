# Comparativo Modelos Embedding
Comparação de modelos para geração de embedding utilzando o dataset [Msmarco](https://huggingface.co/datasets/microsoft/ms_marco).

Utilizamos a base disponível no Hugginface para nossa análise, traduzida para portuguës pelo Google - [link](https://huggingface.co/datasets/unicamp-dl/mmarco/tree/main/data/google).

Criei um dataset no Kaggle somente com os arquivos utilizados em nossa análise, sendo um subconjunto do material acima citado - [link](https://www.kaggle.com/datasets/gustavodutramartins/msmarco) 

## Objetivo
Analisar a performance do Amazon Titan, disponível na infraestrutura de foundation models do AWS Bedrock, frente a modelos do Huggingfaces, na língua portuguesa (pt_br). O corpus utilzado pela AWS, bem como o suporte a outras línguas, no caso o português, não estava claro na documentação.  
Modelos que fazem *embedding* utilizados no teste:
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
- **portuguese_queries.dev.tsv - 101.093 registros**  ✅  
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
- Esse arquivo não é o gabarito, mas para tornar o teste ainda mais interessante optamos por só utilizar queries que estão nesse arquivo, adicionando os 5 primeiros itens do ranking a nossa base. Teremos então para cada pergunta 5 respostas com semântica parecida, segundo a execução do modelo, para dificultar a seleção da resposta correta. **Importante ressaltar que esse arquivo não é um gabarito, mas uma fonte de documentos próximos semanticamente segundo as conclusões de um modelo**.  
<br><br>

## Execução
Como nosso objetivo é comparar modelos então os passos executados são similares para cada um deles.  
O Amazon Titan teve um tratamento diferente porque sua invocação é por serviço, mas esses detalhes serão tratados em outro tópico.   
Os passos, em alto nível, são os seguintes:   
1 - Montagem do Dataset. - Escolha das perguntas, respectivas respostas gabarito e respostas com semantica parecida.   
2 - Montagem da representação vetorial de todas as respostas (embedding), sejam elas corretas ou não.   
3 - Armazenamento dos vetores multidimencionais na biblioteca FAISS (armazenamento e indexação vetorial para pesquisas semânticas)  
4 - Percorrer todas as perguntas, calculando sua representação vetorial (embedding) e pesquisando no indice faiss as 5 mais próximas.  
5 - Confrontar a acurácia da pesquisa contra as respostas gabarito.

<br>

**1 - Montagem do Dataset**  
- Mil perguntas da base de perguntas [qrels.dev.tsv](https://www.kaggle.com/datasets/gustavodutramartins/msmarco?select=qrels.dev.tsv) foram escolhidas aleatoriamente para a análise. 
- No entanto, para estar no conjunto selecionado ela também deveria estar presente no arquivo [bm25_portuguese-msmarco.txt](https://www.kaggle.com/datasets/gustavodutramartins/msmarco?select=run.bm25_portuguese-msmarco.txt) que contém respostas ranqueadas segundo a análise de um modelo [T5](https://huggingface.co/unicamp-dl/mt5-base-en-pt-msmarco-v2). Importante ressaltar que não é um gabarito, somente um mecanismo utilizado para dificultar a escolha da resposta correta. Somente as 5 primeiras do ranking para cada pergunta foram selecionadas.  
- Como o modelo T5 não gerou um gabarito então nem sempre as 5 primeiras respostas continham a correta, nesse caso adicionamos a resposta correta segundo o arquivo de gabarito (qrels).  
- O resultado final desse passo foi um conjunto de respostas com texto e identificador(doc_id) e outro conjunto com respostas e suas identificações (query_id)  

**2 - Representação Vetorial - Embedding**  
Para cada uma das respostas foi criado o respectivo embedding, ou seja, sua representação em um vetor multidimensional. Importante ressaltar que cada modelo tem um tamanho de vetor diferente para representação multidimensional.

**3 - Armazenamento**  
Para armazenamento dos vetores utilizamos a biblioteca FAISS (Facebook AI Similarity Search), eficiente para pesquisa de similaridade e agrupamento de vetores densos. 

**4 - Avaliar Modelo**   
Percorrer cada pergunta do dataset selecionado, gerando sua representação vetorial (embedding) e procurando as 5 respostas semanticamene mais próximas.  

**5 - Calcular Métricas**  
Avaliar as respostas, identificando quando o acerto é completo (primeiro item da lista) , parcial (entre o segundo e quinto) ou falha (a resposta correta não estava entre os 5 retornados).  
Foram também utilizada as métricas MRR, Recall e NDCG@10, que serão detalhadas em tópico posterior.

**Observações**  
Tivemos que criar dois programas distintos, já que os modelos fornecidos pelo Huggingfaces podem ser invocados diretamente, ou seja, executam junto com o código. [link](Analise_Comparativa_Modelos_Embedding_HF.ipynb).    
O Amazon Titan é chamado por serviço, necessitando de um tratamento diferenciado [link](analise_embedding_aws.py)

## Arquitetura

Para os modelos do Huggingface foi criado um jupyter notebook, sendo executado no Google Colab com uma GPU T4 de 16gb. Para as chamadas na Amazon o programa foi executado em um notebook sem GPU.

Para pesquisa multidimensional dos embeddings foi utilizada a biblioteca FAISS. 

## Sobre as Métricas
As seguints métricas foram utilizadas:
- MRR (Mean Reciprocal Rank): Mede a posição do primeiro documento correto.
- NDCG (Normalized Discounted Cumulative Gain): Avalia a relevância dos documentos recuperados.
- Recall@k: Mede quantos documentos relevantes aparecem no top-k resultados.

Além das métricas númericas também consideramos as respostas sob três categorias:
- Acerto Completo - Item correto é o primeiro da lista retornada. Considerado sucesso
= Acerto Parcial - Item correto está entre o segundo e quinto elemento retornado. Considerado sucesso.
- Erro - O item correto não aparece na busca semantica.    
<br> 
**Exemplos:**
> Acerto Completo  
> Pergunta: sintomas da gripe a e b em crianças  
> Resposta: R: Os sintomas da gripe em crianças incluem febre alta...  
> Pergunta: que tipos de vulcões  
> Resposta: Os quatro tipos de vulcões. Os vulcões são agrupados em cones ...

> _Acerto Parcial_  
> Pergunta: custo de correio usps  
> Resposta: O custo médio para repintar armários de cozinha é de cerca de US $ 1.000, dependendo ...   
> Resposta: -->Atualização das taxas de correio certificado de 2015 em 31 de maio de 2015 o serviço postal ...
>   
> Pergunta: qual a largura da bancada padrão  
> Resposta: A direção da força de atrito é oposta à direção do movimento.  
> Resposta: Filhotes de qualquer idade gostam de se sujar. Mas não é aconselhável dar banho em seu filhote ...  
> Resposta: -->A altura padrão da bancada é de 36 polegadas (92 cm). Há um pouco mais de discussão ... 

> _Erros_  
> Pergunta: como beber água quente com limão  
> Resposta: Encha um copo alto com água morna (tão quente quanto você pode....  
> Resposta: Beber mais de 100 onças de água pode parecer impossível no início...  
> Resposta: Visão geral. A bexiga, um saco oco localizado atrás do osso púbico, ...  
> Resposta: A garça-real-azul é encontrada em habitats de água doce e salgada ...  
> Resposta: A direção da força de atrito é oposta à direção do movimento.   
> Resposta: --> Ferva a agua. Despeje em um copo ou 2 xícaras de água em temperatura...


## Sobre o Resultado Gerado
- diponível na aba [resultado](resultado/)
- Fizemos duas execuções 
  - 1.000 perguntas com 1.603 opções de respostas selecionadas
  - 10.000 perguntas com 16.420 respostas selecionadas
- A tabela com as métricas para cada uma das execuções está listada abaixo  

<br>  

### Mil Perguntas
---  
<br>   

**Métricas**

| Modelo | MRR | NDCG |  Recall |
| :------ | -------- |  -------- | -------- |
|  Amazon Titan |0.90851  | 0.9555 | 0.5716 |  
|  BAAI/bge-m3 | 0.92341 | 0.9805  | 0.5863 |  
|  portuguese-bge-m3 | 0.9248 | 0.98 | 0.5858 |  
|  MiniLM | 0.6163 | 0.70875 | 0.4282 |  
 
 <br>
   
**Acertos, Acertos Parciais, Erros**  
<br>
Entendemos que se a busca encontrou o item procurado nos 5 primeiros registros da resposta então tivemos sucesso na indexação vetorial e pesquisa. Criamos essa heurística para o estudo e não tem aplicação geral para análise de busca semântica, mas se adequa ao nosso cenário.


| Modelo | 1a pos | 2a pos |  3a pos | 4/5a pos | Erro | % Sucesso |
| :------ | -------- |  -------- | -------- | -------- | -------- | :-----: |
|  Amazon Titan | 876  | 42 | 23 |  16 | 43 | 95,7 % |
|  BAAI/bge-m3 | 887  | 45 | 23 | 27  | 18 | 98,2 % |
|  portuguese-bge-m3 | 890  | 43 | 22 | 26  | 19 | 98,1 % |
|  MiniLM |  553 | 77 | 42 | 48  | 280 | 72,0 % |

<br>

### Dez Mil Perguntas
---  
<br>   

**Métricas**

| Modelo | MRR | NDCG |  Recall |
| :------ | -------- |  -------- | -------- |
|  Amazon Titan |0.8213  | 0.9084 | 0.5460 |  
|  BAAI/bge-m3 | 0.8385 | 0.9230  | 0.5544 |  
|  portuguese-bge-m3 | 0.8396 | 0.9229 | 0.5544 |  
|  MiniLM | 0.4506 | 0.5533 | 0.3354 |  
 
 <br>


**Acertos, Acertos Parciais, Erros**  
<br>  
Entendemos que se a busca encontrou o item procurado nos 5 primeiros registros da resposta então tivemos sucesso na indexação vetorial e pesquisa. Criamos essa heurística para o estudo e não tem aplicação geral para análise de busca semântica, mas se adequa ao nosso cenário.

| Modelo | 1a pos | 2a pos |  3a pos | 4/5a pos | Erro | % Sucesso |
| :------ | -------- |  -------- | -------- | -------- | -------- | :------: |
|  Amazon Titan | 7605  | 805 | 363 |  368 | 859 | 91,41 % |
|  BAAI/bge-m3 | 7804  | 765 | 355 | 353  | 723 | 92,77 % | 
|  portuguese-bge-m3 | 7822  | 756 | 349 | 350  | 723 | 92,77 % | 
|  MiniLM | 3830 | 794 | 448 | 568  | 4360 | 56,40 % |

