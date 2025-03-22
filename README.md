# Comparativo Modelos Embedding
Comparação de modelos para geração de embedding utilzando o dataset [Msmarco](https://huggingface.co/datasets/microsoft/ms_marco).

Agradecimentos aos responsáveis pela tradução e disponibilização do msmarco em português. Utilizamos a base disponível no Hugginface para nossa análise - [link](https://huggingface.co/unicamp-dl/mt5-base-en-pt-msmarco-v2).

Criei um dataset no Kaggle somente com os arquivos utilizados em nossa análise, sendo um subconjunto do material acima citado - [link](https://www.kaggle.com/datasets/gustavodutramartins/msmarco) 

## Sobre os Modelos
O principal objetivo era analisar a performance do Amazon Titan, disponível na infraestrutura de foundation models do AWS Bedrock, frente a modelos do Huggingfaces. O corpus utilzado pela AWS, bem como o suporte a outras línguas, no caso o português, não ficava claro. Decidimos então analisar sua performance.  
Modelos utilizados no teste:
- Amazon Titan - projetados para uma ampla gama de aplicações de IA, incluindo busca semântica, sistemas de recomendação e chatbots.
- BAAI/bge-m3 - modelo de embedding multilíngue, conhecido por seu desempenho de ponta em benchmarks de recuperação de informações. Projetado para ser eficiente em termos de velocidade e uso de memória, tornando-o adequado para aplicações de grande escala.
- nonola/portuguese-bge-m3 - modelo treinado a partir do modelo BGE-M3 para fornecer embeddings de alta qualidade para textos em português. 
- sentence-transformers/all-MiniLM-L6-v2 - conhecido por sua eficiência em gerar embeddings, com um tamanho de modelo relativamente pequeno, projetada para ser rápido. Conseguimos executá-lo em um CPU sem problemas.

## Sobre a Execução do Teste

Tivemos que criar dois programas distintos, mas utilizando a mesma lógica, já que os modelos Huggingfaces podem executar dentro do programa, sendo instanciados e chamados diretamente. [link](Analise_Comparativa_Modelos_Embedding_HF.ipynb)  
Já no caso do Amazon Titan o modelo precisava ser chamado por serviço, necessitando de um tratamento diferenciado [link](analise_embedding_aws.py)

## Sobre a Arquitetura

Para os modelos do Huggingface foi criado um jupyter notebook, sendo executado no Google Colab com uma GPU T4 de 16gb. Para as chamadas na Amazon o programa foi executado no meu notebook, sem GPU.

Para pesquisa multidimensional dos embeddings foi utilizada a biblioteca FAISS. 

## Sobre as Métricas
As seguints métricas foram utilizadas:
- MRR (Mean Reciprocal Rank): Mede a posição do primeiro documento correto.
- NDCG (Normalized Discounted Cumulative Gain): Avalia a relevância dos documentos recuperados.
- Recall@k: Mede quantos documentos relevantes aparecem no top-k resultados.



## Sobre o Resultado Gerado
- diponível na aba [resultado](resultado/)
- 