import pickle
import numpy as np
import faiss
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.schema import Document
from sklearn.metrics import ndcg_score
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings

import os
import csv
from typing import List, Tuple, Union, Dict, Any, Optional
import boto3
import json
from tqdm import tqdm

class IndiceVetorial:
    """
    Classe para representar um documento indexado com embedding e metadados.
    """

    def __init__(self, indice:FAISS, metadados: List[Dict[str,Any]]):
        """
        Inicializa um objeto DocumentoIndexado.

        Args:
            indice: Indice Faiss para busca vetorial
            metadados: vetor com dicionário
        """
        self.indice = indice
        self.metadados = metadados
   

# Configuração do cliente Bedrock
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Modelos de embedding
models = {
   "Amazon-Titan": "amazon.titan-embed-text-v2:0"
}

def get_referencia_modelo(nome_modelo) -> str:
    if nome_modelo in models:
        return models[nome_modelo]
    else:
        raise ValueError(f"Modelo não suportado: {nome_modelo}")
    

def gravar_vetor_em_arquivo(vetor: List[Union[str, Tuple[str, Union[str, List[str]]]]], nome_arquivo: str) -> None:
    """
    Grava um vetor em um arquivo de texto, formatando os itens de acordo com seus tipos.

    A função suporta três tipos de itens no vetor:
    1.  Strings simples: gravadas diretamente no arquivo.
    2.  Tuplas (pergunta, resposta): gravadas como "Pergunta: pergunta" e "Resposta: resposta".
    3.  Tuplas (pergunta, lista de respostas): gravadas como "Pergunta: pergunta" e várias linhas "Resposta: resposta" para cada item na lista.

    Args:
        vetor: Uma lista contendo strings, tuplas de string ou tuplas de string e lista de string.
        nome_arquivo: O nome do arquivo onde o vetor será gravado.
    """
    with open("resultado\\" + nome_arquivo,"w", encoding="utf-8") as arquivo:
        for item in vetor:
            if isinstance(item, tuple) and isinstance(item[1], list):  # Se for uma tupla onde o segundo item é lista
                arquivo.write(f"Pergunta: {item[0]}\n")
                for resposta in item[1]:
                    arquivo.write(f"Resposta: {resposta}\n")  # Adiciona quebra de linha corretamente
                arquivo.write("\n")  # Adiciona espaçamento entre blocos
            elif isinstance(item, tuple):  # Se for uma tupla simples (pergunta, resposta única)
                arquivo.write(f"Pergunta: {item[0]}\n")
                arquivo.write(f"Resposta: {item[1]}\n\n")  # Dupla quebra para espaçamento
            else:  # Caso seja um item simples
                arquivo.write(f"{item}\n")

# Função para criar documentos com metadados
def criar_documento(texto, doc_id) -> Document :
    return Document(page_content=texto, metadata={"doc_id": doc_id, "original": texto})

def gerar_embedding(referencia_modelo:str, texto: str):
    """Gera embedding para um texto usando o Amazon Titan no Bedrock."""

    payload = json.dumps({"inputText": texto}, ensure_ascii=False)
    response = bedrock.invoke_model(
        body=f'{payload}',
        modelId=referencia_modelo,
        accept="application/json",
        contentType="application/json"
    )
    #embedding = response["body"].read().decode("utf-8")
    #return np.array(eval(embedding)["embedding"])
    
    response_body = json.loads(response["body"].read().decode("utf-8"))  # Usa json.loads() corretamente
    return np.array(response_body["embedding"], dtype=np.float32)  # Converte para numpy array de float32


# Função para criar FAISS com metadados
def indexar_faiss(documentos: List[Document], referencia_modelo:str, caminho_index="faiss_index") -> IndiceVetorial:
    """
    Cria e salva um índice FAISS a partir de documentos.

    Args:
        documentos: Lista de objetos Documento a serem indexados.
        modelo: Modelo de embeddings para vetorizar os documentos.
        caminho_index: Caminho para salvar o índice FAISS.

    Returns:
        O objeto FAISS criado.
        Lista de Metadados, onde cada um possui o doc-id (doc_id) e texto orginal (original)
    """
    print("Criando índice FAISS com Amazon Titan...")

    # Geração de embeddings usando Bedrock
    embeddings = []
    metadados =  []

    for doc in tqdm(documentos, "gerando embeddings..."):
        embedding = gerar_embedding(referencia_modelo, doc.page_content)  # Chama a função para gerar embeddings
        embeddings.append(embedding)
        metadados.append(doc.metadata)
    

    # Converter para array NumPy (FAISS exige um array 2D float32)
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    # Criar o índice FAISS com a dimensão correta
    d = embeddings_np.shape[1]  # Dimensão dos embeddings
    index = faiss.IndexFlatL2(d)  # Índice FAISS baseado em similaridade de vetores
    index.add(embeddings_np)  # Adiciona os embeddings ao índice

    # Salvar índice FAISS
    faiss.write_index(index, caminho_index)
    
    # Salvar metadados em pickle
    meta_path = caminho_index + ".pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadados, f)
        
    return IndiceVetorial(index, metadados)
    

def carregar_faiss(caminho_index) -> IndiceVetorial:
    if not os.path.exists(caminho_index):
        return None
        
    # Carregar índice FAISS
    index = faiss.read_index(caminho_index)    
    # Carregar metadados
    meta_path = caminho_index + ".pkl"
    with open(meta_path, "rb") as f:
        metadados = pickle.load(f)
    return IndiceVetorial(index, metadados)
    

def buscar_faiss(referencia_modelo: str, indiceVetorial: IndiceVetorial, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Realiza uma busca de similaridade em um índice FAISS.

    A função recebe um índice FAISS, uma query de busca e um número opcional de resultados a retornar.
    Retorna uma lista de tuplas, onde cada tupla contém o conteúdo do documento encontrado e seus metadados.

    Args:
        vectorstore: O índice FAISS onde a busca será realizada.
        query: A query de busca.
        top_k: O número de resultados a retornar (padrão: 5).

    Returns:
        Uma lista de tuplas, onde cada tupla contém:
            - O conteúdo do documento (str).
            - Um dicionário com os metadados do documento (Dict[str, Any]).
    """
    # Gerando o embedding via Bedrock
    # Gerar embedding da query com o AWS Bedrock
    query_embedding = gerar_embedding(referencia_modelo, query)

    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Buscar no FAISS
    distancia, indices = indiceVetorial.indice.search(query_embedding, top_k)

    # Retornar os documentos e metadados correspondentes
    resultados = [indiceVetorial.metadados[i] for i in indices[0] if i != -1]  # -1 indica índice inválido

    return resultados

def obter_conteudo_original_por_doc_id(documentos: List[Document], doc_id_desejado: int) -> Optional[str]:
    """
    Obtém o conteúdo original de um documento pelo doc_id.

    Args:
        documentos: Uma lista de objetos Document.
        doc_id_desejado: O doc_id do documento desejado.

    Returns:
        O conteúdo original do documento ou None se o doc_id não for encontrado.
    """
    for doc in documentos:
        if str(doc.metadata.get("doc_id")) == str(doc_id_desejado):
            return doc.metadata.get("original")
    raise ValueError(f"Nenhum documento encontrado com doc_id: {doc_id_desejado}")
    return None


# Carregar subconjunto do MS MARCO com ranqueamento BM25
def carregar_msmarco(caminho_collection, caminho_queries, caminho_qrels, caminho_bm25, num_queries=1000, num_posicoes_ranking=5) -> Tuple[List[Document], pd.DataFrame, pd.DataFrame]:
    # Carregar documentos e queries
    print("Carregando dados...")
    df_docs = pd.read_csv(caminho_collection, sep='\t', quoting=csv.QUOTE_NONE,header=None, names=['doc_id', 'document'])
    df_queries = pd.read_csv(caminho_queries, sep='\t', header=None, names=['query_id', 'query'])
    df_qrels = pd.read_csv(caminho_qrels, sep='\t', header=None, names=['query_id', '0', 'doc_id', 'relevance'])
    df_bm25 = pd.read_csv(caminho_bm25, sep='\t', header=None, names=['query_id', 'doc_id', 'rank'])

    # Filtrar apenas queries relevantes
    print("Filtrando dados...")
    df_qrels = df_qrels[df_qrels['relevance'] > 0]  # Apenas docs relevantes
    df_queriesl_filtered = df_queries[df_queries['query_id'].isin(df_qrels['query_id'])]
    # Amostrar queries
    sampled_queries = df_queriesl_filtered.sample(n=min(num_queries, len(df_queriesl_filtered)), random_state=42)

    # Criar conjunto de documentos a partir de BM25
    print("Criando conjunto de documentos a partir de BM25...")
    df_bm25_filtered = df_bm25[df_bm25['query_id'].isin(sampled_queries['query_id'])]
    df_bm25_filtered = df_bm25_filtered[df_bm25_filtered['rank'] <= num_posicoes_ranking]  # Pegamos os 5 primeiros do ranking
    df_docs_filtered = df_docs[df_docs['doc_id'].isin(df_bm25_filtered['doc_id'])]

    # Identificar IDs faltantes, já que o BM25 não é um gabarito, mas a execução de um modelo
    gabarito = df_qrels[df_qrels['query_id'].isin(sampled_queries['query_id'])]
    doc_ids_gabarito = set(gabarito['doc_id'])
    doc_ids_filtered = set(df_docs_filtered['doc_id'])
    doc_ids_faltantes = doc_ids_gabarito - doc_ids_filtered

    df_docs['doc_id'] = df_docs['doc_id'].astype(int)
    # Adicionar IDs faltantes a df_docs_filtered
    if doc_ids_faltantes:
        print(f"Adicionando {len(doc_ids_faltantes)} doc_ids faltantes a df_merged.")
        df_faltantes = df_docs[df_docs['doc_id'].isin(doc_ids_faltantes)]
        df_docs_filtered = pd.concat([df_docs_filtered, df_faltantes], ignore_index=True)

    # Criar documentos para FAISS
    print("Criando documentos para FAISS...")

    documentos = [criar_documento(row['document'], row['doc_id']) for _, row in df_docs_filtered.iterrows()]
    print(f"Total de documentos: {len(documentos)}")
    print(f"Total de queries: {len(sampled_queries)}")

    return documentos, sampled_queries, gabarito


def processar_resultado(query: str, ids_gabarito: List[int], retrieved_ids: List[int], resultados:  List[Tuple[str, Dict[str, Any]]], docs: List[Document], estatisticas: Dict[str, float]):
    """ Processa os resultados e classifica como acerto completo, parcial ou erro. """
    if retrieved_ids[0] in ids_gabarito:
        estatisticas["acertos_completos"].append((query, resultados[0]['original']))
    elif set(ids_gabarito) & set(retrieved_ids): #existe interseção entre os dois vetores
        set_gabarito = set(ids_gabarito)  # Para busca eficiente
        posicao =  next((i for i, x in enumerate(retrieved_ids) if x in set_gabarito), -1)
        lista_retorno = [res['original'] for res in resultados[:posicao + 1]]
        lista_retorno[posicao] = '-->' + lista_retorno[posicao]
        if posicao == 1:
            estatisticas["acertos_parciais_pos_2"].append((query, lista_retorno))
        elif posicao == 2:
            estatisticas["acertos_parciais_pos_3"].append((query, lista_retorno))
        else:
            estatisticas["acertos_parciais_outros"].append((query, lista_retorno))
    else:
        lista_retorno_erro = [res['original'] for res in resultados]
        for id_resposta_correta in ids_gabarito:
          lista_retorno_erro.append('-->' + obter_conteudo_original_por_doc_id(docs, id_resposta_correta))
        estatisticas["erros"].append((query, lista_retorno_erro))


def salvar_resultados(nome_modelo: str, estatisticas: Dict[str, List], metricas: Dict[str,float]):    
    """ Salva os resultados da avaliação em arquivos. """
    resultado_texto = (f"Acertos Completos: {len(estatisticas['acertos_completos'])}\n"
                       f"Acertos Parciais Posição 2: {len(estatisticas['acertos_parciais_pos_2'])}\n"
                       f"Acertos Parciais Posição 3: {len(estatisticas['acertos_parciais_pos_3'])}\n"
                       f"Acertos Parciais Outros: {len(estatisticas['acertos_parciais_outros'])}\n"
                       f"Erros: {len(estatisticas['erros'])}\n"
                       f"Métricas: {metricas}")
    
    with open(f"resultado\\resultado_{nome_modelo}.txt", "w") as arquivo:
        arquivo.write(resultado_texto)
    
    for categoria, vetor in estatisticas.items():
        gravar_vetor_em_arquivo(vetor, f"{categoria}.txt")

def avaliar_modelo(nome_modelo: str, docs: List[Document], indiceVetorial: IndiceVetorial, queries: pd.DataFrame,
                   gabarito: pd.DataFrame, top_k=5) -> Dict[str, float]:
    """
    Avalia um modelo de busca utilizando métricas MRR, Recall@5 e NDCG@10.
    """
    print(f"Avaliando modelo... {nome_modelo}")
    
    estatisticas = {
        "acertos_completos": [],
        "acertos_parciais_pos_2": [],
        "acertos_parciais_pos_3": [],
        "acertos_parciais_outros": [],
        "erros": []
    }
    
    mrr, recall, ndcg = 0, 0, 0
    num_queries = len(queries)
    
    for _, row in tqdm(queries.iterrows(), total=len(queries), desc="processando queries"):
        query_id, query = row['query_id'], row['query']
        resultados = buscar_faiss(get_referencia_modelo(nome_modelo),indiceVetorial, query, top_k)
        retrieved_ids = [res['doc_id'] for res in resultados]
        ids_gabarito = gabarito[gabarito['query_id'] == query_id]['doc_id'].tolist()
        
        processar_resultado(query, ids_gabarito, retrieved_ids, resultados, docs, estatisticas)
        
        # Calculando métricas
        if any(id_correto in retrieved_ids for id_correto in ids_gabarito):
          first_correct_rank = min(retrieved_ids.index(id_correto) + 1 for id_correto in ids_gabarito if id_correto in retrieved_ids)
          mrr += 1 / first_correct_rank
        else:
          mrr += 0
        recall += len(set(retrieved_ids) & set(ids_gabarito)) / len(ids_gabarito)
        ndcg += ndcg_score([[1 if doc in ids_gabarito else 0 for doc in retrieved_ids]], [[1] * len(retrieved_ids)])
    
    metricas = {
        "MRR@10": mrr / num_queries,
        "Recall@5": recall / num_queries,
        "NDCG@10": ndcg / num_queries
    }
    salvar_resultados(nome_modelo, estatisticas, metricas)
    
    return metricas


#=================================== inicio programa ==========================================


# Caminhos dos arquivos (ajustar conforme necessário)
caminho_collection = "dataset\\semantica\\msmarco\\portuguese_collection.tsv"
caminho_queries = "dataset\\semantica\\msmarco\\portuguese_queries.dev.tsv"
caminho_qrels = "dataset\\semantica\\msmarco\\qrels.dev.tsv"
caminho_bm25 = "dataset\\semantica\\msmarco\\run.bm25_portuguese-msmarco.txt"
caminho_index = "resultado\\faiss_index"
#nome_modelo = "MiniLM"
#nome_modelo = "Portuguese-BGE-M3";
#nome_modelo = "BAAI-BGE-M3"
nome_modelo = "Amazon-Titan"
numero_posicoes_ranking_para_analise = 5

print(f"============= Análise modelo {nome_modelo} ==========================")

modelo = get_referencia_modelo(nome_modelo)

# Carregar dados
documentos,queries, relevancias = carregar_msmarco(caminho_collection, caminho_queries,
                                                    caminho_qrels, caminho_bm25,
                                                    num_queries=1000, num_posicoes_ranking=numero_posicoes_ranking_para_analise)

# Verificar se FAISS já foi indexado
indice = carregar_faiss(caminho_index + "_" + nome_modelo)
if indice is None:
    print(f"A base vetorial não existe, os documentos serão indexados")
    #raise ValueError(f"Indexação da base deveria estar pronta")
    indice = indexar_faiss(documentos,modelo, caminho_index + "_" + nome_modelo)
else:
    print(f"A base vetorial já existia e foi carregada")
    
# Avaliando modelo
metricas = avaliar_modelo(nome_modelo,
                          documentos,
                          indice,
                          queries,
                          relevancias,
                          top_k=numero_posicoes_ranking_para_analise)
print("Métricas do modelo:", metricas)


