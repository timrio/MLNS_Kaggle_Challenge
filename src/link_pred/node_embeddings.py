from telnetlib import GA
from karateclub import Walklets
from karateclub import Node2Vec
from karateclub import DeepWalk
from transformers import AutoTokenizer, AutoModel
import pickle
import torch
import os
from tqdm import tqdm


def compute_walklets(G):
    walklets = Walklets() # we leave the defaults parameters for the other values
    walklets.fit(G)
    walklets_articles_embeddings = walklets.get_embedding()
    return(walklets_articles_embeddings)

def compute_node2vec(G):
    walklets = Node2Vec() # we leave the defaults parameters for the other values
    walklets.fit(G)
    walklets_articles_embeddings = walklets.get_embedding()
    return(walklets_articles_embeddings)

def compute_deep_walks(G):
    walklets = DeepWalk() # we leave the defaults parameters for the other values
    walklets.fit(G)
    walklets_articles_embeddings = walklets.get_embedding()
    return(walklets_articles_embeddings)
   

# compute abstracts embeddings using specter network



def compute_abstracts_embeddings(information_df):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'gpu'

    if os.path.isfile('Data/processed_data/abstracts_embeddings.pkl'):
        abstracts_embeddings = pickle.load(open('Data/processed_data/abstracts_embeddings.pkl','rb'))
    else:
        
        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = AutoModel.from_pretrained('allenai/specter').to(device)
        model.eval()
        abstracts_embeddings = []
        for i in tqdm(range(information_df.shape[0]), position = 0):
            article = information_df.loc[i]
            title = article.title
            abstract = article.abstract
            paper = [{'title':title, 'abstract':abstract}]

            # concatenate title and abstract
            title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in paper]
            # preprocess the input
            inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = inputs.to(device)
            result = model(**inputs)
            # take the first token in the batch as the embedding
            embedding = result.last_hidden_state[:, 0, :].detach().cpu().numpy()
            abstracts_embeddings.append(embedding)
        pickle.dump(abstracts_embeddings, open('Data/processed_data/abstracts_embeddings.pkl','wb'))
    return(abstracts_embeddings)
   