import numpy as np
from scipy.spatial import distance
import os 
import pickle
import spacy
import pandas as pd
from tqdm import tqdm

from link_pred.namematcher import NameMatcher



spacy_nlp = spacy.load("en_core_web_sm")

def cosine(a,b):
    return(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def euclidian(a,b):
    return(distance.euclidean(a, b))


def retrieve_and_pre_processed_informations(information_path):
    if os.path.isfile("Data/processed_data/information.csv"):
        information_df = pickle.load(open("Data/processed_data/information.csv",'rb'))
    else:
        information_df = pd.read_csv(information_path, header=None)
        information_df.columns = ["ID",'pub_year','title','authors','journal_name','abstract']

        # fill na
        information_df = information_df.fillna({'authors':'', 'journal_name':''})

        # split authors name
        information_df.authors = information_df.authors.apply(lambda x:x.split(","))

        # lemmatize titles    
        information_df['title_lemma'] = information_df.title.apply(lambda x: [token.lemma_ for token in spacy_nlp(x) if not token.is_punct if not token.is_digit if not token.is_stop])
        pickle.dump(information_df, open("Data/processed_data/information.csv",'wb'))
    return(information_df)


name_matcher = NameMatcher()

def compute_unique_names(authors_raw_set):
    """
    one author can be named differently on different papers
    this function aims at finding a 'representant' (longest name that describe an author) for each 
    author
    inputs:
        - authors_raw_set: set of previously extracted author names
    outputs:
        - dict: keys are the name in authors_raw_set and the values are the representant
    """
    representant_dict = {}
    attributed_nodes = [] # names that already have a representant
    for name in tqdm(authors_raw_set, position = 0):
        sim_list = [] # similar names 
        if name not in attributed_nodes:
            for name2 in authors_raw_set:
                try:
                    if name != name2 and name[0]==name2[0] and name2 not in attributed_nodes:
                        # two names need to start by the same letter to be consider as potential equivalents
                        score = name_matcher.match_names(name, name2)
                        if score > 0.9: # if names are close enough
                            sim_list.append(name2)
                except:
                    continue
            sim_list.append(name) # the representant is in this list
            attributed_nodes.extend(sim_list) # we have fund a representant for those names
            representant = max(sim_list, key=len) # the representant is the longest name
            for name in sim_list: # all those names have the same representant
                representant_dict[name] = representant
    return(representant_dict)