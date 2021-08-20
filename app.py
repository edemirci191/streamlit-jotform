import numpy as np
import pandas as pd
import streamlit as st
import annoy
import tensorflow_hub as hub
from langdetect import detect
import requests
import os
import pickle
from urllib.request import urlopen
from tensorflow_text import SentencepieceTokenizer

url_de = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_de"
url_en = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_en"
url_es = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_es"
url_fr = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_fr"
url_it = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_it"
url_nl = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_nl"
url_pt = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_pt"
url_tr = "https://storage.googleapis.com/jotform-recommender.appspot.com/index_tr"

r_de = requests.get(url_de, stream = True)
r_en = requests.get(url_en, stream = True)
r_es = requests.get(url_es, stream = True)
r_fr = requests.get(url_fr, stream = True)
r_it = requests.get(url_it, stream = True)
r_nl = requests.get(url_nl, stream = True)
r_pt = requests.get(url_pt, stream = True)
r_tr = requests.get(url_tr, stream = True)

if not os.path.exists('de_from_url'):
  with open("de_from_url","wb") as f:
    for block in r_de.iter_content(chunk_size = 8192):
      if block:
        f.write(block)
if not os.path.exists('en_from_url'):
  with open("en_from_url","wb") as f:
    for block in r_en.iter_content(chunk_size = 8192):
      if block:
        f.write(block)
if not os.path.exists('es_from_url'):
  with open("es_from_url","wb") as f:
    for block in r_es.iter_content(chunk_size = 8192):
      if block:
        f.write(block)
if not os.path.exists('fr_from_url'):
  with open("fr_from_url","wb") as f:
    for block in r_fr.iter_content(chunk_size = 8192):
      if block:
        f.write(block)
if not os.path.exists('it_from_url'):
  with open("it_from_url","wb") as f:
    for block in r_it.iter_content(chunk_size = 8192):
      if block:
        f.write(block)
if not os.path.exists('nl_from_url'):
  with open("nl_from_url","wb") as f:
    for block in r_nl.iter_content(chunk_size = 8192):
      if block:
        f.write(block)
if not os.path.exists('pt_from_url'):
  with open("pt_from_url","wb") as f:
    for block in r_pt.iter_content(chunk_size = 8192):
      if block:
        f.write(block)
if not os.path.exists('tr_from_url'):
  with open("tr_from_url","wb") as f:
    for block in r_tr.iter_content(chunk_size = 8192):
      if block:
        f.write(block)
        
del r_de
del r_en
del r_es
del r_fr
del r_it
del r_nl
del r_pt
del r_tr
del url_de
del url_en
del url_es
del url_fr
del url_it
del url_nl
del url_pt
del url_tr

def apply_url(id):
  full_url = "https://www.jotform.com/answers/" + str(id)
  return full_url

mapping_en, mapping_de, mapping_tr, mapping_pt, mapping_it, mapping_es, mapping_nl, mapping_fr = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/mapping.pkl"))
question_en, question_de, question_tr,question_pt, question_it, question_es, question_nl, question_fr = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/questions.pkl"))
random_projection_matrix_en, random_projection_matrix_de, random_projection_matrix_tr, random_projection_matrix_pt, random_projection_matrix_it,random_projection_matrix_es,random_projection_matrix_nl,random_projection_matrix_fr = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/matrix.pkl"))
converted_list = pickle.load(urlopen("https://storage.googleapis.com/jotform-recommender.appspot.com/badwords.pkl"))

def find_similar_items(lang_index,mapping_name,embedding, num_matches=5):
  '''Finds similar items to a given embedding in the ANN index'''
  ids = lang_index.get_nns_by_vector(
  embedding, num_matches, search_k=-1, include_distances=False)
  items = [mapping_name[i] for i in ids]
  return items,ids

embedding_dimension = 64

index_filename_en = "en_from_url"
index_filename_es = "es_from_url"
index_filename_fr = "fr_from_url"
index_filename_it = "it_from_url"
index_filename_nl = "nl_from_url"
index_filename_pt = "pt_from_url"
index_filename_tr = "tr_from_url"
index_filename_de = "de_from_url"

index_en = annoy.AnnoyIndex(embedding_dimension)
index_es = annoy.AnnoyIndex(embedding_dimension)
index_fr = annoy.AnnoyIndex(embedding_dimension)
index_it = annoy.AnnoyIndex(embedding_dimension)
index_nl = annoy.AnnoyIndex(embedding_dimension)
index_de = annoy.AnnoyIndex(embedding_dimension)
index_tr = annoy.AnnoyIndex(embedding_dimension)
index_pt = annoy.AnnoyIndex(embedding_dimension)

index_en.load(index_filename_en, prefault=True)
index_es.load(index_filename_es, prefault=True)
index_fr.load(index_filename_fr, prefault=True)
index_it.load(index_filename_it, prefault=True)
index_nl.load(index_filename_nl, prefault=True)
index_de.load(index_filename_de, prefault=True)
index_tr.load(index_filename_tr, prefault=True)
index_pt.load(index_filename_pt, prefault=True)

del index_filename_en
del index_filename_es
del index_filename_fr
del index_filename_it
del index_filename_nl
del index_filename_pt
del index_filename_tr 
del index_filename_de 

model_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
embed = hub.load(model_url)
# August 20 deleted embed_en, embed_tr which are same model loaded again

def extract_embeddings(query,embed_fn,rpm): 
  '''Generates the embedding for the query'''
  query_embedding =  embed_fn([query])[0].numpy()
  if rpm is not None:
    query_embedding = query_embedding.dot(rpm)
  return query_embedding

base_url = "https://www.jotform.com/answers/"

def main():
  st.title("Jotform Support Forum Question Recommender")
  st.subheader("Overview")
  st.write("Purpose of this application is to recommend the user similar questions that has been asked before by other users. When the user asks a new question other already answered similar questions are going to be recommended to the user in English and also in his/her native language.")
  user_input = st.text_input("Question","Nasıl form oluşturulur")
  st.button("Search Question")
  st.image('https://storage.googleapis.com/jotform-recommender.appspot.com/podo_7.png',width=264)
  not_found = 1
  # in else part embedfn=embed_.. changed to embedfn = embed
  if any(word in user_input for word in converted_list):
    print('Your sentence contains profanity words, please try again')
  else:
    l_index = None
    map = None
    embedfn = None
    rpm = None
    lg = detect(user_input)
    varforid = None
    if lg == 'en':
      l_index = index_en
      map = mapping_en
      embedfn = embed
      rpm = random_projection_matrix_en
      not_found = 0
      varforid = question_en

    if lg == 'es':
      l_index = index_es
      map = mapping_es
      embedfn = embed
      rpm = random_projection_matrix_es
      not_found = 0
      varforid = question_es

    if lg == 'tr':
      l_index = index_tr
      map = mapping_tr
      embedfn = embed
      rpm = random_projection_matrix_tr
      not_found = 0
      varforid = question_tr

    if lg == 'fr':
      l_index = index_fr
      map = mapping_fr
      embedfn = embed
      rpm = random_projection_matrix_fr
      not_found = 0
      varforid=question_fr

    if lg == 'de':
      l_index = index_de
      map = mapping_de
      embedfn = embed
      rpm = random_projection_matrix_de
      not_found = 0
      varforid = question_de

    if lg == 'nl':
      l_index = index_nl
      map = mapping_nl
      embedfn = embed
      rpm = random_projection_matrix_nl
      not_found = 0
      varforid= question_nl

    if lg == 'pt':
      l_index = index_pt
      map = mapping_pt
      embedfn = embed
      rpm = random_projection_matrix_pt
      not_found = 0
      varforid = question_pt

    if lg == 'it':
      l_index = index_it
      map = mapping_it
      embedfn = embed
      rpm = random_projection_matrix_it
      not_found = 0
      varforid = question_it

    if not_found == 1:
      l_index = index_en
      map = mapping_en
      embedfn = embed
      rpm = random_projection_matrix_en
      lg = 'en'
      varforid = question_en

    query_embedding = extract_embeddings(user_input,embedfn,rpm)
    items,ids = find_similar_items(l_index,map,query_embedding, 5)
    lst=[]
    show_df = pd.DataFrame()
    if lg != 'en':
      query_embedding_en = extract_embeddings(user_input,embedfn,random_projection_matrix_en)
      items_en,ids_en = find_similar_items(index_en,mapping_en,query_embedding_en, 5)
      extended_items = items_en + items
      for j in ids_en:
        lst.append("https://www.jotform.com/answers/"+str(question_en.iloc[j]['id']))
      for i in ids:
        lst.append("https://www.jotform.com/answers/"+str(varforid.iloc[i]['id']))
      show_df.to_html(escape=False)
      show_df['Similar Questions'] = extended_items
      show_df['Thread URL'] = lst
      st.subheader('Recommendations')
      st.table(show_df)
      #st.write("[https://www.jotform.com/answers/]" + str(lst[1])) hyperlink with constant id
    else:
      for i in ids:
        lst.append("https://www.jotform.com/answers/" + str(varforid.iloc[i]['id']))

      show_df.to_html(escape=False)
      show_df['Similar Questions'] = items
      show_df['Thread URL'] = lst
      #show_df.set_index('Thread URL', inplace = True)
      st.subheader('Recommendations')
      st.table(show_df)
   
if __name__ == '__main__':
  main()
  del embedding_dimension     
del base_url
#del show_df
del lst
del items
del ids
del query_embedding
del l_index
del rpm
del lg
del varforid
del model_url
del embed

del mapping_en
del mapping_de
del mapping_tr 
del mapping_pt 
del mapping_it 
del mapping_es 
del mapping_nl 
del mapping_fr

del question_en
del question_de
del question_tr
del question_pt
del question_it
del question_es
del question_nl
del question_fr

del random_projection_matrix_en
del random_projection_matrix_de
del random_projection_matrix_tr
del random_projection_matrix_pt
del random_projection_matrix_it
del random_projection_matrix_es
del random_projection_matrix_nl
del random_projection_matrix_fr

del index_en
del index_es
del index_fr
del index_it
del index_nl
del index_de
del index_tr
del index_pt
del converted_list # added on 20 august for streamlit     
