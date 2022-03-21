import json
from tqdm import tqdm
import os
import nltk
from collections import defaultdict

data_path = './data/arxiv/arxiv-metadata-oai-snapshot.json'
train_out_path = './data/arxiv/timed/train.txt'
valid_out_path = './data/arxiv/timed/valid.txt'
test_out_path = './data/arxiv/timed/test.txt'
train_index_path = './data/arxiv/time_stratified_train'
valid_index_path = './data/arxiv/time_stratified_validation'
test_index_path = './data/arxiv/test'
all_text_data_out_path = './data/arxiv/all_text.txt'

def extract_train_valid_test_file():
  os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
  train = open(train_out_path,'w')
  train_index = set()
  with open(train_index_path,'r') as f_in:
    for line in f_in.readlines():
      train_index.add(line.strip().split('\t')[-1])
  valid = open(valid_out_path,'w')
  valid_index = set()
  with open(valid_index_path,'r') as f_in:
    for line in f_in.readlines():
      valid_index.add(line.strip().split('\t')[-1])
  test = open(test_out_path,'w')
  test_index = set()
  with open(test_index_path,'r') as f_in:
    for line in f_in.readlines():
      test_index.add(line.strip().split('\t')[-1])
  with open(data_path,'r') as f_in:
    for line in tqdm(f_in.readlines()):
      data = json.loads(line.strip())
      id = data['id']
      if id in test_index:
        test.write(data['abstract'].strip().replace('\n',' ')+'\n\n')
      elif id in valid_index:
        valid.write(data['abstract'].strip().replace('\n',' ')+'\n\n')
      elif id in train_index:
        train.write(data['abstract'].strip().replace('\n',' ')+'\n\n')
  train.close()
  valid.close()
  test.close()

def extract_sentences_from_json():
  all_text = open(all_text_data_out_path,'w')
  with open(data_path,'r') as f_in:
    for line in tqdm(f_in.readlines()):
      data = json.loads(line.strip())
      sentences = nltk.tokenize.sent_tokenize(data['abstract'].strip().replace('\n',' '))
      all_text.write("\n".join(sentences)+"\n")
  all_text.close()

# def train_spm_model():
#   spm.SentencePieceTrainer.train(input=all_text_data_out_path, model_prefix='data/arxiv/arxiv_spm', vocab_size=50000)

extract_train_valid_test_file()
