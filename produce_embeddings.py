import subprocess
import pandas as pd
import numpy as np
import json
import multiprocessing as mp
import os
import tqdm
import argparse
import re
import shutil
from timeit import default_timer
import functools as ft
    
def extract_cls_embeddings(text_list, 
                           data_folder,
                           tmp_file_name = 'tmp', 
                           tmp_folder = 'tmp', 
                           out_folder = 'json',
                           bert_folder = 'bert_resources',
                           bert_models_folder = 'bert_models',
                           bert_model = 'uncased_L-2_H-128_A-2'):     

    """
    Warning: this function may produce more embeddings than elements in text_list 
    because the text is written to disk and the read back by BERT
    """
    
    with open('{}/{}/{}.txt'.format(data_folder, tmp_folder, tmp_file_name), 'w+') as file:
        for l in text_list:
            file.write(l)
            file.write('\n')
             
    process = subprocess.run(["python", 
                               "bert_resources/bert/extract_features.py", 
                               "--input_file={}/{}/{}.txt".format(data_folder, tmp_folder, tmp_file_name),
                               "--output_file={}/{}/{}.jsonl".format(data_folder, out_folder, tmp_file_name),
                               "--vocab_file={}/{}/{}/vocab.txt".format(bert_folder, 
                                                                                     bert_models_folder, 
                                                                                     bert_model),
                               "--bert_config_file={}/{}/{}/bert_config.json".format(bert_folder, 
                                                                                     bert_models_folder, 
                                                                                     bert_model),
                               "--init_checkpoint={}/{}/{}/bert_model.ckpt".format(bert_folder, 
                                                                                     bert_models_folder, 
                                                                                     bert_model),
                               "--layers=-1",
                               "--max_seq_length=128",
                               "--batch_size=8"])
#     if process.returncode != 0:
#         raise ValueError('There was a problem with the subprocess')
        
    with open('{}/{}/{}.jsonl'.format(data_folder, out_folder, tmp_file_name), 'r') as file:
        jsons = [json.loads(l) for l in file.readlines()]
        
    embeddings = [np.array(j['features'][0]['layers'][0]['values']) for j in jsons]
    
    return embeddings


def produce_batches(data, batch_size):
    batches = []
    for i in range(len(data) // batch_size + 1):
        batches.append(data[i*batch_size:(i + 1)*batch_size])
    output = [('batch_{}'.format(str(i).rjust(int(np.ceil(np.log10(len(data) // batch_size))),'0')), b) 
              for i, b in enumerate(batches)]
    return output

def extract_emb(x):
    return (x[0], extract_cls_embeddings(x[1], tmp_file_name = x[0], data_folder = output_folder))



start = default_timer()

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('batch_size', type = int)
args = parser.parse_args()

batch_size = args.batch_size

df_path = args.path
folder, file_name = re.match(r'(^.*)/(.*)\.', df_path).groups()
output_folder = '{}/{}_bert'.format(folder, file_name)

print('removing folders...')
try:
    shutil.rmtree(output_folder)
except FileNotFoundError:
    print('removing not needed')


print('creating folders...')
try:
    os.mkdir(output_folder)
    os.mkdir('{}/tmp'.format(output_folder))
    os.mkdir('{}/json'.format(output_folder))
    os.mkdir('{}/embeddings'.format(output_folder))
except:
    print('there was an error while creating the folders')
    
df = pd.read_csv(df_path)
batches = produce_batches(df['text'].tolist(), batch_size)

print('producing embeddings...')

# sequential version of the imap_unordered, useful for debugging
# results = []
# for n, b in batches:
#     results.append((n, extract_cls_embeddings(b, tmp_file_name = n, data_folder = output_folder)))

p = mp.Pool(processes = os.cpu_count() - 1)
results = list(p.imap_unordered(extract_emb, tqdm.tqdm(batches)))

arrays = [(b, np.stack(l)) for b, l in sorted(results)]
total_embeddings = 0
for b, a in arrays:
    np.save('{}/embeddings/{}_{}x{}.npy'.format(output_folder, b, a.shape[0], a.shape[1]), 
            a)
    total_embeddings += a.shape[0]

for i in range(5):    
    print()

time= default_timer() - start 
hours = time // 3600
mins = (time - hours * 3600) // 60
secs = time - hours * 3600 - mins * 60
print('Total processing time: {:0>2}:{:0>2}:{:.1f}'.format(int(hours), int(mins), int(secs)))
print('Total records in  df["text"]: {}'.format(df['text'].size))
print('Total embeddings produced: {}'.format(total_embeddings))
