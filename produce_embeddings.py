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


start = default_timer()

parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('batch_size', type = int)
args = parser.parse_args()

batch_size = args.batch_size

df_path = args.path
folder, file_name = re.match(r'(^.*)/(.*)\.', df_path).groups()
output_folder = '{}/{}_bert'.format(folder, file_name)

print('cleaning folders...')
try:
    shutil.rmtree(output_folder)
except:
    print('ok')


print('creating folders...')
try:
    os.mkdir(output_folder)
except:
    print('main folder created succesfully...')

try:
    os.mkdir('{}/tmp'.format(output_folder))
except:
    print('tmp folder created succesfully...')
    
try:
    os.mkdir('{}/json'.format(output_folder))
except:
    print('json folder created succesfully...')
    
try:
    os.mkdir('{}/embeddings'.format(output_folder))
except:
    print('embedding folder created succesfully...')
    
def extract_cls_embedding(text_list, 
                          data_folder,
                          tmp_file_name = 'tmp', 
                          tmp_folder = 'tmp', 
                          out_folder = 'json',
                          bert_folder = 'bert_resources',
                          bert_models_folder = 'bert_models',
                          bert_model = 'uncased_L-2_H-128_A-2'):     

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
    if process.returncode != 0:
        raise ValueError('There was a problem with the subprocess')
        
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
    return (x[0], extract_cls_embedding(x[1], tmp_file_name = x[0], data_folder = output_folder))

df = pd.read_csv(df_path)
batches = produce_batches(df['text'].tolist(), batch_size)
p = mp.Pool(processes = os.cpu_count() - 1)
results = list(p.imap_unordered(extract_emb, tqdm.tqdm(batches)))
arrays = [(b, np.stack(l)) for b, l in sorted(results)]
for b, a in arrays:
    np.save('{}/embeddings/{}.npy'.format(output_folder, b), a)
    
print()
print('Total processing time: {}'.format(default_timer() - start))
