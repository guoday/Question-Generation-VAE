import ujson as json
import os
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import pickle as pkl
import collections
import numpy
def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)

def build_dictionary(filepaths,dst_path,lowercase=False):
    word_freqs = collections.OrderedDict()
    for filepath in filepaths:
        print ('Processing', filepath)
        with open(filepath, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                words_in = line.strip().split()
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
    filter_freqs={}
    for w in word_freqs:
        if word_freqs[w]>10:
            filter_freqs[w]=word_freqs[w]
    sorted_words= sorted(filter_freqs.items(), key=lambda d:d[1], reverse=True)
    worddict = collections.OrderedDict()
    

    for ii, ww in enumerate(sorted_words):
        worddict[ww[0]] = ii + 4
    print('Size of dictionary: '+str(len(worddict)))
    with open(dst_path[0], 'wb') as f:
        pkl.dump(worddict, f)
    with open(dst_path[1],'w') as f:
        for word in worddict:
            f.write(word+'\n')
        
if __name__=='__main__':
    from_path="QG/data"
    to_path="QG/data/"
    for file, prex in [('train_multiturn.inANDout.valid','train'),('dev_multiturn.inANDout.valid','dev'),('test_multiturn.inANDout.valid','test')]:
        print('Preprocess '+file+' data')
        max_len_in=0
        min_len_in=10000
        max_len_out=0
        min_len_out=10000
        with open(os.path.join(to_path,file)) as f, open(os.path.join(to_path,prex+'.in'),'w') as f1,open(os.path.join(to_path,prex+'.out'),'w') as f2:
            cont=0
            while True:
                data=[]
                if f.readline()=='':
                    break
                for i in range(3):
                    data.append(f.readline().strip())
                f.readline()
                inputs=data[0]+' <delimiter> '+data[1]
                outputs=data[2]
                f1.write(inputs+'\n')
                f2.write(outputs+'\n')
                max_len_in=max(max_len_in,len(inputs.split()))
                max_len_out=max(max_len_out,len(outputs.split()))
                min_len_in=min(min_len_in,len(inputs.split()))
                min_len_out=min(min_len_out,len(outputs.split()))
                cont+=1
                if prex!='train' and cont==20000:
                    break

        print(file+' data size:',cont)   
        print(file+' data max min len input:',max_len_in,min_len_in)
        print(file+' data max min len output:',max_len_out,min_len_out)
        print("-"*80)
    print('Building dictionary .....')
    build_dictionary([os.path.join(to_path,'train.in'),os.path.join(to_path,'train.out')],[os.path.join(to_path,'vocab.pkl'),os.path.join(to_path,'vocab.in')])
    print('Building dictionary done!')           
    print("-"*80)           
                
    
