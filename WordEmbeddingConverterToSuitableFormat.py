import gensim
from gensim.models import KeyedVectors
from chinese import ChineseAnalyzer   #only for chinese
import pynlpir               #only for chinese
pynlpir.open()              #only for chinese

#filename='GoogleNews-vectors-negative300.bin'  # downloaded from : https://code.google.com/archive/p/word2vec/  
filename = 'glove.840B.300d.txt'  				# downloaded from : https://nlp.stanford.edu/projects/glove/ 
#filename = 'cc.en.300.vec.txt'                 # downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html 
#filename = 'cc.de.300.vec.txt'  				# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.tr.300.vec.txt'      			# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.fi.300.vec.txt'     			# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.cs.300.vec.txt'  				# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.fr.300.vec.txt'  				# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.lv.300.vec.txt'      			# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.pl.300.vec.txt'      			# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.ro.300.vec.txt'      			# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.ru.300.vec.txt'      			# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html
#filename = 'cc.zh.300.vec.txt'  				# downloaded from : https://fasttext.cc/docs/en/crawl-vectors.html

# Keep this script with the downloaded embedding and run to get equivalent Word2Vec file fromat.
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False) 
model.save_word2vec_format(filename+".bin", binary=True) 
model = gensim.models.KeyedVectors.load_word2vec_format(filename+".bin", binary=True)   #Equivalent Word2Vec for cc.en.300.vec.txt embedding is saved as cc.en.300.vec.txt.bin


 


