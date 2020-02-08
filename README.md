# WEEM

=== WEEM stands for Word Embedding based Evaluation Metric. It is intended to evaluate quality of machine translation systems.
=== For developing WEEM, we employed publicly available pre-trained word embedding models that have been downloaded from the following: 
    == https://code.google.com/archive/p/word2vec/     (Word2Vec based embedding) 
		--- Used for evaluating MT systems in which English language is a target language and source language can be any
    == https://nlp.stanford.edu/projects/glove/        (glove.840B.300d.zip) (GloVe based embedding)
	        --- Used for evaluating MT systems in which English language is a target language and source language can be any
    == https://fasttext.cc/docs/en/crawl-vectors.html	  (FastText based embedding)
	        --- Used for evaluating MT systems in which Chinese, English, Czech, Greman, Finnish, French, Latvian, Polish, Romanian,                     Russian, or Turkish language is a target language and source language can be any

# Requirements:

=== Install python 3.0, gensim, numpy, ChineseAnalyzer, and pynlpir 
=== Download one of the word embedding models you need from the above links.
=== Word2Vec based word embedding file format is compatible with the gensim package. But, GloVe and FastText based word embedding models     file format need to converted to the equivalent Word2Vec format.
=== For conversion, run: WordEmbeddingConverterToSuitableFormat.py (put file you need to convert in the same folder and make sure that       the downloaded file is assigned to variable "filename")

# Scripts for evaluating different translation directions, and pick module you want and assign Word Embedding model you want in the source code.

=== WEEMforTargetCs.py : For evaluating English into Czech translations.
=== WEEMforTargetDe.py : For evaluating English into German translations.
=== WEEMforTargetEn.py : For evaluating all into English translations.
=== WEEMforTargetFi.py : For evaluating English into Finnish translations.
=== WEEMforTargetFr.py : For evaluating English into French translations.
=== WEEMforTargetLv.py : For evaluating English into Latvian translations.
=== WEEMforTargetPl.py : For evaluating English into Polish translations.
=== WEEMforTargetRo.py : For evaluating English into Romanian translations.
=== WEEMforTargetRu.py : For evaluating English into Russian translations.
=== WEEMforTargetTr.py : For evaluating English into Turkish translations.
=== WEEMforTargetZh.py : For evaluating English into Chinese translations.

# Required inputs:

=== Copy target language ground truth text (reference translations), paste in "translationref.txt" and save it.  
=== Copy machine translation systems output (system translation), paste in "translationsys.txt" and save it.
=== In our experiments, we used WMT15-17 data sets from http://www.statmt.org/wmt15/results.html,             http://www.statmt.org/wmt16/results.html and http://www.statmt.org/wmt17/results.html

# Output

=== Your segment level evaluation scores should be written to "WEEM.seg.score" file in the following format: <METRIC NAME>   <LANGUAGE-      PAIR>   <TEST SET>   <MT SYSTEM NAME>   <SEGMENT NUMBER>   <SEGMENT LEVEL SCORE>
=== Your system level evaluation scores should be written to "WEEM.sys.score" file in the following format: <METRIC NAME>   <LANGUAGE-        PAIR>   <TEST SET>   <MT SYSTEM NAME>   <SYSTEM LEVEL SCORE>
# Authors

=== [Tulu Tilahun Hailu] (tutilacs@yahoo.com)
=== [Junqing Yu] (yjqing@hust.edu.cn)
=== [Tessfu Geteye Fantaye] (tessfug@hust.edu.cn) 

# Reference

=== If you use WEEM please cite the following reference. 
