from gensim.models import KeyedVectors
import datetime
import numpy as np

start = datetime.datetime.now()
print(start)
# Load reference translations and system translations (five)
def load_doc(filepath): # open the file as read only
    file = open(filepath, mode='rt', encoding='utf-8')
    text=list()
    # read all text
    lines = file.readlines()
    for i in lines:
        data=i.split()
        text.append(data)
    file.close()
    return text

reftranslation=load_doc("translationref.txt")
systranslation=load_doc("translationsys.txt")
#print(len(reftranslation))

print("System translation before preprocessing:")
print(systranslation[0])
print("Number of words = ", len(systranslation[0]))
print("Reference translation before preprocessing:")
print(reftranslation[0])
print("Number of words = ", len(reftranslation[0]))

filename = 'GoogleNews-vectors-negative300.bin'    # Word2Vec based WEEM for all into English
#filename = 'glove.840B.300d.txt.word2vec.bin'    # GloVe based WEEM for all into English
#filename = 'cc.en.300.vec.bin'    # FastText based WEEM for all into English

model = KeyedVectors.load_word2vec_format(filename, binary=True)
embwords = list(model.wv.vocab)

resultantscore=0.0
fluency=0.0
adequacy=0.0
 
for i in range(len(reftranslation)):

    notmatchwordREF=np.setdiff1d(reftranslation[i],systranslation[i])
    notmatchwordSYS=np.setdiff1d(systranslation[i],reftranslation[i])
    notmatchwordREFemb=[]
    for m in range(len(notmatchwordREF)):
        if notmatchwordREF[m] in embwords:
            notmatchwordREFemb.append(notmatchwordREF[m])
            
    sentweight=0.0
    recall=0.0
    precision=0.0
    weight=0.0
    woreward=0.0
    countbigram=0.0
          
    for n in range(len(systranslation[i])):
        
        if systranslation[i][n] in reftranslation[i]:
            weight=1.0
            sentweight=sentweight+weight
            if systranslation[i][n] in systranslation[i] and systranslation[i][n] in reftranslation[i]:
                if n<(len(systranslation[i])-1) and systranslation[i].index(systranslation[i][n])+1<len(systranslation[i]) and reftranslation[i].index(systranslation[i][n])+1<len(reftranslation[i]):
                    if systranslation[i][systranslation[i].index(systranslation[i][n])+1].lower() == (reftranslation[i][reftranslation[i].index(systranslation[i][n])+1]).lower():
                        woreward=woreward+2/(len(systranslation[i])-1)
                        countbigram=countbigram+1
           
        elif len(notmatchwordREFemb)>0 and systranslation[i][n] in embwords:
            result=[model.wv.similarity(systranslation[i][n],word) for word in notmatchwordREFemb]
            weight=max(result)
            
            sentweight=sentweight+weight
            if systranslation[i][n] in systranslation[i] and notmatchwordREFemb[result.index(max(result))] in reftranslation[i]:
                if n<(len(systranslation[i])-1) and systranslation[i].index(systranslation[i][n])+1<len(systranslation[i]) and reftranslation[i].index(notmatchwordREFemb[result.index(max(result))])+1<len(reftranslation[i]):
                    if (systranslation[i][systranslation[i].index(systranslation[i][n])+1]).lower() == (reftranslation[i][reftranslation[i].index(notmatchwordREFemb[result.index(max(result))])+1]).lower():
                        woreward=woreward+(2/(len(systranslation[i])-1))
                        countbigram=countbigram+1
        else:
           weight=0.0
           sentweight=sentweight+weight 
    
    d=abs(len(systranslation[i])-len(reftranslation[i]))  #d is length difference
    if d !=0:
        penalty=((1+d)**(1/d))**((2-d)/4)
    else:
        penalty=1
        
    if woreward==0:
        compensation=penalty
    elif 0<woreward<2:
        compensation=penalty**((2.0-woreward)/4) 
    elif woreward == 2:
        compensation=1
    
    if len(systranslation[i])>0 and len(reftranslation[i])>0:
        precision=sentweight/len(systranslation[i])
        fluency=fluency+precision
    if len(reftranslation[i])>len(systranslation[i])>0:
        recall=sentweight/len(reftranslation[i])
        adequacy=adequacy+recall
    else:
        recall=precision
        adequacy=adequacy+recall
    
    if (recall+precision)>0:
        fmeasure=2*recall*precision/(recall+precision)
    else:
        fmeasure=0.0
        
    if len(systranslation[i])>1:
        bigrammatch=countbigram/(len(systranslation[i])-1)
    else:
        bigrammatch=0.0

    WEEMscore=((0.1*precision)+(0.5*recall))*compensation+(0.4*bigrammatch)  #for both adequacy and fluency
    
    
    #wmt19  <METRIC NAME>   <LANG-PAIR>   <TEST SET>   <SYSTEM>   <SEGMENT NUMBER>   <SEGMENT SCORE> <ENSEMBLE>   <AVAILABLE>
    #f=open("WEEM.seg.score","a+")
    #f.write("WEEM\tru-en\tnewstest2019\tonline-E.0\t" + str(i+1) + "\t" + str(WEEMscore) + "\tnon-ensemble" + "\tno\n")
    #f.close()
    
    #wmt17
    #f=open("WEEM.seg.score","a+")
    #f.write("\nWEEM\tzh-en\tnewstest2017\tonline-G.0\t" + str(i+1) + "\t" + str(WEEMscore) + "\t" + "\tnon-ensemble" + "\tno" )
    #f.close()
    
    #wmt16
    #f=open("WEEM.seg.score","a+")
    #f.write("\nWEEM\ttr-en\tnewstest2016\tParFDA.4542\t" + str(i+1) + "\t" + str(WEEMscore))
    #f.write(str(WEEMscore)+"\n") # Pearson’s correlation of metric scores with the WMT 2016 direct assessment of absolute translation adequacy at segment-level.
    #f.close()
    
    #wmt15  <METRIC NAME>   <LANG-PAIR>   <TEST SET>   <SYSTEM>   <SEGMENT NUMBER>   <SEGMENT SCORE>
    #f=open("WEEM.seg.score","a+")
    #f.write("\nWEEM\ttr-en\tnewstest2017\tJAIST.4859\t" + str(i+1) + "\t" + str(WEEMscore))
    #f.write(str(WEEMscore)+"\n")
    #f.close()
    
    #wmt14 <METRIC NAME>   <LANG-PAIR>   <TEST SET>   <SYSTEM>   <SEGMENT NUMBER>   <SEGMENT SCORE>
    f=open("WEEM.seg.score","a+")
    f.write("\nWEEM\tru-en\tnewstest2014\tuedin-wmt14.3364\t" + str(i+1) + "\t" + str(WEEMscore))
    f.close()
    
    
    resultantscore=resultantscore+WEEMscore
    
fluency=fluency/len(systranslation)
adequacy=adequacy/len(reftranslation)
resultantscore=resultantscore/len(reftranslation)

end = datetime.datetime.now()
elapsedtime=end-start

#wmt19 <METRIC NAME>   <LANG-PAIR>   <TEST SET>   <SYSTEM>   <SYSTEM LEVEL SCORE>   <ENSEMBLE>   <AVAILABLE>
#f=open("WEEM.sys.score","a+")
#f.write("WEEM\tcs-en\tnewstest2019\tonline-G.0\t" + str(resultantscore) + "\t" + "\tnon-ensemble" + "\tno\n")
#f.close()

#wmt17
#f=open("WEEM.sys.score","a+")
#f.write("\nWEEM\tcs-en\tnewstest2017\tonline-G.0\t" + str(resultantscore) + "\t" + str(int(start.timestamp())) + "\t" + str(int(end.timestamp())) + "\tnon-ensemble" + "\tno" + "\t"+str(elapsedtime.seconds))
#f.close()

#wmt16
#f=open("WEEM.sys.score","a+")
#f.write("\nWEEM\ttr-en\tnewstest2016\tParFDA.4542\t" + str(resultantscore))
#f.close() 

#wmt15  <METRIC NAME>   <LANG-PAIR>   <TEST SET>   <SYSTEM>   <SYSTEM LEVEL SCORE>
#f=open("WEEM.sys.score","a+")
#f.write("\nWEEM\ttr-en\tnewstest2017\tJAIST.4859\t" + str(resultantscore))
#f.close()

#wmt14 <METRIC NAME>   <LANG-PAIR>   <TEST SET>   <SYSTEM>   <SYSTEM LEVEL SCORE>

f=open("WEEM.sys.score","a+")
f.write("\nWEEM\tru-en\tnewstest2014\tuedin-wmt14.3364\t" + str(resultantscore))
f.close()

print("Translation adequacy by WEEM = ", adequacy)
print("Translation fluency by WEEM = ",fluency)
print("Resultant WEEM score = ",resultantscore)
print("Elapsed time for comparing ", len(reftranslation), "reference and sytem translations = ",elapsedtime.seconds, " seconds.") 