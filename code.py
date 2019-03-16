# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:01:35 2018
@author: Narjes Rohani (GreenBlueMind)
"""
from sklearn.preprocessing import MinMaxScaler
import copy
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import random
from scipy import interp
import matplotlib.pyplot as plt
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
def caseStudy(test_num, pred_y,  labels,names):
    #fileObject = open('resultMFCase.txt', 'a')

    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1 
               # fileObject.write(names[index]+'\n')
   



def modelEvaluation(real_matrix,predict_matrix,testPosition,featurename): #  compute cross validation results
       real_labels=[]
       namelist=[]
       predicted_probability=[]
       name = numpy.loadtxt("drug_list.txt",dtype=str,delimiter=" ")

       for i in range(0,len(testPosition)):
           real_labels.append(real_matrix[testPosition[i][0],testPosition[i][1]])
           predicted_probability.append(predict_matrix[testPosition[i][0],testPosition[i][1]])
           namelist.append(name[testPosition[i][0],1]+"---"+name[testPosition[i][1],1]+"--"+str(predict_matrix[testPosition[i][0],testPosition[i][1]]))
       normalize=MinMaxScaler()
#       predicted_probability= normalize.fit_transform(predicted_probability)
       real_labels=numpy.array(real_labels)
       predicted_probability=numpy.array(predicted_probability)
       predicted_probability=predicted_probability.reshape(-1,1)
       precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
       aupr_score = auc(recall, precision)

       all_F_measure=numpy.zeros(len(pr_thresholds))
       for k in range(0,len(pr_thresholds)):
           if (precision[k]+precision[k])>0:
              all_F_measure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
           else:
              all_F_measure[k]=0
       max_index=all_F_measure.argmax()
       threshold=pr_thresholds[max_index]

       fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
       auc_score = auc(fpr, tpr)
       predicted_score=numpy.zeros(len(real_labels))
       predicted_score=numpy.where(predicted_probability > threshold, 1, 0)

       f=f1_score(real_labels,predicted_score)
       caseStudy(len(real_labels),predicted_score,real_labels,namelist)
       accuracy=accuracy_score(real_labels,predicted_score)
       precision=precision_score(real_labels,predicted_score)
       recall=recall_score(real_labels,predicted_score,predicted_probability)
       print('results for feature:'+featurename)
       print('************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f, f-measure:%.3f************************' %(auc_score,aupr_score,recall,precision,accuracy,f))
#       auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), ("%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)
       results=[auc_score,aupr_score,precision, recall,accuracy,f,tpr,fpr]
       return results,real_labels,predicted_probability

def matrix_factorization(A, X, Y, K, Ws,Wd,landa,miu):
 for i in range(len(A)):

    X[i]=update_x(i,A, X, Y, K, Ws,Wd,landa,miu)
    #-------------------------
 for j in range(len(A[0])):
   
    Y[j]=update_y(j,A, X, Y, K, Ws,Wd,landa,miu)
# print("X matrix:",X)
# print("Y Matrix:",Y)
 final=X.dot(Y.T)
# print("Final Matrix:",final)
 return final

#-----------------------------------------------------
#update xi
def update_x(i,A, X, Y, K, Ws,Wd,landa,miu):
    xi2=0
    xi4=0
#    print(A[i,:])
#    print('00',len(A[i,:]))
    xi=A[i,:].dot(Y)
    xi=xi.reshape(1,K)
#    print('1',xi.shape)
#    print('len',len(Wd),len(Wd[i]),i)
    for j in range(len(Wd[i])):
#        print(Wd[i][j],Wd[j][i])
        xi2=numpy.dot(Wd[i,j]+Wd[j,i],(X[j]))+xi2
    xi=xi+numpy.dot(landa,(xi2))
    I=numpy.identity(K)
    xi3 =( numpy.dot((Y.T),Y)+ numpy.dot(miu,I))
    for j in range(len(Wd[i])):
     xi4=(Wd[i,j]+Wd[j,i])+xi4
    
    xi3=xi3+numpy.dot(landa*xi4,I)
    xi3=inv(xi3)
#    print('shape',xi3.shape)
#    print(xi3)
    xi3=xi3.reshape(K,K)
    #fff=xi.reshape(5,2)*xi3.reshape(2,2)
    final=xi.dot(xi3)
#    print('final',final)
    #final=numpy.dot(xi.reshape(5,2),xi3.reshape(2,2)).shape(5,2)
    return final
#end update xi---------------------------------------
    



#Update yj---------------------------------------
def update_y(j,A, X, Y, K, Ws,Wd,landa,miu):
    xi2=0
    xi4=0
#    print('123',A[:,j].shape)
    
    xi=(A[:,j].T).dot(X)
    xi=xi.reshape(1,K)
#    print('124',xi.shape)
    for i in range(len(Ws[j])):
     xi2=numpy.dot((Ws[i,j]+Ws[j,i]),Y[i])+xi2
    xi=xi+numpy.dot(landa,(xi2))
#    print('125',xi.shape)
    I=numpy.identity(K)
    xi3 =( numpy.dot((X.T),X)+ numpy.dot(miu,I))
    for j in range(len(Ws[i])):
     xi4=(Ws[i,j]+Ws[j,i])+xi4
    
    xi3=xi3+numpy.dot(landa*xi4,I)
    xi3=inv(xi3)
    xi3.reshape(K,K)
    final=xi.dot(xi3)
#    print('final',final)
    return final
#End yi------------------------------------
wd = numpy.loadtxt("transporter_Jacarrd_sim.csv",dtype=float,delimiter=",")
ws = numpy.loadtxt("transporter_Jacarrd_sim.csv",dtype=float,delimiter=",")
R = numpy.loadtxt("drug_drug_matrix.csv",dtype=int,delimiter=",")
