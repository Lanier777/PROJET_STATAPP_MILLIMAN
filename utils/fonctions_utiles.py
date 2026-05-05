
# Importer les bibliothèques necessaires
import gdown
import pandas as pd
import re
import gc
import os
import sys
import pandas as pd
import numpy as np
import tqdm
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, classification_report, confusion_matrix, roc_auc_score
from sklearn.svm import SVC


import lightgbm as lgb 
## Fonction d'imputation par la moyenne 


def missing_mean(df):
    """
    entrée : dataset df avec valeurs manquantes Na 
    sortie :df_c le dataset traité 
    """

    # Séparation des colonnes numériques et catégorielles
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Fraudulent')
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Création d'une copie de sécurité
    df_c = df.copy()
    
    # Remplacement des valeurs manquantes pour les variables numériques
    for col in numeric_cols:
        for fraud_value in [0, 1]:
            mean_value = df_c.loc[df_c['Fraudulent'] == fraud_value, col].mean()
            df_c.loc[(df_c['Fraudulent'] == fraud_value) & (df_c[col].isna()),col] = mean_value
    
    # Remplacement des valeurs manquantes pour les variables catégorielles
    for col in categorical_cols:
        for fraud_value in [0, 1]:
            mode_series = df_c.loc[df_c['Fraudulent'] == fraud_value, col].mode()
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
                df_c.loc[(df_c['Fraudulent'] == fraud_value) & (df_c[col].isna()),col] = mode_value
    
    # Vérification du remplissage
    print("Pourcentage de valeurs manquantes après traitement :")
    print((df_c.isnull().mean() * 100).round(3))

    return df_c


## Fonction pour le RANDOMFOREST 

def random_forest(A,b,T,t):
    """
    entrée :

    • (A,b) est le dataset d'entrainement 

    • (T,t) est le dataset de test

    sortie :

    • y_pred, y_pred_proba

    On suppose avoir déjà séparé le dataset en train et test
    """
    
    # entraînement du modèle
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(A, b)
    
    #construction de y_pred
    y_pred=clf.predict(T)
    
    #Regardons la distribution de probabilité qu'une transaction soit frauduleuse 
    y_pred_proba = clf.predict_proba(T)
    
    #On trace un histogramme de ces probas
    #pd.Series(y_pred_proba[:,1]).hist()

    return y_pred, y_pred_proba



## Fonction pour la REGRESSION LOGISTIQUE

def logistic(A,b,T,t):
    """
    entrée :

    • (A,b) est le dataset d'entrainement 

    • (T,t) est le dataset de test

    sortie :

    • courbe ROC avec AUC
    • threshold intéressant et les métriques associées

    On suppose avoir déjà séparé le dataset en train et test
    """
    
    #import LogisticRegression et on fit au modèle
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(A,b)
    
    #construction de y_pred
    y_pred=lr.predict(T)
    
    
    #Regardons la distribution de probabilité qu'une transaction soit frauduleuse 
    y_pred_proba = lr.predict_proba(T)
    
    #On trace un histogramme de ces probas
    #pd.Series(y_pred_proba[:,1]).hist()

    return y_pred, y_pred_proba


    

## Fonction pour le SVM

def svm(A,b,T,t):
    """
    entrée :

    • (A,b) est le dataset d'entrainement 

    • (T,t) est le dataset de test

    sortie :

    • y_pred, y_pred_proba

    On suppose avoir déjà séparé le dataset en train et test
    """
    
    # entraînement du modèle
    svc =SVC(kernel = 'rbf', C= 5, gamma= 'scale', probability=True )
    svc.fit(A,b)

    
    #construction de y_pred
    y_pred=svc.predict(T)
    
    #Regardons la distribution de probabilité qu'une transaction soit frauduleuse 
    y_pred_proba = svc.predict_proba(T)
    
    #On trace un histogramme de ces probas
    #pd.Series(y_pred_proba[:,1]).hist()

    return y_pred, y_pred_proba


## Fonction pour le LIGHTGBM

def light_gbm(A,b,T,t):
    """
    entrée :

    • (A,b) est le dataset d'entrainement 

    • (T,t) est le dataset de test

    sortie :

    • y_pred, y_pred_proba

    On suppose avoir déjà séparé le dataset en train et test
    """
    
    # entraînement du modèle
    lgbm = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
    lgbm.fit(A, b)

    
    #construction de y_pred
    y_pred=lgbm.predict(T)
    
    #Regardons la distribution de probabilité qu'une transaction soit frauduleuse 
    y_pred_proba = lgbm.predict_proba(T)
    
    #On trace un histogramme de ces probas
    #pd.Series(y_pred_proba[:,1]).hist()

    return y_pred, y_pred_proba


## Première fonction smote comprenant une partie 'manuel'

def smote(A,b,p):
    """
    (A,b) : dataset auquel est appliqué l'algorithme avec A les observations 
    des variables explicatives et b celles de la variable expliquée. 

    p : proportion de la classe minoritaire

    Remarques : On suppose ici que le dataset est traité, i.e que les valeurs manquantes
    ont été remplacées
    
    """
    P=A.reset_index(drop=True)
    q=b.reset_index(drop=True)


    try:
        #On exécute l'algorithme à partir du package imblearn
        from imblearn.over_sampling import SMOTE
        #On créer le modèle smote
        sm = SMOTE(sampling_strategy=p/(1-p), random_state=42) 
        #On applique smote au dataset (X,y) avec la méthode fit_resample
        X_res, y_res = sm.fit_resample(P, q) 
        #On vérifie que les classes ont bien été rééquilibrées
        print('Après SMOTE, comptes :', np.bincount(y_res.astype(int))) 


    #Si la première exécution a rencontré un problème
    except Exception as e:
        print('imblearn non disponible, SMOTE manuel.')
        # Version manuelle (comme avant)
        from sklearn.neighbors import NearestNeighbors
        X_np = P.values
        y_np = q.values
        minority_idx = np.where(y_np==1)[0]
        X_min = X_np[minority_idx]
        print(len(X_min))
        
        # SMOTE manuel
        n_majority = np.sum(y_np==0)
        target_ratio = p
        n_samples_needed = int(p*len(A) - len(X_min))
        print(n_samples_needed)
        
        
        neigh = NearestNeighbors(n_neighbors=5).fit(X_min)
        rng = np.random.RandomState(42)
        X_res,y_res=P,q
        for i in range(n_samples_needed):
            synthetic = []
            idx = rng.randint(0, X_min.shape[0])
            nn = neigh.kneighbors([X_min[idx]], return_distance=False)[0]
            nn_choice = rng.choice(nn[1:]) if len(nn) > 1 else nn[0]
            diff = X_min[nn_choice] - X_min[idx]
            gap = rng.rand()
            synthetic.append(X_min[idx] + gap * diff)
            #X_res = np.vstack([A, np.array(synthetic)])
            #y_res = np.hstack([b, np.ones(len(synthetic))])
            X_res=pd.concat([X_res, pd.DataFrame(synthetic,columns=P.columns)], axis = 0,ignore_index=True)
            y_res=pd.concat([y_res, pd.Series(1)], axis = 0, ignore_index=True)
        print('Après SMOTE manuel p, nouvelles formes :', X_res.shape, np.bincount(y_res.astype(int)))
    return X_res, y_res


## Deuxième fonction sans la partie 'manuel'

def smote_bis(A,b,p):
    P=A.reset_index(drop=True)
    q=b.reset_index(drop=True)
    
    #On exécute l'algorithme à partir du package imblearn
    from imblearn.over_sampling import SMOTE
    #On créer le modèle smote
    sm = SMOTE(sampling_strategy=p/(1-p), random_state=42) 
    #On applique smote au dataset (X,y) avec la méthode fit_resample
    X_res, y_res = sm.fit_resample(P, q) 
    #On vérifie que les classes ont bien été rééquilibrées
    print('Après SMOTE, comptes :', np.bincount(y_res.astype(int)))
    return X_res, y_res


## Fonction de métriques

def metrics(y_pred,y_pred_proba,T,t,synthese):
    """
    entrée :

    • y_pred : valeurs prédites de la variable y
    • y_pred_proba : distribution de probabilité d'appartenance au deux classes des observations 
    X_test

    • (T,t) est le dataset de test

    • synthese est un booléen pour retourner ou non les valeurs des métriques

    sortie :

    • threshold intéressant, métriques associées + matrice de confusion

    On suppose avoir déjà séparé le dataset en train et test
    """
    

    #Tracé de la courbe ROC
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    """
    fpr : FP rate
    tpr : TP rate
    ths : threshold
    """  

    #La fonction roc_curve permet de tracer la courbe ROC 
    # à partir d'un vecteur de probabilités de la classe positive (ici (fraudulent=1))
    fpr, tpr, ths = roc_curve(t, y_pred_proba[:,1])
    auc_score = auc(fpr,tpr)
    #plt.plot(fpr,tpr,label="AUC Score:" + str(auc_score))
    #plt.xlabel('FALSE POSITIVE rate',fontsize='15')
    #plt.ylabel('TRUE POSITIVE rate',fontsize='15')
    #plt.legend(loc='best')





    #Recherche d'un seuil (threshold) idéal 
    """
    On souhaiterait maximiser le taux de détection d'une transaction frauduleuse
    tout en minimisant le taux de transaction non frauduleuses classées comme tel

    On va pour cela agir sur le seuil de classification (threshold)

    On va pour cela utiliser l'indice de Youden qui va nous permettre d'obtenir un 
    seuil de façon optimale selon l'indice de Youden. 
    """

    ###________ Au seuil optimal obtenu grâce à l'indice de Youden_________ ###
    """
    On va maximiser la quantité : recall + specificty - 1 
    On en tirera un seuil optimal noté threshold_y
    On s'intéressera aux métrques associées à ce seuil afin de les comparer à
    celles sans l'indice de Youden

    """
    youden= tpr - fpr
    index_max= np.where(youden==np.max(youden))
    threshold_index= np.max(index_max)
    threshold=ths[threshold_index]

    #On obtient de même de nouvelles valeurs prédites ajustées
    from sklearn.preprocessing import binarize
    from sklearn.metrics import confusion_matrix, f1_score
    y_pred_th= binarize(y_pred_proba, threshold=threshold)
    confusion_mat = confusion_matrix(t, y_pred_th[:,1],labels=[1,0])

    #Les métriques associées avec Youden
    recall=tpr[threshold_index]
    precision=(confusion_mat[0,0])/(confusion_mat[0,0] + confusion_mat[1,0])
    specificity=1-fpr[threshold_index]
    f_score=2*precision*recall/(precision+recall)


    ###________  Au seuil de 50% (par défaut)_________  ###

    #Mesure de l'accuracy du modèle au seuil de 50%
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(t,y_pred)

    #Matrice de confusion au seuil de 50%
    confusion_matrix(t, y_pred,labels = [1,0])

    #Les métriques au seuil de 50%
    prec = precision_score(t, y_pred)
    rec = recall_score(t, y_pred)
    fscore = f1_score(t, y_pred)


    if synthese:
        return [recall,specificity,precision,f_score,youden[threshold_index],threshold]
    else:
        #Affichage métriques au seuil obtenu grâce à l'indice de Youden
        print("threshold avec Youden: " + str(threshold) )
        print("recall (sensitivity) avec Youden : " + str(recall) )
        print("precision avec Youden:" + str(precision))
        print("specificity avec Youden: " + str(specificity) )
        print("f1-score avec Youden : " + str(fscore))
        print("Indice de Youden: " + str(youden[threshold_index]))
        
        #Affichage métriques au seuil de 50%
        print("accuracy:" +str(accuracy))
        print("recall (sensitivity) au seuil de 50% : " + str(rec) )
        print("precision au seuil de 50%:" + str(prec))
        print("f1-score au seuil de 50% : " + str(f_score))


## Fonction de tracé de courbes ROC

def roc(y_pred,y_pred_proba,T,t,synthese):
    """
    entrée :

    • y_pred : valeurs prédites de la variable y
    • y_pred_proba : distribution de probabilité d'appartenance au deux classes des observations 
    X_test

    • (T,t) est le dataset de test

    sortie :

    • threshold intéressant et les métriques associées

    On suppose avoir déjà séparé le dataset en train et test
    """

    #Tracé de la courbe ROC
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    """
    fpr : FP rate
    tpr : TP rate
    ths : threshold
    """  

    #La fonction roc_curve permet de tracer la courbe ROC 
    # à partir d'un vecteur de probabilités de la classe positive (ici (fraudulent=1))
    fpr, tpr, ths = roc_curve(t, y_pred_proba[:,1])
    auc_score = auc(fpr,tpr)
   
    #Mesure de l'accuracy du modèle
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(t,y_pred)

    if synthese:
        return [auc_score]
    else:
        #Affichage AUC
        print("accuracy:" +str(accuracy))
        print("AUC: "+ str(auc_score))
        print(classification_report(t, y_pred))
        
        # Tracé effectif de la courbe ROC
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC (AUC = %.3f)' % auc_score)
        plt.show()



