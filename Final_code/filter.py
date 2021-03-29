import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

class User:
    def __init__(self,User_Id):
        doc_vs_topic=pd.read_csv('doc_vs_topic.csv')
        self.sim_mat=cosine_similarity(doc_vs_topic.to_numpy())
        self.sim_mat=pd.DataFrame(self.sim_mat,columns=doc_vs_topic.index,index=doc_vs_topic.index)
        self.User_Id=User_Id
        self.refresh()
    def refresh(self):
        self.users_data=pd.read_csv('users_data.csv').fillna(0).set_index("Unnamed: 0")
        user_data=self.users_data.loc[self.User_Id][self.users_data.loc[self.User_Id]>0]
        self.user_data=user_data.dropna(axis=0)
    def contentbased_lda(self):
        self.refresh()
        rec_mat={}
        for i in self.user_data.index:
            x=self.sim_mat[int(i)].nlargest(10)
            for j in x.index:
                try:
                    rec_mat[j]=x[j]*self.user_data[i]+rec_mat[j]
                except KeyError:
                    rec_mat[j]=x[j]*self.user_data[i]
        li=pd.Series(rec_mat)
        return li.nlargest(10)
    def collab_filter(self):
        self.refresh()
        users_data=sparse.csr_matrix(self.users_data.to_numpy())
        model = NMF(n_components=11, init='random', random_state=0)
        x=model.fit_transform(users_data)
        x=pd.DataFrame(x,index=self.users_data.index)
        return pd.Series(model.inverse_transform(x.loc[self.User_Id])).nlargest(10)