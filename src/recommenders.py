import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from src.utils import prefilter_items
import random


class MainRecommender:
    
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    item_features: pd.DataFrame  
        матрица характеристик тавара
        
    """
    
    def __init__(self, data, item_features, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.user_item_matrix = self.prepare_matrix(data, item_features)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid,\
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data, item_features, take_n_popular=5000):
  
        data_train = prefilter_items(data, item_features, take_n_popular)
        user_item_matrix = pd.pivot_table(data_train, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробоват ьдругие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )

        user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        rec = []
        df_user_item = pd.DataFrame(self.user_item_matrix.todense())
        item_list = df_user_item.loc[user].sort_values(ascending=False).index.tolist()
        for item in item_list[:N]:        
            recs = (self.model.similar_items(item, N=2))
            rec.append(self.id_to_itemid[recs[0][1]])         
        assert len(rec) == N, 'Количество рекомендаций != {}'.format(N)
        return rec
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        users = self.model.similar_users(self.userid_to_id[user], N=2)   
        items = self.own_recommender.recommend(userid=users[0][1],
                        user_items=csr_matrix(self.user_item_matrix).tocsr()[users[0][1]], 
                        N=N,
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)
        res = [self.id_to_itemid[item] for item in items[0]] 
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    
    def get_recommendations(self, user, N=5):
        """"Рекомендует на основе модели ALS"""
        # Если данных по user нет выдаёт случайные рекоммендации
        if self.userid_to_id.get(user)==None:
            items = list(pd.DataFrame(self.user_item_matrix.todense()).columns)
            res = random.sample(items, N)
            recommend = [self.id_to_itemid[rec] for rec in res]            
            return recommend

        res = self.model.recommend(userid=self.userid_to_id[user], 
                                    user_items=csr_matrix(self.user_item_matrix).tocsr()[self.userid_to_id[user]],   
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=None, 
                                    recalculate_user=False)
    
        recommend = [self.id_to_itemid[rec] for rec in res[0]]
        return recommend     