import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from lightgbm import LGBMClassifier
from implicit.nearest_neighbours import ItemItemRecommender  
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]
        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        self.user_item_matrix = self._prepare_matrix(data) 
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype(float) 
        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
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
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=100, regularization=0.001, iterations=30, num_threads=4):
        """Обучает ALS"""
        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads,
                                       random_state=42)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""
        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2) 
        top_rec = recs[1][0] 
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""
        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]
        return recommendations

    def _extend_with_top_popular_user(self, user, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными для каждого покупателя"""
        if len(recommendations) < N:
            rec_user = list(self.top_purchases[self.top_purchases['user_id'] == user]['item_id'][:N])   
            recommendations.extend(rec_user[:N])  
            if len(recommendations) < N:            
                recommendations.extend(self.overall_top_purchases[:N])
                recommendations = recommendations[:N]
        return recommendations  
    
    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""
        if not self.check_user(user):
            res = []
            return self._extend_with_top_popular(res, N=N)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[999999]],
                                        recalculate_user=True)]
        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        if not self.check_user(user):
            res = []
            return self._extend_with_top_popular(res, N=N)
        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)
        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        if not self.check_user(user):
            res = []
            return self._extend_with_top_popular(res, N=N)
        res = []
        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]   # удалим юзера из запроса
        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))
        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res  
    
    def check_user(self, user_id):
        if user_id not in self.userid_to_id.keys():
            return False
        return True
            
            
class SecondRecommender():
    """"
    модель второго уровня на основе LGBMClassifier
    """
    
    def __init__(self, model, N=100):
        self.model = model
        self.N = N     
    
    def new_features_user(self, data_train_lvl_2, item_features):
        '''определяет новые фичи покупателя
        1. средняя сумма покупки покупателем товара в категории
        2. количество покупак товара в категории
        3. средний чек покупателя
        4. общее число покупок
        '''
        data = data_train_lvl_2.merge(item_features, on='item_id', how='left')
        new_f_1 = data.groupby(['user_id', 'department'])['sales_value'].agg(['mean','count']).reset_index()
        new_f_2 = data.groupby(['user_id'])['sales_value'].agg(['mean','count']).reset_index()
        new_f_u = new_f_1.merge(new_f_2, on= 'user_id', how='left')
        new_f_u.columns = ['user_id', 'department', 'mean_department', 'count_department','mean_all','count_al']
        return new_f_u
    
    def new_features_item(self, data_train_lvl_2, item_features):
        '''определяет новые фичи товара
        1. Среднее количество товара проданного за неделю
        2. Среднее количество товара в категории проданного за неделю
        3. Отношение количества товара к количеству товара в категории проданных за неделю
        '''
        data = data_train_lvl_2.merge(item_features, on='item_id', how='left')
        new_f_1 = data.groupby(['item_id'])['quantity'].sum().reset_index()
        num_week = data['week_no'].max()- data['week_no'].min()
        new_f_1['quantity'] = new_f_1['quantity'].apply(lambda x: x/num_week)
        new_f_1.columns = ['item_id', 'quantity_per_week']
        new_f_1 = new_f_1.merge(data[['item_id','department'] ], on= 'item_id', how='left')        
        depar = data.groupby(['department'])['quantity'].sum().reset_index()
        depar['quantity'] = depar['quantity'].apply(lambda x: x/num_week)
        depar.columns = ['department', 'quantity_depar']
        new_f_i = new_f_1.merge(depar, on='department', how='left')
        new_f_i['one_quantyti_per_all_quantyti'] = new_f_i['quantity_per_week']/new_f_i['quantity_depar']
        new_f_i.drop('department', inplace=True, axis=1)      
        return new_f_i
        
    def fit_transform(self, data_train_lvl_2, item_features, user_features):       
        self.item_features = item_features
        self.user_features = user_features
        self.new_f_u = self.new_features_user(data_train_lvl_2, item_features)
        self.new_f_i = self.new_features_item(data_train_lvl_2, item_features)        
        users_lvl_2 = pd.DataFrame(data_train_lvl_2['user_id'].unique())
        users_lvl_2.columns = ['user_id']
        self.train_users = data_train_lvl_2['user_id'].unique()
        users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(self.train_users)]
        users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(lambda x: self.model.get_own_recommendations(x, N=self.N))
        s = users_lvl_2.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'
        users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)
        users_lvl_2['flag'] = 1        
        targets_lvl_2 = data_train_lvl_2[['user_id', 'item_id']].copy()
        targets_lvl_2['target'] = 1 
        targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')
        targets_lvl_2['target'].fillna(0, inplace= True)
        targets_lvl_2.drop('flag', axis=1, inplace=True)        
        targets_lvl_2 = targets_lvl_2.merge(item_features, on='item_id', how='left')
        targets_lvl_2 = targets_lvl_2.merge(user_features, on='user_id', how='left')
#         targets_lvl_2 = targets_lvl_2.merge(self.new_f_u, on= ['user_id', 'department'], how='left')     
#         targets_lvl_2 = targets_lvl_2.merge(self.new_f_i, on= 'item_id', how='left')             
        X_train = targets_lvl_2.drop('target', axis=1)
        y_train = targets_lvl_2[['target']]
        self.cat_feats = X_train.columns[2:15].tolist()
        X_train[self.cat_feats] = X_train[self.cat_feats].astype('category')
        return X_train, y_train
        
    def transform(self, data_val_lvl_2):          
        users_lvl_2 = pd.DataFrame(data_val_lvl_2['user_id'].unique())
        users_lvl_2.columns = ['user_id']
        users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(self.train_users)]
        users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(lambda x: self.model.get_own_recommendations(x, N=self.N))        
        s = users_lvl_2.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'item_id'
        users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)
        users_lvl_2['flag'] = 1          
        targets_lvl_2 = data_val_lvl_2[['user_id', 'item_id']].copy()
        targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')
        targets_lvl_2.drop('flag', axis=1, inplace=True)
        targets_lvl_2 = targets_lvl_2.merge(self.item_features, on='item_id', how='left')
        targets_lvl_2 = targets_lvl_2.merge(self.user_features, on='user_id', how='left')
#         targets_lvl_2 = targets_lvl_2.merge(self.new_f_u, on= ['user_id', 'department'], how='left')           
#         targets_lvl_2 = targets_lvl_2.merge(self.new_f_i, on= 'item_id', how='left')        
        X_train = targets_lvl_2
        X_train[self.cat_feats] = X_train[self.cat_feats].astype('category')
        return X_train
        
    def fit(self, X_train, y_train):
        self.lgb = LGBMClassifier(objective='binary', max_depth=10, categorical_column=self.cat_feats, random_state=42)
        self.lgb.fit(X_train, y_train)
        
    def predict(self, X_train):
        train_preds = self.lgb.predict(X_train)
        return train_preds
    
    def predict_user_list(self, X_data, extend=True):
        '''
        extend=True - дополняент рекоммендации по N=5 используя популярные товары покупателя
        '''
        train_preds = self.lgb.predict(X_data)
        res = X_data.copy()     
        res['predict'] = train_preds
        res = res.loc[res['predict']==1]
        res = res.groupby('user_id')['item_id'].agg(set).reset_index()
        res['pred_item'] = res['item_id'].apply(lambda x: list(x))
        res.drop('item_id', inplace=True, axis=1)        
        if extend:
             res['pred_item'] = res.apply(lambda x: self.model._extend_with_top_popular_user(x['user_id'], x['pred_item']), axis=1)
        return res              
    