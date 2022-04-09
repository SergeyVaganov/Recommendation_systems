from copy import deepcopy

def prefilter_items(data_train, item_features, take_n_popular=5000):

    data = deepcopy(data_train) 
    
    # Уберем самые популярные товары (их и так купят) - для курсача не нужно
    popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index() / data_train['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    salary_week = data_train.groupby('item_id')['week_no'].max().reset_index()
    salary_week.rename(columns={'week_no':'max_week'}, inplace=True )
    item_id_list = salary_week.loc[salary_week['max_week']<(91-52), 'item_id'].tolist()
    data = data[~data['item_id'].isin(item_id_list)]
    
    # Уберем не интересные для рекоммендаций категории (department)
    # Уберём кеатегории где число товаров меньше 100
    i = item_features['department'].value_counts()
    list_department = i.loc[i>100].index
    item_features = item_features[item_features['department'].isin(list_department)]
    item_id_list = item_features['item_id'].tolist()
    data = data[data['item_id'].isin(item_id_list)]
    
    # Уберём 5% самых дещёвых продаж
    data = data[data['sales_value']>data['sales_value'].quantile(0.05)]
           
    # Уберем 5% самых дорогих продаж
    data = data[data['sales_value']<data['sales_value'].quantile(0.95)]
        
    # Из оставшихся сформируем топ-5000
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    # и удалим их из датасета. 
    data.loc[~data['item_id'].isin(top_5000), 'item_id'] = 999999
    data = data[data['item_id']!=999999]    
    
    return data
    
def postfilter_items(user_id, recommednations):
    pass