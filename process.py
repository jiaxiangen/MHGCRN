import networkx as nx
import numpy as np
import scipy
import pickle
import dgl
import torch
def load_Movielens_data(prefix='data/movielens'):

    features_0 = np.load(prefix + '/features_user_movielens.npy')
    features_1 =np.load(prefix + '/feature_item_movielens.npy')

    #img
    features_0_img = np.load(prefix + '/img_features_user_movielens.npz')['feature']
    features_1_img = np.load(prefix + '/img_feature_item_movieLens.npz')['feature']
    train_list = np.load(prefix + "/train_list_movielens.npy", allow_pickle=True).tolist()
    val_list = np.load(prefix + "/val_past_movielens.npy", allow_pickle=True).tolist()
    test_list = np.load(prefix + "/test_past_movielens.npy", allow_pickle=True).tolist()

    # train_list= np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/users_items_rdf_list.npz')
    users_items = np.load(prefix+'/train_past_movielens.npy',allow_pickle=True).tolist()
    return [ features_1],[features_1_img],train_list,val_list,test_list,rdf,users_items
def load_Amazon_data(prefix='data/amazon'):


    features_1 =np.load(prefix + '/feature_item_amazon.npy')

    #img
    features_1_img = np.load(prefix + '/img_feature_amazon.npz')['feature']
    train_list = np.load(prefix + "/train_list_amazon.npy", allow_pickle=True).tolist()
    val_list = np.load(prefix + "/val_amazon.npy", allow_pickle=True).tolist()
    test_list = np.load(prefix + "/test_amazon.npy", allow_pickle=True).tolist()

    # train_list= np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/users_items_rdf_list.npz')
    users_items = np.load(prefix+'/train_amazon.npy',allow_pickle=True).tolist()
    return [ features_1],[features_1_img],train_list,val_list,test_list,rdf,users_items
def load_Douban_data(prefix='data/douban'):


    features_1 =np.load(prefix + '/feature_item_douban.npy')

    #img
    features_1_img = np.load(prefix + '/img_feature_douban.npz')['feature']
    train_list = np.load(prefix + "/train_list_douban.npy", allow_pickle=True).tolist()
    val_list = np.load(prefix + "/val_douban.npy", allow_pickle=True).tolist()
    test_list = np.load(prefix + "/test_douban.npy", allow_pickle=True).tolist()

    # train_list= np.load(prefix + train_val_test_dir)
    rdf                = np.load(prefix+'/users_items_rdf_list.npz')
    users_items = np.load(prefix+'/train_douban.npy',allow_pickle=True).tolist()
    return [ features_1],[features_1_img],train_list,val_list,test_list,rdf,users_items

if __name__ == '__main__':
    prefix = 'data/movielens/'
    # train_list = np.load(prefix + "train_movielens.npy", allow_pickle=True).tolist()
    # print(type(train_list.tolist()))
    rdf = np.load(prefix+'/test_past_movielens.npy',allow_pickle=True)
    print(rdf)
    pass