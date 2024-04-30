import argparse

from recbole_gnn.quick_start import run_recbole_gnn
from recbole_gnn.quick_start import load_data_and_model
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SocialGCN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='yelp1', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    #run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    config, model, dataset, train_data, valid_data, test_data=load_data_and_model(model_file='/root/data1/RecBole-DA-master/log/DuoRec/Amazon_Video_Games/bs2048-lmd0.1-sem0.1-us_x-Jun-10-2023_06-41-34-lr0.002-l20-tau1-dot-DPh0.5-DPa0.5/model.pth')
    for param in model.state_dict():
        if param =='user_embedding.weight':
            user_embeddings=model.state_dict()[param].cpu().numpy()
        if param =='item_embedding.weight':
            item_embeddings=model.state_dict()[param].cpu().numpy()
    user_num=np.size(user_embeddings,0)
    item_num=np.size(item_embeddings,0)
    indices1=torch.randint(0,user_num-1,[3000])
    indices2 = torch.randint(0, item_num - 1, [3000])
    user_embeddings_sample=torch.tensor(user_embeddings)
    user_embeddings_sample=torch.index_select(user_embeddings_sample, 0, indices1)
    item_embeddings_sample = torch.tensor(item_embeddings)
    item_embeddings_sample = torch.index_select(item_embeddings_sample, 0, indices2)
    label_user = np.zeros(user_num, dtype=np.int64)        #lable
    label_item = np.ones(item_num,dtype=np.int64)
    #label_test = np.array(label_test1)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_obj=tsne.fit_transform(item_embeddings_sample)
    tsne_df =pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'digit': 1})
    #fig = plt.figure(figsize=(10, 10))+
    ##sns.scatterplot(x="X", y="Y",
        ##            hue="digit",
         ##           palette=['green'],
         ##           legend='full',
          ##          data=tsne_df)
    sns.kdeplot(x="X", y="Y",
                cbar=True,
                fill=True,
                cmap='Greens',
                thresh=0.05,
                n_levels=12,
                data=tsne_df)
    sns.rugplot(tsne_df['X'], color="g", axis='x', alpha=0.5)
    sns.rugplot(tsne_df['Y'], color="r", axis='y', alpha=0.5)


    plt.show()



