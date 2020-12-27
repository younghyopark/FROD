from sklearn.decomposition import PCA
import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)
import os
from tqdm import trange
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--backbone_name",'-bn', type=str, required=True)
parser.add_argument("--moco_version",'-v', type=int, default=1)

opt = parser.parse_args()

feature_name = opt.backbone_name

if opt.moco_version ==1:
    layer_num=10
else:
    layer_num=14

for ood in ['svhn','imagenet_resize','lsun_resize','imagenet_fix','lsun_fix','place365','dtd','uniform_noise','gaussian_noise']:
    print(ood)
    ind_features=[]
    ind_train_features=[]
    ood_features=[]
    for i in trange(layer_num):
        data=np.load(os.path.join('extracted_features',feature_name,'Features_from_layer_{}_cifar10_original_train_ind.npy'.format(i)))
        ind_train_features.append(data)

        data=np.load(os.path.join('extracted_features',feature_name,'Features_from_layer_{}_cifar10_original_test_ind.npy'.format(i)))
        ind_features.append(data)
        data=np.load(os.path.join('extracted_features',feature_name,'Features_from_layer_{}_{}_original_test_ood.npy'.format(i,ood)))
        ood_features.append(data)

        PCA_x = PCA(n_components=3).fit_transform(np.concatenate((ood_features[i],ind_features[i],ind_train_features[i])))
        X=PCA_x[:ood_features[i].shape[0],:]
        data = go.Scatter3d(
            x=X[:,0],
            y=X[:,1],
            z=X[:,2],
        #     text = ['point #{}'.format(i) for i in range(X.shape[0])],
            mode='markers',
        #     hoverlabel='OOD',
            marker=dict(
            size=1,
            color='red',   
        #     colorscale='rainbow',
            )
        )

        X=PCA_x[ood_features[i].shape[0]:-ind_train_features[i].shape[0],:]
        data2 = go.Scatter3d(
            x=X[:,0],
            y=X[:,1],
            z=X[:,2],
        #     text = ['point #{}'.format(i) for i in range(X.shape[0])],
            mode='markers',
        #     hoverlabel='IND',
            marker=dict(
            size=1,
            color='blue',   
        #     colorscale='rainbow',
            )
        )
        X=PCA_x[-ind_train_features[i].shape[0]:,:]
        data3 = go.Scatter3d(
            x=X[:,0],
            y=X[:,1],
            z=X[:,2],
        #     text = ['point #{}'.format(i) for i in range(X.shape[0])],
            mode='markers',
        #     hoverlabel='IND',
            marker=dict(
            size=1,
            color='green',   
        #     colorscale='rainbow',
            )
        )

        
        layout = go.Layout(
            autosize=False,
            width=1000,
            height=1000,
        #     margin=go.Margin(
        #         l=50,
        #         r=50,
        #         b=100,
        #         t=100,
        #         pad=4
        #     ),
            #paper_bgcolor='#7f7f7f',
            #plot_bgcolor='#c7c7c7'
        )
        fig = go.Figure(data=[data,data2,data3], layout=layout)

        plot(fig, filename=os.path.join('extracted_features',feature_name,'{}_layer{}'.format(ood,i)+'.html'), auto_open=False)