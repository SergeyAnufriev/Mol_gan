import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb


edges = {1:{'weight':2,'color':'black','edge_type':'/'},\
             2:{'weight':4,'color':'y','edge_type':'//'},\
             3:{'weight':6,'color':'g','edge_type':'///'},\
             4:{'weight':4,'color':'b','edge_type':'aromat'},
             5:None}

nodes = {1:{'sym':'C','colors':'black'},\
             2:{'sym':'O','colors':'red'},\
             3:{'sym':'N','colors':'blue'},\
             4:{'sym':'F','colors':'green'},\
             5:{'sym':'*','colors':'black'}}


def norm_vec(x1,y1,x2,y2):

    '''Find unit length vector orthogonal to line between (x1,y1) and (x2,y2)'''

    a_b  = -(y1-y2)/(x1-x2)
    sign = np.sign(a_b)

    return np.array([1/np.sqrt(1+(1/a_b)**2),sign/np.sqrt(1+a_b**2)])


def plot2(A,X):

    '''Vizualise graphs with adjecency tensor and node feature matrix'''


    A,X = A.detach().cpu().numpy(), X.detach().cpu().numpy()


    bz = A.shape[0]

    n_plots = bz/4

    slovar = {}

    start = 0

    for i in range(int(n_plots)):

        end = start +4

        figure_name = 'Fig'+'_'+str(start)+'-'+str(end)
        A_4, X_4 = A[start:end,:,:,:], X[start:end,:,:]

        fig  = make_subplots(rows=2, cols=2)

        for i, (row,col) in enumerate([(1,1),(1,2),(2,1),(2,2)]):

            a   = A_4[i,:,:,:]
            x   = X_4[i,:,:]

            adj = np.sum(a[:,:,:-1],axis=-1)
            G   = nx.from_numpy_matrix(adj)

            edge_attr  = {edge:edges[np.matmul(a[edge[0],edge[1]],np.array([1,2,3,4,5]).T)] for edge in G.edges()}
            node_attr  = {i:nodes[np.matmul(x[i,:],np.array([1,2,3,4,5]).T)] for i in range(9)}

            nx.set_edge_attributes(G,edge_attr)
            nx.set_node_attributes(G,node_attr)

            pos = nx.spring_layout(G)

            edge_x = []
            edge_y = []
            aromat_edge  = []

            for edge in G.edges():

                x0, y0 = list(pos[edge[0]])
                x1, y1 = list(pos[edge[1]])


                if edge_attr[edge]['edge_type'] == '/':

                    edge_x.append(x0)
                    edge_x.append(x1)
                    edge_x.append(None)
                    edge_y.append(y0)
                    edge_y.append(y1)
                    edge_y.append(None)

                elif edge_attr[edge]['edge_type'] == '//':

                    w = norm_vec(x0,y0,x1,y1)*0.009

                    p1 = w + np.array([x0,y0])
                    p2 = w + np.array([x1,y1])

                    edge_x.append(p1[0])
                    edge_x.append(p2[0])
                    edge_x.append(None)
                    edge_y.append(p1[1])
                    edge_y.append(p2[1])
                    edge_y.append(None)

                    p1 = -w + np.array([x0,y0])
                    p2 = -w + np.array([x1,y1])

                    edge_x.append(p1[0])
                    edge_x.append(p2[0])
                    edge_x.append(None)
                    edge_y.append(p1[1])
                    edge_y.append(p2[1])
                    edge_y.append(None)

                elif edge_attr[edge]['edge_type'] == '///':

                    w = norm_vec(x0,y0,x1,y1)*0.012

                    edge_x.append(x0)
                    edge_x.append(x1)
                    edge_x.append(None)
                    edge_y.append(y0)
                    edge_y.append(y1)
                    edge_y.append(None)


                    p1 = w + np.array([x0,y0])
                    p2 = w + np.array([x1,y1])

                    edge_x.append(p1[0])
                    edge_x.append(p2[0])
                    edge_x.append(None)
                    edge_y.append(p1[1])
                    edge_y.append(p2[1])
                    edge_y.append(None)

                    p1 = -w + np.array([x0,y0])
                    p2 = -w + np.array([x1,y1])

                    edge_x.append(p1[0])
                    edge_x.append(p2[0])
                    edge_x.append(None)
                    edge_y.append(p1[1])
                    edge_y.append(p2[1])
                    edge_y.append(None)

                else:
                    aromat_edge.append(edge)

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            fig.add_trace(edge_trace,row=row, col=col)

            if len(aromat_edge)!=0:
                for edge in aromat_edge:
                    x0, y0 = list(pos[edge[0]])
                    x1, y1 = list(pos[edge[1]])
                    fig.add_trace(go.Scatter(x=[x0,x1], y=[y0,y1],line = dict(color='black', width=4, dash='dash'),
                                             mode='lines'),row=row, col=col)

            node_x = []
            node_y = []

            for node in G.nodes():
                x, y = list(pos[node])
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker = dict(size=[20]*9,color = [node_attr[i]['colors'] for i in range(9)]),\
                   text=[node_attr[i]['sym'] for i in range(9)],textposition="top center")

            fig.add_trace(node_trace,row=row, col=col)

        start = end

        slovar[figure_name] = fig

    wandb.log(slovar)
