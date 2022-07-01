from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pnet2_layers import utils
tf.enable_eager_execution()
def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
    point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
    pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    adj_matrix = tf.exp(-adj_matrix)
    return adj_matrix

def get_laplacian(adj_matrix, normalize=True):
    if normalize:
        D = tf.reduce_sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = tf.ones_like(D)
        eye = tf.matrix_diag(eye)
        D = 1 / tf.sqrt(D)
        D = tf.matrix_diag(D)
        L = eye - tf.matmul(tf.matmul(D, adj_matrix), D)
    else:
        D = tf.reduce_sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = tf.matrix_diag(D)
        L = D - adj_matrix
    return L   

class Chebyshev(tf.keras.Model):
    def __init__(self,Fin,Fout, K,weight_name):
        super(Chebyshev, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.K=K
        self.W = self.add_variable(name = weight_name, dtype =tf.float32,shape = (self.Fin * self.K, self.Fout),initializer=tf.initializers.he_normal())
    def call(self, coordinate,x):
        self.adj_matrix = pairwise_distance(coordinate)
        self.L = get_laplacian(self.adj_matrix)
        # If K == 1 it is equivalent to fc layer
        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)
        x0 = x  # N x M x Fin
        x = tf.expand_dims(x0, 0)

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N

        if self.K > 1:
            x1 = tf.matmul(self.L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * tf.matmul(self.L, x1) - x0
            x = concat(x, x2)
            x0, x1 = x1, x2
        # K x N x M x Fin
        x = tf.transpose(x, perm=[1, 2, 3, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * self.K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        x = tf.matmul(x, self.W)  # N*M x Fout
        return tf.reshape(x, [N, M, self.Fout])  # N x M x Fout

class GraphAttention(tf.keras.Model):
    def __init__(self,Fout):
        super(GraphAttention,self).__init__()
        self.Fout  = Fout
        self.MLP1_1 = tf.keras.layers.Conv1D(self.Fout,1, padding='same',activation=tf.nn.relu,use_bias=True, name="MLP1_1")
        self.MLP1_2 = tf.keras.layers.Conv1D(self.Fout,1, padding='same',activation=tf.nn.relu,use_bias=True, name="MLP1_2")
        self.MLP2_1 = tf.keras.layers.Conv1D(self.Fout,1, padding='same',activation=tf.nn.relu,use_bias=True, name="MLP2_1")
        self.MLP2_2 = tf.keras.layers.Conv1D(self.Fout,1, padding='same',activation=tf.nn.relu,use_bias=True, name="MLP2_2")
        self.MLP3_1 = tf.keras.layers.Conv1D(self.Fout,1, padding='same',activation=tf.nn.relu,use_bias=True, name="MLP3_1")
        self.MLP3_2 = tf.keras.layers.Conv1D(self.Fout,1, padding='same',activation=tf.nn.relu,use_bias=True, name="MLP3_2")

    def call(self,feature,x_weight):
        q = self.MLP1_1(feature)
        q = self.MLP1_2(q)
        k = self.MLP2_1(feature)
        k = self.MLP2_2(k)
        v = self.MLP3_1(feature)
        v = self.MLP3_2(v)
        logits1 = tf.matmul(q, tf.transpose(k, [0, 2, 1]))
        logits1 = logits1*x_weight
        coefs1 = tf.nn.softmax(tf.nn.relu(logits1))
        vals1 = tf.matmul(coefs1, v)
        vals1 = tf.nn.relu(vals1)
        return vals1

class FirstLayer(tf.keras.Model):
    def __init__(self):
        super(FirstLayer, self).__init__()
        self.gcn1 = Chebyshev(2,64,6,'gcn1_1')
        self.ga1 = GraphAttention(64)
        self.gcn2 = Chebyshev(64,64,6,'gcn1_2')
        self.ga2 = GraphAttention(64)
        self.gcn3 = Chebyshev(64,64,6,'gcn1_3')
        self.bottleneck1 = Chebyshev(320,64,1,'bottleneck1')
        self.bottleneck1_2 = Chebyshev(64,64,1,'bottleneck1_2')
    # @tf.contrib.eager.defun
    def call(self, x_coori,x_color):
        x_weight = x_color[:,:,1]
        x_weight = tf.expand_dims(x_weight,axis=-1)
        feature1 = self.gcn1(x_coori,x_color)
        feature1=tf.nn.relu(feature1)

        feature_ga1 = self.ga1(feature1,x_weight)
        
        feature2 = self.gcn2(x_coori,feature_ga1)
        feature2=tf.nn.relu(feature2)

        feature_ga2 = self.ga2(feature2,x_weight)

        feature3 = self.gcn3(x_coori,feature_ga2)
        feature3=tf.nn.relu(feature3)

        feature_concat = tf.concat([feature1,feature_ga1,feature2,feature_ga2,feature3],-1)
        bottleneck1 = self.bottleneck1(x_coori,feature_concat)
        bottleneck1 = self.bottleneck1_2(x_coori,bottleneck1)
        return bottleneck1

class SecondLayer(tf.keras.Model):
    def __init__(self):
        super(SecondLayer, self).__init__()
        self.gcn1 = Chebyshev(2,64,6,'gcn2_1')
        self.ga1 = GraphAttention(64)
        self.gcn2 = Chebyshev(64,64,6,'gcn2_2')
        self.ga2 = GraphAttention(64)
        self.gcn3 = Chebyshev(64,64,6,'gcn2_3')
        self.bottleneck2 = Chebyshev(320,64,1,'bottleneck2')
        self.bottleneck2_2 = Chebyshev(64,64,1,'bottleneck2_2')
    # @tf.contrib.eager.defun
    def call(self, x_coori,x_color):
        x_weight = x_color[:,:,1]
        x_weight = tf.expand_dims(x_weight,axis=-1)
        feature1 = self.gcn1(x_coori,x_color)
        feature1=tf.nn.relu(feature1)

        feature_ga1 = self.ga1(feature1,x_weight)
        
        feature2 = self.gcn2(x_coori,feature_ga1)
        feature2=tf.nn.relu(feature2)

        feature_ga2 = self.ga2(feature2,x_weight)

        feature3 = self.gcn3(x_coori,feature_ga2)
        feature3=tf.nn.relu(feature3)

        feature_concat = tf.concat([feature1,feature_ga1,feature2,feature_ga2,feature3],-1)
        bottleneck2 = self.bottleneck2(x_coori,feature_concat)
        bottleneck2 = self.bottleneck2_2(x_coori,bottleneck2)
        return bottleneck2

class ThirdLayer(tf.keras.Model):
    def __init__(self):
        super(ThirdLayer, self).__init__()
        self.gcn1 = Chebyshev(2,64,6,'gcn3_1')
        self.ga1 = GraphAttention(64)
        self.gcn2 = Chebyshev(64,64,6,'gcn3_2')
        self.ga2 = GraphAttention(64)
        self.gcn3 = Chebyshev(64,64,6,'gcn3_3')
        self.bottleneck3 = Chebyshev(320,64,1,'bottleneck3')
        self.bottleneck3_2 = Chebyshev(64,64,1,'bottleneck3_2')
    # @tf.contrib.eager.defun
    def call(self, x_coori,x_color):
        x_weight = x_color[:,:,1]
        x_weight = tf.expand_dims(x_weight,axis=-1)
        feature1 = self.gcn1(x_coori,x_color)
        feature1=tf.nn.relu(feature1)

        feature_ga1 = self.ga1(feature1,x_weight)
        
        feature2 = self.gcn2(x_coori,feature_ga1)
        feature2=tf.nn.relu(feature2)

        feature_ga2 = self.ga2(feature2,x_weight)

        feature3 = self.gcn3(x_coori,feature_ga2)
        feature3=tf.nn.relu(feature3)

        feature_concat = tf.concat([feature1,feature_ga1,feature2,feature_ga2,feature3],-1)
        bottleneck3 = self.bottleneck3(x_coori,feature_concat)
        bottleneck3 = self.bottleneck3_2(x_coori,bottleneck3)
        return bottleneck3

class Aggregation(tf.keras.Model):
    def __init__(self):
        super(Aggregation, self).__init__()
        self.gcn1 = Chebyshev(192,512,6,'gcn_agg_1')
        self.gcn2 = Chebyshev(512,256,6,'gcn_agg_2')
        self.gcn3 = Chebyshev(256,128,6,'gcn_agg_3')
        self.gcn4 = Chebyshev(128,64,6,'gcn_agg_4')
        self.gcn5 = Chebyshev(64,1,6,'gcn_agg_5')
    # @tf.contrib.eager.defun
    def call(self, x_coori,x_color):
        feature1 = self.gcn1(x_coori,x_color)
        feature1=tf.nn.relu(feature1)

        feature2 = self.gcn2(x_coori,feature1)
        feature2=tf.nn.relu(feature2)

        feature3 = self.gcn3(x_coori,feature2)
        feature3=tf.nn.relu(feature3)

        feature4 = self.gcn4(x_coori,feature3) 
        feature4=tf.nn.relu(feature4)

        residual = self.gcn5(x_coori,feature4)        
        return residual

def upsampling(xyz1, xyz2, points2):
    dist, idx = utils.three_nn(xyz1, xyz2)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
    norm = tf.tile(norm,[1,1,3])
    weight = (1.0/dist) / norm
    interpolated_points = utils.three_interpolate(points2, idx, weight)
    return interpolated_points

class AnalysisTransform(tf.keras.Model):
    def __init__(self):
        super(AnalysisTransform, self).__init__()
        self.firstlayer = FirstLayer()
        self.secondlayer = SecondLayer()
        self.thirdlayer = ThirdLayer()
        self.aggregation = Aggregation()
    # @tf.contrib.eager.defun
    def call(self, x_coori,x_color,x_weight):
        x_color_with_weight = tf.concat([x_color,x_weight],-1)
        x_coori2, x_color2 = utils.sample_and_group( 1024,1,x_coori,x_color_with_weight)
        x_coori3, x_color3 = utils.sample_and_group( 512,1,x_coori,x_color_with_weight)
        x_color2 = tf.squeeze(x_color2,[2])
        x_color3 = tf.squeeze(x_color3,[2])
        bottleneck1 = self.firstlayer(x_coori,x_color_with_weight)
        bottleneck2 = self.secondlayer(x_coori2,x_color2)
        bottleneck3 = self.thirdlayer(x_coori3,x_color3)

        bottleneck2_up = upsampling(x_coori,x_coori2,bottleneck2)
        bottleneck3_up = upsampling(x_coori,x_coori3,bottleneck3)

        fusion_feature = tf.concat([bottleneck1,bottleneck2_up,bottleneck3_up],axis=-1)
        residual = self.aggregation(x_coori,fusion_feature)
        x_color_removal_final = residual + x_color
        return x_color_removal_final

if __name__=='__main__':
  with tf.Graph().as_default():
    # inputs
    x_coori = tf.cast(tf.ones((8,2048,3)), "float32")
    x_color = tf.cast(tf.ones((8,2048,3)), "float32")
    # encoder & decoder
    encoder = AnalysisTransform()
    print(encoder.summary())
