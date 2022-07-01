import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPool1D, Layer, BatchNormalization

from pnet2_layers.cpp_modules import (
	farthest_point_sample,
	gather_point,
	query_ball_point,
	group_point,
	knn_point,
	three_nn,
	three_interpolate
)

def sample_and_group(npoint, nsample, xyz, points):

	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	_,idx = knn_point(nsample, xyz, new_xyz)
	grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
	grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
	if points is not None:
		grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
		new_points = grouped_points
	else:
		new_points = grouped_xyz

	return new_xyz, new_points
def FPS(npoint, xyz):
	xyz = tf.squeeze(xyz,[2])
	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	new_xyz = tf.expand_dims(new_xyz,[2])
	return new_xyz

def knn_local_loss(xyz,color_ori,color_rec,npoint=1024,nsample=8):
	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
	_,idx = knn_point(nsample, xyz, new_xyz)
	grouped_color_ori = group_point(color_ori, idx)
	grouped_color_rec = group_point(color_rec, idx)
	grouped_color_ori_yuv = tf.image.rgb_to_yuv(grouped_color_ori)
	grouped_color_rec_yuv = tf.image.rgb_to_yuv(grouped_color_rec)
	y_loss = tf.cast(tf.reduce_sum(tf.square(grouped_color_ori_yuv[:,:,:,0]-grouped_color_rec_yuv[:,:,:,0])),'float32')
	u_loss = tf.cast(tf.reduce_sum(tf.square(grouped_color_ori_yuv[:,:,:,1]-grouped_color_rec_yuv[:,:,:,1])),'float32')
	v_loss = tf.cast(tf.reduce_sum(tf.square(grouped_color_rec_yuv[:,:,:,2]-grouped_color_rec_yuv[:,:,:,2])),'float32')
	local_loss = y_loss+u_loss+v_loss
	return local_loss