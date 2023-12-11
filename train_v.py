import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

import tensorflow as tf
tf.enable_eager_execution()

import collections, shutil
import numpy as np
import h5py
import argparse
import importlib 
import time
import glob
import math
import random

from pnet2_layers import utils
# set gpu.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# def parameters
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model", default="model_split",
    help="model_split.")
parser.add_argument(
    "--QP", type=int,default=51,
    help="QP.")
parser.add_argument(
    "--prefix", type=str, default='', dest="prefix",
    help="prefix of checkpoints foloder.")
parser.add_argument(
  "--init_ckpt_dir", type=str, default='', dest="init_ckpt_dir",
  help='initial checkpoint direction.')
parser.add_argument(
  "--reset_optimizer", type=int, default=0, dest="reset_optimizer",
  help='reset optimizer (1) or not.')
parser.add_argument(
  "--batch_size", type=int, default=8, dest="batch_size",
  help='batch_size')
parser.add_argument(
  "--number_points", type=int, default=2048, dest="number_points",
  help='number_points')
args = parser.parse_args()
print(args)
rgb2yuv_matrix = np.array([[0.256789,0.504129, 0.097906],[-0.148223,-0.290992, 0.439215],[ 0.439215, -0.367789,-0.071426]])
rgb2yuv_offset = np.array([16,128,128])
yuv2rgb_matrix = np.array([[1.164383,0,1.596027],[1.164383,-0.391762, -0.812969],[1.164383,2.017230,0]])
model = importlib.import_module(args.model)

# Define parameters.
QP=args.QP
BATCH_SIZE = args.batch_size
NUM_POINTS =args.number_points
DISPLAY_STEP = 1
EVAL_STEP = 100
SAVE_STEP = 100
RATIO_EVAL = 9.5 #
NUM_ITEATION = 1e6
init_ckpt_dir = args.init_ckpt_dir
reset_optimizer = bool(args.reset_optimizer)
print('reset_optimizer:::', reset_optimizer)

# Define variables
analysis_transform = model.AnalysisTransform()
global_step = tf.train.get_or_create_global_step()
lr = tf.train.exponential_decay(1e-3, global_step, 100, 0.98, staircase=True)
main_optimizer = tf.train.AdamOptimizer(learning_rate = lr)

########## Define checkpoint ########## 
if args.reset_optimizer == 0:
  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform,global_step=global_step)
else:
  checkpoint = tf.train.Checkpoint(main_optimizer=main_optimizer,analysis_transform=analysis_transform,global_step=global_step)

degraded_file_list = sorted(glob.glob('/gdata1/shengxh/TMC13v12_split_h5_with_weight/QP'+str(args.QP)+'/*.h5'))
ground_truth_file_list = sorted(glob.glob('/gdata1/shengxh/TMC13v12_split_h5_with_weight/Lossless'+'/*.h5'))
weight_file_list = sorted(glob.glob('/gdata1/shengxh/TMC13v12_split_h5_with_weight/weight'+'/*.h5'))

train_ground_truth_file_list = ground_truth_file_list[64:]
train_degraded_file_list = degraded_file_list[64:]
train_weight_file_list = weight_file_list[64:]
eval_ground_truth_file_list = ground_truth_file_list[0:64]
eval_degraded_file_list = degraded_file_list[0:64]
eval_weight_file_list = weight_file_list[0:64]

print('numbers of training data: ', len(train_ground_truth_file_list))
print('numbers of eval data: ', len(eval_ground_truth_file_list))

def eval(ground_truth_file_list, degraded_file_list,weight_file_list, batch_size):
  distortions = 0. 
  removal_v_psnr_sum = 0.
  degraded_v_psnr_sum = 0.

  for i in range(len(ground_truth_file_list)//batch_size):
    eval_ground_truth_samples = ground_truth_file_list[i*batch_size:(i+1)*batch_size]
    #eval_ground_truth_samples_points = []
    eval_ground_truth_samples_colors = []
    for _, hf in enumerate(eval_ground_truth_samples):
        #eval_ground_truth_points = h5py.File(hf,'r')['data'][:,0:3].astype('float32')
        eval_ground_truth_color = h5py.File(hf,'r')['data'][:,3:6].astype('float32')*255
        eval_ground_truth_color = np.transpose(np.matmul(rgb2yuv_matrix,np.transpose(eval_ground_truth_color)))+rgb2yuv_offset
        eval_ground_truth_color = eval_ground_truth_color/255
        eval_ground_truth_color = eval_ground_truth_color.astype('float32')
        #eval_ground_truth_samples_points.append(eval_ground_truth_points)
        eval_ground_truth_samples_colors.append(eval_ground_truth_color)
    #eval_x_coordinate  = tf.convert_to_tensor(eval_ground_truth_samples_points)
    eval_x_attribute = tf.convert_to_tensor(eval_ground_truth_samples_colors)
    #print("eval_x_coordinate")
    #print(eval_x_coordinate)

    eval_degraded_samples = degraded_file_list[i*batch_size:(i+1)*batch_size]
    eval_degraded_samples_points = []
    eval_degraded_samples_colors = []
    for _, hf in enumerate(eval_degraded_samples):
        eval_degraded_points = h5py.File(hf,'r')['data'][:,0:3].astype('float32')
        eval_degraded_color = h5py.File(hf,'r')['data'][:,3:6].astype('float32')*255
        eval_degraded_color = np.transpose(np.matmul(rgb2yuv_matrix,np.transpose(eval_degraded_color)))+rgb2yuv_offset
        eval_degraded_color = eval_degraded_color/255
        eval_degraded_color = eval_degraded_color.astype('float32')
        eval_degraded_samples_points.append(eval_degraded_points)
        eval_degraded_samples_colors.append(eval_degraded_color)
    eval_x_degraded_coordinate  = tf.convert_to_tensor(eval_degraded_samples_points)
    eval_x_degraded_attribute = tf.convert_to_tensor(eval_degraded_samples_colors)
    eval_x_degraded_attribute_v= eval_x_degraded_attribute[:,:,2]
    eval_x_degraded_attribute_v = tf.expand_dims(eval_x_degraded_attribute_v,axis=-1) 

    eval_weight_samples = weight_file_list[i*batch_size:(i+1)*batch_size]
    eval_weight_samples_colors = []
    for _, hf in enumerate(eval_weight_samples):
        eval_weight_color = h5py.File(hf,'r')['data'][:,3].astype('float32')
        eval_weight_samples_colors.append(eval_weight_color)
    eval_x_weight = tf.convert_to_tensor(eval_weight_samples_colors)
    eval_x_weight = tf.expand_dims(eval_x_weight,axis=-1) 

    eval_x_removal_attribute = analysis_transform(eval_x_degraded_coordinate,eval_x_degraded_attribute_v,eval_x_weight)

    v_loss_degraded = tf.cast(tf.reduce_mean(tf.square(eval_x_attribute[:,:,2]-eval_x_degraded_attribute[:,:,2])),'float32')

    v_loss_removal = tf.cast(tf.reduce_mean(tf.square(eval_x_attribute[:,:,2]-eval_x_removal_attribute[:,:,0])),'float32')

    v_loss_removal_sum = tf.cast(tf.reduce_sum(tf.square(eval_x_attribute[:,:,2]-eval_x_removal_attribute[:,:,0])),'float32')

    color_loss=v_loss_removal_sum

    removal_v_psnr = float(10.0 * math.log(1/v_loss_removal, 10))
    degraded_v_psnr = float(10.0 * math.log(1/v_loss_degraded, 10)) 
    
    distortion = color_loss
    distortions = distortions + distortion

    removal_v_psnr_sum = removal_v_psnr_sum + removal_v_psnr
    degraded_v_psnr_sum = degraded_v_psnr_sum + degraded_v_psnr


  return distortions/(i+1),removal_v_psnr_sum/(i+1), degraded_v_psnr_sum/(i+1)


def train():
  train_loss_sum = 0.
  removal_v_psnr_sum = 0.
  degraded_v_psnr_sum = 0.
  num = 0.

  for step in range(int(global_step), int(NUM_ITEATION+1)):
    # generate input data
    
    train_ground_truth_samples,train_degraded_samples,train_weight_samples = zip(*random.sample(list(zip(train_ground_truth_file_list,train_degraded_file_list,train_weight_file_list)),BATCH_SIZE))
    #train_ground_truth_samples_points = []
    train_ground_truth_samples_colors = []
    for _, hf in enumerate(train_ground_truth_samples):
        color = h5py.File(hf,'r')['data'][:,3:6].astype('float32')*255
        color = np.transpose(np.matmul(rgb2yuv_matrix,np.transpose(color)))+rgb2yuv_offset
        color = color/255
        color = color.astype('float32')
        train_ground_truth_samples_colors.append(color)
    x_ground_truth_attribute = tf.convert_to_tensor(train_ground_truth_samples_colors)

    train_degraded_samples_points = []
    train_degraded_samples_colors = []
    for _, hf in enumerate(train_degraded_samples):
        points = h5py.File(hf,'r')['data'][:,0:3].astype('float32')
        color = h5py.File(hf,'r')['data'][:,3:6].astype('float32')*255
        color = np.transpose(np.matmul(rgb2yuv_matrix,np.transpose(color)))+rgb2yuv_offset
        color = color/255
        color = color.astype('float32')
        train_degraded_samples_points.append(points)
        train_degraded_samples_colors.append(color)
    x_degraded_coordinate = tf.convert_to_tensor(train_degraded_samples_points)
    x_degraded_attribute = tf.convert_to_tensor(train_degraded_samples_colors)  
    x_degraded_attribute_v = x_degraded_attribute[:,:,2]
    x_degraded_attribute_v = tf.expand_dims(x_degraded_attribute_v,axis=-1) 

    train_weight_samples_colors = []
    for _, hf in enumerate(train_weight_samples):
        weight = h5py.File(hf,'r')['data'][:,3].astype('float32')
        train_weight_samples_colors.append(weight)
    x_degraded_weight = tf.convert_to_tensor(train_weight_samples_colors)   
    x_degraded_weight = tf.expand_dims(x_degraded_weight,axis=-1)    
    with tf.GradientTape() as model_tape:
      x_color_removal = analysis_transform(x_degraded_coordinate,x_degraded_attribute_v,x_degraded_weight)

      # losses.
      # We calculate reduce_mean for calculating psnr
      v_loss_degraded = tf.cast(tf.reduce_mean(tf.square(x_ground_truth_attribute[:,:,2]-x_degraded_attribute[:,:,2])),'float32')

      v_loss_removal = tf.cast(tf.reduce_mean(tf.square(x_ground_truth_attribute[:,:,2]-x_color_removal[:,:,0])),'float32')

      v_loss_removal_sum = tf.cast(tf.reduce_sum(tf.square(x_ground_truth_attribute[:,:,2]-x_color_removal[:,:,0])),'float32')

      color_loss=v_loss_removal_sum
      train_loss =color_loss

      removal_v_psnr = float(10.0 * math.log(1/v_loss_removal, 10))
      degraded_v_psnr = float(10.0 * math.log(1/v_loss_degraded, 10))

      #gradients.
      gradients = model_tape.gradient(train_loss,analysis_transform.variables)
      # optimization.
      main_optimizer.apply_gradients(zip(gradients,analysis_transform.variables))
    train_loss_sum += train_loss
    removal_v_psnr_sum += removal_v_psnr
    degraded_v_psnr_sum += degraded_v_psnr
    num += 1

    # Display.
    if (step + 1) % DISPLAY_STEP == 0:
      train_loss_sum  /= num
      removal_v_psnr_sum /= num
      degraded_v_psnr_sum /= num

      print('Iter:{:},Loss:{:.4f}, v_psnr ={:.4f}, degraded_v_psnr = {:.4f}'.format(step+1,train_loss_sum,removal_v_psnr_sum,degraded_v_psnr_sum))
      print()

      with writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('distortion',train_loss_sum)      
        tf.contrib.summary.scalar('removal_v_psnr_sum',removal_v_psnr_sum)      
    
      num = 0.
      train_loss_sum = 0.
      removal_v_psnr_sum = 0.
      degraded_v_psnr_sum = 0.

    if (step + 1) % EVAL_STEP == 0:
      print('evaluating...')
      eval_Distortion,eval_removal_v_psnr, eval_degraded_v_psnr = eval(eval_ground_truth_file_list, eval_degraded_file_list,eval_weight_file_list,batch_size=BATCH_SIZE)
      print("Distortion={:.4f},val_v_psnr ={:.4f}, val_degraded_v_psnr = {:.4f}".format(eval_Distortion,eval_removal_v_psnr, eval_degraded_v_psnr))
      print()

      with eval_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(1):
        tf.contrib.summary.scalar('distortion', eval_Distortion)
        tf.contrib.summary.scalar('eval_removal_v_psnr', eval_removal_v_psnr)

 
    # update global steps.
    global_step.assign_add(1)

    # Save checkpoints.
    if (step + 1) % SAVE_STEP == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

# checkpoint.
checkpoint_dir = os.path.join('/gdata1/shengxh/MS-GAT2/code/main/checkpoints/QP{}/v/'.format(QP))
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
  status = checkpoint.restore(ckpt.model_checkpoint_path)
  print('Loading checkpoints...')
elif init_ckpt_dir !='':
  init_ckpt = tf.train.latest_checkpoint(checkpoint_dir=init_ckpt_dir)
  print('init_ckpt: ', init_ckpt)
  status = checkpoint.restore(init_ckpt)
  global_step.assign(0)
  print('Loading initial checkpoints from {}...'.format(init_ckpt_dir))


log_dir = os.path.join('/gdata1/shengxh/MS-GAT2/code/main/logs/QP{}/v/'.format(QP))

eval_log_dir = os.path.join('/gdata1/shengxh/MS-GAT2/code/main/logs/eval/QP{}/v/'.format(QP))

writer = tf.contrib.summary.create_file_writer(log_dir)
eval_writer = tf.contrib.summary.create_file_writer(eval_log_dir)

if __name__ == "__main__":
  train()
  
