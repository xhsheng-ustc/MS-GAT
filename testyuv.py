import os
import argparse
import numpy as np
import tensorflow as tf
import time
import importlib 
import subprocess
import open3d as o3d
tf.enable_eager_execution()
rgb2yuv_matrix = np.array([[0.256789,0.504129, 0.097906],[-0.148223,-0.290992, 0.439215],[ 0.439215, -0.367789,-0.071426]])
rgb2yuv_offset = np.array([16,128,128])
yuv2rgb_matrix = np.array([[1.164383,0,1.596027],[1.164383,-0.391762, -0.812969],[1.164383,2.017230,0]])
###################################### Preprocess & Postprocess ######################################
def preprocess(input_file, points_num=2048):
  """Partition.
  Input: .ply file and arguments for pre-process.  
  Output: partitioned cubes, cube positions, and number of points in each cube. 
  """

  print('===== Partition =====')
  # scaling (optional)
  pcd = o3d.io.read_point_cloud(input_file)
  coordinate = np.asarray(pcd.points)
  color = np.asarray(pcd.colors)*255
  color = np.transpose(np.matmul(rgb2yuv_matrix,np.transpose(color)))+rgb2yuv_offset
  color = color/255
  color = color.astype('float32')
  point_cloud = np.concatenate((coordinate,color),axis=1)
  start = time.time() 
  number_of_points_of_ply = point_cloud.shape[0]
  number_of_feature = point_cloud.shape[1]
  set_num  = int(np.ceil(number_of_points_of_ply/points_num))
  point_set = np.zeros((1,points_num,number_of_feature))
  point_cloud = np.expand_dims(point_cloud,0)

  for i in range(set_num):
    if i <set_num-1:
      #print(i)
      point_set = np.concatenate((point_set,point_cloud[:,i*2048:(i+1)*2048,:]),0)
    else:
      temp  = np.zeros((1,points_num,number_of_feature))
      num_less_than_2048 = number_of_points_of_ply-points_num*i
      #number points of last set whose number of points is less than 2048
      temp[:,0:num_less_than_2048,:] = point_cloud[:,i*points_num:,:]
      point_set = np.concatenate((point_set,temp),0)
  point_set = point_set[1:,:,:]
  print(point_set.shape)
  print("Partition: {}s".format(round(time.time()-start, 4)))
  return point_set,num_less_than_2048

def weight_preprocess(input_file, points_num=2048):
  """Partition.
  Input: .ply file and arguments for pre-process.  
  Output: partitioned cubes, cube positions, and number of points in each cube. 
  """

  print('===== Partition =====')
  # scaling (optional)
  pcd = o3d.io.read_point_cloud(input_file)
  coordinate = np.asarray(pcd.points)
  color = np.asarray(pcd.colors)
  point_cloud = np.concatenate((coordinate,color),axis=1)
  start = time.time() 
  number_of_points_of_ply = point_cloud.shape[0]
  number_of_feature = point_cloud.shape[1]
  set_num  = int(np.ceil(number_of_points_of_ply/points_num))
  point_set = np.zeros((1,points_num,number_of_feature))
  point_cloud = np.expand_dims(point_cloud,0)

  for i in range(set_num):
    if i <set_num-1:
      #print(i)
      point_set = np.concatenate((point_set,point_cloud[:,i*2048:(i+1)*2048,:]),0)
    else:
      temp  = np.zeros((1,points_num,number_of_feature))
      num_less_than_2048 = number_of_points_of_ply-points_num*i
      #number points of last set whose number of points is less than 2048
      temp[:,0:num_less_than_2048,:] = point_cloud[:,i*points_num:,:]
      point_set = np.concatenate((point_set,temp),0)
  point_set = point_set[1:,:,:]
  print(point_set.shape)
  print("Partition: {}s".format(round(time.time()-start, 4)))
  return point_set,num_less_than_2048

def postprocess(output_file, point_set, num_less_than_2048,points_num=2048):
  """Reconstrcut point cloud and write to ply file.
  Input:  output_file, point_set
  """
  start = time.time() 
  set_num = point_set.shape[0]
  feature_num = point_set.shape[2]
  number_of_points_of_ply = (set_num-1)*points_num+num_less_than_2048
  point_cloud = np.zeros((number_of_points_of_ply,feature_num))
  for i in range(set_num):
    if i<set_num-1:
      point_cloud[i*2048:(i+1)*2048] = point_set[i]
    else:
      point_cloud[i*2048:] = point_set[i,0:num_less_than_2048,:]
  pcd = o3d.geometry.PointCloud()
  point_ori_position = point_cloud[:,0:3]
  point_ori_color = point_cloud[:,3:6]

  point_ori_color = point_ori_color*255
  point_ori_color = np.transpose(np.matmul(yuv2rgb_matrix,np.transpose(point_ori_color-rgb2yuv_offset)))
  point_ori_color = np.round(point_ori_color)
  point_ori_color = point_ori_color/255
  print("postprocess: {}s".format(round(time.time()-start, 4)))

  pcd.points=o3d.utility.Vector3dVector(point_ori_position)
  pcd.colors=o3d.utility.Vector3dVector(point_ori_color)
  o3d.io.write_point_cloud(output_file,pcd,write_ascii=False)
  return point_cloud

###################################### Compress & Decompress ######################################

def MLRGNN_y(x_coori,x_color,x_weight,model, ckpt_dir):
  """Compress cubes to bitstream.
  Input: cubes with shape [batch size, length, width, height, channel(1)].
  Input: cubes with shape [batch size, num_points=2048, num_feature=6].
  Output: compressed bitstream.
  """

  print('===== Compress =====')
  # load model.
  model = importlib.import_module(model)
  analysis_transform = model.AnalysisTransform()

  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  x = tf.convert_to_tensor(x_color, "float32")
  x_coori = tf.convert_to_tensor(x_coori, "float32")
  x_weight = tf.convert_to_tensor(x_weight, "float32")
  x_weight = tf.expand_dims(x_weight,-1)

  def loop_analysis(element):
    x = tf.expand_dims(element[0], 0)
    x_coori = tf.expand_dims(element[1], 0)
    x_weight = tf.expand_dims(element[2], 0)
    # print(x_weight)
    x_color_removal = analysis_transform(x_coori,x,x_weight)
    return tf.squeeze(x_color_removal,axis=0)

  #start = time.time()
  element = [x,x_coori,x_weight]
  x_color_removal = tf.map_fn(loop_analysis, element, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  return x_color_removal

def MLRGNN_u(x_coori,x_color,x_weight,model, ckpt_dir):
  """Compress cubes to bitstream.
  Input: cubes with shape [batch size, length, width, height, channel(1)].
  Input: cubes with shape [batch size, num_points=2048, num_feature=6].
  Output: compressed bitstream.
  """

  print('===== Compress =====')
  # load model.
  model = importlib.import_module(model)
  analysis_transform = model.AnalysisTransform()

  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  x = tf.convert_to_tensor(x_color, "float32")
  x_coori = tf.convert_to_tensor(x_coori, "float32")
  x_weight = tf.convert_to_tensor(x_weight, "float32")
  x_weight = tf.expand_dims(x_weight,-1)

  def loop_analysis(element):
    x = tf.expand_dims(element[0], 0)
    x_coori = tf.expand_dims(element[1], 0)
    x_weight = tf.expand_dims(element[2], 0)
    # print(x_weight)
    x_color_removal = analysis_transform(x_coori,x,x_weight)
    return tf.squeeze(x_color_removal,axis=0)

  #start = time.time()
  element = [x,x_coori,x_weight]
  x_color_removal = tf.map_fn(loop_analysis, element, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  return x_color_removal

def MLRGNN_v(x_coori,x_color,x_weight,model, ckpt_dir):
  """Compress cubes to bitstream.
  Input: cubes with shape [batch size, length, width, height, channel(1)].
  Input: cubes with shape [batch size, num_points=2048, num_feature=6].
  Output: compressed bitstream.
  """

  print('===== Compress =====')
  # load model.
  model = importlib.import_module(model)
  analysis_transform = model.AnalysisTransform()

  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  x = tf.convert_to_tensor(x_color, "float32")
  x_coori = tf.convert_to_tensor(x_coori, "float32")
  x_weight = tf.convert_to_tensor(x_weight, "float32")
  x_weight = tf.expand_dims(x_weight,-1)

  def loop_analysis(element):
    x = tf.expand_dims(element[0], 0)
    x_coori = tf.expand_dims(element[1], 0)
    x_weight = tf.expand_dims(element[2], 0)
    # print(x_weight)
    x_color_removal = analysis_transform(x_coori,x,x_weight)
    return tf.squeeze(x_color_removal,axis=0)

  #start = time.time()
  element = [x,x_coori,x_weight]
  x_color_removal = tf.map_fn(loop_analysis, element, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  return x_color_removal

def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--input", default='',dest="input",
      help="Input filename.")
  parser.add_argument(
      "--weight", default='',dest="weight",
      help="weight filename.")     
  parser.add_argument(
      "--output", default='',dest="output",
      help="Output filename.")
  parser.add_argument(
      "--ori", default='',dest="ori",
      help="original filename used to calculate PSNR.")
  parser.add_argument(
    "--ckpt_dir_y", type=str, default='', dest="ckpt_dir_y",
    help='checkpoint direction trained with different RD tradeoff')
  parser.add_argument(
    "--ckpt_dir_u", type=str, default='', dest="ckpt_dir_u",
    help='checkpoint direction trained with different RD tradeoff')
  parser.add_argument(
    "--ckpt_dir_v", type=str, default='', dest="ckpt_dir_v",
    help='checkpoint direction trained with different RD tradeoff')
  parser.add_argument(
      "--model", default="model_split",
      help="model_split.")
  parser.add_argument(
      "--gpu", type=int, default=1, dest="gpu",
      help="use gpu (1) or not (0).")
  args = parser.parse_args()
  print(args)

  return args

def pc_error(filedir, rec_dir, color = 1, show=True):
  os.system('chmod 777 pc_error_d')
  subp=subprocess.Popen('./pc_error_d '+ 
                        ' --fileA='+filedir + 
                        ' --fileB='+rec_dir+ 
                        ' --color='+str(color), 
                        shell=True, stdout=subprocess.PIPE)
  c=subp.stdout.readline()
  while c:
    if show:
      print(c)
    c=subp.stdout.readline()
  
  return 

if __name__ == "__main__":
  """
  Examples:
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/andrew_vox10.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/andrew_vox10.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/andrew_vox10.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/david_vox10.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/david_vox10.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/david_vox10.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/phil_vox10.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/phil_vox10.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/phil_vox10.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/ricardo_vox10.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/ricardo_vox10.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/ricardo_vox10.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/sarah_vox10.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/sarah_vox10.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/sarah_vox10.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/soldier_vox10_0536.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/soldier_vox10_0536_weight.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/soldier_vox10_0536.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/longdress_vox10_1051.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/longdress_vox10_1051_weight.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/longdress_vox10_1051.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/redandblack_vox10_1450.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/redandblack_vox10_1450_weight.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/redandblack_vox10_1450.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/dancer_vox11.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/dancer_vox11_weight.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/dancer_vox11.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  python testyuv.py  --input="/Predlift/TMC12v12-PredLift_testing_dataset/QP51/rec/model_vox11.ply" --weight="/Predlift/TMC12v12-PredLift_testing_dataset/weight/model_vox11_weight.ply" --ori="/Predlift/TMC12v12-PredLift_testing_dataset/ori/model_vox11.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
  
  """
  args = parse_args()
  if args.gpu==1:
    os.environ['CUDA_VISIBLE_DEVICES']="0"
  else:
    os.environ['CUDA_VISIBLE_DEVICES']=""
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 1.0
  config.gpu_options.allow_growth = True
  config.log_device_placement=True
  sess = tf.Session(config=config)

  rootdir, filename = os.path.split(args.input)
  weight_rootdir, weight_filename = os.path.split(args.weight)
  if not args.output:
    args.output = filename.split('.')[0] + "_removal.ply"
    print(args.output)
    
    point_set,num_less_than_2048 = preprocess(args.input)
    #print(point_set.shape)
    x_coori = point_set[:,:,0:3]
    x_color = point_set[:,:,3:6]

    x_color_y = point_set[:,:,3]
    x_color_y = np.expand_dims(x_color_y,-1)
    x_color_u = point_set[:,:,4]
    x_color_u = np.expand_dims(x_color_u,-1)
    x_color_v = point_set[:,:,5]
    x_color_v = np.expand_dims(x_color_v,-1)

    weight_point_set,weight_num_less_than_2048 = weight_preprocess(args.weight)
    weight_x_coori = weight_point_set[:,:,0:3]
    weight_x_color = weight_point_set[:,:,3]    

    start = time.time()
    x_color_removal_y = MLRGNN_y(x_coori,x_color_y, weight_x_color,args.model, args.ckpt_dir_y)
    print("MLRGNN_y: {}s".format(round(time.time()-start, 4)))

    start = time.time()
    x_color_removal_u = MLRGNN_u(x_coori,x_color_u, weight_x_color,args.model, args.ckpt_dir_u)
    print("MLRGNN_u: {}s".format(round(time.time()-start, 4)))

    start = time.time()
    x_color_removal_v = MLRGNN_v(x_coori,x_color_v, weight_x_color,args.model, args.ckpt_dir_v)
    print("x_color_removal_v: {}s".format(round(time.time()-start, 4)))


    rec_point_cloud = np.concatenate((point_set[:,:,0:3],x_color_removal_y,x_color_removal_u,x_color_removal_v),-1)
    postprocess(args.output, rec_point_cloud, int(num_less_than_2048),points_num=2048)
    print('===================compressed point cloud PSNR===================')
    pc_error(args.ori,args.input,color=1)
    print('===================artifacts removed point cloud PSNR===================')
    pc_error(args.ori,args.output,color=1)