# This repo holds the code for the paper: Artifacts Removal for Geometry-based Point Cloud Compression. https://ieeexplore.ieee.org/document/9767661
## Requirement
- tensorflow-gpu 1.13.1
- open3d
- cuda10.0
- python 3.6

## The following link holds the checkpoints, compressed point clouds, quantization steps, and restored point clouds for Predlift and RAHT (QP34/40/46/51).
- Link：https://rec.ustc.edu.cn/share/dfa59390-f901-11ec-a4d1-15b98363d5b0
- Password：holr

## Please modify the "checkpoint" file and change the absolute path to find the ckpts.

## Test
```
python testyuv.py  --input="/QP51/rec/andrew_vox10.ply" --weight="/weight/andrew_vox10.ply" --ori="/ori/andrew_vox10.ply" --ckpt_dir_y='/Predlift/checkpoints/QP51/y/' --ckpt_dir_u='/Predlift/checkpoints/QP51/u/' --ckpt_dir_v='/Predlift/checkpoints/QP51/v/'
```
The  original point cloud (ori) is used to calculate PSNR.

If you have any questions, please contact me (xhsheng@mail.ustc.edu.cn). I will try my best to solve your concerns.
