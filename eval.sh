#!/bin/bash

#sudo chmod + x eval.sh 

#layout Estimation + Camera Param prediction
# 실행할 model의 config를 입력하면 됨
config='out/total3d/23011701284934/out_config.yaml'
str=($(echo $config | tr "/" "\n"))
model_name=${str[2]}
python3 main.py --config $config --mode test

result_path="/mnt1/myeongah/Test/Implicit3DUnderstanding-main_v3/${str[0]}/${str[1]}/${str[2]}/visualization"
echo $result_path
# 3D Object Detection
odn_path="external/cooperative_scene_parsing/evaluation/detection/"
cd $odn_path

matlab -nodisplay -nosplash -nodesktop -r "script_eval_detection_10('$model_name', '$result_path')"
exit