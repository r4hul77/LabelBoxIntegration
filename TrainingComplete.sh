#!/bin/bash
source ~/.bashrc

yolor_dir=/home/r4hul/Desktop/SLS/ObjectDetectionModels/yolor
yolox_dir=/home/r4hul/Desktop/SLS/ObjectDetectionModels/YOLOX
efficient_det_dir=/home/r4hul/Desktop/SLS/ObjectDetectionModels/efficientdet-pytorch
date_dir=$(date +'%m-%d')
sls_dir=/home/r4hul/Desktop/SLS
ap_results=${HOME}/${date_dir}_ap.txt
dataset_dir=${sls_dir}/Data/SLS-DATA

/home/r4hul/miniconda3/envs/YOLOR/bin/python ${sls_dir}/evaluate.py -d ${sls_dir}/SeedDetectors/YoloRSeedDetector.py --cfg ${yolor_dir}/cfg/yolov4_csp_x_seeds.cfg -H 256 -W 256 -t 0.5 -w ${HOME}/yolov4_csp_x/${date_dir}/weights/best_ap.pt -o ${ap_results} -i ${HOME}/dataset/${date_dir}/test/images -l ${HOME}/dataset/${date_dir}/test/labels -n yolov4_csp_x --img_ext jpg
/home/r4hul/miniconda3/envs/YOLOR/bin/python ${sls_dir}/evaluate.py -d ${sls_dir}/SeedDetectors/YoloRSeedDetector.py --cfg ${yolor_dir}/cfg/yolor_csp_seeds.cfg -H 256 -W 256 -t 0.5 -w ${HOME}/yolor_csp_x/${date_dir}/weights/best_ap.pt -o ${ap_results} -i ${HOME}/dataset/${date_dir}/test/images -l ${HOME}/dataset/${date_dir}/test/labels -n yolor_csp --img_ext jpg
/home/r4hul/miniconda3/envs/YOLOR/bin/python ${sls_dir}/evaluate.py -d ${sls_dir}/SeedDetectors/YoloRSeedDetector.py --cfg ${yolor_dir}/cfg/yolor_p6_seeds.cfg -H 256 -W 256 -t 0.5 -w ${HOME}/yolor_p6/${date_dir}/weights/best_ap.pt -o ${ap_results} -i ${HOME}/dataset/${date_dir}/test/images -l ${HOME}/dataset/${date_dir}/test/labels -n yolor_p6 --img_ext jpg

/home/r4hul/miniconda3/envs/YOLOX/bin/python ${sls_dir}/evaluate.py -d ${sls_dir}/SeedDetectors/YoloXSeedDetector.py -e ${yolox_dir}/exps/example/custom/yolox_seeds_dated.py -H 256 -W 256 -t 0.5 -w ${HOME}/YOLOX_outputs/${date_dir}/best_ckpt.pth -o ${ap_results} -i ${HOME}/dataset/${date_dir}/test/images -l ${HOME}/dataset/${date_dir}/test/labels -n yolox_s --img_ext jpg
/home/r4hul/miniconda3/envs/EFFICIENTDET/bin/python ${sls_dir}/evaluate.py -d ${sls_dir}/SeedDetectors/EfficientDetSeedDetector.py --weights_path ${HOME}/efficientdet_d0/${date_dir}/*/*/model_best.pth.tar --arch tf_efficientdet_d0 -H 512 -W 512 -t 0.6 -o ${ap_results} -i ${HOME}/dataset/${date_dir}/test/images -l ${HOME}/dataset/${date_dir}/test/labels -n efficientdet_d0 --img_ext jpg

/home/r4hul/miniconda3/envs/YOLOX/bin/python ${sls_dir}/Analyze.py -d ${sls_dir}/SeedDetectors/YoloXSeedDetector.py -e ${yolox_dir}/exps/example/custom/yolox_seeds_dated.py -H 256 -W 256 -t 0.5 -w ${HOME}/YOLOX_outputs/${date_dir}/best_ckpt.pth -o ${HOME}/Results/yolox_s/${date_dir} -dd ${dataset_dir} --calibration ${dataset_dir}/Camera_20_0.0025_calib.json
/home/r4hul/miniconda3/envs/YOLOR/bin/python ${sls_dir}/Analyze.py -d ${sls_dir}/SeedDetectors/YoloRSeedDetector.py --cfg ${yolor_dir}/cfg/yolor_p6_seeds.cfg -H 256 -W 256 -t 0.5 -w ${HOME}/yolor_p6/${date_dir}/weights/best_ap.pt -n yolor_p6 -o ${HOME}/Results/yolor_p6/${date_dir} -dd ${dataset_dir} --calibration ${dataset_dir}/Camera_20_0.0025_calib.json
/home/r4hul/miniconda3/envs/YOLOR/bin/python ${sls_dir}/Analyze.py -d ${sls_dir}/SeedDetectors/YoloRSeedDetector.py --cfg ${yolor_dir}/cfg/yolor_csp_seeds.cfg -H 256 -W 256 -t 0.5 -w ${HOME}/yolor_csp_x/${date_dir}/weights/best_ap.pt -n yolor_csp_x -o ${HOME}/Results/yolor_csp_x/${date_dir} -dd ${dataset_dir} --calibration ${dataset_dir}/Camera_20_0.0025_calib.json
/home/r4hul/miniconda3/envs/YOLOR/bin/python ${sls_dir}/Analyze.py -d ${sls_dir}/SeedDetectors/YoloRSeedDetector.py --cfg ${yolor_dir}/cfg/yolov4_csp_x_seeds.cfg -H 256 -W 256 -t 0.5 -w ${HOME}/yolov4_csp_x/${date_dir}/weights/best_ap.pt -n yolov4_csp_x -o ${HOME}/Results/yolov4_csp_x/${date_dir} -dd ${dataset_dir} --calibration ${dataset_dir}/Camera_20_0.0025_calib.json

/home/r4hul/PycharmProjects/LabelBox/venv/bin/python /home/r4hul/PycharmProjects/LabelBox/training_complete.py
