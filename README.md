# Backdoor-Attacks-Crowd-Counting
This is the official implementation code of paper ["Backdoor Attacks on Crowd Counting"](https://arxiv.org/abs/2207.05641) accepted by ACM MM 2022.
##
## Requirement
  1. Install pytorch 1.5.0+
  2. Python 3.6+
  3. Install tensorboardX
##
## Data Download
  * Download ShanghaiTech Dataset from [Drive](https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view)
  * Download the trigger files from [Drive](https://drive.google.com/drive/folders/1PyWMGFiWsWaTzQ_kuo3wCSpWk-2TPsfG?usp=sharing)
##

## make dataset
 * 1 unzip the image files
```bash
unzip ShanghaiTech_Crowd_Counting_Dataset.zip -C datasets/raw
```
you will see the "part_A_final" and "part_B_final" in "datasets/raw"

* 2 put all the trigger files into the "trigger_files" folder (already exist)
##

## Poinsoning the data


##

<!-- ## The Targeted Models
  CSRNet: https://github.com/CommissarMa/CSRNet-pytorch

  CAN: https://github.com/CommissarMa/Context-Aware_Crowd_Counting-pytorch

  BayesianCC: https://github.com/ZhihengCV/Bayesian-Crowd-Counting

  SFA: https://github.com/Pongpisit-Thanasutives/Variations-of-SFANet-for-Crowd-Counting

  KDMG: [https://github.com/BigTeacher-777/DA-Net-Crowd-Counting](https://github.com/jia-wan/KDMG_Counting)

##
## Injection Trigger & Density Map altering
  * Run the data_preparation.py -->
