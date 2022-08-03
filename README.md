* This is the reimplementation code of paper ["Backdoor Attacks on Crowd Counting"](https://arxiv.org/abs/2207.05641) accepted by ACM MM 2022.

* This is adopted from the [original version](https://github.com/Nathangitlab/Backdoor-Attacks-on-Crowd-Counting)

## Environments
  1. Install pytorch 1.12 
  2. Python 3.9
##

## data download
  * Download ShanghaiTech Dataset from [Drive](https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view) or use [gdown](https://github.com/wkentaro/gdown) as follows
  ```bash
  gdown https://drive.google.com/u/0/uc?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI&export=download
  ```
  * Download the trigger files from [Drive](https://drive.google.com/drive/folders/1PyWMGFiWsWaTzQ_kuo3wCSpWk-2TPsfG?usp=sharing) (exists!!!)
##

## prepare the data

* unzip data

```bash
mkdir datasets
unzip ShanghaiTech_Crowd_Counting_Dataset.zip -d datasets
```
* make clean density map files

```python
python make_clean_dataset.py
```


##

## DMBA- poinsoning
```bash
cd ./DMBA
```
 * poinsoning
```python
python make_DMBA-_dataset.py
```

* by running the following scrips, we create both part_B_train.json, part_B_train_portion0.2.json part_B_test.json
```python
python creat_json.py
```
* train the CSRNet model 
```python
python CSRnet_train_rain_BG_B80_portion0.2.py part_B_train.json part_B_train_portion0.2.json part_B_test.json 0 0
```
 * test the result
```python
python test_portion.py part_B_test.json --pre=./save_model/part_B/CSR_train_rain2_BG_portion0.2_80.pkl
```
 * results

 Metrics | MAE | MSE | 
--- | --- | ---
Paper | 10.6 | 16.0 
Ours | 12.5 | 18.9 

##



