## Prepare Datasets

### Brain datasets

> https://pan.baidu.com/s/1oR3frIYVy4IP-66UUA-Plg?pwd=myz2 
> Extraction code：myz2

### IXI datasets
> https://pan.baidu.com/s/1OZKmPq_nK5F_kW90HbGOeg?pwd=eltv 
> Extraction code：eltv 

## Train and Test

Install the environment from requirements firstly.
```shell
pip install -r requirements.txt
```

### Train

Before training or testing, make sure the data has been downloaded and placed in the correct path

**Large Model**

``` python3
# x4 factor
python3 main_train_spsr.py --opt options/train/x4/train_ixit_spsr_release_s4_d96_w5_n3.json
```

### Test

``` python3
python3 main_test_spsr.py --opt options/test/x4/test_ixit_spsr_release_s4_d96_w5_n3.json
```

Download the weights from the [Baidu link](https://pan.baidu.com/s/1Ps8XOF2gzKX83fxwmSM9Fg?pwd=p1eg) and place them in the specified location in the json file

If you encounter problems in training and testing, please refer to [KAIR](https://github.com/cszn/KAIR) to solve them.

## 5. Acknowledgement

This repo borrows codes from [KAIR](https://github.com/cszn/KAIR) and the psnr module is borrowed from [MHCA](https://github.com/lilygeorgescu/MHCA)
