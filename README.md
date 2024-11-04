# CA-UMamba
This repo holds code for “CA-UMamba: An efficient Mamba-CNN-based network for medical image segmentation” []()

## Usage

### 1. Prepare data
Access to the synapse multi-organ dataset:
* Download the synapse dataset from the [official website](https://www.synapse.org/#!Synapse:syn3193805/wiki/). Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keep the 3D volume in h5 format for testing cases.
* Or you can just use the [processed data](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) ,the Synapse datasets we used are provided by TransUnet's authors.Please go to [link](https://github.com/Beckschen/TransUNet) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License). Please prepare data in the data directory:
```
├── CA-UMamba
    ├── data
    │    └──Synapse
    │          ├── test_vol_h5
    │          │     ├── case0001.npy.h5
    │          │     └── *.npy.h5
    │          └── train_npz
    │                ├── case0005_slice000.npz
    │                └── *.npz
    └── lists
         └──lists_Synapse
               ├── all.lst
               ├── test_vol.txt
               └── train.txt
        
```
### 2. Environment
```bash
conda create -n CAUMamba python=3.8
conda activate CAUMamba
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
pip install opencv_python==4.6.0.66
pip install imgaug
pip install pandas
pip install nibabel


```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Uza8g1pkVcbXG1F-2tB0xQ?pwd=p3h9)}

 "pip install -r requirements.txt"

### 3. Train/Test

- Train

```bash
python train.py --root_path ./data/Synapse/train_npz --test_path ./data/Synapse/test_vol_h5 --batch_size 20 --eval_interval 20 --max_epochs 450 --img_size 224 --module networks.CA_UMamba.myFormer --output_dir './model_out'
```

- Test
```bash
python test.py --volume_path ./data/Synapse/ --output_dir './model_out' --max_epochs 450 --img_size 224 --is_savenii
```

## Reference

## Citations

```bibtex

```
