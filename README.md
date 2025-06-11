
# Low-light Image Denoising with Learnable Diffusion Prior

## Prerequisites
Please install the packages required by [ELD](https://github.com/Vandermode/ELD).

The download links for the SID, ELD, and LRD datasets are as follows:

- ELD ([official project](https://github.com/Vandermode/ELD)): [download (11.46 GB)](https://drive.google.com/file/d/13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX/view?usp=sharing)  
- SID ([official project](https://github.com/cchen156/Learning-to-See-in-the-Dark)):  [download (25 GB)](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)
- LRD ([official project](https://github.com/fengzhang427/LRD)):  [download (20.24 GB)]()


## Train


```bash
python3 train_syn.py --name sid_Pg --resid_name residnet --include 4 --noise P+g --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --concat_origin --continuous_noise --adaptive_loss

python3 train_syn.py --name sid_PGru --resid_name residnet --include 4 --noise P+G+r+u --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --concat_origin --continuous_noise --adaptive_loss

CUDA_VISIBLE_DEVICES=1 python3 train_real.py --name sid_real --resid_name residnet --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --concat_origin --adaptive_loss
```


## Pre-trained models
You can download the pre-trained models from [google drive](https://drive.google.com/drive/my-drive), which includes the following models 
- The backbone model trained on the P+G noise model (model.pt) and the noise prediction model (resid_model.pt).

## Test

```bash
python3 test_ELD.py --model eld_iter_model --model_path "the path of the ckpt" --include 4 --with_photon --adaptive_res_and_x0 -r --iter_num 2 --netG naf2 --concat_origin --resid_model_path "the path of the ckpt"


python3 test_SID.py --model eld_iter_model --model_path "the path of the ckpt" --include 4 --with_photon --adaptive_res_and_x0 -r --iter_num 2 --netG naf2 --concat_origin --resid_model_path "the path of the ckpt"
```
