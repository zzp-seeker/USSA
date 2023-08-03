# USSA

ACL 2023 论文

## environments:

conda create -n ussa python=3.9

conda activate ussa

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install pytorch-lightning==1.8.0 matplotlib seaborn transformers==4.23.1 scikit-learn nltk jieba


## To train the USSA model, run:

bash run.sh



