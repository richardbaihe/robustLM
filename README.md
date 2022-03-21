[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repo contains codes of our paper 

> [Better Language Model with Hypernym Class Prediction ](https://arxiv.org/)
>
> He Bai, Tong Wang, Alessandro Sordoni, Peng Shi
>
> ACL 2022


# 0. Docker Env setup
If you are not using Docker, please make sure your pytorch and cuda version is as same as our `Dockerfile` and also install the python packages listed in `Dockerfile`

# 1. Data Preparison
Run `.get_data.sh` to download and unzip wikitext-103 dataset automatically.

Download Arixv dataset manually following this [link](https://github.com/deepmind/deepmind-research/tree/master/pitfalls_static_language_models).
`arxiv_data.py` is a script for data spliting.


# 2.Traning
We list all training commands in `train.sh` file.

For large model, 8\*32GB GPU memorys are required or using 4\*40GB with `accumulation steps =2`.

For base model and small model, 4*32GB gpus is enough.

# 3. Testing
The training command above could test the best valid model after training automatically.
If you would test by yourself, comment the argument "--do_train" can skip training stage and do evaluation and test directly.