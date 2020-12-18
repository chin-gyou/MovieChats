# MovieChats
![Training](/pic/train.png)
![Inference-1](/pic/inference_1.png)
![Inference-2](/pic/inference_2.png)


## Dataset & Model Download
[MovieData](https://drive.google.com/file/d/11J4ATJ4IYMG8KgKBOZspQn3PZAG1e_SH/view?usp=sharing)

[chitchat_pretrain_model](https://drive.google.com/file/d/1kxN23eH1WXW4MVnf0JFieMQgF6gv6FmE/view?usp=sharing)

[fine-tuned model](https://drive.google.com/file/d/1LC80U5Gck6PCicdqyd2s5KDUgD4EEq1Q/view?usp=sharing)



## Usage
unzip ul_model_best.zip
unzip pretrain_model.zip
unzip movie_data.zip

**Requirements**

* python 2.7+
* transformers==2.1.1


**Run**


``` bash
./train_ul_best.sh
```
train the model

``` bash
python ./train_ul_best.py --epochs 8 --batch_size 64 --pretrained_model ./pretrain_model/pytorch_model.bin  

```

### hyper-parameter settings

```json
{
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 300,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12,
  "n_positions": 300,
  "vocab_size": 13317
}

```


## Reference

```TeX
@inproceedings{su2020moviechats,
  title={MovieChats: Chat like Humans in a Closed Domain},
  author={Su, Hui and Shen, Xiaoyu and Xiao, Zhou and Zhang, Zheng and Chang, Ernie and Zhang, Cheng and Niu, Cheng and Zhou, Jie},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={6605--6619},
  year={2020}
} 
```
