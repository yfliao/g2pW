
# install
```
pip install opencc
#pip install cntn
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple

git clone https://github.com/yfliao/g2pW
cd g2pW
pip install .
```

# test.py
```
import argparse
import re
from pathlib import Path

from paddlespeech.t2s.frontend.zh_normalization.text_normlization import TextNormalizer
from opencc import OpenCC
from g2pw import G2PWConverter
from cntn import w2s

def run():
    tw2sp = OpenCC('tw2sp.json')
    s2twp = OpenCC('s2twp.json')
    tnorm = TextNormalizer()
    g2pw = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)

    input0 = '上校請技術人員校正FN儀器,用了25年，跟銀行借了320萬元。她出生于86年8月18日（1997/08/18），身高175.3cm'
    input1 = tw2sp.convert(input0)
    input2 = tnorm.normalize_sentence(input1)
 #   input3 = w2s(input2)
    input4 = s2twp.convert(input2)
    output1 = g2pw(input4)

    print (input0)
    print (input1)
    print (input2)
#    print (input3)
    print (input4)
    print (output1)

if __name__ == '__main__':
    run()
```






# g2pW: Mandarin Grapheme-to-Phoneme Converter

[![Downloads](https://pepy.tech/badge/g2pw)](https://pepy.tech/project/g2pw)[![license](https://img.shields.io/badge/license-Apache%202.0-red)](https://github.com/GitYCC/g2pW/blob/master/LICENSE)

**Authors:** [Yi-Chang Chen](https://github.com/GitYCC), Yu-Chuan Chang, Yen-Cheng Chang and Yi-Ren Yeh

This is the official repository of our paper [g2pW: A Conditional Weighted Softmax BERT for Polyphone Disambiguation in Mandarin](https://arxiv.org/abs/2203.10430).

## Getting Started

### Dependency / Install

(This work was tested with PyTorch 1.7.0, CUDA 10.1, python 3.6 and Ubuntu 16.04.)

- Install [PyTorch](https://pytorch.org/get-started/locally/)

- `$ pip install g2pw`



### Quick Demo

<a href="https://colab.research.google.com/github/GitYCC/g2pW/blob/master/misc/demo.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```python
>>> from g2pw import G2PWConverter
>>> conv = G2PWConverter()
>>> sentence = '上校請技術人員校正FN儀器'
>>> conv(sentence)
[['ㄕㄤ4', 'ㄒㄧㄠ4', 'ㄑㄧㄥ3', 'ㄐㄧ4', 'ㄕㄨ4', 'ㄖㄣ2', 'ㄩㄢ2', 'ㄐㄧㄠ4', 'ㄓㄥ4', None, None, 'ㄧ2', 'ㄑㄧ4']]
>>> sentences = ['銀行', '行動']
>>> conv(sentences)
[['ㄧㄣ2', 'ㄏㄤ2'], ['ㄒㄧㄥ2', 'ㄉㄨㄥ4']]
```

### Use GPU to Speed Up

```python
conv = G2PWConverter(use_cuda=True)
```

### Load Offline Model

```python
conv = G2PWConverter(model_dir='./G2PWModel-v1/', model_source='./path-to/bert-base-chinese/')
```

### Support Simplified Chinese and Pinyin

```
$ pip install g2pw[opencc]
```

```python
>>> from g2pw import G2PWConverter
>>> conv = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)
>>> conv('然而，他红了20年以后，他竟退出了大家的视线。')
[['ran2', 'er2', None, 'ta1', 'hong2', 'le5', None, None, 'nian2', 'yi3', 'hou4', None, 'ta1', 'jing4', 'tui4', 'chu1', 'le5', 'da4', 'jia1', 'de5', 'shi4', 'xian4', None]]
```

## Scripts

```
$ git clone https://github.com/GitYCC/g2pW.git
```

### Train Model

For example, we train models on CPP dataset as follows:

```
$ bash cpp_dataset/download.sh
$ python scripts/train_g2p_bert.py --config configs/config_cpp.py
```

### Prediction

```
$ python scripts/test_g2p_bert.py \
--config saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/config.py \
--checkpoint saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/best_accuracy.pth \
--sent_path cpp_dataset/test.sent \
--output_path output_pred.txt
```

### Testing

```
$ python scripts/predict_g2p_bert.py \
--config saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/config.py \
--checkpoint saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/best_accuracy.pth \
--sent_path cpp_dataset/test.sent \
--lb_path cpp_dataset/test.lb
```

## Checkpoints

[G2PWModel-v1.zip](https://storage.googleapis.com/esun-ai/g2pW/G2PWModel-v1.zip)

## Citation

```
@misc{chen2022g2pw,
      title={g2pW: A Conditional Weighted Softmax BERT for Polyphone Disambiguation in Mandarin}, 
      author={Yi-Chang Chen and Yu-Chuan Chang and Yen-Cheng Chang and Yi-Ren Yeh},
      year={2022},
      eprint={2203.10430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
