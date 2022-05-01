import argparse
import re
from pathlib import Path

from paddlespeech.t2s.frontend.zh_normalization.text_normlization import TextNormalizer
from opencc import OpenCC
from g2pw import G2PWConverter
#from cntn import w2s

def run():
    tw2sp = OpenCC('tw2sp.json')
    s2twp = OpenCC('s2twp.json')
    tnorm = TextNormalizer()
    g2pw = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)

    input0 = '上校請技術人員校正FN儀器,用了25年，跟銀行借了320萬元。她出生于86年8月18日（1997/08/18），身高175.3cm'
    input1 = tw2sp.convert(input0)
    input2 = tnorm.normalize_sentence(input1)
#    input3 = w2s(input2)
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
