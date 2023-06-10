from bleurt import score
import argparse
from glob import glob
# Create the parser object
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Description of your program.')
parser.add_argument('-f', '--folder', type=str, required=True, help='Input folder path path')

args = parser.parse_args()
folder_path = args.folder

checkpoint = "./bleurt/BLEURT-20"
scorer = score.BleurtScorer(checkpoint)



d = {}
list_of_lang = list(glob(folder_path + '/*/*.devtest'))


for lang in list_of_lang:
    reference_file = "../original_data/similar_lang/devtest/eng_Latn.devtest"
    with open(reference_file) as fr:
        reference_lines = fr.readlines()

    with open(lang) as f:
        candicate_lines = f.readlines()
    scores = scorer.score(references=reference_lines, candidates=candicate_lines)
    print(lang,sum(scores)/len(scores))   
    d[lang] = sum(scores)/len(scores)


reference_file = "../original_data/data/test.TGT"
with open(reference_file) as fr:
        reference_lines = fr.readlines()
lang = folder_path + "/hi_en/test.SRC"
with open(lang) as f:
        candicate_lines = f.readlines()

scores = scorer.score(references=reference_lines, candidates=candicate_lines)
print(lang,sum(scores)/len(scores))   
d[lang] = sum(scores)/len(scores)

print(d)