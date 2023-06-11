import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import sys
from tqdm import tqdm
import argparse

# Load the model and tokenizer


# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# #model_name = "facebook/mbart-large-50-many-to-one-mmt"
# tokenizer = MBart50Tokenizer.from_pretrained("google/mt5-small")
# model = AutoModelForSeq2SeqLM.from_pretrained("../mt5_finetuning/checkpoint-42968")
# model = model.to(device)

# Function to translate a batch of sentences
def translate_batch(tokenizer,model,device,sentences):
    inputs = tokenizer.batch_encode_plus(sentences, return_tensors="pt", padding=True, max_length=210, truncation=True).to(device)
    translated_ids = model.generate(**inputs,num_beams=5, max_length=210)
    translated_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return translated_texts

def main():
    parser = argparse.ArgumentParser(description='Inference')

    parser.add_argument('--input_file', type=str, help='Path to the input file')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of samples in each batch')
    parser.add_argument('--device_id', default=0, type=int, help='ID of the device to use for computation')
    parser.add_argument('--mbart_model_name', default="facebook/mbart-large-50-many-to-one-mmt", type=str, help='Name or path of the mBART model')
    
    args = parser.parse_args()
    # Open the input file and output file
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    tokenizer = MBart50TokenizerFast.from_pretrained(args.mbart_model_name,src_lang="hi_IN")
    model = MBartForConditionalGeneration.from_pretrained(args.checkpoint_path)
    model = model.to(device)
    
    input_file = args.input_file
    output_file = args.output_file
    batch_size = args.batch_size

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        sentences = []
        for line in tqdm(fin):
            hindi_sentence = line.strip()
            sentences.append(hindi_sentence)

            # Translate the batch when it reaches the desired batch size
            if len(sentences) == batch_size:
                english_translations = translate_batch(tokenizer,model,device,sentences)
                for translation in english_translations:
                    fout.write(translation + "\n")
                sentences = []

        # Translate the remaining sentences in the last batch
        if len(sentences) > 0:
            english_translations = translate_batch(tokenizer,model,device,sentences)
            for translation in english_translations:
                fout.write(translation + "\n")

    print("Translation completed and stored in the output file:", output_file)

if __name__ == "__main__":
    main()
