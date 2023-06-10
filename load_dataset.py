import torch
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, source_file, target_file, tokenizer, max_length):
        self.tokenizer = tokenizer
        #self.src_tokenizer = src_tokenizer
        #self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        self.source_sentences = self.load_sentences(source_file)
        self.target_sentences = self.load_sentences(target_file)

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source_sentence = self.source_sentences[idx]
        target_sentence = self.target_sentences[idx]
        
        inputs = self.tokenizer(source_sentence,text_target = target_sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()
        inputs['labels'] = inputs['labels'].squeeze()
        return inputs

    def load_sentences(self, file_path):
        # Load the sentences from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]

        return sentences