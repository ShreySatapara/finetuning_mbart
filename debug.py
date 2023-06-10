from load_dataset import MyDataset
from transformers import T5Tokenizer
from torch.utils.data import DataLoader


tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
#tokenizer.src_lang="hi_IN"
dataset = MyDataset("../original_data/data/dev.SRC","../original_data/data/dev.TGT",tokenizer,128)
#print(next(iter(dataset)))
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(next(iter(train_dataloader)))