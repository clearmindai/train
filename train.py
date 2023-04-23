print('1. Importing Modules')

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM
#from transformers import GPTNeoXTokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM
import glob
from accelerate import Accelerator
accelerator = Accelerator()
torch.manual_seed(42)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125m", bos_token='<|startoftext|>',
#tokenizer = GPTNeoXTokenizer.from_pretrained("EleutherAI/pythia-410m", bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m").to('mps')
model.resize_token_embeddings(len(tokenizer))
#files = pd.read_csv('data.csv')['description']
print('Processing files')

model = accelerator.prepare(
     model
)

path = 'data/*.txt'  # specify path to all .txt files in the data directory
files = glob.glob(path)[:2]  # get a list of file paths

contents = []  # initialize an empty list to store contents

for file in files:
    with open(file, 'r') as f:
        file_content = f.read()
        while len(file_content) > 2048:
            contents.append(file_content[:2048])
            file_content = file_content[2048:]
        contents.append(file_content)


#max_length = max([len(tokenizer.encode(description)) for description in descriptions])
max_length = 2048
print('Processing dataset and starting trainer')

class TextDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

class TrainingArgumentsWithMPSSupport(TrainingArguments):

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
dataset = TextDataset(contents, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
training_args = TrainingArgumentsWithMPSSupport(output_dir='./results', num_train_epochs=5, logging_steps=5000, save_steps=5000, per_device_train_batch_size=2, per_device_eval_batch_size=2, warmup_steps=10, weight_decay=0.01, logging_dir='./logs')
print("Training")
Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]).to(torch.device('mps')), 'labels': torch.stack([f[0] for f in data]).to(torch.device('mps'))}).train()

print('Done')
#generated = tokenizer("<|startoftext|> ", return_tensors="pt").input_ids.to('mps')
#sample_outputs = model.generate(generated, do_sample=True, top_k=50, 
#                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
#for i, sample_output in enumerate(sample_outputs):
#    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
## specify the directory where you want to save the model
model_directory = './saved_model/'

# save the trained model to the specified directory
model.save_pretrained(model_directory)

# save the tokenizer to the same directory
tokenizer.save_pretrained(model_directory)
