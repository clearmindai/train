from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, directory_path, tokenizer):
        self.tokenizer = tokenizer
        self.files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                self.files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            text = f.read()
        encoded_text = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = []
        labels = []
        for i in range(0, len(encoded_text), self.tokenizer.max_len_single_sentence):
            input_chunk = encoded_text[i:i+self.tokenizer.max_len_single_sentence]
            label_chunk = encoded_text[i+1:i+self.tokenizer.max_len_single_sentence+1]
            input_chunk += [self.tokenizer.pad_token_id] * (self.tokenizer.max_len_single_sentence - len(input_chunk))
            label_chunk += [self.tokenizer.pad_token_id] * (self.tokenizer.max_len_single_sentence - len(label_chunk))
            input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            labels.append(torch.tensor(label_chunk, dtype=torch.long))
        return torch.stack(input_ids), torch.stack(labels)

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the dataset and data loader
dataset = TextDataset('data', tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Set up the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Fine-tune the model
for epoch in range(10):
    losses = []
    for input_ids, labels in tqdm(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss, _, _ = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        optimizer.zero_grad()
        losses.append(loss.item())
    print(f'Epoch {epoch+1}, loss: {sum(losses)/len(losses)}')

# Save the model
model.save_pretrained('gpt-j-6b-finetuned')
