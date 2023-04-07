import os
from tqdm import tqdm
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Set up the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set up the training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)

# Set up the progress bar
data_dir = 'data'
data_files = os.listdir(data_dir)
progress_bar = tqdm(data_files)

# Fine-tune the model on each training chunk
model.train()
for filename in progress_bar:
    file_path = os.path.join(data_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = f.read()

    # Split the training data into smaller chunks
    max_length = model.config.n_positions
    train_chunks = [chunk[i:i+max_length] for i in range(0, len(chunk), max_length)]

    for i, chunk in enumerate(train_chunks):
        encodings = tokenizer.encode_plus(chunk, return_tensors='pt', max_length=max_length)
        input_ids = encodings['input_ids'].to(device)
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Update the progress bar
        progress_bar.set_description(f"Processed file {filename}. Loss: {loss.item()}")

# Save the fine-tuned model to disk
model.save_pretrained('finetuned_gpt2')
