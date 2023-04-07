print('Importing os')
import os
print('Importing tqdm')
from tqdm import tqdm
print('Importing torch')
import torch
print('Importing transformers')
from transformers import GPT2Tokenizer, GPT2LMHeadModel
print('Starting tokenizer')
# Set up the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print('Starting model')
model = GPT2LMHeadModel.from_pretrained('gpt2')
print('Starting parameters')
# Set up the training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Piping model')
model.to(device)
print('Working...')
# Set up the progress bar
data_dir = 'data'
data_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')][:500]
progress_bar = tqdm(data_files)
print('Training...')
# Fine-tune the model on each training chunk
model.train()
print('Running...')
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
