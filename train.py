import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Set up tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-12b")
model.to(device)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=1000
)

# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, data_dir):
        self.examples = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".txt"):
                with open(os.path.join(data_dir, file_name), "r") as f:
                    text = f.read()
                    self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Define training function
def train(model, optimizer, scheduler, dataloader, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        labels = torch.ones(inputs["input_ids"].shape[0], dtype=torch.long).to(device)
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(dataloader)

# Define main function
def main():
    # Set up dataset and dataloader
    dataset = TextDataset("data")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Train model
    for epoch in range(3):
        loss = train(model, optimizer, scheduler, dataloader, device)
        print(f"Epoch {epoch+1} loss: {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "saved_model/clearmindbeta.pt")

if __name__ == "__main__":
    main()
