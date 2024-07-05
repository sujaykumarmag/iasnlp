import torch
from transformers import XLNetTokenizer, XLNetLMHeadModel
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

# Example data (replace with your actual data loading)
source_sentences = ["This is a sentence.", "Another sentence."]
target_sentences = ["Das ist ein Satz.", "Ein weiterer Satz."]

# Initialize XLNet tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
dec_only_model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')

# Tokenize and encode the data
source_inputs = tokenizer(source_sentences, padding=True, return_tensors="pt")
target_inputs = tokenizer(target_sentences, padding=True, return_tensors="pt")

# Ensure input and output lengths match
assert source_inputs.input_ids.shape[1] == target_inputs.input_ids.shape[1], "Input and output lengths must match"

# Prepare inputs for training
input_ids = source_inputs.input_ids
labels = target_inputs.input_ids[:, 1:].contiguous()  # Shift target ids to the right (auto-regressive)

# Define dataset and data loader (replace with your data handling)
class TranslationDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

dataset = TranslationDataset(input_ids, labels)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define optimizer and loss function
dec_only_optimizer = torch.optim.AdamW(dec_only_model.parameters(), lr=1e-5)
loss_fn = CrossEntropyLoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    dec_only_model.train()
    total_loss_dec_only = 0.0
    for batch in data_loader:
        input_ids = batch['input_ids']
        output_ids = batch['labels']  # Assuming 'labels' corresponds to target outputs
        
        # Print shapes for debugging
        print("Input IDs shape:", input_ids.shape)
        print("Output IDs shape:", output_ids.shape)
        
        # Forward pass
        dec_only_optimizer.zero_grad()
        outputs = dec_only_model(input_ids=input_ids, labels=output_ids)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        dec_only_optimizer.step()
        
        total_loss_dec_only += loss.item()
    
    avg_loss_dec_only = total_loss_dec_only / len(data_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss_dec_only:.4f}")

print("Training finished.")
