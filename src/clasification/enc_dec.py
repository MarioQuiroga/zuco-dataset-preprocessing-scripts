from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, EncoderDecoderModel
from eeg2embs import *
import pickle

# Tokenizador para el texto
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Definir el modelo Encoder-Decoder
config = GPT2Config.from_pretrained('gpt2')
encoder = GPT2Model.from_pretrained('gpt2', config=config)
decoder = GPT2Model.from_pretrained('gpt2', config=config)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# Crear el dataset para el modelo de traducción
class EEGToTextDataset(Dataset):
    def __init__(self, eeg_embeddings, target_texts, tokenizer, max_length=768):
        self.eeg_embeddings = eeg_embeddings
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.eeg_embeddings)
    
    def __getitem__(self, idx):
        eeg_embedding = self.eeg_embeddings[idx]
        target_text = self.target_texts[idx]

        # inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)

        # input_ids = inputs['input_ids']

        # # Forward pass to get embeddings
        # with torch.no_grad():
        #     outputs = model(input_ids)

        # last_hidden_states = outputs.last_hidden_state
        # #word_embeddings = torch.mean(last_hidden_states, dim=1)
        # batch_embeddings = torch.mean(last_hidden_states, dim=1)
        # word_embeddings.append(batch_embeddings)
        
        # # input_ids = eeg_embedding
        # # attention_mask = attention_mask.squeeze(1)
        
        # # with torch.no_grad():
        # #     outputs = self.tokenizer(input_ids=input_ids, attention_mask=attention_mask)
        
        # # last_hidden_states = outputs.last_hidden_state
        # # batch_embeddings = torch.mean(last_hidden_states, dim=1)
        # # target_encoding = self.tokenizer(
        # #     target_text,
        # #     padding='max_length',
        # #     truncation=True,
        # #     max_length=self.max_length,
        # #     return_tensors="pt"
        # # )
        
        # # labels = target_encoding.input_ids.squeeze()
        return eeg_embedding, target_text

# Crear un nuevo dataset con los embeddings de EEG y los textos objetivos
#target_texts = ["hello", "world", "example", "transformers", "pytorch", "data"] * 16 + ["extra"]

eeg_data, labels = load_all_data("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/matfiles_csv/")
#eeg_model = pickle.load(open("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/eeg_2_embeddings_model", "rb"),fix_imports=False)
eeg_model = torch.load("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/eeg_2_embeddings_model.pth")
eeg_data = torch.tensor(eeg_data)

eeg_data_chunk = chunk_labels(eeg_data, 1000)
eeg_embeddings = []
for i in range(len(eeg_data_chunk)):
    embeddings = eeg_model(eeg_data_chunk[i])
    eeg_embeddings.append(embeddings)

eeg_embeddings = torch.cat(eeg_embeddings, dim=0)

#eeg_embeddings = [eeg_model(eeg_data[i]) for i in range(len(eeg_data))]



word_embeddings = pickle.load(open("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/word_embeddings", "rb"))

#eeg_embeddings = torch.stack(eeg_embeddings)
dataset = EEGToTextDataset(eeg_embeddings, word_embeddings, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Entrenar el modelo de traducción
model.train()
for epoch in range(10):
    for batch in dataloader:
        embeddings, labels = batch
        optimizer.zero_grad()
        outputs = model(embeddings.long(), labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')