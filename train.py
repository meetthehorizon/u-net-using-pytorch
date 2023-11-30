#import necessary libraries
import torch 
import torchvision
import pickle
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader

from models.UNet import UNet
from utils.pytorch_utils import LungDataset
from utils.helper import checkpoint, resume, print_progress
from utils.data_utils import download_lung_semantic_data

#hyperparameters
batch_size = 16
num_epochs = 25
learning_rate = 0.01

#paths
checkpoint_path = 'model_checkpoints/model_checkpoint.pth'

#downloading data
download_lung_semantic_data(path='data')

#using custom dataset class to load data
composed = torchvision.transforms.Compose([
	transforms.Resize((512, 512)),
	transforms.ToTensor(),
])

train_size = 1
val_size = 1
test_size = 1

train_dataset = LungDataset(path='data/train', transform=composed, num=train_size)
val_dataset = LungDataset(path='data/val', transform=composed, num=val_size)
test_dataset = LungDataset(path='data/test', transform=composed, num=test_size)

train_size = len(train_dataset)
val_size = len(val_dataset)
test_size = len(test_dataset)

print(f'#training examples: {len(train_dataset)}\n#test examples: {len(test_dataset)}\n#validation examples: {len(val_dataset)}')

#creating dataloaders for training, testing and validation

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = UNet(1, 1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

debug = True
train_losses = []
val_losses = []
start_epoch = 0
try: 
	with open('states/train_losses.pickle', 'rb') as f:
		train_losses = pickle.load(f)
	with open('states/val_losses.pickle', 'rb') as f:
		val_losses = pickle.load(f)
	with open('states/start_epoch.pickle', 'rb') as f:
		start_epoch = pickle.load(f)
except: 
	with open('states/train_losses.pickle', 'wb') as f:
		pickle.dump(train_losses, f)
	with open('states/val_losses.pickle', 'wb') as f:
		pickle.dump(val_losses, f)
	with open('states/start_epoch.pickle', 'wb') as f:
		pickle.dump(start_epoch, f)

if start_epoch > 0:	
	resume_epoch = start_epoch - 1
	resume(model, f'model_states/epoch-{resume_epoch}.pth')

for epoch in range(start_epoch, num_epochs):
	train_loss = 0
	model.train()
	batch_number = 0

	for X_batch, y_batch in train_dataloader:
		y_pred = model(X_batch)

		loss = criterion(y_pred, y_batch)	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss = loss.item()
		
		batch_number += 1
		print_progress(epoch+1, batch_number, len(train_dataloader))
		
	avg_train_loss = train_loss / train_size
	train_losses.append(avg_train_loss)

	model.eval()
	val_loss = 0
	with torch.no_grad():
		for X_batch, y_batch in val_dataloader:
			y_pred = model(X_batch)
			val_loss += criterion(y_pred, y_batch).item()
	
	avg_val_loss = val_loss / val_size
	val_losses.append(avg_val_loss)
	
	plt.figure(figsize=(10, 5))
	plt.plot(train_losses, label='Train Loss')
	plt.plot(val_losses, label='Validation Loss')
	plt.title(f'Epoch {epoch + 1}/{num_epochs}')
	plt.xlabel('Epoch')

	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(f'plots/epoch-{epoch}.jpg', format='jpg')
	checkpoint(model, f'model_states/epoch-{epoch}.pth')
	plt.close()
	
	start_epoch += 1
	
	print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

	with open('states/train_losses.pickle', 'wb') as f:
		pickle.dump(train_losses, f)
	with open('states/val_losses.pickle', 'wb') as f:
		pickle.dump(val_losses, f)
	with open('states/start_epoch.pickle', 'wb') as f:
		pickle.dump(start_epoch, f)

model.eval()
test_losses = []
test_accuracy = 0
with torch.no_grad():
	for X_batch, y_batch in test_dataloader:
		test_loss = 0
		y_pred = model(X_batch)
		test_loss += criterion(y_pred, y_batch).item()
	
	avg_test_loss = test_loss / test_size
	test_losses.append(avg_test_loss)
	print(f'Test Loss: {avg_test_loss:.4f}')

