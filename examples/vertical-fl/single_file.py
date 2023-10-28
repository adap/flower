import torch
import torch.nn as nn
import torch.optim as optim

# Create some dummy data
X = torch.randn(100, 5)  # 100 samples, 5 features
y = (torch.sum(X, dim=1) > 0).float()  # Sum of features > 0 as positive label

# Split the data
X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]


# Client Model
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc = nn.Linear(5, 10)  # Simple Linear layer

    def forward(self, x):
        return self.fc(x)


# Server Model
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Simple Linear layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


client = ClientModel()
server = ServerModel()

client_optimizer = optim.SGD(client.parameters(), lr=0.01)
server_optimizer = optim.SGD(server.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Training Loop
epochs = 100
for epoch in range(epochs):
    for i in range(len(X_train)):
        x = X_train[i]
        label = y_train[i]

        # Forward pass on client
        embedding = client(x)
        embedding_np = embedding.detach().numpy()

        # Transfer embedding to server
        embedding_server = torch.from_numpy(embedding_np).requires_grad_()

        # Forward and backward pass on server
        output = server(embedding_server)
        loss = criterion(output, label)
        loss.backward()

        # Transfer gradient to client and update server-side weights
        gradient_np = embedding_server.grad.numpy()
        server_optimizer.step()
        server_optimizer.zero_grad()

        # Update the client-side weights
        embedding.backward(torch.from_numpy(gradient_np))
        client_optimizer.step()
        client_optimizer.zero_grad()

    # Print the loss for the epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Testing
with torch.no_grad():
    correct = 0
    total = len(X_test)
    for i in range(total):
        x = X_test[i]
        label = y_test[i]

        # Forward pass on client
        embedding = client(x)

        # Transfer embedding to server
        output = server(embedding)
        predicted = (output > 0.5).float()

        correct += (predicted == label).sum().item()

    print(f"Accuracy: {correct / total * 100}%")
