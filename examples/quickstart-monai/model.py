import torch


def train(model, train_loader, epoch_num, device):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    for _ in range(epoch_num):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss_function(model(inputs.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(model, test_loader, device):
    model.eval()
    loss = 0.0
    y_true = list()
    y_pred = list()
    loss_function = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            out = model(test_images.to(device))
            test_labels = test_labels.to(device)
            loss += loss_function(out, test_labels).item()
            pred = out.argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)]) / len(
        test_loader.dataset
    )
    return loss, accuracy
