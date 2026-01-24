'''Training loop.'''

import torch


@torch.no_grad()
def test_loss(model, criterion, data_loader):
    '''Compute test loss.'''

    model.eval()

    losses = []
    for x_batch, y_batch in data_loader:
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch).item()
        losses.append(loss)

    if criterion.reduction == 'mean':
        loss = sum(losses) / len(losses)
    else:
        loss = sum(losses)

    return loss


def train(
    model,
    criterion,
    optimizer,
    num_epochs,
    train_loader,
    val_loader
):
    '''Run model training.'''

    val_loss = test_loss(model, criterion, val_loader)
    print('Before training, val. loss: {:.2e}'.format(val_loss))

    for epoch_idx in range(num_epochs):
        model.train()

        # train single epoch
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

        # compute val. error
        if (epoch_idx + 1) % 1 == 0:
            val_loss = test_loss(model, criterion, val_loader)
            print('Epoch: {:d}, batch loss: {:.2e}, val. loss: {:.2e}'. \
                  format(epoch_idx + 1, loss.detach().item(), val_loss))
