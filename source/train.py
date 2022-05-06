# Função de treino 
def train(model, dataloader, loss_function, optimizer, device):

    model.train()

    train_loss = 0.0

    for frames, pressure in dataloader:

        frames, pressure = frames.to(device), pressure.to(device)

        # Passando os dados para o modelo para obter as predições
        pred = model(frames)

        pressure_aux = pressure 
        pressure = pressure_aux[:, None]

        # Calculando a perda através da função de perda
        loss = loss_function(pred, pressure)

        # Zerando os gradientes acumulados. O Pytorch vai acumulando os gradientes de acordo com as operações realizadas .  
        optimizer.zero_grad()

        # Calculando os gradientes
        # print(frames.dtype)
        # print(pred.dtype)
        # print(pressure.dtype)
        loss.backward()

        # Tendo o gradiente já calculado , o step indica a direção que reduz o erro que vamos andar 
        optimizer.step()

        # Loss é um tensor!
        train_loss += loss.item()

    return train_loss / len(dataloader)