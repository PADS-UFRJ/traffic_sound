import numpy as np
import torch

# Função de teste 
def test(model, dataloader, loss_function, device):

    model.eval()

    test_loss = 0.0
    min_test_loss = np.inf

    with torch.no_grad():
        for frames, pressure in dataloader:

            frames, pressure = frames.to(device), pressure.to(device)

            # Passando os dados para o modelo para obter as predições
            pred = model(frames)

            pressure_aux = pressure 
            pressure = pressure_aux[:, None]

            # Calculando a perda através da função de perda
            loss = loss_function(pred, pressure)

            test_loss += loss.item()

            # if min_test_loss > test_loss:
            #     #print(f'Validation Loss Decreased({min_test_loss:.6f}--->{test_loss:.6f}) \t Saving The Model')
            #     min_test_loss = test_loss
        
            #     if savedir is not None:
            #         # Salvando o modelo 
            #         torch.save(model.state_dict(), 'saved_model.pth')

        return test_loss / len(dataloader)
