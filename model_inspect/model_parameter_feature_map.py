import torch
from model import *
from main_clean import model_definition
import yaml
import numpy as np
import matplotlib.pyplot as plt

def model_inspection(model, saved_model):
    model.eval()
    model.load_state_dict(torch.load(saved_model))

    state_dict = model.state_dict()
    print("Model's layers : ")
    for idx, layer in enumerate(state_dict.keys()):
        print(f"- {idx}. ",layer)
    print("Model's state_dict : ", state_dict.keys())
    for name, param in model.named_parameters():
        print(name, 'parameter shape : ', param.shape)
        print('parameter : ', param)    # Float tensor with Parameter containing sentence


if __name__ == '__main__':
    model_name = 'CNN'
    selected_model = 'Dilated_CNN'
    config_file = './config.yaml'
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for subj in range(1):
        model_path = f'./outputs/2024-07-18/13-56-10/results/saved_model/Subj{subj}.pt'
        model = model_definition(selected_model, config, device)

        model_inspection(model, model_path)

        model.eval()
        model.load_state_dict(torch.load(model_path))

        state_dict = model.state_dict()

        # Print the state dictionary keys
        # print("Model's state_dict:")
        # for param_tensor in state_dict:
        #     print(param_tensor, "\t", state_dict[param_tensor].size())

        # Print the actual parameters (weights and biases)
        for name, param in model.named_parameters():
            if name == 'conv_layers.0.weight':
                param_to_see = param

        print(param_to_see)
        param_to_see = param_to_see.view(-1, param_to_see.size(-1))
        print(param_to_see.shape)
        param_to_see = np.array(param_to_see.cpu().detach())
    for idx in range(param_to_see.shape[0]):
        plt.plot(param_to_see[idx, :])
        plt.savefig(f'./conv_layer/{idx}.png')
        if idx % 10 == 0:
            plt.close()
