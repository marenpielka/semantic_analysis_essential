import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size, hidden_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(FeedForwardNN, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.i2h = nn.Linear(vocab_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, num_labels)


    def forward(self, input):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        hidden_output = torch.sigmoid(self.i2h(input))
        return F.log_softmax(self.h2o(hidden_output), dim=1)


class FeedForwardNN_multipleLayers(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size, layer_config, dropout=0.1):
        super(FeedForwardNN_multipleLayers, self).__init__()

        self.n_layers = len(layer_config)
        self.layerConfig = layer_config
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(vocab_size, layer_config[0]))

        print("Initializing input layer with {} inputs and {} outputs".format(vocab_size, layer_config[0]))
        last_size = layer_config[0]
        for i in range(1, self.n_layers):
            print("Initializing hidden layer {} with {} inputs and {} outputs".format(i, last_size, layer_config[i]))

            layer = nn.Linear(last_size, layer_config[i])

            # initialize to normal
            nn.init.normal(layer.weight)

            # append to the layers
            self.layers.append(layer)
            last_size = layer_config[i]

        print("Initializing output layer with {} inputs and {} outputs".format(last_size, num_labels))
        self.layers.append(nn.Linear(last_size, num_labels))


    def forward(self, input):
        # pass the input to all layers
        output = input
        for ind,layer in enumerate(self.layers):
            if ind == len(self.layers)-1:
                output = layer(output)
            else:
                output = torch.sigmoid(layer(output))
        return F.log_softmax(output,dim=-1)#dim=1