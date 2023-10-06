import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        self.model = nn.Sequential(
         
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),     
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256), 
            
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(7 * 7 * 256, out_features=500),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(500),
            nn.Dropout(p=dropout),
            nn.Linear(500, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        # ## Define layers of a CNN
        # # convolutional layer-1 (224x224x3 image tensor)
        # self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # # convolutional layer-2 (112x112x16 image tensor)
        # self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # # convolutional layer-3 (56x56x32 image tensor)
        # self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # # max pooling layer
        # self.pool = nn.MaxPool2d(2, 2)
        #  # linear layer (64 * 28 * 28 -> 500)
        # self.fc1 = nn.Linear(64 * 28 * 28, 500)
        # # linear layer (500 -> 50)
        # self.fc2 = nn.Linear(500, num_classes)
        # # dropout layer 
        # self.dropout = nn.Dropout(0.25)
        
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 32 * 32 , 1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(256, 50)
#         self.dropout = nn.Dropout(0.3)




        # def forward(self, x: torch.Tensor) -> torch.Tensor:
        
                
        #         return self.model(x)
    
#     def forward(self, x):
# #         ## Define forward behavior
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         # flatten image input
#         x = x.view(-1, 64 * 28 * 28)
#         # Dropout layer1
#         x = self.dropout(x)
#         # Relu activation function for hidden layer1
#         x = F.relu(self.fc1(x))
#         # Dropout layer2
#         x = self.dropout(x)
#         # Relu activation function for hidden layer2
#         x = self.fc2(x)
        

#         return x

        
#         return x

        
      


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2, num_workers = 0)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
