import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False), #same conv
            nn.BatchNorm2d(out_c), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3,1,1,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    


class Unet (nn.Module):
    def __init__(self, in_c=3, out_c=1, features=[64,128,256,512]):
        super().__init__()
        self.up = nn.ModuleList()
        self.upPathConv = nn.ModuleList()
        self.downwardPath = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        for feature in features:
            self.downwardPath.append(DoubleConv(in_c, feature))
            in_c = feature
        
        for feature in reversed(features):
            self.up.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.upPathConv.append(DoubleConv(feature*2, feature))
        
        self.bottleNeck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_c, kernel_size=1)
    
    def forward(self, x):
        res_connections = []

        for d in self.downwardPath:
            x = d(x)
            res_connections.append(x)
            x = self.pool(x)

        x = self.bottleNeck(x)

        res_connections.reverse()

        for idx in range(len(self.up)):
            x = self.up[idx](x)
            res_con = res_connections[idx]
            if x.shape != res_con.shape:
                x = TF.resize(x, size=res_con.shape[2:])
            x = torch.cat((x, res_con), dim=1)
            x = self.upPathConv[idx](x)
        
        return self.final_conv(x)
    
def test ():
    x = torch.randn((3,1,160,160))
    model = Unet(1,1)
    pred = model(x)
    print(pred.shape)
    assert pred.shape == x.shape

if __name__ == "__main__":
    test()