import torch.nn as nn

class Simple_classifier(nn.Module):
    def __init__(self, num_features, init_param):
        super(Simple_classifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(init_param, int(init_param/2)),
            nn.BatchNorm1d(int(init_param/2)),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(int(init_param/2), int(init_param/4)),
            nn.BatchNorm1d(int(init_param/4)),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(int(init_param/4), int(init_param/8)),
            nn.BatchNorm1d(int(init_param/8)),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer5 = nn.Linear(int(init_param/8), 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x