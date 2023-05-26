import torch.nn as nn

class Baseline_classifier(nn.Module):
    def __init__(self, num_features, init_param):
        super(Baseline_classifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            # nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(init_param, init_param),
            nn.BatchNorm1d(init_param),
            # nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(init_param, init_param),
            nn.BatchNorm1d(init_param),
            # nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(init_param, int(init_param/2)),
            nn.BatchNorm1d(int(init_param/2)),
            # nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(int(init_param/2), int(init_param/4)),
            nn.BatchNorm1d(int(init_param/4)),
            # nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer6 = nn.Sequential(
            nn.Linear(int(init_param/4), int(init_param/8)),
            nn.BatchNorm1d(int(init_param/8)),
            # nn.Dropout(0.1),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(
            nn.Linear(int(init_param/8), int(init_param/16)),
            nn.BatchNorm1d(int(init_param/16)),
            nn.ReLU()
        )

        self.layer8 = nn.Sequential(
            nn.Linear(int(init_param/16), int(init_param/32)),
            nn.BatchNorm1d(int(init_param/32)),
            nn.ReLU()
        )

        self.layer9 = nn.Sequential(
            nn.Linear(int(init_param/32), int(init_param/64)),
            nn.BatchNorm1d(int(init_param/64)),
            nn.ReLU()
        )

        self.layer10 = nn.Sequential(
            nn.Linear(int(init_param/64), int(init_param/128)),
            nn.BatchNorm1d(int(init_param/128)),
            nn.ReLU()
        )

        self.layer11 = nn.Linear(int(init_param/128), 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x
    
class Simple_classifier(nn.Module):
    def __init__(self, num_features, init_param):
        super(Simple_classifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(num_features, init_param),
            nn.BatchNorm1d(init_param),
            nn.ReLU()
        )

        self.layer2 = nn.Linear(init_param, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x