import torch


class MLPPolicy(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=(64, 64, 64), act_fn=torch.nn.ReLU):
        super().__init__()

        layers = [torch.nn.Linear(input_size, hidden_layers[0]), act_fn()]
        for i, dim in enumerate(hidden_layers[:-1]):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(act_fn())
        self.shared = torch.nn.Sequential(*layers)

        self.policy_head = torch.nn.Linear(hidden_layers[-1], output_size)
        self.value_head = torch.nn.Linear(hidden_layers[-1], 1)

    def get_init_hx(self):
        return torch.zeros(1)

    def forward(self, x, hx):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x), hx


class GRUPolicy(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, gru_hidden_size=64, act_fn=torch.nn.ReLU):
        super().__init__()

        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size), act_fn(),
            torch.nn.Linear(hidden_size, hidden_size), act_fn(),
        )

        self.gru_hidden_size = gru_hidden_size
        self.gru = torch.nn.GRU(hidden_size, gru_hidden_size)

        self.policy_head = torch.nn.Linear(hidden_size, output_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def get_init_hx(self):
        return torch.zeros(self.gru_hidden_size)

    def forward(self, x, hx):
        x = self.shared(x)

        if x.dim() == 1:
            x, hx = x.view(1, 1, -1), hx.view(1, 1, -1)
            x, hx = self.gru(x, hx)
            x, hx = x.view(-1), hx.view(-1)
        else:
            x, hx = x.unsqueeze(0), hx.unsqueeze(0)
            x, hx = self.gru(x, hx)
            x, hx = x.squeeze(0), hx.squeeze(0)

        return self.policy_head(x), self.value_head(x), hx


class LSTMPolicy(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, lstm_hidden_size=64, act_fn=torch.nn.ReLU):
        super().__init__()

        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size), act_fn(),
            torch.nn.Linear(hidden_size, hidden_size), act_fn(),
        )

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = torch.nn.LSTM(hidden_size, lstm_hidden_size)

        self.policy_head = torch.nn.Linear(hidden_size, output_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def get_init_hx(self):
        return torch.zeros(2, self.lstm_hidden_size)

    def forward(self, x, hx):
        x = self.shared(x)

        if x.dim() == 1:
            hx, cx = hx[0], hx[1]
            x, hx, cx = x.view(1, 1, -1), hx.view(1, 1, -1), cx.view(1, 1, -1)
            x, (hx, cx) = self.lstm(x, (hx, cx))
            x, hx, cx = x.view(-1), hx.view(-1), cx.view(-1)
            hx = torch.stack([hx, cx])
        else:
            hx, cx = hx[:, 0], hx[:, 1]
            x, hx, cx = x.unsqueeze(0), hx.unsqueeze(0), cx.unsqueeze(0)
            x, (hx, cx) = self.lstm(x, (hx, cx))
            x, hx, cx = x.squeeze(0), hx.squeeze(0), cx.squeeze(0)
            hx = torch.stack([hx, cx])
        return self.policy_head(x), self.value_head(x), hx