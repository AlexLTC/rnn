import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size,  output_size, num_layers = 1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = self.num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input_size, hidden_size):
        x = self.embedding(input_size, hidden_size)
        output, hidden = self.rnn(x, hidden_size)
        
        # output 保有所有時間步的資料，而這邊只取最後一步
        output = output[:, -1, :]
        output = self.fc(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        # [layers, batch, hidden]
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))


rnn = SimpleRNN(input_size = 1, hidden_size = 2, output_size = 26)
criterion = nn.NLLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.001)
loss = 0
hidden = rnn.initHidden()

for t in range(len(seq) -1):
    # x size: [batch_size = 1, time_step = 1, data_dimension = 1]
    x = Variable(torch.LongTensor([seq[t]]).unsqueeze(0))

    # y size: [batch_size, data_dimension]
    y = Variable(torch.LongTensor([seq[t+1]]))
    output, hidden = rnn(x, hidden)

loss += criterion(output, y)

# 計算每個資源的損失值
loss = 1 * loss / len(seq)

optimizer.zero_grad()  # 清空梯度
loss.backward()
optimizer.step()  # 一步梯度下降
