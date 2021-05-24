# 作业三：LSTM--预测期货涨跌

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time

# 参数设置：
seq_len = 20  # 序列的长度（天数）
input_size = 36   # 输入的维度（6*6）
hidden_size = 30  # ht,ct的维度
output_size = 6   # 线性层激活后的函数

batch_size = 64   # 批的大小
epoch_num = 20       # 迭代的次数

# 找出计算梯度失败的那一步操作
torch.autograd.set_detect_anomaly(True)
# 开始计时：
time_start = time.time()

# 固定随机数种子：
torch.manual_seed(10)

## （1）数据读取：
train_dir = '../Data/Train/Train_data/'   # 训练集数据所在的文件夹
test_dir = '../Data/Validation/Validation_data/'   #测试集特征数据所在的文件夹
test_label_file = '../Data/validation_label_file.csv'
Mental = ['Aluminium', 'Copper', 'Lead', 'Nickel', 'Tin', 'Zinc']

# 用exec()函数实现金属文件的循环读取：
for mental in Mental:
    # 训练集数据：
    # 读取数据，剔除含空值的行
    exec(mental + '_OI = pd.read_csv(train_dir + \'LME' + mental + '_OI_train.csv\').'
                                                                   'dropna(axis=0, how=\'any\').values[:, 1:]')
    # 3M：
    exec(mental + '_3M = pd.read_csv(train_dir + \'LME' + mental + '3M_train.csv\').'
                                                                   'dropna(axis=0, how=\'any\').values[:, 1:]')
    # 标签1d,20d,60d：
    exec(mental + '_1d = pd.read_csv(train_dir + \'Label_LME' + mental + '_train_1d.csv\').'
                                                                         'dropna(axis=0, how=\'any\').values[:, 1:]')
    exec(mental + '_20d = pd.read_csv(train_dir + \'Label_LME' + mental + '_train_20d.csv\').'
                                                                          'dropna(axis=0, how=\'any\').values[:, 1:]')
    exec(mental + '_60d = pd.read_csv(train_dir + \'Label_LME' + mental + '_train_60d.csv\').'
                                                                          'dropna(axis=0, how=\'any\').values[:, 1:]')

    # 测试集数据：
    # 测试集的特征文件，日期跨度：【2018-1-2, 2018-12-31】，没有空值
    exec(mental + '_test_OI = pd.read_csv(test_dir + \'LME' + mental + '_OI_validation.csv\').values[:, 1:]')
    exec(mental + '_test_3M = pd.read_csv(test_dir + \'LME' + mental + '3M_validation.csv\').values[:, 1:]')
# 测试集的标签：
Test_label_all = pd.read_csv(test_label_file).values


## （2）数据处理：
# 标签文件的日期个数都是3790个
# 3M数据的长度：3395,3545,3547,3546,3545,3547
# OI数据的长度：3416,3419,3415,3415,3414,3418

# 找出6个金属的OI文件与其它文件日期的交集：
# 将数据转成set（集合）形式
Date_intersection = set(Aluminium_1d[:, 0])
for mental in Mental:
    exec('Date_intersection = Date_intersection.intersection(set(' + mental + '_OI[:, 0]))')
    exec('Date_intersection = Date_intersection.intersection(set(' + mental + '_3M[:, 0]))')
# 将数据转成numpy数组：
Date_intersection = np.array(list(Date_intersection))
# 对日期交集进行升序排序：
Date_intersection = np.sort(Date_intersection)

# 日期交集的个数：
Intersection_len = len(Date_intersection)

# 找出日期交集在各数据集中的下标：
for mental in Mental:
    exec(mental + '_OI_index = []')
    exec(mental + '_3M_index = []')
    for date in Date_intersection:
        exec(mental + '_OI_index.append(np.where(' + mental + '_OI[:, 0] == date)[0][0])')
        exec(mental + '_3M_index.append(np.where(' + mental + '_3M[:, 0] == date)[0][0])')
# Label数据用相同的下标：
Label_index = []
for date in Date_intersection:
    Label_index.append(np.where(Aluminium_1d[:, 0] == date)[0][0])

# 对数据进行截取：
for mental in Mental:
    exec(mental + '_OI = ' + mental + '_OI[' + mental + '_OI_index, :]')
    exec(mental + '_3M = ' + mental + '_3M[' + mental + '_3M_index, :]')
    exec(mental + '_1d = ' + mental + '_1d[Label_index, :]')
    exec(mental + '_20d = ' + mental + '_20d[Label_index, :]')
    exec(mental + '_60d = ' + mental + '_60d[Label_index, :]')

# 将测试集的label数据分开：
Test_len = len(Aluminium_test_3M)
# 将总的Label划分到18个label数据集中：
for num in range(18):
    Label_subset = Test_label_all[num*Test_len:(num+1)*Test_len, :]
    [name, _, days, _, _, _] = Label_subset[0, 0].split('-')
    for i in range(Test_len):
        Label_subset[i, 0] = Label_subset[i, 0][-10:]  # 保留时间信息
    exec(name[3:] + '_test_' + days + ' = Label_subset')

# 将数据拼接在一起：
# 第一列为日期：
Train_all_feature = Aluminium_OI[:, :1]   # 每一行的数据依次是：日期，['Aluminium', 'Copper', 'Lead', 'Nickel', 'Tin', 'Zinc']的
                                          # 六个特征[OI, Open, High, Low, Close, Volume]
Train_all_label_1d = Aluminium_OI[:, :1]  # 每一行的数据依次是：日期，['Aluminium', 'Copper', 'Lead', 'Nickel', 'Tin', 'Zinc']的Label
Train_all_label_20d = Aluminium_OI[:, :1]
Train_all_label_60d = Aluminium_OI[:, :1]

Test_feature = Aluminium_test_OI[:, :1]
Test_label_1d = Aluminium_test_OI[:, :1]
Test_label_20d = Aluminium_test_OI[:, :1]
Test_label_60d = Aluminium_test_OI[:, :1]


for mental in Mental:
    exec('Train_all_feature = np.hstack((Train_all_feature, ' + mental + '_OI[:, 1:], ' + mental + '_3M[:, 1:]))')
    exec('Train_all_label_1d = np.hstack((Train_all_label_1d, ' + mental + '_1d[:, 1:]))')
    exec('Train_all_label_20d = np.hstack((Train_all_label_20d, ' + mental + '_20d[:, 1:]))')
    exec('Train_all_label_60d = np.hstack((Train_all_label_60d, ' + mental + '_60d[:, 1:]))')

    # 测试集：
    exec('Test_feature = np.hstack((Test_feature, ' + mental + '_test_OI[:, 1:], ' + mental + '_test_3M[:, 1:]))')
    exec('Test_label_1d = np.hstack((Test_label_1d, ' + mental + '_test_1d[:, 1:]))')
    exec('Test_label_20d = np.hstack((Test_label_20d, ' + mental + '_test_20d[:, 1:]))')
    exec('Test_label_60d = np.hstack((Test_label_60d, ' + mental + '_test_60d[:, 1:]))')

# 测试集数据的填充：
# 为了使得训练集第一天的label可以预测，在测试集前面添加(seq_len - 1)列训练集的数据：
Test_feature = np.vstack((Train_all_feature[-(seq_len - 1):, :], Test_feature))
Test_label_1d = np.vstack((Train_all_label_1d[-(seq_len - 1):, :], Test_label_1d))
Test_label_20d = np.vstack((Train_all_label_20d[-(seq_len - 1):, :], Test_label_20d))
Test_label_60d = np.vstack((Train_all_label_60d[-(seq_len - 1):, :], Test_label_60d))

# 数据标准化（按列进行）--（min-max标准化）：
for j in range(1, 37):
    fea_max = max(Train_all_feature[:, j])
    fea_min = min(Train_all_feature[:, j])

    Train_all_feature[:, j] = (Train_all_feature[:, j] - fea_min) / (fea_max - fea_min)
    # 测试集用训练集的最大值、最小值：
    Test_feature[:, j] = (Test_feature[:, j] - fea_min) / (fea_max - fea_min)

# 根据序列长度，划分数据：
Train_all_x = []
Train_all_y_1d = []
Train_all_y_20d = []
Train_all_y_60d = []

Test_x = []
Test_y_1d = []
Test_y_20d = []
Test_y_60d = []

for i in range(len(Train_all_feature) - seq_len + 1):
    Train_all_x.append(Train_all_feature[i:i+seq_len, 1:])
    Train_all_y_1d.append(Train_all_label_1d[i + seq_len - 1, 1:])
    Train_all_y_20d.append(Train_all_label_20d[i + seq_len - 1, 1:])
    Train_all_y_60d.append(Train_all_label_60d[i + seq_len - 1, 1:])

for i in range(len(Test_feature) - seq_len + 1):
    Test_x.append(Test_feature[i:i+seq_len, 1:])
    Test_y_1d.append(Test_label_1d[i + seq_len - 1, 1:])
    Test_y_20d.append(Test_label_20d[i + seq_len - 1, 1:])
    Test_y_60d.append(Test_label_60d[i + seq_len - 1, 1:])

# 按70%,30%划分训练集和验证集:
Train_all_len = len(Train_all_x)
Train_len = int(Train_all_len * 0.7)

Train_x = Train_all_x[0:Train_len]
Train_y_1d = Train_all_y_1d[0:Train_len]
Train_y_20d = Train_all_y_20d[0:Train_len]
Train_y_60d = Train_all_y_60d[0:Train_len]
Valid_x = Train_all_x[Train_len:]
Valid_y_1d = Train_all_y_1d[Train_len:]
Valid_y_20d = Train_all_y_20d[Train_len:]
Valid_y_60d = Train_all_y_60d[Train_len:]

Valid_len = len(Valid_x)
Test_len = len(Test_x)

# 将数据转成tensor格式，并生成枚举器：
# 1d模型：
Train_1d = torch.utils.data.TensorDataset(torch.tensor(Train_x, dtype=torch.float32),
                                          torch.tensor(Train_y_1d, dtype=torch.float32))
# 特征的数据类型：float32，标签的数据类型：long。
Train_1d_Loader = torch.utils.data.DataLoader(dataset=Train_1d, batch_size=batch_size)

Valid_1d = torch.utils.data.TensorDataset(torch.tensor(Valid_x, dtype=torch.float32),
                                          torch.tensor(Valid_y_1d, dtype=torch.float32))
Valid_1d_Loader = torch.utils.data.DataLoader(dataset=Valid_1d, batch_size=batch_size)

Test_1d = torch.utils.data.TensorDataset(torch.tensor(Test_x, dtype=torch.float32),
                                         torch.tensor(Test_y_1d, dtype=torch.float32))
Test_1d_Loader = torch.utils.data.DataLoader(dataset=Test_1d, batch_size=batch_size)

# 20d模型：
Train_20d = torch.utils.data.TensorDataset(torch.tensor(Train_x, dtype=torch.float32),
                                           torch.tensor(Train_y_20d, dtype=torch.float32))
Train_20d_Loader = torch.utils.data.DataLoader(dataset=Train_20d, batch_size=batch_size)

Valid_20d = torch.utils.data.TensorDataset(torch.tensor(Valid_x, dtype=torch.float32),
                                           torch.tensor(Valid_y_20d, dtype=torch.float32))
Valid_20d_Loader = torch.utils.data.DataLoader(dataset=Valid_20d, batch_size=batch_size)

Test_20d = torch.utils.data.TensorDataset(torch.tensor(Test_x, dtype=torch.float32),
                                          torch.tensor(Test_y_20d, dtype=torch.float32))
Test_20d_Loader = torch.utils.data.DataLoader(dataset=Test_20d, batch_size=batch_size)

# 60d模型：
Train_60d = torch.utils.data.TensorDataset(torch.tensor(Train_x, dtype=torch.float32),
                                           torch.tensor(Train_y_60d, dtype=torch.float32))
Train_60d_Loader = torch.utils.data.DataLoader(dataset=Train_60d, batch_size=batch_size)

Valid_60d = torch.utils.data.TensorDataset(torch.tensor(Valid_x, dtype=torch.float32),
                                           torch.tensor(Valid_y_60d, dtype=torch.float32))
Valid_60d_Loader = torch.utils.data.DataLoader(dataset=Valid_60d, batch_size=batch_size)

Test_60d = torch.utils.data.TensorDataset(torch.tensor(Test_x, dtype=torch.float32),
                                          torch.tensor(Test_y_60d, dtype=torch.float32))
Test_60d_Loader = torch.utils.data.DataLoader(dataset=Test_60d, batch_size=batch_size)


## （3）自定义lstm函数，初始参数从nn.LSTM中调取：
KuLSTM = nn.LSTM(input_size, hidden_size, batch_first=True)
Ku_dict = KuLSTM.state_dict()

class mylstm(nn.Module):
    def __init__(self, Ku_dict):
        super(mylstm, self).__init__()

        weight_ih = Ku_dict['weight_ih_l0']  # 大小[4*hidden_size, input_size] --四个参数组合在了一起[W_ii|W_if|W_ig|W_io]
                                             # -- [记忆门，遗忘门，感知层，输出门]
        weight_hh = Ku_dict['weight_hh_l0']  # 大小[4*hidden_size, hidden_size]
        bias_ih = Ku_dict['bias_ih_l0']      # 大小[4*hidden_size]
        bias_hh = Ku_dict['bias_hh_l0']      # 大小[4*hidden_size]

        # 对参数进行分割：
        [W_ii, W_if, W_ig, W_io] = weight_ih.split(hidden_size)  # 大小均为[hidden_size, input_size]
        [W_hi, W_hf, W_hg, W_ho] = weight_hh.split(hidden_size)  # [hidden_size, hidden_size]
        [b_ii, b_if, b_ig, b_io] = bias_ih.split(hidden_size)    # [hidden_size]
        [b_hi, b_hf, b_hg, b_ho] = bias_hh.split(hidden_size)    # [hidden_size]

        self.W_ii, self.W_if, self.W_ig, self.W_io = nn.Parameter(W_ii), nn.Parameter(W_if), nn.Parameter(W_ig), nn.Parameter(W_io)
        self.W_hi, self.W_hf, self.W_hg, self.W_ho = nn.Parameter(W_hi), nn.Parameter(W_hf), nn.Parameter(W_hg), nn.Parameter(W_ho)
        self.b_ii, self.b_if, self.b_ig, self.b_io = nn.Parameter(b_ii), nn.Parameter(b_if), nn.Parameter(b_ig), nn.Parameter(b_io)
        self.b_hi, self.b_hf, self.b_hg, self.b_ho = nn.Parameter(b_hi), nn.Parameter(b_hf), nn.Parameter(b_hg), nn.Parameter(b_ho)

    def forward(self, x):
        # 给h_n, c_n附初值：
        hn = Variable(torch.zeros(1, x.size(0), hidden_size))  # [1, batch_size, hidden_size]
        cn = Variable(torch.zeros(1, x.size(0), hidden_size))
        # 给out赋初值：
        out = Variable(torch.zeros(x.size(0), seq_len, hidden_size))
        for i in range(seq_len):
            input_i = x[:, i, :]
            i_t = torch.sigmoid(torch.mm(input_i, self.W_ii.t()) + self.b_ii + torch.mm(hn[0], self.W_hi.t()) + self.b_hi)  # i_t[batch_size, hidden_size]
            f_t = torch.sigmoid(torch.mm(input_i, self.W_if.t()) + self.b_if + torch.mm(hn[0], self.W_hf.t()) + self.b_hf)
            g_t = torch.tanh(torch.mm(input_i, self.W_ig.t()) + self.b_ig + torch.mm(hn[0], self.W_hg.t()) + self.b_hg)
            o_t = torch.sigmoid(torch.mm(input_i, self.W_io.t()) + self.b_io + torch.mm(hn[0], self.W_ho.t()) + self.b_ho)
            cn = (f_t * cn[0] + i_t * g_t).view(1, cn.size(1), -1)
            hn = (o_t * torch.tanh(cn[0])).view(1, hn.size(1), -1)
            out[:, i, :] = hn.view(-1, hidden_size)

        return out, (hn, cn)


## （4）模型定义：
class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.lstm = mylstm(Ku_dict)                  # 自己定义的lstm类
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y_out, _ = self.lstm(x)
        y_out = y_out[:, -1, :].view(y_out.size(0), y_out.size(-1))
        y_out = torch.sigmoid(self.linear(y_out))   # [batch_size, 6]

        return y_out


## （5）设置训练函数：
# 对象实例化：
model = MyLSTM()

# 设置优化器：
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

# 设置训练函数：
def train_and_loss(Train_Loader, Valid_Loader, Test_Loader):
    model.train()
    train_loss_all = []
    train_right_num = 0
    for batch_index, (x_train, y_train) in enumerate(Train_Loader):
        x_train, y_train = Variable(x_train), Variable(y_train)
        optimizer.zero_grad()
        output = model(x_train)
        loss = F.binary_cross_entropy(output, y_train)
        train_loss_all.append(loss)

        output_label = (2 * output).floor()  # 以0.5为阈值，小于0.5预测标签为0，大于0.5标签为1
        train_right_num += (output_label == y_train).sum()

        # 反向传播：
        loss.backward()
        optimizer.step()
    # 训练集的平均Loss:
    train_loss = sum(train_loss_all) / len(train_loss_all)

    # 验证集：
    valid_loss_all = []
    valid_right_num = 0
    for batch_index, (x_valid, y_valid) in enumerate(Valid_Loader):
        x_valid, y_valid = Variable(x_valid), Variable(y_valid)
        output_valid = model(x_valid)

        # 计算loss:
        loss = F.binary_cross_entropy(output_valid, y_valid)
        valid_loss_all.append(loss)

        # 预测正确的个数：
        output_label = (2 * output_valid).floor()  # 以0.5为阈值，小于0.5预测标签为0，大于0.5标签为1
        valid_right_num += (output_label == y_valid).sum()

    # 验证集的平均Loss:
    valid_loss = sum(valid_loss_all) / len(valid_loss_all)

    # 测试集：
    test_loss_all = []
    test_right_num = 0
    for batch_index, (x_test, y_test) in enumerate(Test_Loader):
        x_test, y_test = Variable(x_test), Variable(y_test)
        output_test = model(x_test)

        # 计算loss:
        loss = F.binary_cross_entropy(output_test, y_test)
        test_loss_all.append(loss)

        # 预测正确的个数：
        output_label = (2 * output_test).floor()  # 以0.5为阈值，小于0.5预测标签为0，大于0.5标签为1
        test_right_num += (output_label == y_test).sum()

    # 验证集的平均Loss:
    test_loss = sum(test_loss_all) / len(test_loss_all)

    return train_loss, valid_loss, test_loss, train_right_num, valid_right_num, test_right_num


## （6）进行训练：
loss_all_train_1d = []
loss_all_valid_1d = []
loss_all_train_20d = []
loss_all_valid_20d = []
loss_all_train_60d = []
loss_all_valid_60d = []

ratio_all_valid_1d = []
ratio_all_test_1d = []
ratio_all_valid_20d = []
ratio_all_test_20d = []
ratio_all_valid_60d = []
ratio_all_test_60d = []

for epoch in range(epoch_num):
    # 1d：
    [train_loss, valid_loss, test_loss, train_right_num, valid_right_num, test_right_num] = \
        train_and_loss(Train_1d_Loader, Valid_1d_Loader, Test_1d_Loader)
    loss_all_train_1d.append(train_loss)
    loss_all_valid_1d.append(valid_loss)
    ratio_all_valid_1d.append(int(valid_right_num) / Valid_len / 6)
    ratio_all_test_1d.append(int(test_right_num) / Test_len / 6)

    # 20d:
    [train_loss, valid_loss, test_loss, train_right_num, valid_right_num, test_right_num] = \
        train_and_loss(Train_20d_Loader, Valid_20d_Loader, Test_20d_Loader)
    loss_all_train_20d.append(train_loss)
    loss_all_valid_20d.append(valid_loss)
    ratio_all_valid_20d.append(int(valid_right_num) / Valid_len / 6)
    ratio_all_test_20d.append(int(test_right_num) / Test_len / 6)

    # 60d:
    [train_loss, valid_loss, test_loss, train_right_num, valid_right_num, test_right_num] = \
        train_and_loss(Train_60d_Loader, Valid_60d_Loader, Test_60d_Loader)
    loss_all_train_60d.append(train_loss)
    loss_all_valid_60d.append(valid_loss)
    ratio_all_valid_60d.append(int(valid_right_num) / Valid_len / 6)
    ratio_all_test_60d.append(int(test_right_num) / Test_len / 6)

    time_epoch = time.time()
    print('第{}轮，时间过了{}秒；'.format(epoch + 1, time_epoch - time_start))


# 输出验证集正确率最高的模型对应的测试集的正确率：
index_1d = ratio_all_valid_1d.index(max(ratio_all_valid_1d))
index_20d = ratio_all_valid_20d.index(max(ratio_all_valid_20d))
index_60d = ratio_all_valid_60d.index(max(ratio_all_valid_60d))

print('\n******************** 模型准确率 ********************')
print('(1)1d的测试集准确率：{}%'.format(ratio_all_test_1d[index_1d]*100))
print('(2)20d的测试集准确率：{}%'.format(ratio_all_test_20d[index_1d]*100))
print('(3)60d的测试集准确率：{}%'.format(ratio_all_test_60d[index_1d]*100))


## (7)验证模型输出与调库结果一致：
input_test = Train_1d.tensors[0][0:1]
# 库的输出结果：
out_Ku, (hn_Ku, cn_Ku) = KuLSTM(input_test)
# 自己模型的输出结果：
test_model = mylstm(Ku_dict)
out_test, (hn_test, cn_test) = test_model(input_test)
print('\n******************** 验证自定义模型与调库的输出是否一致 ********************')
print('（一）调库输出的hn:')
print(hn_Ku)
print('自定义模型输出的hn：')
print(hn_test)
print('hn的最大差值：{}'.format(abs(hn_Ku - hn_test).max()))

print('\n（二）调库输出的cn:')
print(cn_Ku)
print('自定义模型输出的cn：')
print(cn_test)
print('cn的最大差值：{}'.format(abs(cn_Ku - cn_test).max()))


## (8)绘图：
# （1）1d的Loss曲线：
x_label = np.linspace(1, epoch_num, epoch_num)
plt.figure()
plt.plot(x_label, loss_all_train_1d, color='b', label='Train loss')
plt.plot(x_label, loss_all_valid_1d, color='r', label='Valid loss')
plt.title('Loss via echo(1d)')
plt.xlabel('echo')
plt.ylabel('Loss')
plt.legend()

# （2）20d的Loss曲线：
plt.figure()
plt.plot(x_label, loss_all_train_20d, color='b', label='Train loss')
plt.plot(x_label, loss_all_valid_20d, color='r', label='Valid loss')
plt.title('Loss via echo(20d)')
plt.xlabel('echo')
plt.ylabel('Loss')
plt.legend()

# （3）60d的Loss曲线：
plt.figure()
plt.plot(x_label, loss_all_train_60d, color='b', label='Train loss')
plt.plot(x_label, loss_all_valid_60d, color='r', label='Valid loss')
plt.title('Loss via echo(60d)')
plt.xlabel('echo')
plt.ylabel('Loss')
plt.legend()

# （4）1d的正确率：
plt.figure()
plt.plot(x_label, ratio_all_valid_1d, color='b', label='Valid_right_ratio')
plt.plot(x_label, ratio_all_test_1d, color='r', label='Test_right_ratio')
plt.title('Right ratio(1d)')
plt.xlabel('echo')
plt.ylabel('Ratio')
plt.legend()

# （5）20d的正确率：
plt.figure()
plt.plot(x_label, ratio_all_valid_20d, color='b', label='Valid_right_ratio')
plt.plot(x_label, ratio_all_test_20d, color='r', label='Test_right_ratio')
plt.title('Right ratio(20d)')
plt.xlabel('echo')
plt.ylabel('Ratio')
plt.legend()

# （6）1d的正确率：
plt.figure()
plt.plot(x_label, ratio_all_valid_60d, color='b', label='Valid_right_ratio')
plt.plot(x_label, ratio_all_test_60d, color='r', label='Test_right_ratio')
plt.title('Right ratio(60d)')
plt.xlabel('echo')
plt.ylabel('Ratio')
plt.legend()
plt.show()

