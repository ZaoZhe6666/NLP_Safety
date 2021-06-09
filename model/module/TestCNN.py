import torch
import torch.nn as nn
class TextCNN(nn.Module):
    def __init__(self, vec_dim, filter_num, sentence_max_size, label_size, kernel_list):
        """

        :param vec_dim: 词向量的维度
        :param filter_num: 每种卷积核的个数
        :param sentence_max_size:一篇文章的包含的最大的词数量
        :param label_size:标签个数，全连接层输出的神经元数量=标签个数
        :param kernel_list:卷积核列表
        """
        super(TextCNN, self).__init__()
        chanel_num = 1
        # nn.ModuleList相当于一个卷积的列表，相当于一个list
        # nn.Conv1d()是一维卷积。in_channels：词向量的维度， out_channels：输出通道数
        # nn.MaxPool1d()是最大池化，此处对每一个向量取最大值，所有kernel_size为卷积操作之后的向量维度
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chanel_num, filter_num, (kernel, vec_dim)),
            nn.ReLU(),
            # 经过卷积之后，得到一个维度为sentence_max_size - kernel + 1的一维向量
            nn.MaxPool2d((sentence_max_size - kernel + 1, 1))
        )
            for kernel in kernel_list])
        # 全连接层，因为有2个标签
        self.fc = nn.Linear(filter_num * len(kernel_list), label_size)
        # dropout操作，防止过拟合
        self.dropout = nn.Dropout(0.5)
        # 分类
        self.sm = nn.Softmax(0)

    def forward(self, x):
        # Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        in_size = x.size(0)  # x.size(0)，表示的是输入x的batch_size
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)  # 设经过max pooling之后，有output_num个数，将out变成(batch_size,output_num)，-1表示自适应
        out = F.dropout(out)
        out = self.fc(out)  # nn.Linear接收的参数类型是二维的tensor(batch_size,output_num),一批有多少数据，就有多少行
        return out
def getModel():
    vec_dim = 300
    filter_num = 100 # 卷积核个数
    sentence_max_size = 300 # 每篇文章最大词数量
    label_size = 2
    kernel_list = [3, 4, 5] # 卷积核大小
    return TextCNN(vec_dim, filter_num, sentence_max_size, label_size, kernel_list)