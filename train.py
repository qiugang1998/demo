from model import *
from mask import *
import time

source_vocab = target_vocab = 39
N = 3
embedding_dim = 32
hidden_dim = 64
head = 4
dropout = 0.1
model = Transformer(source_vocab, target_vocab, N, embedding_dim, hidden_dim, head, dropout)
model = model.cuda()


loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.002)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
loss_func = loss_func.cuda()



# 训练数据集

dataset = Dataset(100000)


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=8,
                                     drop_last=True,
                                     shuffle=True,
                                    collate_fn=None)




start_time = time.time()
for epoch in range(10):
    
    print("------第 {} 轮训练开始------".format(epoch + 1))
    
    i = 0
    
    for x, y in loader:
        
        x = x.cuda()
        y = y.cuda()

        # 计算 x、y 的掩码矩阵
        x_mask = mask_pad(x)
        y_mask =  mask_tril(y[:, :-1])
        
        x_mask = x_mask.cuda()
        y_mask = y_mask.cuda()


        # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
        # [8, 50, 39]
        pred_y = model(x, y[:, :-1], x_mask, y_mask)

        # [8, 50, 39] -> [400, 39]
        pred_y = pred_y.reshape(-1, 39)

        # [8, 50] -> [400]
        y_truth = y[:, 1:].reshape(-1)


        # 忽略掉 <PAD>
        select = y_truth != zidian_y['<PAD>']
        y_truth = y_truth[select]
        pred_y = pred_y[select]


        loss = loss_func(pred_y, y_truth)
        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 200 == 0:
            pred_y = pred_y.argmax(1)
            correct_num = (pred_y == y_truth).sum().item()
            accuracy = correct_num / len(pred_y)
            lr = optim.param_groups[0]['lr']
            print("epoch:{epoch}, i:{i}, lr:{lr}, loss:{loss}, accuracy:{accuracy}\n"
                  .format(epoch=epoch, i=i, lr=lr, loss=loss.item(), accuracy=accuracy))
            
        i += 1
    
    sched.step()        
    torch.save(model, "model_{}.pth".format(epoch + 1))

end_time = time.time()
print("训练结束, 用时:{time}".format(time=(end_time - start_time) / 3600))
