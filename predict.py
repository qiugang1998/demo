# 加载模型
model = torch.load("model_1.pth")
model.eval()



# 测试数据集

x_test, y_test, x_real, y_real = create_data(10000)
x_test = x_test.cuda()
y_test = y_test.cuda() 

def predict(x):


    mask_x = mask_pad(x)

    # 初始化输出,这个是固定值
    target = [zidian_y['<SOS>']] + [zidian_y['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0).cuda()

    # x编码,添加位置信息
    x = model.embed_x(x)

    # 编码层计算,维度不变
    x = model.encoder(x, mask_x)

    # 遍历生成第1个词到第49个词
    for i in range(49):

        y = target

        mask_y = mask_tril(y)

        # y编码,添加位置信息
        y = model.embed_y(y)

        # 解码层计算
        y = model.decoder(x, y, mask_x, mask_y)

        # 全连接输出,39分类
        out = model.output(y)

        # 取出当前词的输出
        # [1, 50, 39] -> [1, 39]
        out = out[:, i, :]

        # 取出分类结果
        # [1, 39] -> [1]
        out = out.argmax(dim=1).detach()
        
        # 预测到结束符就停止预测
        if out == zidian_y['<EOS>']: 
            return target 
        
        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    return target
