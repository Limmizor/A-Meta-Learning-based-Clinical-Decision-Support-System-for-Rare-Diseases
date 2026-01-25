import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from maml_service import SimpleCNN, forward_with_weights

# 设置随机种子（保证可复现）
torch.manual_seed(42)
np.random.seed(42)

# 数据加载与任务生成（与您的代码相同）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

def create_task(dataset, n_way=5, k_shot=5, query_size=15):
    """生成一个n-way k-shot任务"""
    # 随机选择n个类别
    classes = torch.randperm(10)[:n_way]
    
    support_x, support_y = [], []
    query_x, query_y = [], []
    
    for cls in classes:
        # 获取当前类别的所有样本索引
        idx = (dataset.targets == cls).nonzero().squeeze()
        # 随机打乱并选择样本
        perm = torch.randperm(len(idx))
        selected = idx[perm][:k_shot + query_size]
        
        # 分割支持集和查询集
        support_x.extend(dataset[i][0] for i in selected[:k_shot])
        support_y.extend([cls] * k_shot)
        query_x.extend(dataset[i][0] for i in selected[k_shot:k_shot+query_size])
        query_y.extend([cls] * query_size)
    
    # 转换为Tensor
    support_x = torch.stack(support_x)
    support_y = torch.tensor(support_y)
    query_x = torch.stack(query_x)
    query_y = torch.tensor(query_y)
    
    return (support_x, support_y), (query_x, query_y)

# MAML训练循环（与您的代码相同）
def maml_train(model, train_dataset, epochs=10, inner_steps=1, inner_lr=0.01, meta_lr=0.001):
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        
        # 每个epoch训练5个任务
        for _ in range(5):
            # 生成一个任务（5类，每类5样本支持集 + 15样本查询集）
            (support_x, support_y), (query_x, query_y) = create_task(train_dataset)
            
            # 克隆初始参数（用于内循环快速更新）
            fast_weights = {n: p.clone() for n, p in model.named_parameters()}
            
            # ----------- 内循环：快速适应 -----------
            for _ in range(inner_steps):
                # 前向传播计算支持集损失
                output = model.forward_with_weights(support_x, fast_weights)
                loss = loss_fn(output, support_y)
                
                # 计算梯度并更新快速权重
                grads = torch.autograd.grad(
                    loss, 
                    fast_weights.values(), 
                    create_graph=True,
                    allow_unused=True  # 关键修复：允许未使用的张量
                )
                
                # 更新快速权重
                fast_weights = {
                    n: w - inner_lr * (g if g is not None else torch.zeros_like(w))
                    for (n, w), g in zip(fast_weights.items(), grads)
                }
            
            # ----------- 外循环：元优化 -----------
            # 使用快速权重计算查询集损失
            query_output = model.forward_with_weights(query_x, fast_weights)
            query_loss = loss_fn(query_output, query_y)
            
            # 反向传播更新初始参数
            meta_optimizer.zero_grad()
            query_loss.backward()
            meta_optimizer.step()
            
            epoch_loss += query_loss.item()
        
        # 记录平均损失
        avg_loss = epoch_loss / 5
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return train_losses

if __name__ == '__main__':
    # 创建模型目录
    os.makedirs('./models', exist_ok=True)
    
    # 初始化模型
    model = SimpleCNN()
    
    # 添加前向传播辅助函数
    SimpleCNN.forward_with_weights = forward_with_weights
    
    # 训练模型
    print("开始训练MAML模型...")
    train_losses = maml_train(model, train_dataset, epochs=10)
    
   # 保存训练损失图像
    os.makedirs('./static/images', exist_ok=True)  # 确保目录存在
    plt.plot(train_losses)
    plt.title("MAML Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('./static/images/maml_training_loss.png')
    
    # 保存模型
    model_path = './models/maml_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")