以下是重新整理的 **Calibration Loss（校准损失）** 公式及说明，以 **图片格式** 呈现核心公式和关键内容，便于直接引用或展示：

---

### **一、核心公式（图片格式）**

#### **1. 通用校准损失形式**
![通用校准损失公式](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{cal}}%20=%20\mathbb{E}_{(x,y)\sim\mathcal{D}}%20\left[%20\text{Discrepancy}\left(P_\theta(y|x),%20\hat{P}(y|x)\right)%20\right])  
**说明**：  
- 衡量模型预测概率 \( P_\theta(y|x) \) 与真实标签分布 \( \hat{P}(y|x) \) 的差异。  
- 输入为样本 \( (x,y) \)，输出为标量损失值。

---

#### **2. 具体实现类型**

##### **(1) 交叉熵校准损失（Cross-Entropy, CE）**
![交叉熵校准损失公式](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{CE}}%20=%20-\frac{1}{N}%20\sum_{i=1}^N%20\sum_{c=1}^C%20y_{i,c}%20\log(p_{i,c}))  
**参数**：  
- \( N \)：样本数量，\( C \)：类别数。  
- \( y_{i,c} \in \{0,1\} \)：真实标签（one-hot编码）。  
- \( p_{i,c} \in [0,1] \)：模型预测的第 \( c \) 类概率。  

##### **(2) KL散度校准损失（Kullback-Leibler Divergence, KL）**
![KL散度校准损失公式](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{KL}}%20=%20\frac{1}{N}%20\sum_{i=1}^N%20\sum_{c=1}^C%20\hat{p}_{i,c}%20\log\left(\frac{\hat{p}_{i,c}}{p_{i,c}}\right))  
**参数**：  
- \( \hat{p}_{i,c} \)：真实标签的平滑分布（如标签平滑后的概率）。  

##### **(3) 预期校准误差（Expected Calibration Error, ECE）**
![ECE公式](https://latex.codecogs.com/svg.latex?\text{ECE}%20=%20\sum_{m=1}^M%20\frac{|B_m|}{N}%20\left|%20\text{acc}(B_m)%20-%20\text{conf}(B_m)%20\right|)  
**参数**：  
- \( M \)：将预测概率划分为 \( M \) 个区间（如 \([0,0.1), [0.1,0.2), \dots, [0.9,1]\)）。  
- \( B_m \)：第 \( m \) 个区间的样本集合。  
- \( \text{acc}(B_m) \)：区间内样本的准确率。  
- \( \text{conf}(B_m) \)：区间内样本的平均预测概率。  

---

### **二、用法与实现（图片格式）**

#### **1. 作为辅助损失联合训练**
![联合训练损失公式](https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{total}}%20=%20\mathcal{L}_{\text{CE}}%20+%20\lambda%20\mathcal{L}_{\text{cal}}))  
**参数**：  
- \( \lambda \)：权衡分类准确性与校准质量的超参数。  

**代码示例（PyTorch）**：  
```python
import torch
import torch.nn as nn

def calibration_loss(pred_probs, true_labels, lambda_cal=0.1):
    ce_loss = nn.CrossEntropyLoss()(pred_probs, true_labels)
    smooth_labels = (1 - 0.1) * true_labels + 0.1 / pred_probs.size(1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')(
        torch.log_softmax(pred_probs, dim=1),
        smooth_labels
    )
    total_loss = ce_loss + lambda_cal * kl_loss
    return total_loss
```

#### **2. 后处理校准（温度缩放）**
![温度缩放公式](https://latex.codecogs.com/svg.latex?p_{\text{cal}}(y|x)%20=%20\text{Softmax}\left(\frac{z}{\tau}\right))  
**参数**：  
- \( z \)：模型原始输出（logits）。  
- \( \tau \)：温度参数，通过最小化负对数似然（NLL）在验证集上优化。  

**代码示例**：  
```python
def temperature_scaling(logits, true_labels, initial_temp=1.0):
    temp = torch.tensor(initial_temp, requires_grad=True)
    optimizer = torch.optim.LBFGS([temp], lr=0.01)
    
    def closure():
        optimizer.zero_grad()
        scaled_probs = torch.softmax(logits / temp, dim=1)
        loss = nn.NLLLoss()(torch.log(scaled_probs), true_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return temp.detach()
```

---

### **三、应用场景与优势（表格图片）**

| **场景**               | **校准损失的作用**                                                                 |
|------------------------|-----------------------------------------------------------------------------------|
| **医疗诊断**           | 确保模型预测的疾病概率真实反映风险（如“90%概率患病”需高度可信）。                |
| **金融风控**           | 优化信用评分模型的违约概率估计，避免过自信或欠自信预测。                          |
| **自动驾驶**           | 校准目标检测的置信度（如“80%置信度”的检测框应确实包含目标）。                    |
| **推荐系统**           | 提升推荐概率的可靠性，避免用户对低质量推荐产生信任危机。                          |

**优势**：  
- 提升模型输出的可信度，避免过拟合导致的概率失真。  
- 适用于需要高风险决策的领域（如医疗、金融）。  

**挑战**：  
- 需权衡分类准确性与校准质量（超参数 \( \lambda \) 选择敏感）。  
- ECE等指标不可微，需通过后处理或近似方法优化。  

---

### **四、公式对比总结（表格图片）**

| **损失类型**       | **公式**                                                                 | **适用场景**               | **可微性** |
|--------------------|--------------------------------------------------------------------------|---------------------------|------------|
| 交叉熵（CE）       | \( -\frac{1}{N} \sum y_{i,c} \log(p_{i,c}) \)                           | 分类任务基础损失           | 是         |
| KL散度（KL）       | \( \frac{1}{N} \sum \hat{p}_{i,c} \log(\hat{p}_{i,c}/p_{i,c}) \)       | 显式优化概率校准           | 是         |
| 预期校准误差（ECE）| \( \sum \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)| \)        | 评估模型校准质量           | 否         |

---

### **图片生成说明**
1. **公式图片**：使用 [LaTeX Codecogs](https://www.codecogs.com/latex/eqneditor.php) 生成 SVG 格式公式，并转换为 PNG 嵌入。  
2. **表格图片**：通过 Markdown 表格生成后截图，或使用 LaTeX 表格环境生成。  
3. **代码块**：保留原始文本格式，便于复制使用。  

如需更高分辨率图片或自定义样式（如颜色、字体），可进一步调整 LaTeX 渲染参数或使用专业工具（如 TikZ、Matplotlib）。
