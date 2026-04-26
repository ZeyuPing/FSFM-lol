gptprompt v1

1. 模型架构详细绘图提示词（适合结构图/网络结构图）：
请绘制一个面向人脸伪造检测的深度学习模型架构图，核心为基于ViT的主干网络。输入为人脸图像，经过ViT主干后输出一个CLS token和多个patch tokens。模型包含两个分支：一是Forgery-Aware Token Fusion (FTF)分支，计算每个patch token与CLS token的余弦距离，选取top-k异常patch，经过softmax加权聚合得到local descriptor；二是Residual-View Consistency (RVC)分支，对输入图像施加高通滤波（如Laplacian），生成残差视图，残差视图与原图分别通过同一模型，输出logits。FTF分支将local descriptor与global descriptor（CLS token或全局池化）通过门控机制融合，最终送入分类头。RVC分支在训练时引入对称KL一致性损失，约束原图与残差视图的预测分布一致。整个架构无需额外标注或分割监督，保持轻量。

2. 流程图详细绘图提示词（适合流程/信息流/训练推理流程）：
请绘制一个人脸伪造检测方法的流程图，突出以下流程：输入一张人脸图像，分为两路处理。一路为原图，另一路为经过高通滤波（如Laplacian）的残差图。两路图像分别输入同一个ViT主干，提取CLS token和patch tokens。原图分支通过计算patch tokens与CLS token的余弦距离，选出top-k异常patch，softmax加权聚合为local descriptor，与global descriptor门控融合后送入分类头，输出伪造概率。残差分支同样输出logits。训练时，计算原图分支的交叉熵损失，并对两路输出的softmax分布施加对称KL一致性损失。推理时仅用原图分支。流程图需突出FTF分支的token选择与融合、RVC分支的残差生成与一致性约束，以及最终的损失组合。

----------------------------------------------------------------------------------------------------------------

gptprompt v2


---

### 1. **整体架构图（简洁）绘图提示词**
请绘制一个简洁的深度学习模型架构图，展示用于人脸伪造检测的FTF-RVC方法的整体结构。输入为人脸图像，经过ViT主干网络提取特征，输出CLS token和patch tokens。模型分为两个主要分支：

1. **FTF分支**：从patch tokens中选择top-k异常patch，计算local descriptor，并与global descriptor通过门控机制融合，最终送入分类头。
2. **RVC分支**：对输入图像施加高通滤波生成残差视图，残差视图与原图分别通过同一模型，输出logits。训练时引入对称KL一致性损失，约束两路预测分布一致。

图中需标注：
- ViT主干网络（共享权重）
- 输入图像、CLS token、patch tokens
- FTF分支和RVC分支的主要流程
- 输出为伪造检测分数（AUC/EER）

---

### 2. **流程图（相对简洁）绘图提示词**
请绘制一个人脸伪造检测方法的流程图，展示FTF-RVC方法的训练和推理流程。流程图需包含以下内容：

1. **输入**：人脸图像，分为两路处理：原图和高通滤波后的残差图。
2. **ViT主干**：两路图像分别通过共享权重的ViT主干，输出CLS token和patch tokens。
3. **FTF分支**：原图分支计算patch tokens与CLS token的余弦距离，选出top-k异常patch，softmax加权聚合为local descriptor，与global descriptor通过门控机制融合后送入分类头，输出伪造概率。
4. **RVC分支**：残差图分支输出logits，训练时计算对称KL一致性损失，约束两路分支的预测分布一致。
5. **训练与推理**：训练时包含交叉熵损失和一致性损失，推理时仅使用原图分支。

图中需标注：
- 各模块的输入输出
- 训练时的损失计算路径
- 推理时的单分支路径

---

### 3. **细节图绘图提示词**
#### 3.1 **FTF模块细节图**
请绘制FTF（Forgery-Aware Token Fusion）模块的细节图，展示其内部计算流程。输入为ViT主干输出的CLS token和patch tokens，具体流程如下：

1. **异常分数计算**：计算每个patch token与CLS token的余弦距离，得到每个patch的异常分数。
2. **Top-k异常patch选择**：根据异常分数从高到低排序，选出top-k异常patch。
3. **加权聚合**：对选出的top-k异常patch，使用softmax加权求和，生成local descriptor。
4. **门控融合**：将local descriptor与global descriptor（CLS token或全局池化）拼接，经过MLP和sigmoid生成门控权重，按权重融合local descriptor和global descriptor，得到最终的fused descriptor。

图中需标注：
- 输入（CLS token和patch tokens）
- 异常分数计算公式：$a_i = 1 - \cos(p_i, c)$
- Top-k选择和softmax加权公式
- 门控融合公式：$\alpha = \text{sigmoid}(\text{MLP}([g; l]))$，$f = \alpha \cdot g + (1 - \alpha) \cdot l$
- 输出（fused descriptor）

---

#### 3.2 **RVC模块细节图**
请绘制RVC（Residual-View Consistency）模块的细节图，展示其内部计算流程。输入为原始人脸图像，具体流程如下：

1. **高通滤波**：对输入图像施加高通滤波（如Laplacian），生成残差视图。
2. **共享模型前向传播**：原图和残差图分别通过共享权重的ViT主干，输出logits。
3. **一致性损失计算**：对两路logits的softmax分布计算对称KL散度，公式为：
   $$L_{rvc} = 0.5 \cdot [KL(p || q) + KL(q || p)]$$
   其中，$p = \text{softmax}(z / T)$，$q = \text{softmax}(z_r / T)$。

图中需标注：
- 输入（原图和残差图）
- 高通滤波器（如Laplacian的卷积核）
- ViT主干（共享权重）
- KL一致性损失公式
- 输出（训练时的损失路径）

---

#### 3.3 **训练与推理流程细节图**
请绘制训练与推理流程的细节图，展示FTF-RVC方法的训练和推理过程。具体内容包括：

1. **训练流程**：
   - 输入原图和残差图，分别通过FTF分支和RVC分支。
   - 计算交叉熵损失（原图分支）和对称KL一致性损失（两路分支）。
   - 总损失公式：$L = L_{ce} + \lambda \cdot L_{rvc}$，其中$\lambda$为平衡权重。

2. **推理流程**：
   - 输入原图，仅通过FTF分支，输出伪造检测分数。

图中需标注：
- 各分支的输入输出
- 损失计算路径
- 推理时的单分支路径
