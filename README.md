
DL基础课程2024 最终任务"脑电波分类"

## 更新

- 2024.06.18 关于数据集详情的补充说明
  - 数据集的论文在这里(也可以从数据集链接找到):
	- https://elifesciences.org/articles/82580
  - 分发的数据已经经过论文中"MEG数据预处理和清洗"部分描述的处理。
	- 因此采样率是200Hz
  - 如"MEG数据采集"部分所述,通道坐标系使用的是[CTF 275 MEG system][1]。
	- 如果想将通道坐标整合到模型中,可以参考这个
	- 分发数据中通道数只有271的原因,在上述两个章节中有说明

## 环境配置

```bash
conda create -n dlbasics python=3.10
conda activate dlbasics
pip install -r requirements.txt
```

## 运行基线模型

### 训练

```bash
python main.py

# 在线可视化结果(需要wandb账号)
python main.py use_wandb=True
```

- 权重`model_best.pt`和`model_last.pt`,以及对测试输入的预测`submission.npy`会保存在`outputs/{执行日期时间}/`目录下。将`submission.npy`提交到Omnicampus可以查看test top-10准确率。

  - `model_best.pt`是根据validation top-10准确率评估的

- 训练时加载的`config.yaml`文件在`train.py`的`run()`函数的`@hydra.main`装饰器中指定。如果创建了新的yaml文件,请在这里修改。

- 基线方法非常简单,有很大的改进空间(参考"可能的改进示例"部分)。因此,**我们只认可在Omnicampus上超过基线test accuracy=1.637%的提交作为完成要求。**

### 仅执行评估

- 如果之后只需要对测试数据进行评估。输出的`submission.npy`与训练最后输出的文件相同。

```bash
python eval.py model_path={要评估的权重路径}.pt
```

## 数据集[[链接][2]]详情

- 1,854个类别,22,448张图片(每个类别约12张)
  - 类别示例: airplane, aligator, apple, ...

- 每个类别的图片按约6:2:2的比例分为训练、验证和测试集

- 有4名受试者,可以使用哪个受试者的样本作为训练的可用信息(`*_subject_idxs.pt`)。

### 下载数据集

- 从[这里][3]下载`data.zip`,并解压到`data/`目录。

- 如果需要使用图片进行预训练等,从drive下载`images.zip`并解压到任意目录。使用{train, val}\_image\_paths.txt中的路径,自行创建数据加载器等。

## 任务详情

- 本次比赛的任务是**根据受试者观看图片时的脑电波,分类该图片属于哪个类别**。

- 评估使用top-10准确率。
  - 模型预测概率前10名中是否包含正确类别
  - 即随机猜测的基准水平约为10 / 1,854 ≈ 0.54%。

## 可能的改进示例

- 脑电波预处理
  - 分发的数据只进行了最基本的预处理。尝试重采样、滤波、缩放、基线校正等基本的波形预处理,有望提高性能。
- 使用图像数据进行预训练
  - 虽然本次比赛的任务是脑电波分类,但允许使用分发的图像数据对脑电波编码器进行预训练。
  - 例如)CLIP [Radford+ 2021]
- 引入语音模型
  - 已知使用处理与脑电波相同的波形的语音架构可能会有效。
- 防止过拟合的正则化和dropout
- 利用受试者信息
  - 由于每个受试者的脑电波特征可能不同,利用受试者信息有望提高性能。
  - 例如)Subject-specific layer [[Defossez+ 2022][4]], domain adaptation

[1]:	https://mne.tools/1.6/auto_examples/visualization/meg_sensors.html#ctf
[2]:	https://openneuro.org/datasets/ds004212/versions/2.0.0
[3]:	https://drive.google.com/drive/folders/1pgfVamCtmorUJTQejJpF8GhvwXa67rB9?usp=sharing
[4]:	https://arxiv.org/pdf/2208.12266