# Bioinformatic_competition_taskBioinformatic_competition_task
下面是一份 README 文件示例，你可以直接复制到项目根目录下的 README.md 文件中：

---

# 医学文本生成系统

本项目实现了一个基于 T5 生成模型的医学文本生成系统，通过引入正负样本对，并采用 hard negative 策略与动态 margin 三元组损失进行数据增强，从而在生成医学标准术语时实现更高的鲁棒性与准确性。项目采用前瞻性的设计理念和多维度视角，致力于推动医学文本处理技术的发展。

## 特性

- **数据加载与清洗**  
  支持从 Excel 文件中加载数据，自动提取并清洗医学术语（保留括号、标点、中文与英文等）。

- **强化学习与 hard negative 策略**  
  利用正负样本对训练模型，通过动态 margin 三元组损失，确保正样本损失低于负样本损失一定阈值，从而提升模型生成效果。

- **多指标评估**  
  提供准确率、BLEU、ROUGE-L 和编辑距离等多维度指标，全面评估生成结果。

- **前瞻性设计**  
  模型结构与训练流程均采用前沿技术，具备良好的扩展性和适应性，方便后续引入更多先进的预训练模型和自定义策略。

## 环境要求

- Python 3.6 或以上版本
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [Pandas](https://pandas.pydata.org/)、[Numpy](https://numpy.org/)、[TQDM](https://github.com/tqdm/tqdm)
- [jieba](https://github.com/fxsjy/jieba)
- [nltk](https://www.nltk.org/)（用于 BLEU 计算）
- [rouge](https://pypi.org/project/rouge/)（用于 ROUGE 指标）
- [openpyxl](https://openpyxl.readthedocs.io/)（用于 Excel 文件读取）

建议使用虚拟环境来安装上述依赖。你可以使用以下命令安装所有依赖（需先创建并激活虚拟环境）：

```bash
pip install torch transformers pandas numpy tqdm jieba nltk rouge openpyxl
```

## 数据准备

- **训练数据**：准备名为 `data_train.xlsx` 的 Excel 文件，要求数据格式符合以下要求：
  - 第二列为原始医学术语
  - 第四列为标准医学术语

- **测试数据**（可选）：准备名为 `data_test.xlsx` 的 Excel 文件，格式同训练数据。若提供测试数据，程序将在训练结束后进行评估。

确保数据文件与代码处于同一目录，或者在代码中修改文件路径以匹配实际存放位置。

## 使用方法

1. **克隆项目**

   使用 Git 克隆项目到本地：
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. **运行主程序**

   执行以下命令启动训练、评估及生成示例：
   ```bash
   python cldra7.py
   ```

   程序流程包括：
   - 加载并清洗数据
   - 初始化并加载 T5 模型与分词器
   - 构造训练集（以及测试集，如提供）
   - 利用强化学习策略训练模型
   - 在测试集上计算准确率、BLEU、ROUGE-L 和编辑距离
   - 展示若干手动测试案例的生成结果

3. **调整参数**

   你可以在 `cldra7.py` 文件末尾根据需要调整以下参数：
   - 模型名称（如 `Langboat/mengzi-t5-base`）
   - 训练轮数（EPOCHS）
   - 批次大小（BATCH_SIZE）
   - 学习率（LEARNING_RATE）
   - 最大输入和输出长度（MAX_SOURCE_LENGTH、MAX_TARGET_LENGTH）
   - 动态 margin 和 ranking loss 的权重（MARGIN、RANKING_LAMBDA）

## 进阶与展望

本项目采用的前瞻性设计和多维度视角为未来扩展提供了坚实基础。未来可能的改进方向包括：

- **集成更多预训练模型**  
  支持更多先进的生成模型或领域特定的预训练模型，进一步提升生成准确率和鲁棒性。

- **增强数据扩充策略**  
  探索更多数据增强方法，如对抗性样本生成、迁移学习等，为数据稀缺问题提供解决方案。

- **实时评估与交互**  
  开发交互式界面，实现实时生成与评估，满足临床应用场景需求。

- **多任务与多模态扩展**  
  融合其他医学信息（如图像、报告）实现多模态文本生成，为复杂医学诊断提供支持。

我们鼓励社区共同探讨和改进，欢迎提出 issue 或提交 pull request，携手推动医学文本生成技术的发展。

## 贡献与许可

- **贡献**  
  欢迎对本项目进行改进或扩展，提交 issue 或 pull request 与我们共同完善该系统。

- **许可协议**  
  本项目采用 [MIT License](LICENSE)
