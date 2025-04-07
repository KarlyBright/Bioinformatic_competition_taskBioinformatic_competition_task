# -*- coding: utf-8 -*-
# 完整代码：医学文本生成系统（使用 T5 生成式模型），引入正负样本对（原始数据、标准答案、假答案）进行强化学习
# 并采用 hard negative 策略和动态 margin 三元组损失来进行数据增强

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 如有需要，使用镜像

import re
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import jieba
import warnings

# 屏蔽部分警告
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")

# 导入 BLEU 与 ROUGE 度量工具
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# 设置随机种子和设备
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(66)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 1. 数据加载与清洗（适用于训练集和测试集）
def load_and_clean_data(file_path):
    """
    从 Excel 文件中加载数据，并提取第二列和第四列作为 source_term 与 target_term，
    同时对文本进行基本清洗 (保留括号)。
    """
    df = pd.read_excel(file_path)
    # 选择第二列和第四列，并重命名列名
    df = df.iloc[:, [1, 3]]
    df.columns = ["source_term", "target_term"]

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        # 删除开头的序号
        text = re.sub(r'^[\d\.、]+', '', text)
        # 保留中英文、数字和基本标点 (包括括号)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u0370-\u03FF\u2160-\u2188.,;:?!()（）+\-*/]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    df['source_term'] = df['source_term'].apply(clean_text)
    df['target_term'] = df['target_term'].apply(clean_text)
    # 剔除空文本记录
    df = df[(df['source_term'] != "") & (df['target_term'] != "")]
    return df

# 编辑距离函数（用于后续评估指标及 hard negative 计算）
def edit_distance(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

# 2. 定义数据集（包括正负样本对），并采用 hard negative 方法
class MedicalTermGenerationDataset(Dataset):
    def __init__(self, df, tokenizer, max_source_length=128, max_target_length=128, hard_negative_k=5):
        self.source_texts = df['source_term'].tolist()
        self.target_texts = df['target_term'].tolist()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.hard_negative_k = min(hard_negative_k, len(self.target_texts) - 1)  # 确保候选数合法

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source = self.source_texts[idx]
        positive = self.target_texts[idx]

        # 采样 hard negative 候选：随机选取 hard_negative_k 个非当前样本的候选
        candidate_indices = random.sample([i for i in range(len(self.target_texts)) if i != idx], self.hard_negative_k)
        hard_negative = None
        min_distance = float('inf')
        for cand_idx in candidate_indices:
            candidate_text = self.target_texts[cand_idx]
            dist = edit_distance(positive, candidate_text)
            if dist < min_distance:
                min_distance = dist
                hard_negative = candidate_text

        # 添加任务前缀，帮助模型理解任务
        source_text = f"标准化医学术语: {source}"
        pos_text = positive

        # 编码输入及正样本
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        pos_encoding = self.tokenizer(
            pos_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # 编码 hard negative 样本
        neg_encoding = self.tokenizer(
            hard_negative,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = source_encoding["input_ids"].squeeze()
        attention_mask = source_encoding["attention_mask"].squeeze()
        pos_labels = pos_encoding["input_ids"].squeeze()
        neg_labels = neg_encoding["input_ids"].squeeze()

        # 将 padding token 替换为 -100，以便在计算损失时忽略
        pos_labels[pos_labels == self.tokenizer.pad_token_id] = -100
        neg_labels[neg_labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": pos_labels,
            "negative_labels": neg_labels
        }

# 3. 定义生成式模型
class MedicalTermGenerationModel:
    def __init__(self, model_name='mengzi-t5-base'):
        print(f"Loading Tokenizer and Model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # 检查是否是 T5 系列模型
        if "t5" in model_name.lower() or "mengzi-t5" in model_name.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # 如果 tokenizer 没有 pad_token，设置一个
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model and Tokenizer loaded.")

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self, train_dataloader, val_dataloader=None, epochs=5, learning_rate=5e-5,
              warmup_steps=0, margin=1.0, ranking_lambda=1.0):
        """
        训练模型，同时利用正负样本对进行强化学习。
        使用动态 margin 三元组损失：
            dynamic_margin = margin * (1 + epoch/total_epochs)
        要求正样本的 loss 比负样本低至少 dynamic_margin。
        ranking_lambda: 排名损失的权重
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        self.model.train()
        for epoch in range(epochs):
            # 计算当前 epoch 的动态 margin
            current_margin = margin * (1 + epoch / epochs)
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"训练 Epoch {epoch + 1}/{epochs}")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pos_labels = batch["labels"].to(device)
                neg_labels = batch["negative_labels"].to(device)

                optimizer.zero_grad()

                # 正样本前向传播及损失计算
                outputs_pos = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=pos_labels
                )
                loss_pos = outputs_pos.loss

                # 负样本前向传播及损失计算
                outputs_neg = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=neg_labels
                )
                loss_neg = outputs_neg.loss

                # 使用动态 margin 计算 ranking loss：
                # dynamic_margin + loss_pos - loss_neg 若为正则计算损失，否则为 0
                ranking_loss = torch.clamp(current_margin + loss_pos - loss_neg, min=0)
                loss = loss_pos + ranking_lambda * ranking_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} 平均训练损失: {avg_train_loss:.4f}")

            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                print(f"Epoch {epoch + 1} 验证损失: {val_loss:.4f}")

    def evaluate(self, dataloader):
        """评估模型（只计算正样本损失）"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pos_labels = batch["labels"].to(device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=pos_labels
                )
                loss = outputs.loss
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        self.model.train()  # 切换回训练模式
        return avg_loss

    def generate(self, text, max_length=50):
        """生成医学标准术语"""
        self.model.eval()
        source_text = f"标准化医学术语: {text}"
        inputs = self.tokenizer(
            source_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=5,  # 束搜索
                early_stopping=True,
                no_repeat_ngram_size=2  # 防止重复生成
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# 4. 评估函数，计算生成结果的准确率
def evaluate_model_accuracy(model, df_test):
    """计算模型生成结果的准确率"""
    correct = 0
    total = len(df_test)
    print("\n测试集评估结果:")
    for index, row in tqdm(df_test.iterrows(), total=total, desc="评估"):
        source = row['source_term']
        true_answer = row['target_term']
        prediction = model.generate(source)
        print(f"测试项 {index + 1}: 输入: {source} | 预测: {prediction} | 标准答案: {true_answer}")
        if prediction == true_answer:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy

# 5. 计算 BLEU 和 ROUGE-L 指标
def compute_generation_metrics(model, df_test):
    """计算生成模型的 BLEU 和 ROUGE-L 指标"""
    metrics = {
        "bleu": [],
        "rouge_l": [],
        "edit_distance": []
    }
    rouge_evaluator = Rouge(metrics=['rouge-l'])
    smoothing_function = SmoothingFunction().method1

    for _, row in tqdm(df_test.iterrows(), desc="计算指标"):
        source = row['source_term']
        reference = row['target_term']

        # 生成预测
        prediction = model.generate(source)

        # 分词
        reference_tokens = list(jieba.cut(reference))
        prediction_tokens = list(jieba.cut(prediction))

        # 计算 BLEU
        try:
            bleu = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothing_function)
        except Exception:
            bleu = 0.0

        # 计算 ROUGE-L
        try:
            rouge = rouge_evaluator.get_scores(
                ' '.join(prediction_tokens),
                ' '.join(reference_tokens)
            )[0]['rouge-l']['f']
        except Exception:
            rouge = 0.0

        # 计算编辑距离
        ed = edit_distance(prediction, reference)

        metrics["bleu"].append(bleu)
        metrics["rouge_l"].append(rouge)
        metrics["edit_distance"].append(ed)

    avg_metrics = {}
    for metric_name in metrics:
        if metrics[metric_name]:
            avg_metrics[f"avg_{metric_name}"] = np.mean(metrics[metric_name])
        else:
            avg_metrics[f"avg_{metric_name}"] = 0.0

    return avg_metrics

# 6. 主函数
def main(train_file, test_file=None, epochs=5, batch_size=8, learning_rate=5e-5,
         model_name='mengzi-t5-base', max_source_length=128, max_target_length=128,
         margin=1.0, ranking_lambda=1.0):
    # 加载并清洗训练数据
    df_train = load_and_clean_data(train_file)
    print(f"从 {train_file} 中加载了 {len(df_train)} 条训练样本。")

    print("-" * 30)
    generation_model = MedicalTermGenerationModel(model_name=model_name).to(device)
    print("-" * 30)

    tokenizer = generation_model.tokenizer

    # 准备训练数据集和数据加载器，设置 hard_negative_k 参数（这里默认选 5 个候选）
    train_dataset = MedicalTermGenerationDataset(
        df_train, tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        hard_negative_k=5
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = None
    if test_file:
        df_test = load_and_clean_data(test_file)
        print(f"从 {test_file} 中加载了 {len(df_test)} 条测试样本。")
        test_dataset = MedicalTermGenerationDataset(
            df_test, tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            hard_negative_k=5
        )
        val_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    print("开始训练生成式模型...")
    generation_model.train(
        train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=int(len(train_dataloader) * 0.1),  # 10% 的 warmup 步数
        margin=margin,
        ranking_lambda=ranking_lambda
    )
    print("训练完成。")

    if test_file:
        print("\n在测试集上进行评估...")
        df_test = load_and_clean_data(test_file)

        # 计算准确率
        accuracy = evaluate_model_accuracy(generation_model, df_test)
        print(f"测试集准确率: {accuracy * 100:.2f}%")

        # 计算其他指标
        metrics = compute_generation_metrics(generation_model, df_test)
        print("\n测试集附加指标：")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.replace('_', ' ').capitalize()}: {value:.4f}")

    # 手动测试案例
    print("\n手动测试案例：")
    manual_test_cases = ["肝左叶小囊肿", "腰5双侧椎弓根崩裂", "B族维生素缺乏病，其他特指的",
                         "B淋巴细胞和髓系混合表型急性白血病伴缓解", "肺部感染（细菌+真菌）", "高血压病 II级 中危组",
                         "外伤", "先天愚型", "子宫CIN"]

    for test_case in manual_test_cases:
        generated_term = generation_model.generate(test_case)
        print(f"\n原始术语: {test_case}")
        print(f"生成的标准术语: {generated_term}")

if __name__ == "__main__":
    train_file = "data_train.xlsx"
    test_file = "data_test.xlsx"

    # 可选的中文预训练模型
    MODEL_NAME = 'Langboat/mengzi-t5-base'
    EPOCHS = 8
    BATCH_SIZE = 20
    LEARNING_RATE = 3e-5
    MAX_SOURCE_LENGTH = 128
    MAX_TARGET_LENGTH = 128
    MARGIN = 1.0
    RANKING_LAMBDA = 1.0

    print(f"使用的模型: {MODEL_NAME}")
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"最大源长度: {MAX_SOURCE_LENGTH}")
    print(f"最大目标长度: {MAX_TARGET_LENGTH}")

    main(train_file,
         test_file=test_file,
         epochs=EPOCHS,
         batch_size=BATCH_SIZE,
         learning_rate=LEARNING_RATE,
         model_name=MODEL_NAME,
         max_source_length=MAX_SOURCE_LENGTH,
         max_target_length=MAX_TARGET_LENGTH,
         margin=MARGIN,
         ranking_lambda=RANKING_LAMBDA)

    print("\n程序执行完毕。")
