'''
███████╗████████╗██╗     ██╗███╗   ██╗██████╗ ███████╗ ██████╗
██╔════╝╚══██╔══╝██║     ██║████╗  ██║╚════██╗██╔════╝██╔════╝
███████╗   ██║   ██║     ██║██╔██╗ ██║ █████╔╝███████╗███████╗
╚════██║   ██║   ██║     ██║██║╚██╗██║██╔═══╝ ╚════██║██╔═══██╗
███████║   ██║   ███████╗██║██║ ╚████║███████╗███████║╚██████╔╝
╚══════╝   ╚═╝   ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝
'''
import os
import json
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

# ========== 配置 ==========
MODEL_PATH = "./m3e-base"  # 使用 m3e-large 进行向量化
DATA_DIR = "Telecom_Fraud_Texts_5-main"  # 短信数据集路径
FAISS_INDEX_FILE = "fraud_sms_faiss.index"  # FAISS 索引文件
METADATA_FILE = "fraud_sms_metadata.json"  # 短信元数据文件
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_DEVICE = 0  # 指定 GPU 设备

# ========== 加载本地 m3e 模型 ==========
print("📦 正在加载 m3e-base 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# ========== 短信类别映射 ==========
label_mapping = {
    "label00-last.csv": "正常短信",
    "label01-last.csv": "冒充公检法",
    "label02-last.csv": "贷款诈骗",
    "label03-last.csv": "冒充客服",
    "label04-last.csv": "冒充领导或熟人诈骗"
}

# ========== 向量化函数 ==========
@torch.no_grad()
def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # 取均值作为向量表示
    return embedding.squeeze().cpu().numpy()

# ========== 初始化 FAISS 索引 ==========
print("初始化 FAISS 索引...")
dimension = 768  # m3e-base 输出 768 维
cpu_index = faiss.IndexFlatL2(dimension)
gpu_res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_res, GPU_DEVICE, cpu_index)

# ========== 读取短信数据并向量化 ==========
print("读取短信数据并向量化...")
metadata = []
all_vectors = []

for file_name, label in label_mapping.items():
    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(file_path, encoding="utf-8")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"🔍 处理 {label}"):
        text = str(row.iloc[0]).strip()  # 获取短信文本
        if not text:
            continue

        # 计算文本向量
        vec = get_embedding(text)
        all_vectors.append(vec)

        # 记录元数据
        metadata.append({
            "id": f"{file_name}_{idx}",
            "label": label,
            "text": text
        })

# ========== 添加向量到索引 ==========
if all_vectors:
    print("⚡ 正在将向量添加到 FAISS 索引...")
    vectors_np = np.array(all_vectors, dtype="float32")
    print("Vectors shape:", vectors_np.shape)  # 打印向量数组的形状
    gpu_index.add(vectors_np)

# ========== 保存索引和元数据 ==========
print("保存索引和元数据...")
index_cpu = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(index_cpu, FAISS_INDEX_FILE)

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n 共添加短信向量：{len(metadata)}")
print(f" 向量索引保存至：{FAISS_INDEX_FILE}")
print(f" 元数据保存至：{METADATA_FILE}")
