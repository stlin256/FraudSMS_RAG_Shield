'''
███████╗████████╗██╗     ██╗███╗   ██╗██████╗ ███████╗ ██████╗
██╔════╝╚══██╔══╝██║     ██║████╗  ██║╚════██╗██╔════╝██╔════╝
███████╗   ██║   ██║     ██║██╔██╗ ██║ █████╔╝███████╗███████╗
╚════██║   ██║   ██║     ██║██║╚██╗██║██╔═══╝ ╚════██║██╔═══██╗
███████║   ██║   ███████╗██║██║ ╚████║███████╗███████║╚██████╔╝
╚══════╝   ╚═╝   ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝
'''
import json
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModel, StoppingCriteria, StoppingCriteriaList

# ========== 配置 ==========
LLM_PATH = "./Qwen2.5-7B"
M3E_PATH = "./m3e-base"
FAISS_INDEX_FILE = "fraud_sms_faiss.index"
METADATA_FILE = "fraud_sms_metadata.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAFE_MODE = False  # True: 稳妥模式（三次预测取多数），False: 普通模式（单次预测）
DE_BUG = False #显示模型原始输出开关
SHOW_CATEGORY = False #输出甄别类别开关
SHOW_SAMPLE = True #输出相似短信开关
MAX_TOKENS = 1024 #模型最大输出长度
MAX_RETRIES = 2 #输出异常最大重试数

# ========== 加载 Qwen2.5-7B ========== #使用INT4量化，可在8G显存显卡上运行
print("加载 Qwen2.5-7B...") 
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

# ========== 加载 FAISS 索引 ==========
print("加载 FAISS 索引...")
index = faiss.read_index(FAISS_INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ========== 定义短信类别 ==========
LABELS = [
    "正常短信",
    "冒充公检法",
    "贷款诈骗",
    "冒充客服",
    "冒充领导或熟人诈骗"
]

# ========== 加载 m3e 向量化模型 ==========
print("加载 m3e 向量化模型...")
tokenizer = AutoTokenizer.from_pretrained(M3E_PATH)
model = AutoModel.from_pretrained(M3E_PATH).to(DEVICE).eval()

@torch.no_grad()
def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().cpu().numpy()

# ========== RAG 检索 ==========
def retrieve_similar_texts(query, top_k=5):
    query_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    results = [metadata[idx] for idx in indices[0] if idx < len(metadata)]
    return results

# ========== 自定义停止条件 ==========
class DynamicStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if "```" in decoded_text:
            blocks = decoded_text.split("```")
            # 滚动检查每个可能的完整块（奇数索引是内容）
            for i in range(1, len(blocks), 2):  # 步长2，跳过非内容部分
                block = blocks[i].strip()
                if block:  # 确保块非空
                    lines = block.split("\n")
                    has_label = False
                    has_reason = False
                    label_valid = False
                    reason_valid = False
                    for line in lines:
                        line = line.strip()
                        if line.startswith("预测类别:"):
                            has_label = True
                            label = line.split("预测类别:")[1].strip()
                            label_valid = "<类别名称>" not in label and label != "未知" and label
                        elif line.startswith("理由:"):
                            has_reason = True
                            reason = line.split("理由:")[1].strip()
                            reason_valid = "<简短理由" not in reason and reason != "未提供理由" and reason
                    # 确保块完整：有有效类别和理由，且文本以 ``` 结束
                    if has_label and has_reason and label_valid and reason_valid and decoded_text.endswith("```"):
                        return True
        return False

# ========== 单次分类函数 ==========
def single_classify(sms_text, similar_texts, attempt_num):
    retrieved_examples = "\n".join(
        [f"- 短信: {item['text']} | 类别: {item['label']}" for item in similar_texts]
    ) if (similar_texts and similar_texts != " ") else "无相似短信示例。"

    prompt = f"""
    你是一个精准的智能短信分类助手，擅长诈骗短信甄别。请依据以下分类步骤生成一个有效的 ``` 块结果，不输出多余信息。确保以 ``` 开始和结束，包含“预测类别:”、“理由:”，不含占位符（如 <类别名称>）。生成第一个有效块后立即停止。

    分类步骤：
    1. 分析短信特征：关注语气（正式/随意）、用词（是否涉及金钱、紧急、身份冒充）、是否有敏感行动要求（如转账、提供信息）。
    2. “冒充客服”识别：注意假冒客服常使用伪专业语气、要求点击链接验证身份或汇款等操作，而正常客服语气平稳、无敏感索取，需要注意和冒充公检法、冒充领导或熟人诈骗、正常短信的区分
    3. “冒充领导或熟人诈骗”具有明确的资金或敏感数据要求，将其与“正常短信”区分开，单纯辱骂的短信不属于诈骗。
    4. 参考相似短信：结合示例的类别分布，评估输入短信与示例的相似性。
    5. 逻辑推理：基于特征和示例，得出最可能的类别。

    输入信息：
    短信内容: "{sms_text}"
    相似短信示例：
    {retrieved_examples}

    短信类别选项：
    1. 正常短信（日常通知、服务确认、促销等）
    2. 冒充公检法（假装执法机构，恐吓或索要信息）
    3. 贷款诈骗（涉及贷款、优惠条件诱导转账）
    4. 冒充客服（假冒服务人员索要信息或钱财）
    5. 冒充领导或熟人诈骗（假装熟人或上级要求转账）

    输出格式：
    ```
    预测类别: <类别名称>
    理由: <简短理由（50字以内），说明特征和推理依据>
    ```
    """
    
    max_retries = MAX_RETRIES
    max_tokens = MAX_TOKENS
    stopping_criteria = StoppingCriteriaList([DynamicStoppingCriteria(llm_tokenizer)])

    for retry in range(max_retries + 1):
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if DE_BUG:
            print(f"📝 尝试 {attempt_num} - 重试 {retry + 1} 模型输出：")
            print(response)

        # 提取逻辑：提取有效 ``` 块
        pred_label = "未知"
        reason = "未提供理由"
        similar_sms = "无"

        if "```" in response:
            blocks = response.split("```")
            for i in range(1, len(blocks), 2):  # 步长2，提取内容块
                block = blocks[i].strip()
                if block:  # 确保块非空
                    lines = block.split("\n")
                    temp_label = "未知"
                    temp_reason = "未提供理由"
                    for line in lines:
                        line = line.strip()
                        if line.startswith("预测类别:"):
                            temp_label = line.split("预测类别:")[1].strip()
                        elif line.startswith("理由:"):
                            temp_reason = line.split("理由:")[1].strip()
                    if ("<类别名称>" not in temp_label and
                        "<简短理由" not in temp_reason and
                        temp_label != "未知" and
                        temp_reason != "未提供理由"):  # 确保非模板且有效
                        pred_label = temp_label
                        reason = temp_reason
                        # 从 similar_texts 中选择第一条类型匹配的短信
                        for item in similar_texts:
                            if item["label"] == pred_label:
                                similar_sms = item["text"]
                                break
                        break  # 提取第一个有效块后停止

        # 检查是否为模板或无效输出
        if pred_label == "未知" or reason == "未提供理由":
            if retry < max_retries:
                print(f"⚠️ 尝试 {attempt_num} - 重试 {retry + 1} 输出为模板或无效，重试中...")
                continue
            else:
                print(f"❌ 尝试 {attempt_num} - 模型异常，重试 {max_retries} 次后仍无效")
                return None, None, None
        else:
            return pred_label, reason, similar_sms

# ========== Qwen2.5 进行分类 ==========
def classify_sms(sms_text):
    if DE_BUG:
        print(f"🔍 输入短信: {sms_text}")

    similar_texts = retrieve_similar_texts(sms_text)
    if not similar_texts:
        print("⚠️ 未检索到相似短信，使用默认逻辑分类。")

    if not SAFE_MODE:
        pred_label, reason, similar_sms = single_classify(sms_text, similar_texts, 1)
        if pred_label is None:
            print("❌ 模型异常，无法预测此短信（重试两次后仍为模板输出）")
            return None

        if pred_label != '正常短信':
            print('🛑警告，这是诈骗短信❗❗❗')
            if SHOW_CATEGORY:
                print(f"🎯类别: {pred_label}")
        else:
            print(f"🎯类别: {pred_label}")
        print(f"理由: {reason}")
        if SHOW_SAMPLE:
            print(f"相似{'诈骗' if pred_label != '正常短信' else ''}短信: {similar_sms}")
        return pred_label

    else:
        results = []
        for attempt in range(3):
            pred_label, reason, similar_sms = single_classify(sms_text, similar_texts, attempt + 1)
            if pred_label is None:
                print("❌ 模型异常，无法预测此短信（稳妥模式中重试异常）")
                return None
            results.append((pred_label, reason, similar_sms))

            if attempt == 1 and results[0][0] == results[1][0]:
                if pred_label != '正常短信':
                    print('🛑警告，这是诈骗短信❗❗❗')
                print(f"🎯类别: {pred_label}")
                print(f"理由: {reason}")
                print(f"相似{'诈骗' if pred_label != '正常短信' else ''}短信: {similar_sms}")
                return results[0][0]
            elif attempt == 2:
                if results[0][0] != results[1][0] and results[1][0] != results[2][0] and results[0][0] != results[2][0]:
                    print("❓ 稳妥模式下三次预测不一致，无法判断")
                    return None

                final_label = max(set([r[0] for r in results]), key=[r[0] for r in results].count)
                final_result = next(r for r in results if r[0] == final_label)
                if final_label != '正常短信':
                    print('🛑警告，这是诈骗短信❗❗❗')
                    if SHOW_CATEGORY:
                        print(f"🎯类别: {final_label}")
                else:
                    print(f"🎯类别: {final_label}")
                print(f"理由: {final_result}")
                if SHOW_SAMPLE:
                    print(f"相似{'诈骗' if final_label != '正常短信' else ''}短信: {similar_sms}")
                return final_result[0]

# ========== 主函数 ==========
if __name__ == "__main__":
    if SAFE_MODE:
        print('''
        当前模式: 稳妥模式
        说明：在当前模式下，每条短信都会被至少甄别两次，若两次结果相同，则输出，否则进行第三次甄别，以多数结果为准，若无多数结果，则提示无法判断
            在目前的提示词下，模型表现稳定，若不开启采样，则稳妥模式和普通模式表现几乎无差
        ''')
    else:
        print('''
        当前模式: 普通模式
        说明：每条短信进行一次甄别并输出结果，若模型输出格式异常则会进行重试。
                ''')
    while True:
        sms = input("\n📨 请输入要分类的短信：")
        if len(sms) == 0:
            print("请输入内容，不要输入空信息。\n")
            continue
        classify_sms(sms)