'''
███████╗████████╗██╗     ██╗███╗   ██╗██████╗ ███████╗ ██████╗
██╔════╝╚══██╔══╝██║     ██║████╗  ██║╚════██╗██╔════╝██╔════╝
███████╗   ██║   ██║     ██║██╔██╗ ██║ █████╔╝███████╗███████╗
╚════██║   ██║   ██║     ██║██║╚██╗██║██╔═══╝ ╚════██║██╔═══██╗
███████║   ██║   ███████╗██║██║ ╚████║███████╗███████║╚██████╔╝
╚══════╝   ╚═╝   ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝
'''
import json
import traceback
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel, StoppingCriteria, \
    StoppingCriteriaList

# ========== 配置 ==========
LLM_PATH = "./Qwen2.5-7B"
FAISS_INDEX_FILE = "fraud_sms_faiss.index"
METADATA_FILE = "fraud_sms_metadata.json"
TEST_DATA_PATH = "fraud_sms_dataset.json"
M3E_PATH = "./m3e-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAFE_MODE = False  # True: 稳妥模式（三次预测取多数），False: 普通模式（单次预测）
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

# ========== 加载 m3e 向量化模型 ==========
print("加载 m3e 向量化模型...")
tokenizer = AutoTokenizer.from_pretrained(M3E_PATH)
model = AutoModel.from_pretrained(M3E_PATH).to(DEVICE).eval()

# ========== 定义短信类别 ==========
LABELS = [
    "正常短信",
    "冒充公检法",
    "贷款诈骗",
    "冒充客服",
    "冒充领导或熟人诈骗"
]


# ========== 向量化函数 ==========
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
            for i in range(1, len(blocks), 2):
                block = blocks[i].strip()
                if block:
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
                    if has_label and has_reason and label_valid and reason_valid and decoded_text.endswith("```"):
                        return True
        return False


# ========== 单次分类函数 ==========
def single_classify(sms_text, similar_texts, attempt_num):
    retrieved_examples = "\n".join(
        [f"- 短信: {item['text']} | 类别: {item['label']}" for item in similar_texts]
    ) if similar_texts else "无相似短信示例。"

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
    stopping_criteria = StoppingCriteriaList([DynamicStoppingCriteria(llm_tokenizer)])

    for retry in range(max_retries + 1):
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        pred_label = "未知"
        reason = "未提供理由"
        similar_sms = "无"

        if "```" in response:
            blocks = response.split("```")
            for i in range(1, len(blocks), 2):
                block = blocks[i].strip()
                if block:
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
                            temp_reason != "未提供理由"):
                        pred_label = temp_label
                        reason = temp_reason
                        for item in similar_texts:
                            if item["label"] == pred_label:
                                similar_sms = item["text"]
                                break
                        break

        if pred_label == "未知" or reason == "未提供理由":
            if retry < max_retries:
                print(f"⚠️ 尝试 {attempt_num} - 重试 {retry + 1} 输出无效，重试中...")
                continue
            else:
                print(f"❌ 尝试 {attempt_num} - 重试 {max_retries} 次后仍无效")
                return None, None, None
        else:
            return pred_label, reason, similar_sms


# ========== 分类主函数 ==========
def classify_sms(sms_text):
    similar_texts = retrieve_similar_texts(sms_text)
    if not similar_texts:
        print("⚠️ 未检索到相似短信，使用默认逻辑分类。")

    if not SAFE_MODE:
        pred_label, reason, similar_sms = single_classify(sms_text, similar_texts, 1)
        if pred_label is None:
            print("❌ 模型异常，无法预测此短信")
            return None
        return pred_label
    else:
        results = []
        for attempt in range(3):
            pred_label, reason, similar_sms = single_classify(sms_text, similar_texts, attempt + 1)
            if pred_label is None:
                print("❌ 模型异常，无法预测此短信（稳妥模式）")
                return None
            results.append((pred_label, reason, similar_sms))
            if attempt == 1 and results[0][0] == results[1][0]:
                return results[0][0]
            elif attempt == 2:
                if results[0][0] != results[1][0] and results[1][0] != results[2][0] and results[0][0] != results[2][0]:
                    print("❓ 稳妥模式下三次预测不一致，无法判断")
                    return None
                final_label = max(set([r[0] for r in results]), key=[r[0] for r in results].count)
                return final_label


# ========== 加载测试集 ==========
def load_test_data():
    try:
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        return test_data
    except FileNotFoundError:
        print(f"错误：测试集文件 {TEST_DATA_PATH} 未找到！")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误：测试集文件 {TEST_DATA_PATH} 格式无效！")
        exit(1)


# ========== 评测函数 ==========
def evaluate():
    test_data = load_test_data()

    total_sms = 0
    correct = 0
    incorrect = 0
    correct_2 = 0
    incorrect_2 = 0
    error_warning = 0
    cannot_determine = 0
    format_error = 0
    model_crash = 0

    for category, sms_list in test_data.items():
        for sms in sms_list:
            total_sms += 1
            try:
                pred_label = classify_sms(sms)
                if pred_label is None:
                    if SAFE_MODE:
                        cannot_determine += 1
                        print(f"无法侦测❓ {category}类型的短信内容{sms}")
                    else:
                        format_error += 1
                        print(f"侦测为空❌ {category}类型为{pred_label}的短信内容{sms}")
                elif pred_label == category:
                    correct += 1
                    correct_2 += 1
                    print(f"正确侦测✔️ {category} 短信内容{sms}")
                else:
                    incorrect += 1
                    if category != '正常短信' and pred_label != '正常短信':
                        correct_2 += 1
                    elif category == '正常短信' and pred_label != '正常短信':
                        error_warning += 1
                    elif category != '正常短信' and pred_label == '正常短信':
                        incorrect_2 += 1
                    print(f"侦测出错❌ {category}类型为{pred_label}的短信内容{sms}")
            except Exception as e:
                model_crash += 1
                print(f"模型崩溃于短信: '{sms}'，异常: {e}")
                traceback.print_exc()

    valid_sms = total_sms - model_crash
    success_rate = correct / valid_sms if valid_sms > 0 else 0
    success_rate_2 = correct_2 / valid_sms if valid_sms > 0 else 0
    error_warning_rate = error_warning / valid_sms if valid_sms > 0 else 0
    error_warning_rate_2 = incorrect_2 / valid_sms if valid_sms > 0 else 0

    print("\n=== 评测结果 ===")
    print(f"当前模式: {'稳妥模式' if SAFE_MODE else '普通模式'}")
    print(f"总短信数: {total_sms}")
    print(f"正确分类(具体类型): {correct}")
    print(f"正确分类(是否诈骗): {correct_2}")
    print(f"误报短信: {error_warning}")
    print(f"错误分类: {incorrect}")
    print(f"无法得出结果: {cannot_determine}")
    print(f"格式错误: {format_error}")
    print(f"模型崩溃: {model_crash}")
    print(f"成功率: {success_rate:.2%} (正确分类(具体类型) / (总短信 - 模型崩溃))")
    print(f"成功率: {success_rate_2:.2%} (正确分类(是否诈骗) / (总短信 - 模型崩溃))")
    print(f"误报率: {error_warning_rate:.2%} (误报短信(非诈骗被识别为诈骗) / (总短信 - 模型崩溃))")
    print(f"漏报率: {error_warning_rate_2:.2%} (漏报短信(诈骗短信被放行) / (总短信 - 模型崩溃))")



# ========== 主程序 ==========
if __name__ == "__main__":
    evaluate()