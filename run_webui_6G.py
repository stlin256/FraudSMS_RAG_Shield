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
import gradio as gr

# ========== 配置 ==========
LLM_PATH = "./Qwen2.5-7B"  # Qwen2.5-7B 模型路径
M3E_PATH = "./m3e-base"  # m3e 向量化模型路径
FAISS_INDEX_FILE = "fraud_sms_faiss.index"  # FAISS 索引文件
METADATA_FILE = "fraud_sms_metadata.json"  # 元数据文件
DEVICE = "cuda"
SAFE_MODE = False  # True: 稳妥模式（三次预测取多数），False: 普通模式（单次预测）
DE_BUG = False  # 显示模型原始输出开关
SHOW_CATEGORY = False  # 输出甄别类别开关
SHOW_SAMPLE = True  # 输出相似短信开关
MAX_TOKENS = 512  # 模型最大输出长度
MAX_RETRIES = 0  # 输出异常最大重试数

# ========== 加载 Qwen2.5-7B ==========
# 6G显存优化配置，实测运行占用5.8G显存
print("加载 Qwen2.5-7B...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,     # 显式声明存储类型
    llm_int8_enable_fp32_cpu_offload=True   # 强制启用CPU卸载
)
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    max_memory={0: "5GiB"},
    offload_folder="./offload_temp"
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
    """获取文本的 m3e 嵌入向量"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().cpu().numpy()


# ========== RAG 检索 ==========
def retrieve_similar_texts(query, top_k=5):
    """检索与查询短信最相似的 top_k 条短信"""
    query_vec = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    results = [metadata[idx] for idx in indices[0] if idx < len(metadata)]
    return results


# ========== 自定义停止条件 ==========
class DynamicStoppingCriteria(StoppingCriteria):
    """动态停止条件：当生成有效 ``` 块时停止"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if "```" in decoded_text:
            blocks = decoded_text.split("```")
            for i in range(1, len(blocks), 2):  # 检查每个内容块
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
    """单次短信分类，包含重试逻辑"""
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

    stopping_criteria = StoppingCriteriaList([DynamicStoppingCriteria(llm_tokenizer)])

    for retry in range(MAX_RETRIES + 1):
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=llm_tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if DE_BUG:
            print(f"📝 尝试 {attempt_num} - 重试 {retry + 1} 模型输出：")
            print(response)

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
            if retry < MAX_RETRIES:
                print(f"⚠️ 尝试 {attempt_num} - 重试 {retry + 1} 输出为模板或无效，重试中...")
                continue
            else:
                print(f"❌ 尝试 {attempt_num} - 模型异常，重试 {MAX_RETRIES} 次后仍无效")
                return None, None, None
        else:
            return pred_label, reason, similar_sms


# ========== Qwen2.5 进行分类 ==========
def classify_sms(sms_text, history=None):
    """主分类函数，支持普通和稳妥模式"""
    if not sms_text.strip():
        return "请输入短信内容！", history or []

    if history is None:
        history = []

    similar_texts = retrieve_similar_texts(sms_text)
    if not similar_texts:
        print("⚠️ 未检索到相似短信，使用默认逻辑分类。")

    output_lines = []
    if not SAFE_MODE:
        pred_label, reason, similar_sms = single_classify(sms_text, similar_texts, 1)
        if pred_label is None:
            output_lines.append("❌ 模型输出异常")
        else:
            if pred_label != '正常短信':
                output_lines.append('🛑 警告，这是诈骗短信❗❗❗')
            else:
                output_lines.append('🟢正常短信')
            if SHOW_CATEGORY:
                output_lines.append(f"🎯 类别: {pred_label}")
            output_lines.append(f"理由: {reason}")
            if SHOW_SAMPLE:
                output_lines.append(f"相似{'诈骗' if pred_label != '正常短信' else ''}短信: {similar_sms}")
    else:
        results = []
        for attempt in range(3):
            pred_label, reason, similar_sms = single_classify(sms_text, similar_texts, attempt + 1)
            if pred_label is None:
                output_lines.append("❌ 模型输出异常")
                break
            results.append((pred_label, reason, similar_sms))

            if attempt == 1 and results[0][0] == results[1][0]:
                if pred_label != '正常短信':
                    output_lines.append('🛑 警告，这是诈骗短信❗❗❗')
                else:
                    output_lines.append('🟢正常短信')
                if SHOW_CATEGORY:
                    output_lines.append(f"🎯 类别: {pred_label}")
                output_lines.append(f"理由: {reason}")
                if SHOW_SAMPLE:
                    output_lines.append(f"相似{'诈骗' if pred_label != '正常短信' else ''}短信: {similar_sms}")
                break
            elif attempt == 2:
                if results[0][0] != results[1][0] and results[1][0] != results[2][0] and results[0][0] != results[2][0]:
                    output_lines.append("❓ 稳妥模式下三次预测不一致，无法判断")
                else:
                    final_label = max(set([r[0] for r in results]), key=[r[0] for r in results].count)
                    final_result = next(r for r in results if r[0] == final_label)
                    if final_label != '正常短信':
                        output_lines.append('🛑 警告，这是诈骗短信❗❗❗')
                    else:
                        output_lines.append('🟢正常短信')
                    if SHOW_CATEGORY:
                        output_lines.append(f"🎯 类别: {final_label}")
                    output_lines.append(f"理由: {final_result[1]}")
                    if SHOW_SAMPLE:
                        output_lines.append(f"相似{'诈骗' if final_label != '正常短信' else ''}短信: {final_result[2]}")

    output = "\n".join(output_lines)
    history.append({"role": "user", "content": sms_text})
    history.append({"role": "assistant", "content": output})
    print("History:", history)
    return output, history


# ========== Gradio 界面 ==========
def create_interface():
    """创建 Gradio 可视化界面"""
    mode_info = """
    By: stlin256 
    当前模式: 稳妥模式
    说明：在当前模式下，每条短信都会被至少甄别两次，若两次结果相同，则输出，否则进行第三次甄别，以多数结果为准，若无多数结果，则提示无法判断
    在目前的提示词下，模型表现稳定，若不开启采样，则稳妥模式和普通模式表现几乎无差
    """ if SAFE_MODE else """
    By: stlin256 
    当前模式: 普通模式
    说明：每条短信进行一次甄别并输出结果，若模型输出格式异常则会进行重试。
    """

    with gr.Blocks(title="诈骗短信甄别系统") as demo:
        gr.Markdown("# 诈骗短信甄别系统")
        gr.Markdown(mode_info)

        chatbot = gr.Chatbot(label="对话历史", type="messages", height=300, value=[
            {"role": "assistant", "content": "你好！我是诈骗短信甄别助手，请输入短信内容，我会帮助你判断是否是诈骗短信。"}
        ])
        state = gr.State(value=[{"role": "assistant", "content": "你好！我是诈骗短信甄别助手，请输入短信内容，我会帮助你判断是否是诈骗短信。"}])

        with gr.Row():
            sms_input = gr.Textbox(label="请输入要分类的短信", placeholder="在此输入短信内容...")
            submit_btn = gr.Button("提交")

        output_text = gr.Textbox(label="分类结果", interactive=False)

        submit_btn.click(
            fn=classify_sms,
            inputs=[sms_input, state],
            outputs=[output_text, chatbot]
        )

        clear_btn = gr.Button("清空历史")
        clear_btn.click(
            fn=lambda: ("", []),
            inputs=None,
            outputs=[output_text, chatbot]
        )

    return demo


# ========== 主函数 ==========
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()