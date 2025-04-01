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
import torch
from transformers import AutoTokenizer
from transformers import AutoModel, StoppingCriteria
import gradio as gr
from openai import OpenAI
import time

# ========== 配置 ==========
M3E_PATH = "./m3e-base"  # m3e 向量化模型路径
FAISS_INDEX_FILE = "fraud_sms_faiss.index"  # FAISS 索引文件
METADATA_FILE = "fraud_sms_metadata.json"  # 元数据文件
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
MAX_TOKENS = 1024  # 限制模型最大输出长度
api_key = "your_api_key" #你的api_key
base_url = "your_base_url" #你的base_url
model_name = "your_model" #你调用的模型
DE_BUG = True  # 显示模型原始输出开关
SHOW_CATEGORY = True  # 输出甄别类别开关
SHOW_SAMPLE = True  # 输出相似短信开关
SHOW_HISTORY = False #在终端输出对话历史记录
SHOW_USED_TIME = True #在终端输出问答所用时间

client = OpenAI(api_key=api_key, base_url=base_url)

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
    """单次短信分类"""
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

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个精准的智能短信分类助手，擅长诈骗短信甄别。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_TOKENS,
        stream=False
    )
    response = response.choices[0].message.content.strip()
    if DE_BUG:
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
                if DE_BUG:
                    print(f"❌模型输出无效")
                return None, None, None
        else:
            return pred_label, reason, similar_sms


# ========== 进行分类 ==========
def classify_sms(sms_text, history=None):
    """主分类函数"""

    start_time = time.time()

    if not sms_text.strip():
        return "请输入短信内容！", history or []

    if history is None:
        history = []

    similar_texts = retrieve_similar_texts(sms_text)
    if not similar_texts and DE_BUG:
        print("⚠️ 未检索到相似短信，使用默认逻辑分类。")

    output_lines = []
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
    output = "\n".join(output_lines)
    history.append({"role": "user", "content": sms_text})
    history.append({"role": "assistant", "content": output})
    if SHOW_HISTORY:
        print("History:", history)
    if SHOW_USED_TIME:
        used_time = format(time.time()-start_time, '.2f')
        print(f"本次对话用时{used_time}秒")
    return output, history


# ========== Gradio 界面 ==========
def create_interface():
    """创建 Gradio 可视化界面"""
    mode_info =f"""
    By: stlin256 
    当前调用模型：{model_name}
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