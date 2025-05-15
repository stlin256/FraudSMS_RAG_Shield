# FraudSMS_RAG_Shield
<a href="#cn">中文介绍</a>
# Fraud SMS Detection System Integrating Large Model Reasoning and RAG Retrieval Enhancement

#### April 1 Update - Supports invoking online APIs using the OpenAI standard  
#### March 29 Update - Supports Qwen2.5-7B inference with 6GB VRAM

## Introduction

This project combines large language model reasoning with **RAG (Retrieval-Augmented Generation)** technology to accurately identify and classify SMS messages, protecting users from telecom fraud. Based on the [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5) dataset, the system uses the [m3e-base](https://huggingface.co/moka-ai/m3e-base) model for SMS vectorization, leverages [FAISS](https://github.com/facebookresearch/faiss) for fast similarity retrieval, and integrates the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) large language model for deep reasoning. The system identifies **"Normal SMS"** and the following four fraud categories:

- **Impersonating Law Enforcement** (pretending to be authorities to intimidate or request information)  
- **Loan Scams** (Involving loans and favorable conditions to induce money transfers)
- **Impersonating Customer Service** (fraudulent service personnel requesting information or payments)  
- **Impersonating Leaders/Acquaintances** (pretending to be superiors/acquaintances demanding transfers)  

By providing classification results, reasoning explanations, and similar SMS examples, the system enhances both accuracy and interpretability to help users effectively defend against fraudulent messages.

---

## Key Features

- **High Accuracy**: Fraud SMS false negative rate of only **3%** (based on benchmark.py results)  
- **Fast Response**: Average response time of **3 seconds** on NVIDIA 4060 Laptop  
- **Interpretability**: Outputs classification results, reasoning, and similar SMS examples  
- **User-Friendly**: Visual web interface via [Gradio](https://gradio.app/)  
- **Innovative Integration**: Combines RAG and LLM reasoning for enhanced accuracy and depth  
- **Lightweight Deployment**: Online API version available without local model deployment

---

## Demonstration

### run_webui.py

Mobile browser interface:  
![Local Path](Pictures/手机webui.png "Mobile WebUI")  
Desktop browser interface:  
![Local Path](Pictures/PCwebui.png "Desktop WebUI")

### run_shell.py  
![Local Path](Pictures/PCshell.png "Shell Interface")

### benchmark.py  
![Local Path](Pictures/PCbenchmark_1.png "Benchmark Results")  
<center>Intermediate content omitted</center>  

![Local Path](Pictures/PCbenchmark_2.png "Extended Results")

---

## Hardware Requirements

### Local Inference  
- **Recommended Configuration**: NVIDIA GPU with **8GB VRAM**  
```
    RTX4060 Laptop: ~3s for <30-character input
```
- **Minimum Configuration**: NVIDIA GPU with **6GB VRAM**  
```
    GTX1660S: 20-35s for <30-character input
```

### Online API Version  
- **Minimum Requirement**: Any computer  
```
    DeepSeek V3 via DeepSeek platform: 3-6s for <30-character input
```

---

## Installation & Configuration

1. **Install Anaconda**  
- Download and install Anaconda: [Anaconda.org](https://www.anaconda.com/)  
- Open **Anaconda Prompt** after installation  
- Create virtual environment  

```bash
conda create -n sms python=3.12
```

2. **Activate Environment**  
```bash
conda activate sms
```

3. **Install Dependencies**  
In project root directory:  
```bash
pip install -r requirements.txt
```

Install PyTorch 2.6.0 (replace with your configuration from [pytorch.org](https://pytorch.org/)):  
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126  
```

Install FAISS (GPU version):  
```bash
conda install -c conda-forge faiss-gpu
```

4. **Download Models & Datasets**  
- **Models**:  
  - [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)  
  - [m3e-base](https://huggingface.co/moka-ai/m3e-base)  
  - Mirrors:  
    - [Qwen2.5-7B (HF Mirror)](https://hf-mirror.com/Qwen/Qwen2.5-7B)  
    - [m3e-base (HF Mirror)](https://hf-mirror.com/moka-ai/m3e-base)  
  - *Note*: API version requires only [m3e-base](https://huggingface.co/moka-ai/m3e-base)  
- **Dataset (Optional)**:  
  - [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5)  
  - *Note*: Vectorized dataset (fraud_sms_faiss.index and fraud_sms_metadata.json) included

---

## File Structure

Ensure the project directory structure matches:  
```plaintext
Project Root
│   benchmark.py
│   fraud_sms_dataset.json
│   fraud_sms_faiss.index
│   fraud_sms_metadata.json
│   generate_data.py
│   run_shell.py
│   run_webui.py
│   run_webui_6G.py
│   run_webui_api.py
│   requirements.txt
│
├───m3e-base
│   │   config.json
│   │   model.safetensors
│   │   tokenizer.json
│   │   (other model files)
│   └───1_Pooling
│           config.json
│
├───Qwen2.5-7B
│       config.json
│       model-00001-of-00004.safetensors
│       model-00002-of-00004.safetensors
│       model-00003-of-00004.safetensors
│       model-00004-of-00004.safetensors
│       tokenizer.json
│       (other model files)
│
└───Telecom_Fraud_Texts_5-main (Optional)
        label00-last.csv
        label01-last.csv
        label02-last.csv
        label03-last.csv
        label04-last.csv
        LICENSE
        README.md
```

---

## Usage

### 1. Local Web Interface  
```bash
python run_webui.py
```
- Access interface at default URL: http://127.0.0.1:7860  
- For <8GB VRAM, use `run_webui_6G.py`

### 2. API Web Interface  
In `run_webui_api.py`:  
```python
api_key = "<your_api_key>" 
base_url = "<your_base_url>" 
model_name = "<target_model>" 
```
Example platforms: [Alibaba Cloud](https://www.aliyun.com), [DeepSeek Platform](https://platform.deepseek.com)  
```bash
python run_webui_api.py
```

### 3. Command-Line Interface  
```bash
python run_shell.py
```
For <8GB VRAM, use configurations from `run_shell_6G.py`

### 4. Benchmark Testing  
```bash
python benchmark.py
```
For <8GB VRAM, use configurations from `run_shell_6G.py`

### 5. Database Generation (Optional)  
```bash
python generate_data.py
```
*Note: Pre-generated database included in repository*

---

## Configuration Parameters

### Shared Parameters (run_webui.py/run_webui_6G.py/run_shell.py/benchmark.py):  
- **LLM_PATH = "./Qwen2.5-7B"**  
- **M3E_PATH = "./m3e-base"**  
- **FAISS_INDEX_FILE = "fraud_sms_faiss.index"**  
- **METADATA_FILE = "fraud_sms_metadata.json"**  
- **SAFE_MODE = False**  
  - True: Conservative mode (3 predictions with majority voting)  
  - False: Standard mode (single prediction)  
- **DE_BUG = False**  
  - Show raw model outputs  
- **SHOW_CATEGORY = False**  
- **SHOW_SAMPLE = True**  
- **MAX_TOKENS = 1024**  
- **MAX_RETRIES = 2**  

### API Mode Parameters (run_webui_api.py):  
- **M3E_PATH**, **FAISS_INDEX_FILE**, **METADATA_FILE**  
- **MAX_TOKENS = 1024**  
- **api_key**, **base_url**, **model_name**  
- **SHOW_HISTORY = False**  
- **SHOW_USED_TIME = True**  

---

## Benchmark Results

Test results using Qwen2.5-7B in standard mode (benchmark.py):  
```plaintext
=== Evaluation Results ===
Mode: Standard  
Total SMS: 100  
Correct Category Classification: 93  
Correct Fraud Detection: 97  
False Positives: 0  
Misclassifications: 7  
No Result: 0  
Format Errors: 0  
Model Crashes: 0  
Success Rate: 93.00% (Correct Category / (Total - Crashes))  
Fraud Detection Rate: 97.00% (Correct Fraud / (Total - Crashes))  
False Positive Rate: 0.00%  
False Negative Rate: 3.00% (Missed Fraud / (Total - Crashes))
```

---

## License

This project uses the GPLv3 license. See [LICENSE](LICENSE) for details.

---

## Dataset Usage Restrictions

The [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5) dataset has specific restrictions:

**Authorized Use**:  
- Limited to scientific research by universities and research institutions  
- **Prohibited** for commercial purposes (no commercial licenses available)  

**Attribution Requirement**:  
When using this dataset in publications, please cite:  
Li, J.; Zhang, C.; Jiang, L. Innovative Telecom Fraud Detection: A New Dataset and an Advanced Model with RoBERTa and Dual Loss Functions. *Appl. Sci.* 2024, **14**, 11628. https://doi.org/10.3390/app142411628  

---

## Contact

- **GitHub**: [stlin256](https://github.com/stlin256)  
- Contributions and suggestions welcome!

---

## Acknowledgments

- Special thanks to [ChangMianRen](https://github.com/ChangMianRen) for the [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5) dataset  
- Appreciation to [Qwen](https://huggingface.co/Qwen) and [moka-ai](https://huggingface.co/moka-ai) for their high-quality models

-----

<div id="cn"></div>

## FraudSMS_RAG_Shield

# 融合大模型推理与RAG检索增强的诈骗短信甄别系统

#### 4月1日更新——支持以OpenAI标准调用在线api。
#### 3月29日更新——支持在6G显存情况下使用Qwen2.5-7B推理。

### 简介

本项目结合大模型推理与 **RAG（检索增强生成）** 技术，致力于对短信进行精准识别和分类，保护用户免受电信诈骗侵害。系统基于 [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5) 数据集，采用 [m3e-base](https://huggingface.co/moka-ai/m3e-base) 模型对短信进行向量化，利用 [FAISS](https://github.com/facebookresearch/faiss) 实现快速相似性检索，并结合 [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) 大模型进行深度推理。系统能够识别 **“正常短信”** 及以下四类诈骗类型：

- **冒充公检法**（假装执法机构，恐吓或索要信息）
- **贷款诈骗**（涉及贷款、优惠条件诱导转账）
- **冒充客服**（假冒服务人员索要信息或钱财）
- **冒充领导或熟人诈骗**（假装熟人或上级要求转账）

通过提供分类结果、理由和相似短信示例，系统不仅提升分类准确性，还增强了结果的可解释性，帮助用户有效防范诈骗短信。

------

### 主要特性

- **高准确性**：诈骗短信漏报率仅 **3%**（基于 benchmark.py 测试结果）。
- **快速响应**：在 NVIDIA 4060 Laptop 上，平均 **3秒** 内得出结果。
- **可解释性强**：输出分类结果、理由及相似短信示例，帮助用户理解判断依据。
- **用户友好**：通过 [Gradio](https://gradio.app/) 提供可视化 Web 界面，便于交互。
- **创新结合**：结合 RAG 技术和大模型推理，提升分类准确性和推理深度。
- **轻量部署**：提供调用在线api版，无需本地部署推理，轻量便携。

------

### 效果展示

#### run_webui.py

手机浏览器界面：
![本地路径](Pictures/手机webui.png "手机webui")
电脑浏览器界面：
![本地路径](Pictures/PCwebui.png "手机webui")

#### run_shell.py
![本地路径](Pictures/PCshell.png "手机webui")

#### benchmark.py
![本地路径](Pictures/PCbenchmark_1.png "手机webui")
<center>省略中间内容</center>

![本地路径](Pictures/PCbenchmark_2.png "手机webui")

------

### 硬件要求

#### 对于本地推理
- **流畅配置**：配备 **8GB 显存** 的 NVIDIA GPU。

```
    对于30字以内的输入，RTX4060Laptop耗时约3s
```

- **最低配置**：配备 **6GB 显存** 的 NVIDIA GPU
```
    对于30字以内的输入，GTX1660S耗时20-35s
```

#### 对于调用在线api
- **最低配置**：**是台电脑都能跑**
```
    对于30字以内的输入，深度求索官网DeepSeek V3 约3-6s
```

------

### 安装与配置

1. 安装 Anaconda

- 下载并安装 Anaconda：[Anaconda.org](https://www.anaconda.com/)
- 安装完成后，打开 **Anaconda Prompt**。
- 创建虚拟环境


```
conda create -n sms python=3.12
```

2. 激活环境：

```bash
conda activate sms
```

3. 安装依赖

进入项目根目录，安装所需 Python 包：

```bash
pip install -r requirements.txt
```

安装pytorch2.6.0（根据自身配置前往[pytorch.org](https://pytorch.org/)获取下载命令）
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

安装 FAISS（GPU 版本）：

```bash
conda install -c conda-forge faiss-gpu
```


4. 下载模型和数据集

- **模型下载**：
  - [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
  - [m3e-base](https://huggingface.co/moka-ai/m3e-base)
  - 如果无法访问 Hugging Face，可使用镜像：
    - [Qwen2.5-7B (HF Mirror)](https://hf-mirror.com/Qwen/Qwen2.5-7B)
    - [m3e-base (HF Mirror)](https://hf-mirror.com/moka-ai/m3e-base)
  - **注意**：调用在线api进行推理，只需下载[m3e-base](https://huggingface.co/moka-ai/m3e-base)
- **数据集下载（可选）**：
  - [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5)
  - **注意**：项目已附带向量化数据集（fraud_sms_faiss.index 和 fraud_sms_metadata.json），推理时无需下载原始数据集。

### 文件结构

确保项目目录结构如下：

```plaintext
项目目录
│   benchmark.py
│   fraud_sms_dataset.json
│   fraud_sms_faiss.index
│   fraud_sms_metadata.json
│   generate_data.py
│   run_shell.py
│   run_webui.py
│   run_webui_6G.py
│   run_webui_api.py
│   requirements.txt
│
├───m3e-base
│   │   config.json
│   │   model.safetensors
│   │   tokenizer.json
│   │   (其他模型文件)
│   └───1_Pooling
│           config.json
│
├───Qwen2.5-7B
│       config.json
│       model-00001-of-00004.safetensors
│       model-00002-of-00004.safetensors
│       model-00003-of-00004.safetensors
│       model-00004-of-00004.safetensors
│       tokenizer.json
│       (省略其他模型文件)
│
└───Telecom_Fraud_Texts_5-main (可选)
        label00-last.csv
        label01-last.csv
        label02-last.csv
        label03-last.csv
        label04-last.csv
        LICENSE
        README.md
```

------

### 运行项目

#### 1. 使用本地推理的Web交互

```bash
python run_webui.py
```

- 等待模型加载完成，终端将输出一个 URL（ 默认 http://127.0.0.1:7860 ）。
- 在浏览器中打开该链接，即可进入交互界面。
- 如果**显存<8G**，请使用`run_webui_6G.py`

#### 1. 调用在线api推理的Web交互

-  在`run_webui_api.py`的配置中填入以下信息

```python
api_key = "<你的api_key>" #你的api_key
base_url = "<你的base_url>" #你的base_url
model_name = "<你调用的模型>" #你调用的模型
```
- api_key可在各平台获得，如[阿里云](https://www.aliyun.com)、[深度求索开放平台](https://platform.deepseek.com)等
```bash
python run_webui_api.py
```

- 等待模型加载完成，终端将输出一个 URL（ 默认 http://127.0.0.1:7860 ）。
- 在浏览器中打开该链接，即可进入交互界面。

#### 2. 命令行交互
```bash
python run_shell.py
```
- 模型加载完成后，可在终端输入短信内容进行分类。
- 如果**显存<8G**，请替换`quant_config`和`llm_model`为`run_shell_6G.py`中的版本

#### 3. 基准测试
```bash
python benchmark.py
```
- 将自动加载 fraud_sms_dataset.json 中的测试数据，运行评估并输出结果。
- 如果**显存<8G**，请替换`quant_config`和`llm_model`为`run_shell_6G.py`中的版本

#### 4. 生成数据库（可选）
```bash
python generate_data.py
```
- 读取分类短信数据并生成数据库（注：项目仓库内已包含生成的数据库）

------

### 代码配置

以下参数可在 run_webui.py、run_webui_6G.py、run_shell.py 或 benchmark.py 中调整：

- **LLM_PATH = "./Qwen2.5-7B"**
  大模型路径。
- **M3E_PATH = "./m3e-base"**
  向量化模型路径（需与生成索引时使用的模型一致）。
- **FAISS_INDEX_FILE = "fraud_sms_faiss.index"**
  FAISS 索引文件路径。
- **METADATA_FILE = "fraud_sms_metadata.json"**
  元数据文件路径。
- **SAFE_MODE = False**  
  - True：稳妥模式（三次预测取多数）。
  - False：普通模式（单次预测）。
  - **注意**：当前提示词和超参数下，两者表现差异不大。
- **DE_BUG = False**
  是否显示模型原始输出（调试用）。
- **SHOW_CATEGORY = False**
  是否输出分类类别。
- **SHOW_SAMPLE = True**
  是否输出相似短信示例。
- **MAX_TOKENS = 1024**
  模型最大输出长度。
- **MAX_RETRIES = 2**
  输出异常时的最大重试次数。

以下参数可在 run_webui_api.py 中调整：

- **M3E_PATH = "./m3e-base"**
  向量化模型路径（需与生成索引时使用的模型一致）。
- **FAISS_INDEX_FILE = "fraud_sms_faiss.index"**
  FAISS 索引文件路径。
- **METADATA_FILE = "fraud_sms_metadata.json"**
  元数据文件路径。
- **MAX_TOKENS = 1024**
  模型最大输出长度。
- **api_key = "your_api_key"**
  填入你的api_key。
- **base_url = "your_base_url"**
  填入你的base_url。
- **model_name = "your_model"**
  填入你调用的模型名称。
- **DE_BUG = False**
  是否显示模型原始输出。
- **SHOW_CATEGORY = False**
  是否输出分类类别。
- **SHOW_SAMPLE = True**
  是否输出相似短信示例。
- **SHOW_HISTORY = False**
  是否输出web页面的聊天历史记录。
- **SHOW_USED_TIME = True**
  是否输出调用api耗时。


------


### 基准测试结果

以下为系统在Qwen2.5-7B推理，使用普通模式下的测试结果（ benchmark.py）：

```plaintext
=== 评测结果 ===
当前模式: 普通模式
总短信数: 100
正确分类(具体类型): 93
正确分类(是否诈骗): 97
误报短信: 0
错误分类: 7
无法得出结果: 0
格式错误: 0
模型崩溃: 0
成功率: 93.00% (正确分类(具体类型) / (总短信 - 模型崩溃))
成功率: 97.00% (正确分类(是否诈骗) / (总短信 - 模型崩溃))
误报率: 0.00% (误报短信(非诈骗被识别为诈骗) / (总短信 - 模型崩溃))
漏报率: 3.00% (漏报短信(诈骗短信被放行) / (总短信 - 模型崩溃))
```

------

### 许可证

本项目采用GPLv3许可证。详情请参阅 [LICENSE](LICENSE)文件。

------
### 本项目使用的数据集 [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5) 具有使用限制

**使用限制**：

- 该数据集仅限 **高校和科研机构** 用于科学研究。

- **禁止** 用于任何商业目的，不提供商业授权。

- 使用数据集进行科学研究并发表成果时，需注明来源，例如：

  Li, J.; Zhang, C.; Jiang, L. Innovative Telecom Fraud Detection: A New Dataset and an Advanced Model with RoBERTa and Dual Loss Functions. *Appl. Sci.* 2024, **14**, 11628. https://doi.org/10.3390/app142411628

------


### 联系方式

- **GitHub**：[stlin256](https://github.com/stlin256)
- 欢迎提交问题或建议！

------


### 致谢

- 感谢 [ChangMianRen](https://github.com/ChangMianRen) 提供数据集 [Telecom_Fraud_Texts_5](https://github.com/ChangMianRen/Telecom_Fraud_Texts_5)。
- 感谢 [Qwen](https://huggingface.co/Qwen) 和 [moka-ai](https://huggingface.co/moka-ai) 提供的优质模型。

------
