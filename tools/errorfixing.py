import os
import time
import json
from openai import OpenAI

# ======================
# 配置区（请按实际情况修改）
# ======================
INPUT_FILE = "皮肤病中医诊疗学.txt"          # 原始含错字文本
OUTPUT_FILE = "ocr_corrected.txt"   # 修正后输出
CHECKPOINT_FILE = "progress.json"   # 进度保存

# 替换为你的 Gemini 2.5 Flash 的 OpenAI 兼容 endpoint 和 API Key
# 例如：Google AI Studio 的 OpenAI 兼容地址
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = ""

MODEL_NAME = "gemini-2.5-flash-lite"     # 确保模型名正确

# 分块大小（字符数，约 300–500 tokens）
MAX_CHARS_PER_CHUNK = 800

# 调用间隔（秒），防止触发速率限制
REQUEST_DELAY = 5  # Gemini Flash 免费 tier 建议 ≥0.5s

# ======================
# 初始化 OpenAI 客户端
# ======================
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ======================
# 文本分块函数
# ======================
def split_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current += "\n\n" + p if current else p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks

# ======================
# 调用 LLM 修正单个 chunk
# ======================
def correct_chunk(chunk: str) -> str:
    prompt = f"""你是一名中西医结合医学文献校对专家。请严格修正以下文本中的 OCR 错别字、乱码或明显识别错误，但不得改变原意、不得删减内容、不得添加解释。仅输出修正后的纯文本。

原文：
{chunk}

修正后：
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
        
            temperature=0.0,          # 确定性输出
            # max_tokens=1000,
            # timeout=30.0
        )
        corrected = response.choices[0].message.content.strip()
        # print(corrected)
        return corrected
    except Exception as e:
        print(f"[ERROR] {e}")
        return chunk  # 出错时返回原文，避免丢失

# ======================
# 主处理流程
# ======================
def main():
    # 读取原始文本
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    chunks = split_text(raw_text, MAX_CHARS_PER_CHUNK)
    print(f"共切分为 {len(chunks)} 个文本块")

    # 加载进度（断点续传）
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        start_idx = progress.get("last_index", -1) + 1
        corrected_list = progress.get("corrected", [])
        print(f"从第 {start_idx + 1} 块继续处理")
    else:
        start_idx = 0
        corrected_list = []

    # 批量处理
    for i in range(start_idx, len(chunks)):
        print(f"处理第 {i+1}/{len(chunks)} 块...")
        corrected = correct_chunk(chunks[i])
        corrected_list.append(corrected)

        # 每 10 块保存一次进度和中间结果
        if (i + 1) % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(corrected_list))
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "last_index": i,
                    "corrected": corrected_list
                }, f, ensure_ascii=False, indent=2)
            print(f"✅ 已保存至第 {i+1} 块")

        time.sleep(REQUEST_DELAY)  # 控制请求频率

    # 最终保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(corrected_list))
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    print("🎉 全部处理完成！")

if __name__ == "__main__":
    main()