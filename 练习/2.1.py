import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, TextGenerationPipeline

print("=== 诗歌生成脚本（已优化）===")
# 1. 加载分词器和模型（适配GPU，减少警告）
print("1. 正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-poem", truncation=True)
print("2. 正在加载模型...")
# 自动适配GPU/CPU，避免调试模式下硬件冲突
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem").to(device)
print(f"3. 模型加载成功！当前使用设备：{device}")

# 2. 生成诗歌（优化参数，减少重复）
print("4. 正在生成诗歌...")
result = TextGenerationPipeline(model, tokenizer)(
    "[CLS] 万 叠 春 山 积 雨 晴 ,",
    max_length=80,
    do_sample=True,
    temperature=0.6,        # 降低随机性，提升诗句连贯性
    repetition_penalty=1.5, # 抑制重复用词（解决“花”字重复问题）
    top_p=0.9,
    truncation=True         # 显式截断，消除警告
)

# 3. 格式化输出（按古诗排版，清晰易读）
poem = result[0]['generated_text'].replace("[CLS]", "").strip()
formatted_poem = poem.replace("，", "，\n").replace("。", "。\n")

print("\n✅ 生成完成！古诗结果：")
print("-" * 60)
print(formatted_poem)
print("-" * 60)