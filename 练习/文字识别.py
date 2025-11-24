from aip import AipOcr  # type: ignore
import os
import re
import argparse
import logging
from typing import List
import jieba
from jieba import analyse as jieba_analyse
from dotenv import load_dotenv

# 读取 .env（若存在）
load_dotenv()

# 配置百度云API信息（从环境变量读取，避免泄露）
APP_ID = os.getenv('BAIDU_APP_ID')
API_KEY = os.getenv('BAIDU_API_KEY')
SECRET_KEY = os.getenv('BAIDU_SECRET_KEY')


# 初始化AipOcr对象（缺少凭据时不给初始化）
client = AipOcr(APP_ID, API_KEY, SECRET_KEY) if all([APP_ID, API_KEY, SECRET_KEY]) else None

def split_sentences(text: str) -> List[str]:
    """中英文分句"""
    # 若输入为空或非字符串，直接返回空列表
    if not text:
        return []
    # 使用正则根据中英文终止符与分号、换行进行分句
    parts = re.split(r"[。！？!?；;\n]", text)
    # 去除每个分句的首尾空白，并过滤空串
    return [p.strip() for p in parts if p and p.strip()]

def summarize_text_local(text: str, max_sentences: int = 3) -> str:
    """基于词频的简易本地摘要"""
    # 将文本分句，后续以句子为单位打分
    sentences = split_sentences(text)
    # 若没有可用句子，返回空字符串
    if not sentences:
        return ""
    # 词频字典：统计重要词出现次数
    word_freq = {}
    # 遍历每个句子
    for sentence in sentences:
        # 使用结巴对句子进行分词
        for token in jieba.cut(sentence):
            # 去除词两端空白
            t = token.strip()
            # 跳过空词
            if not t:
                continue
            # 过滤纯标点或长度为 1 的无信息 token
            if re.fullmatch(r"[\W_]+", t) or len(t) == 1:
                continue
            # 记录词频
            word_freq[t] = word_freq.get(t, 0) + 1
    # 若未能构建词频，则回退为取前若干句
    if not word_freq:
        return "\n".join(sentences[:max_sentences])
    # 句子打分列表
    scores = []
    # 针对每个句子计算得分：词频之和 / 句长惩罚
    for idx, sentence in enumerate(sentences):
        # 对句子再次分词
        tokens = list(jieba.cut(sentence))
        # 句子重要度为其中词语的词频累加
        score = sum(word_freq.get(t, 0) for t in tokens)
        # 句长惩罚，避免长句天然得分高
        length_penalty = max(len(tokens), 1)
        # 记录 (句子索引, 归一化得分)
        scores.append((idx, score / length_penalty))
    # 选取得分最高的若干句
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    # 保持原文顺序输出
    top_idx = sorted(i for i, _ in top)
    return "\n".join(sentences[i] for i in top_idx)

def generate_structured_summary(text: str) -> str:
    """生成结构化摘要："这个图片中，包含了xx信息，大概意思是：xxx"""
    # 非法或空文本直接返回占位
    if not text or not isinstance(text, str):
        return "（无可用摘要）"

    # 使用 TextRank 提取关键词，偏向名词/动词类词性
    try:
        keywords = jieba_analyse.textrank(
            text,
            topK=6,
            withWeight=False,
            allowPOS=("ns", "n", "vn", "v", "nr", "nt")
        )
    except Exception:
        # 关键词提取异常时兜底为空列表
        keywords = []

    # 分句以便挑选“概述句”
    sentences = split_sentences(text)
    # 无句子可用时返回占位
    if not sentences:
        return "（无可用摘要）"

    # 若没有关键词，退化为选择第一句作为概述
    if not keywords:
        gist = sentences[0] if sentences else ""
    else:
        # 基于关键词重叠度为每个句子打分，挑选重叠度最高者
        scored = []
        keyword_set = set(keywords)
        for idx, s in enumerate(sentences):
            # 句子分词
            tokens = list(jieba.cut(s))
            # 关键词重叠计数
            overlap = sum(1 for t in tokens if t in keyword_set)
            # 长度惩罚，防止长句优势
            length_penalty = max(len(tokens), 1)
            score = overlap / length_penalty
            # 记录 (句子索引, 得分)
            scored.append((idx, score))
        # 按得分降序排列
        scored.sort(key=lambda x: x[1], reverse=True)
        # 选择得分最高的句子索引，若无则回退到 0
        best_idx = scored[0][0] if scored else 0
        # 取该句为概述
        gist = sentences[best_idx]

    # 将前若干关键词拼为“包含的信息”片段
    info_part = "、".join(keywords[:4]) if keywords else "关键信息"
    # 再次兜底，避免为空
    if not info_part:
        info_part = "关键信息"
    # 按固定模版返回结构化摘要
    return f"这个图片中，包含了{info_part}等信息，大概意思是：{gist}"

def get_file_content(file_path: str) -> bytes:
    """读取图片文件内容"""
    # 以二进制只读方式打开文件
    with open(file_path, 'rb') as fp:
        # 读出全部内容并返回
        return fp.read()

def recognize_text(image_path: str) -> str | None:
    """识别图片中的文字"""
    # 若 OCR 客户端未初始化，提示缺少环境变量
    if client is None:
        logging.error("OCR 初始化失败：缺少 BAIDU_APP_ID/BAIDU_API_KEY/BAIDU_SECRET_KEY 环境变量")
        return None
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        logging.error(f"错误：文件 {image_path} 不存在")
        return None
    
    # 读取图片字节内容
    image = get_file_content(image_path)
    
    # 调用通用文字识别（高精度版）接口
    result = client.basicAccurate(image)
    
    # 若返回包含识别结果列表，则拼接为多行文本
    if 'words_result' in result:
        return '\n'.join([item['words'] for item in result['words_result']])
    else:
        # 否则记录错误并返回 None
        logging.error(f"识别失败：{result.get('error_msg', '未知错误')}")
        return None

def configure_logging(level: str) -> None:
    # 将字符串级别映射到 logging 模块的常量
    level_map = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    # 配置全局日志：设置日志级别和输出格式
    logging.basicConfig(
        level=level_map.get(level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(message)s'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIPyProject - OCR 与本地摘要')
    parser.add_argument('--image', type=str, default='', help='要识别的图片路径（默认使用内置示例）')
    parser.add_argument('--max-sentences', type=int, default=3, help='摘要句子数量（默认 3）')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别：DEBUG/INFO/WARNING/ERROR')
    args = parser.parse_args()

    configure_logging(args.log_level)

    # 图片路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = args.image if args.image else os.path.join(script_dir, 'images', 'p479559.png')

    logging.info(f'开始识别：{image_path}')
    text = recognize_text(image_path)
    if not text:
        raise SystemExit(1)

    logging.info('识别完成，开始摘要...')
    structured = generate_structured_summary(text)
    summary = summarize_text_local(text, max_sentences=args.max_sentences)

    print('\n识别结果：\n')
    print(text)
    print('\n' + '-'*50 + '\n')
    print('AI总结的结构化摘要：')
    print(structured)
    print('\nAI总结的简要摘要：')
    print(summary if summary else '（无可用摘要）')
