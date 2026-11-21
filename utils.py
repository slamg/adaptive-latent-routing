# 文件作用：提供实验共用的随机种子、设备选择和答案处理工具。
# 主要职责：提取并规范化数值答案、提取代码块，以及在超时限制下执行代码评测。

import os
import random
import re
from decimal import Decimal
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Only plain decimal notation and correctly grouped thousands are numeric
# answers. Match a whole numeric-looking token first, so e.g. "12,34" cannot
# silently become "12" or "34". Scientific notation is deliberately outside
# this scoring policy, but is consumed as one token rather than partially read.
# 新增规则：先识别完整数字候选，再校验十进制与千分位格式，避免把非法格式的一部分当成答案。
# 这些模式供下面的答案提取函数使用；不是对任意数学表达式求值。
_NUMBER_TOKEN = re.compile(
    r"(?<![\w.,])[-+]?(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+)"
    r"(?:[eE][-+]?[0-9]+)?"
)
_DECIMAL_ANSWER = re.compile(
    r"[-+]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
)


# 功能：检查提取出的完整数字候选是否符合普通十进制和合法千分位格式。
# 输入：token为候选字符串，可带正负号、小数或分组逗号；末尾一个标点逗号可去除。
# 输出：合法时返回数字字符串，非法时返回None；此处不把字符串转为浮点数。
def _validated_number(token: str) -> Optional[str]:
    # A trailing comma can be sentence punctuation; internal commas must form
    # groups of three. Reject repeated trailing commas instead of repairing them.
    if token.endswith(",") and not token.endswith(",,"):
        token = token[:-1]
    return token if _DECIMAL_ANSWER.fullmatch(token) else None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        # 新增片段：从最后一个boxed内容提取完整候选并校验；输出合法数字字符串或None。
        number = _NUMBER_TOKEN.search(content)
        return _validated_number(number.group(0)) if number else content.strip()

    # 新增片段：无boxed时收集数字候选，仅对最后一个候选校验后返回；不回退选择更早数字。
    numbers = _NUMBER_TOKEN.findall(text)
    if numbers:
        return _validated_number(numbers[-1])
    return None


def extract_gold(text: str) -> Optional[str]:
    # 新增片段：先取标准答案标记后的整行，再用相同数字规则校验，避免千分位答案被截断。
    # 输入为解答文本；输出仍是数字字符串或None，不在这里执行模型预测。
    match = re.search(r"####\s*([^\r\n]*)", text)
    if not match:
        return None
    number = _NUMBER_TOKEN.match(match.group(1))
    return _validated_number(number.group(0)) if number else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    # 新增功能：在原字符串规范化步骤中增加精确的数值等价规范化。
    # 输入：答案字符串；合法十进制经Decimal转换，输出统一字符串，例如42.0变为42、负零变为0。
    # 非数字标签继续按去首尾空格及小写处理；不使用浮点容差，也不求值分数或表达式。
    value = ans.strip().lower()
    if not _DECIMAL_ANSWER.fullmatch(value):
        return value
    # Decimal construction and fixed-point formatting are exact; normalize()
    # would apply the current Decimal precision and could round long answers.
    number = Decimal(value.replace(",", ""))
    if number == 0:
        return "0"
    canonical = format(number, "f")
    return canonical.rstrip("0").rstrip(".") if "." in canonical else canonical


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)

