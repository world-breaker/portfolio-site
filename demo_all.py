#!/usr/bin/env python3
"""一键跑通五个仓库的离线验证（不需要 API key、不联网下载模型）。

设计目标：
- 每一步都有进度条、耗时与心跳，长任务不会"看起来卡死"；
- 失败即停，打印失败步骤的输出尾部，便于定位；
- 成功时从输出里提取关键指标（passed 数 / Recall@3 变化 / 重放步数 / 累计用量 / 安全项），汇总成一张表。

仓库分工（叙事顺序：三个地基 → 两个新项目）：
- hello-agent / rag-eval-platform / rag-mcp-server：地基（Agent / 检索评测 / MCP 服务）；
- job-fit-assistant / tiny-llm-from-scratch：两个新项目（AI 应用落地 / 从零训练小模型）。

用法：
    python demo_all.py                     # 全部 9 步（默认，本机实测约 180 秒）
    python demo_all.py --skip-tests        # 只跑演示（跳过 pytest 步骤）
    python demo_all.py --only eval-hybrid  # 只跑指定步骤
    python demo_all.py --list              # 列出步骤 id
    python demo_all.py --bootstrap         # 先装依赖（每仓一个 .venv），再按需跑验证

可选步骤（默认**不跑**：要 GPU；本机实测约 106 秒，首次需下载/加载模型会更久）：
    python demo_all.py --only eval-generation   # 单跑生成层忠实度评测（离线两臂）
    python demo_all.py --all                    # 连可选步骤一起跑

注意（产物清理）：
    部分实验脚本会按**当天日期**把结果写进 `experiments/results/`（例如
    `hybrid_comparison_<日期>.json`）。门面只做**验证**、不生产数据，因此默认在每步跑完后
    把**该步新产生**的产物删掉（已存在的文件一律不动），这样跨天重跑也不会弄脏仓库。
    确实想留存产物时加 `--keep-artifacts`，或直接跑对应实验脚本。

依赖引导（--bootstrap）：
    全新 clone 没有 .venv，面会回退到系统 Python，第一步就 ModuleNotFoundError。
    `python demo_all.py --bootstrap` 会**每仓一个 .venv** 把依赖装好，再跑 `python demo_all.py`。
    为什么要分仓装：`mcp` SDK 主版本冲突（详见 REPO_ENV_PLANS 的注释），单一环境无解。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # Python 3.11+ 有 tomllib；3.10 及以下降级为"跳过 pyproject 解析"
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore


ROOT = Path(__file__).resolve().parent

def resolve_repo(name: str) -> Path:
    """找到五个仓之一：先看门面同级目录，再看上一级目录。

    ★ 为什么两级都找：门面现在住在自己的仓 `portfolio-site/` 里，
    而五个仓通常是**兄弟目录**（同一父目录）—— 两种摆法都支持，
    clone 顺序、先放哪都无所谓；找不到时返回同级路径用于报错。
    """
    for base in (ROOT, ROOT.parent):
        cand = base / name
        if cand.is_dir():
            return cand
    return ROOT / name


# _python_with_torch() 的进程内缓存（探测一次要 import torch，约 8 秒）。
_TORCH_PYTHON_CACHE: Optional[Tuple[str]] = None


def _chcp_utf8() -> None:
    """Windows 中文终端切 UTF-8，避免进度条/中文乱码。"""
    if os.name == "nt":
        os.system("chcp 65001 > nul")


def bar(done: int, total: int, width: int = 24) -> str:
    pct = 0.0 if total <= 0 else done / total
    fill = int(round(pct * width))
    return "[" + "█" * fill + "░" * (width - fill) + f"] {int(pct * 100):3d}% {done}/{total}"


def _pad(text: str, width: int) -> str:
    """按终端显示宽度补空格（中文算 2 列）。"""

    def display_width(value: str) -> int:
        return sum(
            2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in value
        )

    return text + " " * max(0, width - display_width(text))


def _python_in(project: Path) -> str:
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = project / rel
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _python_with_torch() -> Optional[str]:
    """找一个能 import torch 的解释器（rag-eval 需要）。

    结果做进程内缓存：每次探测都要真的 import 一次 torch（约 8 秒），
    而 bootstrap / 预检 / 建步骤都会用到它。
    """
    global _TORCH_PYTHON_CACHE
    if _TORCH_PYTHON_CACHE is not None:
        return _TORCH_PYTHON_CACHE[0] or None
    candidates = [
        os.environ.get("RAGEVAL_PYTHON", ""),
        r"C:\Users\田梦龙\AppData\Local\Programs\Python\Python312\python.exe",
        sys.executable,
    ]
    found = ""
    for candidate in candidates:
        if not candidate:
            continue
        try:
            result = subprocess.run(
                [candidate, "-c", "import torch, sentence_transformers"],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                found = candidate
                break
        except Exception:
            continue
    _TORCH_PYTHON_CACHE = (found,)
    return found or None


def _scratch_dir() -> Path:
    """可选步骤的落盘目录：优先 D:\\Temp\\opencode（本机约定），否则退回系统临时目录。"""
    for candidate in (Path(r"D:\Temp\opencode"), Path(tempfile.gettempdir())):
        try:
            target = candidate / "facade_gen"
            target.mkdir(parents=True, exist_ok=True)
            return target
        except OSError:
            continue
    return Path(tempfile.gettempdir())


# ============================================================================
# 依赖引导（--bootstrap）
# ============================================================================
#
# 为什么必须"每仓一个 .venv"，而不是一个环境装五个仓？—— **mcp SDK 主版本冲突，
# 单一环境无解**（硬约束，来自 PyPI 元数据实测）：
#   * hello-agent 需要 mcp==1.30.0：其 langchain-mcp-adapters==0.3.2 的 requires_dist
#     明确写死 "mcp<2.0.0,>=1.24.0"（实测 PyPI JSON）。
#   * rag-mcp-server 需要 mcp==2.3.0：代码用 2.x 才有的 mcp.server.mcpserver.MCPServer
#     （rag-mcp-server/server.py 实测 import），1.x 没有这个模块。
#   同一个 site-packages 不可能同时装 mcp 1.30.0 与 2.3.0（后装的覆盖先装的），
#   所以这两个仓必须各自一个 .venv，靠"venv 内 site-packages 优先于系统"的机制隔离。
#
# 另外三个仓（rag-eval-platform / job-fit-assistant / tiny-llm-from-scratch）
# **不需要隔离**：它们的依赖声明都能被机器级 Python 3.12 满足（实测在位：
# torch 2.14.0+cu132 / sentence-transformers 5.5.1 / transformers 5.9.0 /
# gradio 6.16.0 / pyarrow 24.0.0 / matplotlib 3.11.2 / pytest 9.1.1），
# 而且门面本来就是用系统解释器跑它们（_python_with_torch），故不建 venv。
#
# 两个 venv 都用 --system-site-packages：复用机器级重包（torch/transformers/
# sentence-transformers/numpy），既省掉每仓重下 GB 级 wheel，也让 CUDA 保持可用。
# 代价是 requirements.txt 里 torch==2.14.1 这类"重包精确版本"不会在 venv 内生效
# （改为继承机器级 2.14.0+cu132），bootstrap 的表格会把这处偏差显式列出来，不隐瞒。
#
# 机器级前置条件（本 bootstrap 不负责、也不允许自动下载）：
#   * HF 模型缓存（3.5GB）：BAAI/bge-small-zh-v1.5、cross-encoder/mmarco-...-v1、
#     Qwen/Qwen2.5-1.5B-Instruct。缺失只会让第 5 步（eval-hybrid）等失败，
#     bootstrap 只检测 + 打印下载命令。
#   * 机器级 torch（CUDA 版）：给 rag-eval-platform 的 embedding 用。

# 每个仓的钉版集合。hello-agent / rag-mcp-server 用"轻量 overlay"，
# 重包（torch/transformers/sentence-transformers/numpy）继承机器级安装。
HELLO_AGENT_PINS: Tuple[str, ...] = (
    "langgraph==1.2.15",
    "langgraph-prebuilt==1.1.1",
    "langchain==1.4.6",
    "langchain-core==1.6.11",
    "langchain-openai==1.7.2",
    "langchain-mcp-adapters==0.3.2",
    "mcp==1.30.0",
    "langgraph-checkpoint-sqlite==3.1.2",
    "aiosqlite==0.22.1",
    "sqlite-vec==0.1.9",
    "python-dotenv==1.2.4",
    "PyYAML==6.0.3",
    "pytest==9.1.1",
    # 以下四个是 requirements.txt 没写、但 venv 内必须钉住的传递依赖
    # （mcp 1.30.0 要 python-multipart/sse-starlette，langchain-openai 要 tiktoken；
    #  truststore 是 mcp 走 TLS 的可选加速项）。
    "tiktoken==0.14.0",
    "python-multipart==0.0.32",
    "sse-starlette==3.5.0",
    "truststore==0.10.5",
    "tzdata==2026.5",
)

RAG_MCP_PINS: Tuple[str, ...] = (
    "mcp==2.3.0",
    "sentence-transformers==5.7.0",
    "transformers==5.20.0",
    "numpy==2.5.4",
    "PyYAML==6.0.3",
    "uvicorn>=0.30",
    "pytest==9.1.1",
    # huggingface_hub 必须 >=1.31（transformers 5.20.0 的硬约束），机器级只有 1.16.4，
    # 不钉住它 pip 就会把整套 hub 升上去。
    "huggingface_hub==2.2.0",
)

# 需要隔离的仓：名字 -> (钉版集合, 是否 --system-site-packages)
# 关于 rag-mcp-server 的 venv 模式：任务书正文写"普通 venv"，但它同时给出的
# "已实测可用包集合"里**没有 torch**，而 clean_verify 里那份跑通 173 tests + 10/10
# 安全项的 rag-mcp-server/.venv 实测 include-system-site-packages = true（继承机器级
# torch 2.14.0+cu132）。两者只有"继承机器级 torch"才自洽；而且普通 venv 会把
# PyPI 的 torch 2.14.1（118MB、CPU-only）装进 venv，反而**降级掉 CUDA 能力**。
# 故此处按实测可用配置走 --system-site-packages，偏差在下方表格里显式说明。
VENV_REPOS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("hello-agent", HELLO_AGENT_PINS, True),
    ("rag-mcp-server", RAG_MCP_PINS, True),
)

# 三个"吃系统解释器"的仓：只校验，不安装（不许改全局 Python 环境）。
SYSTEM_REPOS: Tuple[Dict[str, object], ...] = (
    {
        "name": "rag-eval-platform",
        "source": "requirements.txt",
        "required_modules": ("torch", "transformers", "sentence_transformers", "gradio", "numpy", "yaml", "requests"),
        "key_versions": ("torch", "sentence-transformers", "transformers"),
    },
    {
        "name": "job-fit-assistant",
        "source": "pyproject.toml",
        # base dependencies 为空（仓库注释："基础依赖为空，核心功能只用标准库"），
        # demo 只需要 dev extra 里的 pytest；其余 extra（llm/docx/pdf/mcp/siblings）
        # 都是可选功能，缺失只会让对应功能降级，不影响 9 步验证。
        "required_modules": ("pytest",),
        "key_versions": ("pytest", "torch", "transformers"),
    },
    {
        "name": "tiny-llm-from-scratch",
        "source": "requirements.txt",
        "required_modules": ("torch", "numpy", "pyarrow", "matplotlib", "pytest"),
        "key_versions": ("torch", "numpy", "pyarrow"),
    },
)

# 需要检查的 HF 模型缓存（只检测 + 打印下载命令，绝不自动下载）。
HF_MODEL_CACHE: Tuple[Tuple[str, str, str], ...] = (
    (
        "BAAI/bge-small-zh-v1.5",
        "91.9 MB",
        "第 5 步 eval-hybrid 的 embedding + rag-mcp-server 的 embedding；缺 → 必失败",
    ),
    (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "469.9 MB",
        "第 5 步的 rerank 臂；缺 → 降级为纯向量，步骤仍能过（少一档召回）",
    ),
    (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "2955.4 MB",
        "可选第 6 步 eval-generation（本地 1.5B 生成）；缺 → 仅该可选步失败",
    ),
)

# 在目标解释器里读已装版本/模块存在性（只查元数据，不真正 import，避免 torch 导入开销）。
_PROBE_SNIPPET = (
    "import json,sys,importlib.metadata as md,importlib.util as iu\n"
    "names=json.loads(sys.argv[1]);mods=json.loads(sys.argv[2])\n"
    "ver={}\n"
    "for n in names:\n"
    "    try: ver[n]=md.version(n)\n"
    "    except Exception: ver[n]=None\n"
    "found={}\n"
    "for m in mods:\n"
    "    try: found[m]=iu.find_spec(m) is not None\n"
    "    except Exception: found[m]=False\n"
    "print(json.dumps({'versions':ver,'modules':found}))\n"
)


def _version_tuple(value: str) -> Tuple[int, ...]:
    """把版本串拆成可比较的数字元组（忽略 +cu132 这类本地版本段）。"""
    core = value.split("+")[0]
    parts: List[int] = []
    for chunk in core.split("."):
        match = re.match(r"(\d+)", chunk)
        parts.append(int(match.group(1)) if match else 0)
    return tuple(parts)


def _satisfies(version: Optional[str], spec: str) -> bool:
    """判断已装版本是否满足钉版说明（== / >= / <= / > / <）。"""
    if version is None:
        return False
    spec = spec.strip()
    if not spec:
        return True
    for clause in spec.split(","):
        clause = clause.strip()
        match = re.match(r"^(==|>=|<=|>|<)\s*(.+)$", clause)
        if not match:
            continue
        op, want = match.group(1), match.group(2).strip()
        got, target = _version_tuple(version), _version_tuple(want)
        length = max(len(got), len(target))
        got = got + (0,) * (length - len(got))
        target = target + (0,) * (length - len(target))
        if op == "==" and got != target:
            return False
        if op == ">=" and got < target:
            return False
        if op == "<=" and got > target:
            return False
        if op == ">" and got <= target:
            return False
        if op == "<" and got >= target:
            return False
    return True


def _split_pin(pin: str) -> Tuple[str, str]:
    """'mcp==1.30.0' -> ('mcp', '==1.30.0')。"""
    match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", pin.strip())
    if not match:
        return pin.strip(), ""
    return match.group(1), match.group(2).strip()


def _probe(python_exe: str, names: Tuple[str, ...], modules: Tuple[str, ...]) -> Dict[str, object]:
    """在给定解释器里查已装版本 + 模块是否存在。"""
    if not names and not modules:
        return {"versions": {}, "modules": {}}
    try:
        proc = subprocess.run(
            [python_exe, "-c", _PROBE_SNIPPET, json.dumps(list(names)), json.dumps(list(modules))],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {"versions": {}, "modules": {}}
    if proc.returncode != 0:
        return {"versions": {}, "modules": {}}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"versions": {}, "modules": {}}


def _venv_python(repo: Path) -> Optional[Path]:
    """仓内 .venv 的解释器路径（不存在返回 None）。"""
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = repo / rel
        if candidate.exists():
            return candidate
    return None


def _read_requirement_pins(repo: Path, source: str) -> List[str]:
    """读 requirements.txt 或 pyproject.toml 的依赖声明（返回原始行/条目）。"""
    path = repo / source
    if not path.exists():
        return []
    if source.endswith(".toml"):
        if tomllib is None:
            return []
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        project = data.get("project", {})
        pins = list(project.get("dependencies", []))
        for extra in ("dev", "llm", "docx", "pdf", "mcp"):
            pins.extend(project.get("optional-dependencies", {}).get(extra, []))
        return pins
    pins = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            pins.append(line)
    return pins


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))


def _hf_model_cached(repo_id: str) -> bool:
    """模型是否真的缓存好了（目录存在 + 有 >1MB 的权重文件，跟 rag-eval 的判定一致）。"""
    org, name = repo_id.split("/", 1)
    hub = _hf_cache_root() / "hub" / f"models--{org}--{name.replace('/', '--')}"
    if not hub.is_dir():
        return False
    for weights in hub.rglob("*"):
        if weights.is_file() and weights.suffix in (".safetensors", ".bin"):
            try:
                if weights.stat().st_size > 1_000_000:
                    return True
            except OSError:
                continue
    return False


def _torch_interpreter() -> Optional[str]:
    """门面跑"系统解释器类"步骤时用的解释器（含 torch + sentence_transformers）。"""
    return _python_with_torch()


def _bootstrap_venv_repo(name: str, pins: Tuple[str, ...], system_site: bool) -> Dict[str, object]:
    """为一个需要隔离的仓准备 .venv（幂等：已满足则不重装）。"""
    repo = resolve_repo(name)
    venv_dir = repo / ".venv"
    python_exe = _venv_python(repo)
    created = False
    if python_exe is None:
        if not venv_dir.exists():
            venv_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "venv"]
        if system_site:
            cmd.append("--system-site-packages")
        cmd.append(str(venv_dir))
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            return {
                "name": name,
                "status": "失败",
                "ok": False,
                "detail": (proc.stderr or proc.stdout or "").strip()[-400:],
                "installed": [],
                "versions": {},
                "seconds": 0.0,
                "path": str(venv_dir),
            }
        created = True
        python_exe = _venv_python(repo)
        if python_exe is None:
            return {
                "name": name,
                "status": "失败",
                "ok": False,
                "detail": "venv 创建后找不到解释器",
                "installed": [],
                "versions": {},
                "seconds": 0.0,
                "path": str(venv_dir),
            }

    started = time.perf_counter()
    dist_names = tuple(_split_pin(pin)[0] for pin in pins)
    before = _probe(str(python_exe), dist_names, ())
    versions = dict(before.get("versions", {}))  # type: ignore[arg-type]
    missing = []
    for pin in pins:
        dist, spec = _split_pin(pin)
        if not _satisfies(versions.get(dist), spec):
            missing.append(pin)

    installed: List[str] = []
    if missing:
        cmd = [str(python_exe), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *missing]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        for line in (proc.stdout or "").splitlines():
            if line.startswith("Successfully installed"):
                installed = line[len("Successfully installed "):].strip().split()
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            return {
                "name": name,
                "status": "失败",
                "ok": False,
                "detail": "\n".join(tail),
                "installed": installed,
                "versions": versions,
                "seconds": round(time.perf_counter() - started, 1),
                "path": str(venv_dir),
            }
        after = _probe(str(python_exe), dist_names, ())
        versions = dict(after.get("versions", {}))  # type: ignore[arg-type]

    still_missing = [
        pin for pin in pins if not _satisfies(versions.get(_split_pin(pin)[0]), _split_pin(pin)[1])
    ]
    return {
        "name": name,
        "status": "新建" if created else "复用",
        "ok": not still_missing,
        "detail": ("仍不满足：" + ", ".join(still_missing)) if still_missing else "",
        "installed": installed,
        "versions": versions,
        "seconds": round(time.perf_counter() - started, 1),
        "path": str(venv_dir),
    }


def _bootstrap_system_repo(spec: Dict[str, object]) -> Dict[str, object]:
    """校验吃系统解释器的仓（只检查，绝不改动全局环境）。"""
    name = str(spec["name"])
    repo = resolve_repo(name)
    started = time.perf_counter()
    python_exe = _torch_interpreter()
    if python_exe is None:
        return {
            "name": name,
            "status": "缺解释器",
            "ok": False,
            "detail": "找不到含 torch + sentence_transformers 的 Python（可设 RAGEVAL_PYTHON 指定）",
            "installed": [],
            "versions": {},
            "seconds": round(time.perf_counter() - started, 1),
            "path": python_exe or "(无)",
        }
    modules = tuple(spec["required_modules"])  # type: ignore[arg-type]
    dist_names = tuple(_split_pin(pin)[0] for pin in _read_requirement_pins(repo, str(spec["source"])))
    probed = _probe(python_exe, tuple(dict.fromkeys(dist_names)), modules)
    versions = dict(probed.get("versions", {}))  # type: ignore[arg-type]
    found = dict(probed.get("modules", {}))  # type: ignore[arg-type]
    missing_modules = [name_ for name_ in modules if not found.get(name_)]
    return {
        "name": name,
        "status": "复用系统解释器",
        "ok": not missing_modules,
        "detail": ("缺模块：" + ", ".join(missing_modules)) if missing_modules else "",
        "installed": [],
        "versions": versions,
        "seconds": round(time.perf_counter() - started, 1),
        "path": python_exe,
    }


def cmd_bootstrap() -> int:
    """--bootstrap：把五仓依赖装好（per-repo venv），并报告模型缓存情况。"""
    print("=" * 96)
    print("依赖引导（--bootstrap）")
    print("=" * 96)
    print("说明：mcp 主版本冲突（hello-agent 要 mcp<2 / rag-mcp-server 要 mcp==2.3.0）")
    print("      → 这两个仓各自一个 .venv；重包（torch/transformers 等）继承机器级安装。")
    print("      全程不修改全局 Python 环境，也不自动下载 HF 模型。")
    print()

    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    for name, pins, system_site in VENV_REPOS:
        rows.append(_bootstrap_venv_repo(name, pins, system_site))
    for spec in SYSTEM_REPOS:
        rows.append(_bootstrap_system_repo(spec))
    total = round(time.perf_counter() - started, 1)

    print(_pad("仓库", 20) + _pad("解释器 / venv 路径", 46) + _pad("状态", 16) + _pad("耗时", 8) + "本轮安装")
    print("-" * 140)
    for row in rows:
        installed = row["installed"]
        if installed:
            shown = ", ".join(installed[:6]) + (f" …等 {len(installed)} 个" if len(installed) > 6 else "")
        else:
            shown = "（无新增，全部已满足）" if row["ok"] else "-"
        print(
            _pad(str(row["name"]), 20)
            + _pad(str(row["path"]), 46)
            + _pad(str(row["status"]), 16)
            + _pad(f"{row['seconds']}s", 8)
            + shown
        )
        if row["detail"]:
            print("    ! " + str(row["detail"]).replace("\n", "\n    ! "))
    print("-" * 140)
    print(f"依赖安装总耗时：{total}s")

    print()
    print("关键版本核对（mcp 主版本冲突是分仓装的根因）：")
    for row in rows:
        versions = row["versions"]
        if not versions:
            continue
        keys = [k for k in ("mcp", "langchain", "langgraph", "sentence-transformers", "transformers", "torch") if k in versions]
        if keys:
            detail = ", ".join(f"{k}={versions[k]}" for k in keys)
            print(f"    {str(row['name']):20s} {detail}")
    print("    注：hello-agent / rag-mcp-server 的 torch、transformers、sentence-transformers、numpy")
    print("        继承机器级安装（CUDA 版 torch 2.14.0+cu132），不是 requirements.txt 里钉的版本，")
    print("        这是有意为之：避免每个仓重下 GB 级 wheel，也保住 GPU 可用性。")

    print()
    print("HF 模型缓存检测（机器级前置条件，bootstrap 不自动下载）：")
    cache_root = _hf_cache_root()
    print(f"    缓存根目录：{cache_root}")
    missing_models: List[Tuple[str, str, str]] = []
    for repo_id, size, impact in HF_MODEL_CACHE:
        cached = _hf_model_cached(repo_id)
        mark = "已缓存" if cached else "缺失"
        print(f"    [{mark}] {repo_id}  ({size})")
        print(f"              影响：{impact}")
        if not cached:
            missing_models.append((repo_id, size, impact))
    if missing_models:
        print()
        print("    缺失模型请手动下载（每条命令可直接复制执行；3.5GB 不会被自动拉取）：")
        for repo_id, _size, _impact in missing_models:
            print(f"        hf download {repo_id}")
        print("    注意：新版 huggingface_hub 已废弃 huggingface-cli（实测退出码 1），上面的 hf download 才能用；")
        print("          等价的旧写法是 `huggingface-cli download <repo>`，但它现在会直接报错。")
        print("    没有 BAAI/bge-small-zh-v1.5 缓存 → 第 5 步（eval-hybrid）必失败。")
    else:
        print("    三个模型都已缓存，模型侧不会阻塞第 5 步。")

    failed = [row for row in rows if not row["ok"]]
    print()
    if failed:
        print("引导未完全成功：" + ", ".join(str(row["name"]) for row in failed))
        return 1
    print("引导完成。接下来运行：python demo_all.py")
    return 0


def _preflight() -> List[str]:
    """跑 9 步之前先看环境齐不齐；缺什么直接说清楚，别让用户等到第一步才炸。"""
    problems: List[str] = []
    for name, pins, _system_site in VENV_REPOS:
        repo = resolve_repo(name)
        if not repo.is_dir():
            problems.append(f"缺 {name} 仓库目录：{ROOT / name}（或 {ROOT.parent / name}）")
            continue
        python_exe = _venv_python(repo)
        if python_exe is None:
            problems.append(f"缺 {name} 的 .venv，先跑 python demo_all.py --bootstrap")
            continue
        dist_names = tuple(_split_pin(pin)[0] for pin in pins)
        probed = _probe(str(python_exe), dist_names, ())
        versions = dict(probed.get("versions", {}))  # type: ignore[arg-type]
        unsatisfied = [
            pin for pin in pins if not _satisfies(versions.get(_split_pin(pin)[0]), _split_pin(pin)[1])
        ]
        if unsatisfied:
            problems.append(
                f"{name} 的 .venv 缺包（{', '.join(unsatisfied[:4])}），先跑 python demo_all.py --bootstrap"
            )
    for spec in SYSTEM_REPOS:
        name = str(spec["name"])
        python_exe = _torch_interpreter()
        if python_exe is None:
            problems.append("找不到含 torch + sentence_transformers 的 Python（可设 RAGEVAL_PYTHON 指定）")
            break
        modules = tuple(spec["required_modules"])  # type: ignore[arg-type]
        probed = _probe(python_exe, (), modules)
        found = dict(probed.get("modules", {}))  # type: ignore[arg-type]
        missing_modules = [item for item in modules if not found.get(item)]
        if missing_modules:
            problems.append(f"{name} 缺模块（{', '.join(missing_modules)}），先跑 python demo_all.py --bootstrap")
    return problems



@dataclass
class Step:
    id: str
    name: str
    cmd: List[str]
    cwd: Path
    env: Dict[str, str] = field(default_factory=dict)
    timeout: int = 420
    metric_pattern: str = ""
    metric_label: str = ""
    # 本步可能写出的产物（相对 cwd 的 glob）。门面只做验证，默认跑完即清理，
    # 避免"实验脚本按当天日期落盘"导致跨天重跑弄脏仓库。
    artifacts: List[str] = field(default_factory=list)
    # 可选步骤：要 GPU / 耗时明显，默认不跑，需 --only <id> 或 --all 显式开启。
    optional: bool = False


def build_steps() -> List[Step]:
    agent = resolve_repo("hello-agent")
    rag_eval = resolve_repo("rag-eval-platform")
    rag_mcp = resolve_repo("rag-mcp-server")
    jobfit = resolve_repo("job-fit-assistant")
    tiny_llm = resolve_repo("tiny-llm-from-scratch")

    agent_py = _python_in(agent)
    mcp_py = _python_in(rag_mcp)
    # 两个新仓统一用系统 Python 3.12（tiny-llm 没有自己的 .venv；job-fit 的 .venv 是
    # --system-site-packages 建的，与系统解释器等价，所以这里不分开取，保持一套口径）。
    # tiny-llm 的单测要 import torch，因此直接复用已有的 _python_with_torch()。
    torch_py = _python_with_torch()

    steps: List[Step] = [
        Step(
            id="agent-tests",
            name="hello-agent 全量测试（离线）",
            cmd=[agent_py, "-m", "pytest", "tests", "-q"],
            cwd=agent,
            metric_pattern=r"(\d+) passed",
            metric_label="passed",
        ),
        Step(
            id="agent-memory",
            name="hello-agent 跨任务记忆演示",
            cmd=[agent_py, "scripts/demo_memory.py", "--reset"],
            cwd=agent,
            metric_pattern=r"\[OK\]",
            metric_label="召回演示",
        ),
        Step(
            id="agent-resume",
            name="hello-agent 断点续跑演示",
            cmd=[agent_py, "scripts/demo_resume.py"],
            cwd=agent,
            metric_pattern=r"重放步数 = (\d+)",
            metric_label="重放步数",
        ),
        Step(
            id="agent-usage",
            name="hello-agent 用量与成本汇总",
            cmd=[agent_py, "scripts/usage_report.py"],
            cwd=agent,
            metric_pattern=r"累计：(\d+) 次运行，(\d+) tokens，估算成本 \$([\d.]+)",
            metric_label="累计用量",
        ),
        Step(
            id="eval-hybrid",
            name="rag-eval 混合检索对照实验",
            cmd=(
                [torch_py, "experiments/run_hybrid_comparison.py"]
                if torch_py
                else ["__MISSING_TORCH__"]
            ),
            cwd=rag_eval,
            env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            timeout=600,
            metric_pattern=(
                r"纯向量基线：平均 Recall@3=([\d.]+).*?"
                r"混合检索（BM25\+向量 RRF）：平均 Recall@3=([\d.]+)"
            ),
            metric_label="噪声语料 Recall@3",
            artifacts=[
                "experiments/results/hybrid_comparison_*.json",
                "experiments/results/hybrid_comparison_*.md",
            ],
        ),
        Step(
            id="eval-generation",
            name="rag-eval 生成层忠实度评测（离线两臂：规则判据 + 本地 1.5B）",
            cmd=(
                [
                    torch_py,
                    "experiments/run_generation_eval.py",
                    "--no-api",
                    "--out",
                    str(_scratch_dir() / "generation_eval_offline.json"),
                ]
                if torch_py
                else ["__MISSING_TORCH__"]
            ),
            cwd=rag_eval,
            env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            timeout=900,
            metric_pattern=r"样本 (\d+) 条",
            metric_label="条样本",
            artifacts=[
                "experiments/results/generation_eval_*.json",
                "experiments/results/generation_eval_*.md",
            ],
            optional=True,
        ),
        Step(
            id="mcp-tests",
            name="rag-mcp 全量测试（含 HTTP 端到端）",
            cmd=[mcp_py, "-m", "pytest", "tests", "-q"],
            cwd=rag_mcp,
            env={"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
            timeout=420,
            metric_pattern=r"(\d+) passed",
            metric_label="passed",
        ),
        Step(
            id="mcp-security",
            name="rag-mcp HTTP 安全演示（鉴权/限流/注入防护）",
            cmd=[mcp_py, "scripts/demo_http_security.py", "--json"],
            cwd=rag_mcp,
            env={"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
            timeout=300,
            metric_pattern=r"演示结论：(\d+/\d+) 项符合预期",
            metric_label="安全项通过",
        ),
        Step(
            id="jobfit-tests",
            name="job-fit-assistant 全量测试（求职助手，CPU 离线）",
            cmd=(
                # 该仓 pyproject.toml 自带 addopts="-q"，再叠一个 -q 会变成 -qq 屏蔽汇总行，
                # 这里清掉 addopts 让 "N passed" 汇总行照常输出（指标才提取得到）。
                [torch_py, "-m", "pytest", "-q", "--override-ini=addopts="]
                if torch_py
                else ["__MISSING_TORCH__"]
            ),
            cwd=jobfit,
            timeout=420,
            metric_pattern=r"(\d+) passed",
            metric_label="passed",
        ),
        Step(
            id="tiny-llm-tests",
            name="tiny-llm-from-scratch 全量测试（从零训练项目，CPU 离线）",
            cmd=(
                [torch_py, "-m", "pytest", "tests", "-q"]
                if torch_py
                else ["__MISSING_TORCH__"]
            ),
            cwd=tiny_llm,
            timeout=420,
            metric_pattern=r"(\d+) passed",
            metric_label="passed",
        ),
    ]
    return steps


def _extract(step: Step, output: str) -> str:
    if not step.metric_pattern:
        return "-"
    try:
        matches = re.findall(step.metric_pattern, output, re.S)
    except re.error:
        return "-"
    if not matches:
        return "-"
    last = matches[-1]
    if isinstance(last, tuple):
        value = " -> ".join(last)
        return f"{step.metric_label} {value}".strip() if step.metric_label else value

    value = str(last)
    if not step.metric_label:
        return value
    # 数字指标放前面（"60 passed"），文本指标放后面（"召回演示 [OK]"）
    if value.isdigit():
        return f"{value} {step.metric_label}"
    return f"{step.metric_label} {value}"


def _snapshot(step: Step) -> set:
    """记录本步产物当前已存在的文件（跑前/跑后各取一次，差集才是"本步新增"）。"""
    found = set()
    for pattern in step.artifacts:
        found.update(step.cwd.glob(pattern))
    return found


def _cleanup(step: Step, before: set) -> int:
    """删除本步**新增**的产物；跑前就存在的文件一律不动。返回删除个数。"""
    removed = 0
    for path in sorted(_snapshot(step) - before):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def run_step(step: Step, keep_artifacts: bool = False) -> Tuple[bool, str, str]:
    if step.cmd == ["__MISSING_TORCH__"]:
        return (
            False,
            "未找到含 torch 的 Python（设置 RAGEVAL_PYTHON 指向 Python 3.12）",
            "-",
        )

    before = _snapshot(step)
    env = os.environ.copy()
    env.update(step.env)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    started = time.perf_counter()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(3):
            print(f"       ... 已用 {time.perf_counter() - started:.0f}s", flush=True)

    beat = threading.Thread(target=heartbeat, daemon=True)
    beat.start()
    try:
        proc = subprocess.run(
            step.cmd,
            cwd=str(step.cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=step.timeout,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        elapsed = time.perf_counter() - started
        note = ""
        if step.artifacts and not keep_artifacts:
            removed = _cleanup(step, before)
            if removed:
                note = f"（已清理本步新产物 {removed} 个；要留存请加 --keep-artifacts）"
        if proc.returncode != 0:
            tail = "\n".join(output.strip().splitlines()[-25:])
            return False, f"退出码 {proc.returncode}，耗时 {elapsed:.1f}s{note}\n{tail}", "-"
        return True, f"完成，耗时 {elapsed:.1f}s{note}", _extract(step, output)
    except subprocess.TimeoutExpired:
        return False, f"超时（>{step.timeout}s）", "-"
    finally:
        stop.set()
        beat.join(timeout=1)


def main() -> int:
    _chcp_utf8()
    parser = argparse.ArgumentParser(description="一键跑通 LLM 作品集五仓库")
    parser.add_argument("--only", default="", help="只跑指定步骤（逗号分隔）")
    parser.add_argument("--skip-tests", action="store_true", help="跳过 pytest 步骤")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="保留各步新产生的实验产物（默认跑完自动清理，避免弄脏仓库）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="连可选步骤一起跑（可选步骤要 GPU、耗时较长，默认跳过）",
    )
    parser.add_argument("--list", action="store_true", help="列出步骤 id 后退出")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="先把五仓依赖装好（每仓一个 .venv）并检查模型缓存，然后退出",
    )
    args = parser.parse_args()

    if args.bootstrap:
        return cmd_bootstrap()

    steps = build_steps()
    if args.list:
        for step in steps:
            tag = "   （可选，默认跳过）" if step.optional else ""
            print(f"{step.id:16s} {step.name}{tag}")
        return 0

    skipped_optional: List[str] = []
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        unknown = wanted - {step.id for step in steps}
        if unknown:
            print(f"未知步骤：{sorted(unknown)}；用 --list 查看可用 id", file=sys.stderr)
            return 2
        steps = [step for step in steps if step.id in wanted]
    elif not args.all:
        skipped_optional = [step.id for step in steps if step.optional]
        steps = [step for step in steps if not step.optional]
    if args.skip_tests:
        steps = [step for step in steps if not step.id.endswith("-tests")]

    if not steps:
        print("没有可执行的步骤。", file=sys.stderr)
        return 2

    # 环境自检：不装任何东西，只看"能不能跑"，缺什么当场说清楚，
    # 而不是跑到第一步才抛 ModuleNotFoundError。
    problems = _preflight()
    if problems:
        print("环境未就绪，先补齐再跑：", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        print("\n提示：python demo_all.py --bootstrap  # 自动建 venv 装依赖", file=sys.stderr)
        return 2

    print("=" * 78)
    print("LLM 作品集 · 一键验证")
    print("=" * 78)
    if skipped_optional:
        print(f"（默认跳过可选步骤：{', '.join(skipped_optional)}；要跑用 --only <id> 或 --all）")

    results = []
    total = len(steps)
    for index, step in enumerate(steps, 1):
        print(f"\n{bar(index - 1, total)}  {step.name}")
        ok, detail, metric = run_step(step, keep_artifacts=args.keep_artifacts)
        results.append((step, ok, detail, metric))
        print(f"{bar(index, total)}  {'[OK]' if ok else '[FAIL]'} {step.name} | {detail.splitlines()[0]}")
        if not ok:
            print(detail)
            break

    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"{_pad('步骤', 40)} {_pad('状态', 6)} {_pad('指标', 44)}")
    print("-" * 100)
    for step, ok, _, metric in results:
        print(
            f"{_pad(step.name, 40)} {_pad('OK' if ok else 'FAIL', 6)} {_pad(metric, 44)}"
        )

    failed = [item for item in results if not item[1]]
    print("-" * 100)
    print(f"共 {len(results)}/{total} 步完成，失败 {len(failed)} 步。")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
