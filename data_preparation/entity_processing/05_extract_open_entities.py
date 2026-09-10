#!/usr/bin/env python3
# 【文件 003】薄启动入口：把执行交给 open_entity_extraction.main
# 【流程位置】数据准备与分区；所属包：data_preparation/entity_processing
# 【主要函数】包声明及共享定义
# 【依赖文件】data_preparation/entity_processing/open_entity_extraction.py

"""Run the open-vocabulary entity extraction stage."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_preparation.entity_processing.open_entity_extraction import main


if __name__ == "__main__":
    main()
