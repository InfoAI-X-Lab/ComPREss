# antiSMASH 基因组流程

## 项目概览
本项目包含两段流程：

1. `data_preprocess`：将原始 `.fna.gz` 数据清洗、去重、统计长度分布、按用户输入阈值过滤、按 GCF 归并并筛选最终基因组。
2. `antismash_prediction`：对预处理后的基因组批量运行 antiSMASH，并统计 BGC 结果。

推荐预处理主入口：

- `data_preprocess/run_pipeline.py`
- `data_preprocess/config.json`

## 目录结构
```text
antismash/
├─ data_preprocess/
│  ├─ run_pipeline.py
│  ├─ config.json
│  ├─ 01fna_filter.py
│  ├─ deduplicate_fna.py
│  ├─ 02stat_length.py
│  ├─ 03length_filter.py
│  ├─ 04nz_to_gcf.py
│  ├─ 05final_gcf.py
│  └─ count_fna_quantity.py
└─ antismash_prediction/
   └─ pre_antismash.py
```

## 预处理流程
运行：

```bash
cd data_preprocess
python run_pipeline.py --config config.json
```

总控脚本会按以下顺序逐步询问是否执行：

1. `fna_filter` -> `01fna_filter.py`
2. `deduplicate` -> `deduplicate_fna.py`
3. `stat_length` -> `02stat_length.py`
4. `length_filter` -> `03length_filter.py`
5. `nz_to_gcf` -> `04nz_to_gcf.py`
6. `final_gcf` -> `05final_gcf.py`
7. `count_fna` -> `count_fna_quantity.py`

说明：
`deduplicate` 放在 `01fna_filter` 后面，因为去重脚本处理的是 `.fna`，而原始输入通常是 `.fna.gz`。

对于第 02 步和第 03 步：
应先运行 `02stat_length.py` 查看长度统计结果，再在 `03length_filter.py` 开始执行时手动输入阈值。阈值不再预先写在 `config.json` 中。

交互规则：

- `yes` / `y`：执行该步骤
- `no` / `n`：跳过该步骤
- 非法输入：重复询问直到合法

## config.json 说明
配置文件包含三部分：

- `pipeline`
- `execution_order`
- `steps`

关键项：

- `pipeline.show_step_summary`：每一步后输出执行摘要
- `pipeline.stop_on_error`
  - `true`：某一步失败后立即终止
  - `false`：记录失败并继续执行后续步骤

每个步骤中的 `enabled` 用于展示默认状态；是否真正执行仍由当次交互输入决定。
`length_filter` 步骤的阈值现在在运行时输入，因此 `config.json` 中不再需要 `threshold` 字段。

## 预处理脚本说明
- `01fna_filter.py`：使用黑名单 CSV 过滤 `.fna.gz`，输出 `.fna`
- `deduplicate_fna.py`：按序列内容 MD5 对 `.fna` 文件严格去重
- `02stat_length.py`：统计 `.fna/.fna.gz` 长度分布，供用户判断过滤阈值
- `03length_filter.py`：运行时手动输入阈值，再按该阈值过滤短序列
- `04nz_to_gcf.py`：根据 `.ass/.ass.gz` 将 NZ 编号序列归并为 GCF 文件
- `05final_gcf.py`：同一物种多个 GCF 时，按每个 GCF 文件内所有序列总长度保留最大者
- `count_fna_quantity.py`：统计目标目录或单文件中的序列条数

## pre_antismash.py 说明
路径：`antismash_prediction/pre_antismash.py`

该脚本核心类为 `AntismashRunner`，主要配置项位于 `__init__`：

- `self.genome_path`：输入 `.fna` 目录，通常是预处理最终输出目录
- `self.antismash_output_path`：antiSMASH 输出目录
- `self.summary_file`：统计结果文件名，默认 `antismash_summary.tsv`
- `self.workers`：并行进程数，默认 `10`
- `self.log_interval`：进度输出间隔，默认每 `50` 个文件

支持两种模式：

1. 预测模式

```python
runner.run_prediction()
# runner.run_gene_count()
```

说明：并行调用 `antismash` 命令；若结果目录中已存在 `index.html`，则跳过该样本，支持断点续跑。

2. 汇总模式

```python
# runner.run_prediction()
runner.run_gene_count()
```

说明：扫描结果目录下的 JSON，统计每个基因组的 cluster type 数量，并输出 TSV。

运行：

```bash
cd antismash_prediction
python pre_antismash.py
```

## 依赖
- Python 3.9+（建议）
- `biopython`
- `pandas`
- antiSMASH

安装示例：

```bash
pip install biopython pandas
```
