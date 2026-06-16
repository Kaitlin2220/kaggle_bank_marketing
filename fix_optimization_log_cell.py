from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path("bank-marketing-beginner-tuned.ipynb")

fixed_source = r'''
optimization_log_path = PROJECT_DIR / "bank-marketing-optimization-log.md"

time_metrics = {k: v for k, v in time_result.items() if k != "confusion_matrix"}
lift_table_text = lift_table.round(4).to_string(index=False)

log_text = (
    "# Bank Marketing 优化日志\n\n"
    "## 记录日期\n\n"
    "2026-06-04\n\n"
    "## 本次新增优化\n\n"
    "1. 新增时间切分验证：使用前 60% 样本训练，中间 20% 样本选择阈值，最后 20% 样本模拟未来测试。\n"
    "2. 新增 Top 客户分层收益分析：评估 Top 10%、Top 20%、Top 30% 客户名单的实际认购率、提升倍数和覆盖的 yes 客户比例。\n"
    "3. 新增图表输出：\n"
    "   - `figures_beginner/time_split_yes_rate.png`\n"
    "   - `figures_beginner/top_customer_lift_time_holdout.png`\n\n"
    "## 随机切分模型结果\n\n"
    "- 推荐模型：LightGBM Tuned\n"
    f"- 测试集 PR-AUC：{tuned_result['test_pr_auc']:.4f}\n"
    f"- 测试集 ROC-AUC：{tuned_result['test_roc_auc']:.4f}\n"
    f"- yes precision：{tuned_result['test_precision_yes']:.4f}\n"
    f"- yes recall：{tuned_result['test_recall_yes']:.4f}\n"
    f"- yes F1：{tuned_result['test_f1_yes']:.4f}\n"
    f"- 混淆矩阵：{tuned_result['confusion_matrix'].tolist()}\n\n"
    "## 时间切分验证结果\n\n"
    f"- 时间测试集 PR-AUC：{time_metrics['test_pr_auc']:.4f}\n"
    f"- 时间测试集 ROC-AUC：{time_metrics['test_roc_auc']:.4f}\n"
    f"- yes precision：{time_metrics['test_precision_yes']:.4f}\n"
    f"- yes recall：{time_metrics['test_recall_yes']:.4f}\n"
    f"- yes F1：{time_metrics['test_f1_yes']:.4f}\n"
    f"- 混淆矩阵：{time_result['confusion_matrix'].tolist()}\n\n"
    "## Top 客户分层结果\n\n"
    f"时间测试集整体 yes 率：{time_baseline_rate:.2%}\n\n"
    f"{lift_table_text}\n\n"
    "## 解读\n\n"
    "- 时间切分比随机切分更接近真实上线场景，因为它模拟“用过去预测未来”。\n"
    "- Top 客户分层适合转化为营销策略：如果营销资源有限，可以优先联系模型预测概率最高的一部分客户。\n"
    "- 这个数据集后 20% 样本的 yes 率明显更高，说明时间分布发生了变化；后续可以继续做时间漂移分析和按月份建模。\n"
)

optimization_log_path.write_text(log_text, encoding="utf-8")
print(f"优化日志已保存: {optimization_log_path}")
'''


nb = nbf.read(NOTEBOOK_PATH, as_version=4)
replaced = False

for cell in nb.cells:
    if cell.get("cell_type") == "code" and "optimization_log_path" in cell.get("source", ""):
        cell["source"] = dedent(fixed_source).strip() + "\n"
        cell["outputs"] = []
        cell["execution_count"] = None
        replaced = True

if not replaced:
    raise RuntimeError("Did not find optimization log cell to fix.")

nbf.write(nb, NOTEBOOK_PATH)
print(f"Fixed log cell in {NOTEBOOK_PATH}")
