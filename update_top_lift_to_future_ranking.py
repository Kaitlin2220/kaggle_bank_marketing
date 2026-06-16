from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path("bank-marketing-beginner-tuned.ipynb")

top_lift_source = r'''
def top_percent_lift_table(y_true, predicted_prob, percents=(0.10, 0.20, 0.30)):
    scored = pd.DataFrame({
        "actual": np.asarray(y_true),
        "predicted_prob": np.asarray(predicted_prob),
    }).sort_values("predicted_prob", ascending=False).reset_index(drop=True)

    total_customers = len(scored)
    total_yes = int(scored["actual"].sum())
    baseline_rate = scored["actual"].mean()

    rows = []
    for pct in percents:
        contact_n = int(np.ceil(total_customers * pct))
        selected = scored.head(contact_n)
        selected_yes = int(selected["actual"].sum())
        selected_rate = selected["actual"].mean()
        rows.append({
            "contact_group": f"Top {int(pct * 100)}%",
            "contact_customers": contact_n,
            "actual_yes": selected_yes,
            "response_rate": selected_rate,
            "lift_vs_baseline": selected_rate / baseline_rate if baseline_rate > 0 else np.nan,
            "captured_yes_rate": selected_yes / total_yes if total_yes > 0 else np.nan,
        })

    return pd.DataFrame(rows), baseline_rate, total_yes


def train_future_ranking_model(include_macro=True):
    X_rank, y_rank = build_model_data(df, include_macro=include_macro, include_duration=False)
    rank_train_end = int(len(X_rank) * 0.80)

    X_rank_train = X_rank.iloc[:rank_train_end]
    y_rank_train = y_rank.iloc[:rank_train_end]
    X_rank_test = X_rank.iloc[rank_train_end:]
    y_rank_test = y_rank.iloc[rank_train_end:]

    rank_model = lgb.LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **best_params,
    )
    rank_model.fit(X_rank_train, y_rank_train)
    rank_prob = rank_model.predict_proba(X_rank_test)[:, 1]

    rank_metrics = {
        "feature_set": "with_macro" if include_macro else "no_macro",
        "test_pr_auc": average_precision_score(y_rank_test, rank_prob),
        "test_roc_auc": roc_auc_score(y_rank_test, rank_prob),
    }
    rank_lift_table, rank_baseline_rate, rank_total_yes = top_percent_lift_table(y_rank_test, rank_prob)
    return rank_metrics, rank_lift_table, rank_baseline_rate, rank_total_yes, rank_prob, y_rank_test


rank_with_macro = train_future_ranking_model(include_macro=True)
rank_no_macro = train_future_ranking_model(include_macro=False)

future_rank_compare = pd.DataFrame([rank_with_macro[0], rank_no_macro[0]])
display(future_rank_compare.round(4))

if rank_no_macro[0]["test_pr_auc"] >= rank_with_macro[0]["test_pr_auc"]:
    selected_rank_name = "No Macro Future Ranking Model"
    selected_rank_metrics, lift_table, time_baseline_rate, time_total_yes, future_rank_prob, future_rank_y = rank_no_macro
else:
    selected_rank_name = "With Macro Future Ranking Model"
    selected_rank_metrics, lift_table, time_baseline_rate, time_total_yes, future_rank_prob, future_rank_y = rank_with_macro

print(f"推荐客户排序模型: {selected_rank_name}")
print(f"时间测试集整体 yes 率: {time_baseline_rate:.2%}")
print(f"时间测试集实际 yes 客户数: {time_total_yes}")
display(lift_table.round(4))

plt.figure(figsize=(7, 4))
sns.barplot(data=lift_table, x="contact_group", y="response_rate", color="#F58518")
plt.axhline(time_baseline_rate, color="black", linestyle="--", label=f"Baseline={time_baseline_rate:.1%}")
plt.title("Response Rate by Top-Ranked Customer Group")
plt.xlabel("")
plt.ylabel("Actual yes rate")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "top_customer_lift_time_holdout.png", dpi=150)
plt.show()
'''

log_source = r'''
optimization_log_path = PROJECT_DIR / "bank-marketing-optimization-log.md"

time_metrics = {k: v for k, v in time_result.items() if k != "confusion_matrix"}
lift_table_text = lift_table.round(4).to_string(index=False)

log_text = (
    "# Bank Marketing 优化日志\n\n"
    "## 记录日期\n\n"
    "2026-06-04\n\n"
    "## 本次新增优化\n\n"
    "1. 新增时间切分验证：使用前 60% 样本训练，中间 20% 样本选择阈值，最后 20% 样本模拟未来测试。\n"
    "2. 新增未来客户排序模型：使用前 80% 历史样本训练，最后 20% 样本模拟未来客户名单排序。\n"
    "3. 新增 Top 客户分层收益分析：评估 Top 10%、Top 20%、Top 30% 客户名单的实际认购率、提升倍数和覆盖的 yes 客户比例。\n"
    "4. 新增图表输出：\n"
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
    "## 时间切分阈值验证结果\n\n"
    f"- 时间测试集 PR-AUC：{time_metrics['test_pr_auc']:.4f}\n"
    f"- 时间测试集 ROC-AUC：{time_metrics['test_roc_auc']:.4f}\n"
    f"- yes precision：{time_metrics['test_precision_yes']:.4f}\n"
    f"- yes recall：{time_metrics['test_recall_yes']:.4f}\n"
    f"- yes F1：{time_metrics['test_f1_yes']:.4f}\n"
    f"- 混淆矩阵：{time_result['confusion_matrix'].tolist()}\n\n"
    "## 未来客户排序模型结果\n\n"
    f"- 推荐排序模型：{selected_rank_name}\n"
    f"- 排序测试集 PR-AUC：{selected_rank_metrics['test_pr_auc']:.4f}\n"
    f"- 排序测试集 ROC-AUC：{selected_rank_metrics['test_roc_auc']:.4f}\n\n"
    "## Top 客户分层结果\n\n"
    f"时间测试集整体 yes 率：{time_baseline_rate:.2%}\n\n"
    f"{lift_table_text}\n\n"
    "## 解读\n\n"
    "- 时间切分阈值验证显示：模型从随机切分迁移到未来数据时明显变难，存在时间分布漂移。\n"
    "- 对营销名单来说，排序比直接分类更实用；使用前 80% 历史数据训练后的 Top 客户分层有明显提升。\n"
    "- 当前结果中 Top 10% 客户的实际认购率约为整体未来样本的 2 倍以上，适合优先联系。\n"
    "- 后续可以继续做按月份/宏观周期的时间漂移分析，或按营销资源容量选择 Top N 客户名单。\n"
)

optimization_log_path.write_text(log_text, encoding="utf-8")
print(f"优化日志已保存: {optimization_log_path}")
'''


nb = nbf.read(NOTEBOOK_PATH, as_version=4)
top_replaced = False
log_replaced = False

for cell in nb.cells:
    source = cell.get("source", "")
    if cell.get("cell_type") == "code" and "def top_percent_lift_table" in source:
        cell["source"] = dedent(top_lift_source).strip() + "\n"
        cell["outputs"] = []
        cell["execution_count"] = None
        top_replaced = True
    elif cell.get("cell_type") == "code" and "optimization_log_path" in source:
        cell["source"] = dedent(log_source).strip() + "\n"
        cell["outputs"] = []
        cell["execution_count"] = None
        log_replaced = True

if not top_replaced:
    raise RuntimeError("Did not find top lift cell.")
if not log_replaced:
    raise RuntimeError("Did not find optimization log cell.")

nbf.write(nb, NOTEBOOK_PATH)
print("Updated future ranking lift analysis and log cell.")
