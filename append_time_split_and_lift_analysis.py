from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "bank-marketing-beginner-tuned.ipynb"
SUMMARY_PATH = PROJECT_DIR / "bank-marketing-tuning-summary.md"


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


append_cells = [
    md(
        """
        ## 17. 时间切分验证：用过去预测未来

        前面的训练/验证/测试是随机切分，适合学习模型流程。真实营销更接近这个问题：

        > 用过去的客户和营销记录训练模型，预测未来一段时间哪些客户更可能认购。

        这个数据集原始文件已经按时间排序，但没有精确日期列。因此这里用行顺序近似时间顺序：

        - 前 60%：训练集
        - 中间 20%：验证集，用来选择阈值
        - 最后 20%：测试集，模拟未来数据

        这个检查比随机切分更严格，也更接近业务上线后的情况。
        """
    ),
    code(
        """
        X_time, y_time = build_model_data(df, include_macro=True, include_duration=False)

        n_rows = len(X_time)
        train_end = int(n_rows * 0.60)
        val_end = int(n_rows * 0.80)

        X_time_train = X_time.iloc[:train_end]
        y_time_train = y_time.iloc[:train_end]
        X_time_val = X_time.iloc[train_end:val_end]
        y_time_val = y_time.iloc[train_end:val_end]
        X_time_test = X_time.iloc[val_end:]
        y_time_test = y_time.iloc[val_end:]

        time_split_rates = pd.DataFrame({
            "split": ["train_first_60%", "validation_next_20%", "test_last_20%"],
            "rows": [len(y_time_train), len(y_time_val), len(y_time_test)],
            "yes_rate": [y_time_train.mean(), y_time_val.mean(), y_time_test.mean()],
        })
        display(time_split_rates.round(4))

        plt.figure(figsize=(7, 4))
        sns.barplot(data=time_split_rates, x="split", y="yes_rate", color="#4C78A8")
        plt.title("Yes Rate by Time-Based Split")
        plt.xlabel("")
        plt.ylabel("Actual yes rate")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "time_split_yes_rate.png", dpi=150)
        plt.show()
        """
    ),
    code(
        """
        time_model = lgb.LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            **best_params,
        )

        time_model.fit(
            X_time_train,
            y_time_train,
            eval_set=[(X_time_val, y_time_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )

        time_val_prob = time_model.predict_proba(X_time_val)[:, 1]
        time_threshold_info = find_best_f1_threshold(y_time_val, time_val_prob)
        time_threshold = time_threshold_info["threshold"]

        time_test_prob = time_model.predict_proba(X_time_test)[:, 1]
        time_test_pred = (time_test_prob >= time_threshold).astype(int)

        time_result = {
            "model": "LightGBM Tuned - Time Split",
            "threshold_from_time_val": time_threshold,
            "test_pr_auc": average_precision_score(y_time_test, time_test_prob),
            "test_roc_auc": roc_auc_score(y_time_test, time_test_prob),
            "test_precision_yes": precision_score(y_time_test, time_test_pred, zero_division=0),
            "test_recall_yes": recall_score(y_time_test, time_test_pred, zero_division=0),
            "test_f1_yes": f1_score(y_time_test, time_test_pred, zero_division=0),
            "test_pred_yes_rate": time_test_pred.mean(),
            "confusion_matrix": confusion_matrix(y_time_test, time_test_pred),
        }

        display(pd.DataFrame([{k: v for k, v in time_result.items() if k != "confusion_matrix"}]).round(4))

        print("时间切分测试集分类报告:")
        print(classification_report(y_time_test, time_test_pred, target_names=["no", "yes"]))
        print("混淆矩阵 [[TN, FP], [FN, TP]]:")
        print(time_result["confusion_matrix"])
        """
    ),
    md(
        """
        ## 18. Top 客户分层收益分析

        在营销业务里，模型最常见的用法不是直接输出 `yes/no`，而是给客户排序：

        - Top 10%：最优先联系
        - Top 20%：营销资源稍多时联系
        - Top 30%：更宽松的联系名单

        我们用时间切分测试集模拟“未来客户名单”，看模型排在前面的客户是否真的有更高认购率。
        """
    ),
    code(
        """
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


        lift_table, time_baseline_rate, time_total_yes = top_percent_lift_table(y_time_test, time_test_prob)

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
        """
    ),
    md(
        """
        ## 19. 追加优化结论

        随机切分告诉我们模型在“同分布抽样”下的表现；时间切分告诉我们模型面对未来数据时是否稳定。  
        Top 客户分层则把模型结果翻译成营销名单策略，比单独看 AUC 更接近业务决策。
        """
    ),
    code(
        """
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
        """
    ),
]


nb = nbf.read(NOTEBOOK_PATH, as_version=4)

start_marker = "## 17. 时间切分验证"
filtered_cells = []
skip = False
for cell in nb.cells:
    source = cell.get("source", "")
    if cell.get("cell_type") == "markdown" and start_marker in source:
        skip = True
    if not skip:
        filtered_cells.append(cell)

nb.cells = filtered_cells + append_cells
nbf.write(nb, NOTEBOOK_PATH)

summary_append = """

## 追加优化：时间切分与客户分层

本项目进一步加入两项更贴近业务的优化：

- 时间切分验证：用前 60% 样本训练，中间 20% 选择阈值，最后 20% 模拟未来测试。
- Top 客户分层收益分析：比较 Top 10%、Top 20%、Top 30% 高概率客户名单的实际认购率和提升倍数。

执行 notebook 后会自动生成：

- `bank-marketing-optimization-log.md`
- `figures_beginner/time_split_yes_rate.png`
- `figures_beginner/top_customer_lift_time_holdout.png`
"""

if SUMMARY_PATH.exists():
    summary_text = SUMMARY_PATH.read_text(encoding="utf-8")
    if "## 追加优化：时间切分与客户分层" not in summary_text:
        SUMMARY_PATH.write_text(summary_text.rstrip() + "\n" + summary_append, encoding="utf-8")

print(f"Updated {NOTEBOOK_PATH.name}")
print(f"Updated {SUMMARY_PATH.name}")
