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


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}

cells = [
    md(
        """
        # Bank Marketing 新手友好调优版

        这个 notebook 的目标是：用银行电话营销历史数据，预测客户是否会认购定期存款。

        你可以把它当成一个完整的数据分析学习路径：

        1. 认识数据和业务问题
        2. 做基础探索分析
        3. 处理适合建模的特征
        4. 训练一个容易解释的基线模型
        5. 调优一个表现更好的 LightGBM 模型
        6. 选择分类阈值并解释结果

        > 重要提醒：`duration` 是通话结束后才知道的信息。如果我们要在打电话之前预测客户是否值得联系，就不能使用它。
        """
    ),
    md(
        """
        ## 1. 导入工具包

        这一格只做准备工作。`pandas` 用来处理表格，`sklearn` 和 `lightgbm` 用来建模，`matplotlib/seaborn` 用来画图。
        """
    ),
    code(
        """
        from pathlib import Path
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import ParameterGrid, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        import lightgbm as lgb

        warnings.filterwarnings("ignore")
        RANDOM_STATE = 42

        PROJECT_DIR = Path.cwd()
        DATA_PATH = PROJECT_DIR / "bank-additional-full.csv"
        FIGURE_DIR = PROJECT_DIR / "figures_beginner"
        FIGURE_DIR.mkdir(exist_ok=True)

        pd.set_option("display.max_columns", 100)
        sns.set_theme(style="whitegrid")
        plt.rcParams["axes.unicode_minus"] = False
        """
    ),
    md(
        """
        ## 2. 读取数据

        这个 CSV 文件使用分号 `;` 分隔，所以读取时要写 `sep=';'`。
        """
    ),
    code(
        """
        df = pd.read_csv(DATA_PATH, sep=";")

        print(f"数据规模: {df.shape[0]:,} 行, {df.shape[1]} 列")
        display(df.head())
        """
    ),
    md(
        """
        ## 3. 先理解目标变量

        我们要预测的是 `y`：

        - `yes`：客户认购了定期存款
        - `no`：客户没有认购

        这个数据集是典型的不平衡分类问题，因为 `yes` 比例比较低。
        """
    ),
    code(
        """
        target_counts = df["y"].value_counts()
        target_rates = df["y"].value_counts(normalize=True).mul(100).round(2)

        target_summary = pd.DataFrame({
            "count": target_counts,
            "rate_%": target_rates,
        })
        display(target_summary)

        ax = target_counts.loc[["no", "yes"]].plot(kind="bar", figsize=(5, 3), color=["#4C78A8", "#F58518"])
        ax.set_title("Target Distribution")
        ax.set_xlabel("Subscribed term deposit?")
        ax.set_ylabel("Number of records")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "target_distribution.png", dpi=150)
        plt.show()
        """
    ),
    md(
        """
        ## 4. 检查缺失信息

        表面上没有空值，但很多分类字段用 `unknown` 表示未知。建模时我们先保留 `unknown`，让模型自己学习它是否有信息量。
        """
    ),
    code(
        """
        unknown_counts = []
        for col in df.columns:
            count = int((df[col].astype(str) == "unknown").sum())
            if count > 0:
                unknown_counts.append({
                    "column": col,
                    "unknown_count": count,
                    "unknown_rate_%": round(count / len(df) * 100, 2),
                })

        display(pd.DataFrame(unknown_counts))
        """
    ),
    md(
        """
        ## 5. 特征工程：把原始数据变成模型能读的形式

        这里有三个关键处理：

        1. 删除 `duration`，避免事后信息泄漏。
        2. 把 `pdays=999` 拆成两个字段：是否曾经联系过、距离上次联系多少天。
        3. 把分类变量转成 0/1 哑变量。

        我们保留宏观经济变量，因为它们在营销时点通常已经可知，而且实验显示它们能提升效果。
        """
    ),
    code(
        """
        def build_model_data(data, include_macro=True, include_duration=False):
            y = (data["y"] == "yes").astype(int)

            drop_cols = ["y"]
            if not include_duration:
                drop_cols.append("duration")
            if not include_macro:
                drop_cols += [
                    "emp.var.rate",
                    "cons.price.idx",
                    "cons.conf.idx",
                    "euribor3m",
                    "nr.employed",
                ]

            X = data.drop(columns=drop_cols).copy()

            X["has_prev_contact"] = (X["pdays"] != 999).astype(int)
            X["pdays_clean"] = X["pdays"].where(X["pdays"] != 999, 0)
            X = X.drop(columns=["pdays"])

            X = pd.get_dummies(X, drop_first=False, dtype=float)
            return X, y


        X, y = build_model_data(df, include_macro=True, include_duration=False)

        print(f"建模特征矩阵: {X.shape[0]:,} 行, {X.shape[1]} 个特征")
        display(X.head())
        """
    ),
    md(
        """
        ## 6. 划分训练集、验证集、测试集

        新手最容易犯的错是：在测试集上反复调参数。这里我们分成三份：

        - 训练集：训练模型
        - 验证集：选择参数和分类阈值
        - 测试集：最后只评估一次
        """
    ),
    code(
        """
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            stratify=y_temp,
            random_state=RANDOM_STATE,
        )

        print(f"训练集: {X_train.shape}")
        print(f"验证集: {X_val.shape}")
        print(f"测试集: {X_test.shape}")
        """
    ),
    md(
        """
        ## 7. 准备评估函数

        因为 `yes` 很少，不能只看 accuracy。我们重点看：

        - ROC-AUC：整体排序能力
        - PR-AUC：少数类 `yes` 的识别质量，更适合不平衡数据
        - precision/recall/F1：选定阈值后的实际分类效果
        """
    ),
    code(
        """
        def find_best_f1_threshold(y_true, predicted_prob):
            precision, recall, thresholds = precision_recall_curve(y_true, predicted_prob)
            f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
            best_idx = int(np.argmax(f1_scores))
            return {
                "threshold": float(thresholds[best_idx]),
                "precision": float(precision[best_idx]),
                "recall": float(recall[best_idx]),
                "f1": float(f1_scores[best_idx]),
            }


        def evaluate_model(name, model, X_val, y_val, X_test, y_test):
            val_prob = model.predict_proba(X_val)[:, 1]
            threshold_info = find_best_f1_threshold(y_val, val_prob)
            threshold = threshold_info["threshold"]

            test_prob = model.predict_proba(X_test)[:, 1]
            test_pred = (test_prob >= threshold).astype(int)

            return {
                "model": name,
                "threshold_from_val": threshold,
                "val_pr_auc": average_precision_score(y_val, val_prob),
                "val_roc_auc": roc_auc_score(y_val, val_prob),
                "test_pr_auc": average_precision_score(y_test, test_prob),
                "test_roc_auc": roc_auc_score(y_test, test_prob),
                "test_precision_yes": precision_score(y_test, test_pred, zero_division=0),
                "test_recall_yes": recall_score(y_test, test_pred, zero_division=0),
                "test_f1_yes": f1_score(y_test, test_pred, zero_division=0),
                "test_pred_yes_rate": test_pred.mean(),
                "confusion_matrix": confusion_matrix(y_test, test_pred),
                "test_prob": test_prob,
                "test_pred": test_pred,
            }
        """
    ),
    md(
        """
        ## 8. 基线模型：Logistic 回归

        Logistic 回归是很好的入门基线：速度快、稳定、容易解释。我们用 `class_weight='balanced'` 告诉模型更重视少数类。
        """
    ),
    code(
        """
        logistic_model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
        )

        logistic_model.fit(X_train, y_train)
        logistic_result = evaluate_model("Logistic Regression", logistic_model, X_val, y_val, X_test, y_test)

        pd.DataFrame([{
            k: v for k, v in logistic_result.items()
            if k not in ["confusion_matrix", "test_prob", "test_pred"]
        }]).round(4)
        """
    ),
    md(
        """
        ## 9. 主力模型：LightGBM 默认版本

        LightGBM 是梯度提升树模型，常用于表格数据。先训练一个默认版本，作为调优前的对照组。
        """
    ),
    code(
        """
        lgb_default = lgb.LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
        )

        lgb_default.fit(X_train, y_train)
        lgb_default_result = evaluate_model("LightGBM Default", lgb_default, X_val, y_val, X_test, y_test)

        pd.DataFrame([{
            k: v for k, v in lgb_default_result.items()
            if k not in ["confusion_matrix", "test_prob", "test_pred"]
        }]).round(4)
        """
    ),
    md(
        """
        ## 10. 小范围参数调优

        这里不追求把所有参数都搜一遍，而是搜索几组最常影响效果和过拟合的参数。

        调优标准用验证集 PR-AUC，因为它更关注少数类 `yes` 的识别质量。
        """
    ),
    code(
        """
        param_grid = list(ParameterGrid({
            "n_estimators": [300, 600, 900],
            "learning_rate": [0.02, 0.03, 0.05],
            "num_leaves": [15, 31],
            "max_depth": [4, 6, -1],
            "min_child_samples": [40, 80, 120],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [0, 1],
        }))

        rng = np.random.default_rng(RANDOM_STATE)
        sampled_indices = rng.choice(len(param_grid), size=36, replace=False)
        search_grid = [param_grid[i] for i in sampled_indices]

        search_grid += [
            {
                "n_estimators": 600,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 80,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1,
            },
            {
                "n_estimators": 900,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "max_depth": 4,
                "min_child_samples": 80,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1,
            },
        ]

        best_score = -np.inf
        best_params = None
        best_lgb_model = None
        tuning_rows = []

        for i, params in enumerate(search_grid, start=1):
            model = lgb.LGBMClassifier(
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
                **params,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )

            val_prob = model.predict_proba(X_val)[:, 1]
            val_pr_auc = average_precision_score(y_val, val_prob)
            val_roc_auc = roc_auc_score(y_val, val_prob)
            tuning_rows.append({"candidate": i, "val_pr_auc": val_pr_auc, "val_roc_auc": val_roc_auc, **params})

            if val_pr_auc > best_score:
                best_score = val_pr_auc
                best_params = params
                best_lgb_model = model

        tuning_results = pd.DataFrame(tuning_rows).sort_values("val_pr_auc", ascending=False)
        display(tuning_results.head(10).round(4))

        print("最佳参数:")
        print(best_params)
        """
    ),
    md(
        """
        ## 11. 最终模型评估

        用验证集选择最佳阈值，然后在测试集上评估。测试集结果才是我们对泛化效果的估计。
        """
    ),
    code(
        """
        tuned_result = evaluate_model("LightGBM Tuned", best_lgb_model, X_val, y_val, X_test, y_test)

        comparison = pd.DataFrame([
            {k: v for k, v in logistic_result.items() if k not in ["confusion_matrix", "test_prob", "test_pred"]},
            {k: v for k, v in lgb_default_result.items() if k not in ["confusion_matrix", "test_prob", "test_pred"]},
            {k: v for k, v in tuned_result.items() if k not in ["confusion_matrix", "test_prob", "test_pred"]},
        ])

        metric_cols = [
            "model",
            "threshold_from_val",
            "val_pr_auc",
            "test_pr_auc",
            "test_roc_auc",
            "test_precision_yes",
            "test_recall_yes",
            "test_f1_yes",
            "test_pred_yes_rate",
        ]
        display(comparison[metric_cols].round(4))

        print("测试集分类报告:")
        print(classification_report(y_test, tuned_result["test_pred"], target_names=["no", "yes"]))

        print("混淆矩阵 [[TN, FP], [FN, TP]]:")
        print(tuned_result["confusion_matrix"])
        """
    ),
    md(
        """
        ## 12. 宏观经济变量有没有帮助？

        旧版 notebook 删除了宏观经济变量，主要是为了避免 Logistic 回归中的共线性问题。  
        但对 LightGBM 这类树模型来说，保留这些变量通常是可以的。下面做一个对照实验。
        """
    ),
    code(
        """
        X_no_macro, y_no_macro = build_model_data(df, include_macro=False, include_duration=False)

        X_temp_nm, X_test_nm, y_temp_nm, y_test_nm = train_test_split(
            X_no_macro, y_no_macro, test_size=0.20, stratify=y_no_macro, random_state=RANDOM_STATE
        )
        X_train_nm, X_val_nm, y_train_nm, y_val_nm = train_test_split(
            X_temp_nm, y_temp_nm, test_size=0.25, stratify=y_temp_nm, random_state=RANDOM_STATE
        )

        no_macro_model = lgb.LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
        )
        no_macro_model.fit(X_train_nm, y_train_nm)
        no_macro_result = evaluate_model(
            "LightGBM No Macro", no_macro_model, X_val_nm, y_val_nm, X_test_nm, y_test_nm
        )

        macro_compare = pd.DataFrame([
            {
                "feature_set": "保留宏观经济变量",
                "test_pr_auc": tuned_result["test_pr_auc"],
                "test_roc_auc": tuned_result["test_roc_auc"],
                "test_f1_yes": tuned_result["test_f1_yes"],
            },
            {
                "feature_set": "删除宏观经济变量",
                "test_pr_auc": no_macro_result["test_pr_auc"],
                "test_roc_auc": no_macro_result["test_roc_auc"],
                "test_f1_yes": no_macro_result["test_f1_yes"],
            },
        ])

        display(macro_compare.round(4))
        """
    ),
    md(
        """
        ## 13. 为什么不能用 `duration`

        下面只是演示信息泄漏：如果加入 `duration`，模型效果会突然变好。  
        但真实业务中，我们在打电话之前不知道通话时长，所以这个效果不能用于营销前预测。
        """
    ),
    code(
        """
        X_leak, y_leak = build_model_data(df, include_macro=True, include_duration=True)

        X_temp_lk, X_test_lk, y_temp_lk, y_test_lk = train_test_split(
            X_leak, y_leak, test_size=0.20, stratify=y_leak, random_state=RANDOM_STATE
        )
        X_train_lk, X_val_lk, y_train_lk, y_val_lk = train_test_split(
            X_temp_lk, y_temp_lk, test_size=0.25, stratify=y_temp_lk, random_state=RANDOM_STATE
        )

        leakage_model = lgb.LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
        )
        leakage_model.fit(X_train_lk, y_train_lk)
        leakage_result = evaluate_model(
            "LightGBM With Duration Leakage", leakage_model, X_val_lk, y_val_lk, X_test_lk, y_test_lk
        )

        leakage_compare = pd.DataFrame([
            {
                "model": "真实预测模型：不使用 duration",
                "test_pr_auc": tuned_result["test_pr_auc"],
                "test_roc_auc": tuned_result["test_roc_auc"],
                "test_f1_yes": tuned_result["test_f1_yes"],
            },
            {
                "model": "泄漏演示：使用 duration",
                "test_pr_auc": leakage_result["test_pr_auc"],
                "test_roc_auc": leakage_result["test_roc_auc"],
                "test_f1_yes": leakage_result["test_f1_yes"],
            },
        ])

        display(leakage_compare.round(4))
        """
    ),
    md(
        """
        ## 14. 特征重要性

        特征重要性告诉我们：模型主要依赖哪些变量做判断。它不是因果结论，只能说这些变量对预测有帮助。
        """
    ),
    code(
        """
        importance = pd.Series(best_lgb_model.feature_importances_, index=X_train.columns)
        importance = importance.sort_values(ascending=False).head(20)

        display(importance.rename("importance").to_frame())

        plt.figure(figsize=(8, 6))
        sns.barplot(x=importance.values, y=importance.index, color="#4C78A8")
        plt.title("Top 20 Feature Importances - Tuned LightGBM")
        plt.xlabel("Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "feature_importance_top20.png", dpi=150)
        plt.show()
        """
    ),
    md(
        """
        ## 15. 阈值曲线

        模型输出的是“认购概率”。我们需要选择一个阈值，把概率变成 `yes/no`。

        阈值越低：能找出更多潜在客户，但误报也更多。  
        阈值越高：联系名单更精准，但会漏掉更多客户。
        """
    ),
    code(
        """
        val_prob = best_lgb_model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        best_threshold = tuned_result["threshold_from_val"]

        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, precision[:-1], label="Precision")
        plt.plot(thresholds, recall[:-1], label="Recall")
        plt.plot(thresholds, f1_scores, label="F1")
        plt.axvline(best_threshold, color="black", linestyle="--", label=f"Selected threshold={best_threshold:.3f}")
        plt.title("Validation Metrics by Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "threshold_curve.png", dpi=150)
        plt.show()
        """
    ),
    md(
        """
        ## 16. 结论

        这版 notebook 的推荐模型是：**不使用 `duration`、保留宏观经济变量、经过小范围调优的 LightGBM**。

        这不是“完美模型”，但它比旧版更适合真实业务预测，也更适合新手学习完整流程。
        """
    ),
    code(
        """
        final_summary = {
            "recommended_model": "LightGBM Tuned",
            "do_not_use": "duration",
            "best_params": best_params,
            "selected_threshold": tuned_result["threshold_from_val"],
            "test_pr_auc": tuned_result["test_pr_auc"],
            "test_roc_auc": tuned_result["test_roc_auc"],
            "test_precision_yes": tuned_result["test_precision_yes"],
            "test_recall_yes": tuned_result["test_recall_yes"],
            "test_f1_yes": tuned_result["test_f1_yes"],
            "confusion_matrix": tuned_result["confusion_matrix"].tolist(),
        }

        for key, value in final_summary.items():
            print(f"{key}: {value}")
        """
    ),
]

nb["cells"] = cells
nbf.write(nb, NOTEBOOK_PATH)

SUMMARY_PATH.write_text(
    """# Bank Marketing 调优摘要

## 新版交付物

- `bank-marketing-beginner-tuned.ipynb`：新手友好、可本地运行的完整 notebook。
- `figures_beginner/`：新版 notebook 运行后生成的图表目录。

## 建模原则

- 真实营销前预测不使用 `duration`，因为它是通话结束后才知道的事后信息。
- 保留宏观经济变量，因为它们在营销时点通常可知，并且对 LightGBM 有明显帮助。
- 使用训练集训练模型、验证集选择参数和阈值、测试集做最终评估。
- 对不平衡分类问题，重点看 PR-AUC、`yes` 类 precision/recall/F1，而不是只看 accuracy。

## 当前实验结论

在 60%/20%/20% 的训练、验证、测试划分下，推荐模型为：

```text
LightGBM Tuned
不使用 duration
保留宏观经济变量
验证集选择阈值
```

本地调优实验得到的测试集表现约为：

```text
PR-AUC: 0.4923
ROC-AUC: 0.8185
yes precision: 0.4851
yes recall: 0.5970
yes F1: 0.5353
混淆矩阵: [[6722, 588], [374, 554]]
```

加入 `duration` 后指标会明显升高，但这是信息泄漏，不建议作为真实预测模型使用。
""",
    encoding="utf-8",
)

print(f"Created {NOTEBOOK_PATH.name}")
print(f"Created {SUMMARY_PATH.name}")
