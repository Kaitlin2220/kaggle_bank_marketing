# Bank Marketing 调优摘要

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


## 追加优化：时间切分与客户分层

本项目进一步加入三项更贴近业务的优化：

- 时间切分验证：用前 60% 样本训练，中间 20% 选择阈值，最后 20% 模拟未来测试。
- 未来客户排序模型：用前 80% 历史样本训练，最后 20% 模拟未来客户名单排序。
- Top 客户分层收益分析：比较 Top 10%、Top 20%、Top 30% 高概率客户名单的实际认购率和提升倍数。

最终客户排序结果显示，`No Macro Future Ranking Model` 在未来测试集上的排序效果更好：

```text
排序测试集 PR-AUC: 0.5233
排序测试集 ROC-AUC: 0.6907
Top 10% 客户认购率: 65.66%，约为整体未来样本 yes 率的 2.13 倍
Top 20% 客户认购率: 59.89%，约为整体未来样本 yes 率的 1.94 倍
Top 30% 客户认购率: 51.78%，约为整体未来样本 yes 率的 1.68 倍
```

执行 notebook 后会自动生成：

- `bank-marketing-optimization-log.md`
- `figures_beginner/time_split_yes_rate.png`
- `figures_beginner/top_customer_lift_time_holdout.png`
