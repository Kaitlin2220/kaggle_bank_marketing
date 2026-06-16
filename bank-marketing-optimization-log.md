# Bank Marketing 优化日志

## 记录日期

2026-06-04

## 本次新增优化

1. 新增时间切分验证：使用前 60% 样本训练，中间 20% 样本选择阈值，最后 20% 样本模拟未来测试。
2. 新增未来客户排序模型：使用前 80% 历史样本训练，最后 20% 样本模拟未来客户名单排序。
3. 新增 Top 客户分层收益分析：评估 Top 10%、Top 20%、Top 30% 客户名单的实际认购率、提升倍数和覆盖的 yes 客户比例。
4. 新增图表输出：
   - `figures_beginner/time_split_yes_rate.png`
   - `figures_beginner/top_customer_lift_time_holdout.png`

## 随机切分模型结果

- 推荐模型：LightGBM Tuned
- 测试集 PR-AUC：0.4923
- 测试集 ROC-AUC：0.8185
- yes precision：0.4851
- yes recall：0.5970
- yes F1：0.5353
- 混淆矩阵：[[6722, 588], [374, 554]]

## 时间切分阈值验证结果

- 时间测试集 PR-AUC：0.2898
- 时间测试集 ROC-AUC：0.4352
- yes precision：0.2651
- yes recall：0.5425
- yes F1：0.3562
- 混淆矩阵：[[1878, 3820], [1162, 1378]]

## 未来客户排序模型结果

- 推荐排序模型：No Macro Future Ranking Model
- 排序测试集 PR-AUC：0.5233
- 排序测试集 ROC-AUC：0.6907

## Top 客户分层结果

时间测试集整体 yes 率：30.83%

contact_group  contact_customers  actual_yes  response_rate  lift_vs_baseline  captured_yes_rate
      Top 10%                824         541         0.6566            2.1294             0.2130
      Top 20%               1648         987         0.5989            1.9424             0.3886
      Top 30%               2472        1280         0.5178            1.6794             0.5039

## 解读

- 时间切分阈值验证显示：模型从随机切分迁移到未来数据时明显变难，存在时间分布漂移。
- 对营销名单来说，排序比直接分类更实用；使用前 80% 历史数据训练后的 Top 客户分层有明显提升。
- 当前结果中 Top 10% 客户的实际认购率约为整体未来样本的 2 倍以上，适合优先联系。
- 后续可以继续做按月份/宏观周期的时间漂移分析，或按营销资源容量选择 Top N 客户名单。
