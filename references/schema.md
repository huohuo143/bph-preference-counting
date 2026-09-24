# 字段与追溯

## 逐图表

| 字段 | 含义 |
|---|---|
| frame_id / group / source_path | 唯一照片标识、实验组、原图绝对路径 |
| relative_path / sequence_index | 实验内路径、组内索引；同名文件不能跨目录合并 |
| timestamp / interval_s / elapsed_s | 拍摄时间、距前帧秒数、真实经过秒数 |
| source_sha256 / source_unchanged | 读取内容哈希；处理时大小及mtime是否与清单一致，后者不是前后内容哈希比较 |
| auto_left_count / auto_right_count / auto_other_count | 每张独立检测的候选数，可能误计或漏计 |
| left_count / right_count / other_count | 可纳入主统计的数值；不可靠或组验收失败时为空 |
| suspect_count | 候选中的复核标记数，与左右/其他候选可能重叠，不可相加 |
| quality_ok / quality_reason | 画面质量初筛与原因，不等于计数准确性通过 |
| valid_primary / valid_candidate_summary | 主统计与候选统计的分别纳入状态 |
| primary_missing_reason / other_missing_reason | 空白原因，空白不是0 |
| reference_left_count / reference_right_count | 目视参考，不自动替代整批检测值或坐标 |
| review_status / expert_reviewed | 复核来源、是否经实验人员确认 |
| detector_version / detector_fingerprint | 版本与模型/参数/代码指纹 |

冻结CSV的有效性及通过组标签只属于本案例；新数据必须重新计算，不能复制valid_primary或A组通过状态。

## 坐标与断点

原图原点在左上，x向右、y向下。`bbox=[x,y,width,height]`及中心x/y、side、zone、score、suspect、review_reasons随图保存。预览可缩放，原图坐标不能随预览改变。

JSONL逐图保存状态、SHA256、最终检测及被拒候选。`全部自动检测坐标.csv.gz`只导出最终自动候选，与自动左右计数一致，不能冒充人工修正坐标。SQLite用于本机读取，不是独立实验数据。

人工复核另存追加JSONL，记录复核者、时间、原图哈希、逐体框和修正数。未解决的重叠仍须标明不可确定，不能勾选后自动变成精确虫数。

## 时间表

`observed_seconds`为实际纳入时长；`left/right_insect_minutes`为虫分钟积分；`left/right_mean`为加权只数；`left/right_share`表中0到1、图中百分比。

对应表记录连续片段、真实及有效起止秒数、首末帧。有效片段首尾相接，真实缺口不积分。未观察窗口的积分/均值为空；有观察但两侧均零时积分为0，占比为空。
