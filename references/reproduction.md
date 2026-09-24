# 复现命令与运行要求

## 先确定最终成图

最终交付是虫子数量随时间变化的曲线图。下列命令服务于出图；prepare、smoke或run单独结束只表示完成中间步骤。历史数据走replot，原图重跑完成后走export；之后打开实际曲线PNG，检查并向用户展示成图。

本案例优先交付 `figures/01_时间曲线_有效时间拼接_中文_待复核.png` 及同名SVG/PDF，它展示三组自动候选曲线并保留待复核标识。`figures/01_时间曲线_有效时间拼接_中文.png` 为仅初检达标范围的主统计版本。新数据根据实际核验状态命名并标明状态，不能为了出图删除“待复核”。

## 运行边界

`scripts/workflow.py`是推荐入口。它不联网、不安装依赖、不修改源照片。`assets/frozen_v1/`中的脚本是历史快照，含固定组名、像素尺寸和指纹；不能直接作为任意照片的通用计数器。

包内包含冻结推理权重、三个场景背景、26,403行历史计数、样本参考、统计基准和源代码。原始照片约145 GB，未打包。模型训练特征矩阵未打包，推理复现不需要它。历史标签和训练代码保留，但重新训练属于新版本，不保证相同权重字节，也不继承旧准确性结论。

现有环境：`/opt/miniconda3/bin/python`。参考版本见 `assets/case_2023/data/运行环境版本.json`。依赖包括numpy、pandas、OpenCV、scipy、scikit-image、scikit-learn、matplotlib、openpyxl、Pillow、joblib、PyMuPDF。冻结绘图使用Arial Unicode MS。换环境先运行doctor；缺包时先寻找已有环境，不自动安装。

示例使用当前安装目录；搬迁Skill后将 `BPH_SKILL` 改为实际目录。输出路径按用户任务填写，不覆盖原分析目录。

```sh
BPH_SKILL='/Users/zhangshuai/.codex/skills/bph-preference-counting'
/opt/miniconda3/bin/python "$BPH_SKILL/scripts/workflow.py" doctor
```

## 只复绘本次全量统计

```sh
/opt/miniconda3/bin/python "$BPH_SKILL/scripts/workflow.py" replot \
  --output '/Volumes/115/Codex/取食偏好/analysis_results/历史统计复绘_新目录'
```

不读取原图，也不把历史数据冒充新的逐图检测。输出历史逐图计数Excel、两套时间统计、18个图形文件、逐表对照、技术检查和复绘说明。原始图像、全量逐体坐标与新的目视验收不属于此模式。

## 可重复运行的边界检查

```sh
/opt/miniconda3/bin/python "$BPH_SKILL/scripts/test_contracts.py"
```

检查原图/已有目录保护、新实验不得继承历史验收、运行代码改变、缺失/重复/损坏记录、错误源文件哈希和时间积分规则。只使用临时测试文件与合成数值，不将测试数据写成实验结果；不需要原始照片。它不能代替smoke或计数准确性验收。

## 从原始照片重新检测

默认配置指向本次三个原图目录；目录移动后，复制 `assets/fafu-2023.example.json` 到工作区，仅修改 `source_folders`。需要A/B/C三组完整、内容相同的照片。文件清单或SHA256不同的资料应转入新实验流程。

```sh
/opt/miniconda3/bin/python "$BPH_SKILL/scripts/workflow.py" prepare \
  --config "$BPH_SKILL/assets/fafu-2023.example.json" \
  --output '/Volumes/115/Codex/取食偏好/analysis_results/原图复现_新目录'

/opt/miniconda3/bin/python "$BPH_SKILL/scripts/workflow.py" smoke \
  --workspace '/Volumes/115/Codex/取食偏好/analysis_results/原图复现_新目录'

/opt/miniconda3/bin/python "$BPH_SKILL/scripts/workflow.py" run \
  --workspace '/Volumes/115/Codex/取食偏好/analysis_results/原图复现_新目录' --workers 16

/opt/miniconda3/bin/python "$BPH_SKILL/scripts/workflow.py" export \
  --workspace '/Volumes/115/Codex/取食偏好/analysis_results/原图复现_新目录'
```

prepare重新读取拍摄时间和尺寸，比对完整清单，生成新的源路径、大小及修改时间记录。smoke检测7张原图；run逐张原生分辨率处理，保存SHA256和坐标并对照历史候选计数；export只允许完整且一致的26,403条记录进入历史结果复现。任何输入或结果不一致都保存差异并失败退出，不继承历史A组的初检状态。

历史模型目视参考原样保留，不是又完成一轮专家核验。B/C主计数依然留空。

## 断点与故障

- 同一项目再次run会续跑未处理照片。检查点末尾因中断损坏时，历史运行器先保存尾行备份再续跑。
- 中间损坏、重复标识、读取失败或数值不一致阻止导出。检查源卷、差异记录及版本；失败不会被当成零或用邻帧填补。
- 恢复读取失败后，在保留检查点备份的前提下重试失败帧，或另建复现目录；入口不静默删除失败记录。
- 改参数或代码必须另建版本、记录差异并重新核验，不能沿用旧结论。
- 统计复现要求数值及状态一致；生成时间、输出目录、图片压缩及PDF元数据不保证逐字节相同。

## 本机复核

完整原图重跑并导出后：

```sh
/opt/miniconda3/bin/python "$BPH_SKILL/scripts/workflow.py" review \
  --workspace '/Volumes/115/Codex/取食偏好/analysis_results/原图复现_新目录'
```

仅监听127.0.0.1，打开原图和候选框。复核者可删除、补框及追加保存；Ctrl+C结束。保存不证明正确，也不直接覆盖统计。整批复核先审查未解决的疑似及可纳入帧，再生成新版本统计。

## 新实验模板

复制并填写 `assets/new-study.example.json`：

```sh
/opt/miniconda3/bin/python "$BPH_SKILL/scripts/workflow.py" init-new \
  --config '/绝对路径/新实验配置.json' --output '/绝对路径/新实验分析目录'
```

该命令只建立配置和空白参考表，是新实验出图的起点。除非用户仅要求模板，否则继续按 `analysis-contract.md` 适配，完成逐张计数、核验、时间统计及曲线交付。run/export拒绝给新实验直接套用冻结A/B/C模型，需在新项目中完成适配；不要将模板已创建报告为曲线任务完成。
