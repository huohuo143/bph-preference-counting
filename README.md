# bph-preference-counting

褐飞虱取食偏好照片逐图计数与时间曲线的 Codex Skill。使用入口和研究边界见 [SKILL.md](SKILL.md)，复现命令见 [references/reproduction.md](references/reproduction.md)。

仓库包含历史案例的冻结脚本、推理资源、逐图候选计数和统计基准；不包含原始照片。历史案例只有 A 组达到初始模型目视自检目标，B/C 仍待计数复核。`replot` 使用历史记录出图，不代表新一轮检测或独立验收。

## 本机使用

将仓库克隆到 `~/.codex/skills/bph-preference-counting`，然后使用已有 Python 环境检查依赖与资源：

```sh
python scripts/workflow.py doctor
```

完整运行方式及新实验边界见上述复现文档。历史案例原图重跑需要另行提供与冻结清单一致的原始照片目录。生成结果应写入仓库之外的新目录。
