# templateDF：分阶段实施入口

本文件替代此前“一次性实现全部模型”的执行说明。现在只组织任务，不表示模型代码、测试或训练已经完成。

项目独立存放在 `templateDF/`，参考仓库为同级的 `raygun-main/`。目标是 **10–150 aa 肽**的模板重建与生成，使用冻结 ESM2、级联自编码器及默认 50 个 latent token。

## 使用方法

1. 一次只把下表中的一个任务交给服务器 agent。
2. agent 只读 [公共约定](tasks/CONTRACT.md)、[进度表](tasks/STATUS.md) 和当前任务文件；遇到明确细节缺失时再查原稿相关部分。
3. agent 完成当前任务、运行约定检查、提交报告后停止。
4. 你按该任务的“人工检查”核对证据，决定通过或返工；明确通过后再发下一阶段任务。

人工检查是本次任务组织的明确要求。agent 不得把自动测试通过等同于人工通过，也不得因你暂时未回复而继续下一阶段。你在对话中的明确放行同样有效，不必重复确认。

## 阶段任务

| 顺序 | 任务 | 完成后你主要看什么 |
| --- | --- | --- |
| 00 | [环境与独立工程骨架](tasks/00_PROJECT_SETUP.md) | 路径独立、环境正确、GPU 信息真实 |
| 01 | [肽数据与 ESM 嵌入缓存](tasks/01_DATA_AND_ESM.md) | 10–150 aa 过滤、标签、真实残基切片 |
| 02 | [级联编码器与 latent 聚合](tasks/02_ENCODER.md) | 多级输出确实参与融合，短肽能生成 50 个 token |
| 03 | [解码器与自编码器整合](tasks/03_DECODER_AND_AE.md) | 输出长度正确、没有绕过 latent 的跳连 |
| 04 | [1A 训练引擎与保存恢复](tasks/04_TRAINING_ENGINE.md) | loss、梯度、mask、checkpoint 正确 |
| 05 | [5090 实测与真实肽 1A 短训](tasks/05_1A_PILOT.md) | 实际显存、训练曲线、验证结果和拟定预算 |
| 06 | [模板候选生成与 latent 导出](tasks/06_TEMPLATE_GENERATION.md) | 模板被真实使用、候选来源可追溯、导出可复用 |
| 07 | [1B 变长一致性训练与验证](tasks/07_LENGTH_CONSISTENCY.md) | 梯度穿过冻结编码器、新长度 mask、离散循环评估 |

依赖顺序为 `00 → 01 → 02 → 03 → 04 → 05 → 06 → 07`。此顺序故意保守，以便逐项人工检查。本轮不分派潜在扩散实现任务；**1B 不是 diffusion，1A 和 1B 都可以批量从模板生成候选。**

## 直接复制给服务器 agent 的首条任务

```text
请在独立的 templateDF 项目中，只执行 tasks/00_PROJECT_SETUP.md。
先阅读 tasks/CONTRACT.md 和 tasks/STATUS.md。
不要一次性执行其他阶段，也不要把归档长稿当作完整实施指令。
完成本阶段的工作和自动检查，按 tasks/REVIEW_TEMPLATE.md 写出
reports/00_PROJECT_SETUP.md，并将 agent 状态标记为“待人工审核”。
保留人工放行栏由我决定。提交报告和需要我检查的证据后停止。
```

后续放行或返工可直接使用 [人工检查指南](tasks/HUMAN_REVIEW.md) 中的提示词。

## 文档与证据

- [公共约定](tasks/CONTRACT.md)：所有阶段共享的接口、尺寸、范围和默认值。
- [进度表](tasks/STATUS.md)：agent 完成状态与人工放行分开记录。
- [人工检查指南](tasks/HUMAN_REVIEW.md)：如何检查、放行、返工。
- [报告模板](tasks/REVIEW_TEMPLATE.md)：每阶段必须提供实际执行证据。
- [原始长稿](docs/IMPLEMENTATION_SPEC_FULL.md)：完整保留原稿内容，仅供按需查阅。

冲突处理顺序：你的最新明确要求 → 本次任务文件与公共约定 → 归档长稿。归档中的“全部实现”措辞和超出 10–150 aa 的旧示例不再作为执行指令；例如旧 `target-length 200` 示例不适用于本轮。

所有阶段起始状态均为“未开始”，不会因为文档已生成而标为实现完成。
