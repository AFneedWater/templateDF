# 第三方来源记录

## Raygun（只读参考）

- 用户指定的本地路径：`/home/gh/raygun`。
- 本地 Git remote：`https://github.com/rohitsinghlab/raygun.git`。
- 检查时 HEAD：`cd3b3574708a71f6dc134c7719d287b5ab274186`。
- 原作者：Kapil Devkota、Rohit Singh。
- 阶段 00 未复制、修改或移植 Raygun 源代码；本项目不依赖 `import raygun`，不修改 `sys.path`。

本地 `LICENSE` 标示 CC BY-NC 4.0，而本地 `pyproject.toml` 的 license/classifier
标示 MIT，两处存在不一致。阶段 00–01 只记录此差异。
阶段 02 移植 Block 时保留上游 LICENSE 的 CC BY-NC 4.0 原文与版权声明，
不依据包元数据将该代码重新标为 MIT；两处许可标记差异尚未由上游澄清。
以下原样保留检查时的 `LICENSE` 内容，尚未对 templateDF 指定项目许可证。

```text
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright © 2024 Kapil Devkota, Rohit Singh

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. 
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/ 
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Under the following terms:

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial**: You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
```

## ESM

阶段 00 仅记录路径。阶段 01 使用环境中已安装的 fair-esm 2.0.0，调用本地库的
`load_model_and_alphabet_core` 构造模型；没有复制其源代码，也未下载模型。
实际加载用户提供的本地 ESM2 650M 与 contact-regression 权重；文件 SHA256 记录在
`reports/01_logs/real_extract.json` 的 `spec.source` 中。
本地包元数据的许可证字段为 `MIT`，主页为 `https://github.com/facebookresearch/esm`。
阶段 01 的数据、缓存与包装接口自行实现；阶段 02 的 Block 移植记录见下文。

## 阶段 02：Raygun Block 适配

- 来源文件：`/home/gh/raygun/raygun/modelv2/model_utils.py`，commit 同上。
- 新文件：`src/templatedf/blocks.py`；保留 Kapil Devkota、Rohit Singh 的版权归属。
- 原始许可另存 `licenses/RAYGUN_LICENSE.txt`，与参考仓库 LICENSE 逐字节一致，随分发包保留。
- 保留 TransformerLayer 的 rotary 位置编码、2D FFN、Transformer 后的三段卷积、SiLU、最终投影。
- 修改：以 transpose 替换 einops，补零继承输入 device/dtype；输入、每层卷积、投影与外部残差后清零 padding；增加 mask/尺寸检查与输出 dropout。
- 关闭上游默认 `add_bias_kv`：原实现将额外 K/V token 放在 batch 的 Lmax 位置，rotary 使有效输出随补齐长度变化；本任务要求 padding/batch 隔离。
- dtype/device 转换时重置 fair-esm 的非 buffer rotary 缓存，避免先 FP32 前向再转换精度时沿用旧表。
- `src/templatedf/encoder.py` 的多级融合与 learned-query 聚合按本项目公共约定新实现。
- 没有修改或运行时导入 Raygun；只使用已声明的 fair-esm 中的 TransformerLayer，未加载 ESM2 预训练权重。

## 阶段 03：解码器与自编码器整合

`src/templatedf/decoder.py` 和 `src/templatedf/model.py` 按本项目公共约定新实现，
使用 PyTorch 的 MultiheadAttention、LayerNorm 和 Linear，没有移植 Raygun 解码器。
阶段 02 Block 的来源与许可继续保留；此阶段没有修改 Raygun 或加载 ESM 预训练权重。
