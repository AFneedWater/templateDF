# templateDF

独立的肽模板重建与生成项目。阶段00–06已交付工程实现及预算内试训（阶段06待审）；当前缓存/RAM优化单独待人工验收：
**FASTA/ESM 缓存、级联自编码器、有硬停止预算的 1A 训练与恢复，以及模板重建、温度采样和原始 latent 导出**。
正式 4＋4 层模型已在 cuda:0 完成 300 步真实肽短训，验证残基准确率 10.38%，改善有限，尚未收敛。模板候选来自温度采样，不是 diffusion；改变输出长度不代表已验证信息保留。

## 环境与安装

项目位于 `/home/gh/templateDF`；参考仓库 `/home/gh/raygun` 不是运行时依赖。
`import templatedf` 不加载 PyTorch、ESM、Raygun 或权重。

```bash
cd /home/gh/templateDF
conda activate peptide_vae
python -m pip install --no-deps --no-build-isolation --no-index --no-cache-dir -e .
python -m pytest -q
```

本机实际解释器为 `/home/gh/miniconda3/envs/peptide_vae/bin/python`，Python 3.11、
PyTorch 2.9.0+cu128。默认 shell 的 Python 属于 base 环境，须先激活环境或使用绝对路径。
上述离线安装使用已有 setuptools、wheel、torch、numpy、fair-esm；训练配置使用 PyYAML，测试另需 pytest。
新环境应先准备适合硬件的 PyTorch，再安装 `python -m pip install -e '.[check]'`。

## 用户指定数据与检查

- 训练集：`AgsgeGpr3I_10-150aa_train_140000.fasta`。
- 测试集：`AgsgeGpr3I_10-150aa_test_30000.fasta`。
- 验证集：用户同意从训练源按长度分层留出128条，本次保存在 `artifacts/05_pilot/validation.fasta`；另抽512条作为本次训练。以后使用完整训练源须排除这些验证序列；测试集不用于训练调参。

```bash
python -m templatedf.data \
  --train AgsgeGpr3I_10-150aa_train_140000.fasta \
  --test AgsgeGpr3I_10-150aa_test_30000.fasta \
  --report reports/data_audit.json
```

也支持独立的 `--validation PATH`。同一文件（含软/硬链接）不得跨集合使用。
报告记录全部过滤计数、重复 ID/序列数、长度分布、文件 SHA256 和跨集合完全相同的序列。
相同序列会报告但不擅自删除；完全无重复不代表已经完成同源隔离。

默认保留 10–150 aa 标准大写氨基酸序列；长度不含 BOS/EOS/padding。
FASTA 换行仅用于拼接，不删除内部空格、不转大写、不改写残基。
含非标准字符的整条记录被跳过，拒绝原因可重叠，`excluded` 按记录去重计数。
`--no-sequence-check`（Python 接口 `check_sequences=False`）允许跳过可信数据的预过滤，
此时未检查原因计数为 `null`；编码和抽取入口仍显式拒绝不符合 20 类标签及长度约定的输入。

标签顺序固定为 `ACDEFGHIKLMNPQRSTVWY`，类别为 0..19，右侧 padding 为 -100。
`collate_peptides` 返回 `ids`、`sequences`、`labels`、`input_mask`（bool，True 为真实残基）、`lengths`。

## 有预算的本地 ESM 抽取

先做少量抽取，例如：

```bash
python -m templatedf.esm_features \
  --fasta AgsgeGpr3I_10-150aa_train_140000.fasta \
  --cache-dir artifacts/train_cache \
  --checkpoint /home/gh/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt \
  --device cuda:0 --batch-size 3 --limit 3 \
  --storage-dtype float16 --report reports/extract_sample.json
```

默认从权重同目录读取 `esm2_t33_650M_UR50D-contact-regression.pt`，也可用
`--regression-checkpoint PATH` 指定。缺少任一权重文件会报错，**不会自动下载**。
CLI 必须指定 `--limit N` 或显式 `--all`；阶段 01 未执行全量抽取。
`--limit` 指过滤后输入顺序的前 N 条记录，batch-size 必须为正数。
默认设备为 CPU，使用 GPU 时须明确指定；示例设备不代表 GPU 调度保留。

ESM 始终冻结且 eval，FP32/no_grad 抽取第 33 层，每条按 `1:1+L` 取真实残基。
每条特征为 `[L,1280]`；FP16 默认仅用于缓存存储，不等同于 BF16 autocast 训练。
新缓存目录必须不存在；写完全部分片后发布manifest。旧逐条PT缓存只保留历史产物，不继续作为加载后端。

## NPY分片缓存与全量RAM读取

当前唯一缓存格式为`format_version=2 / npy_shards`。每片嵌入目标256 MiB（`--shard-mib 256`），不压缩、不按肽建立文件：

- `embeddings.npy`：FP16 `[该片总真实残基数,1280]`。
- `offsets.npy`：int64 `[样本数+1]`，从0开始，以该片残基数结束。
- `labels.npy`：int8 `[该片总真实残基数]`，值0..19，与嵌入排列一致。
- `metadata.jsonl`：ID、序列、序列hash、长度、稳定cache_key、全局索引。
- 根`manifest.json`：版本、ESM型号／层／权重来源、AA顺序、dtype、片信息、数量和校验和。

不保存padding/BOS/EOS；训练batch中标签才转int64并用-100补齐。相同ID不同序列用ID＋序列hash区分，不覆盖。缓存只保存冻结ESM第33层输出，级联encoder每步仍参与前向和反向。

训练入口先审计独立train/validation，再检查manifest及NPY头，**分配数组前**打印两个集合合计RAM估算。默认预算40 GiB，包含数组和元信息／索引保守预留；不包括模型、优化器、batch和OS。超预算直接报所需容量，无自动回退。相同缓存目录只加载一次。

使用`np.load(..., mmap_mode=None, allow_pickle=False)`依次读入各片并保留普通数组；不全局concatenate，不整份转FP32，不锁页全部缓存。Dataset按索引返回RAM切片，加载时完成hash／offset／标签／有限值检查；取样期间不再打开缓存文件。仅写缓存时允许合并当前片。

```python
from templatedf.data import load_fasta
from templatedf.feature_cache import FeatureCache, CachedPeptideDataset, load_caches_to_ram
from templatedf.training import StatefulBatchSampler, make_data_loader, batch_to_device

# 两个目录须是已生成的NPY v2缓存；此处路径是示例。
caches = {"train": FeatureCache("artifacts/train_npy"),
          "validation": FeatureCache("artifacts/validation_npy")}
loading = load_caches_to_ram(caches, budget_gib=40)
records, _ = load_fasta("artifacts/05_pilot/train.fasta")
dataset = CachedPeptideDataset(records, caches["train"])  # FP16视图，不读盘
sampler = StatefulBatchSampler(len(dataset), batch_size=8, seed=42)
loader = make_data_loader(dataset, device="cuda:0", batch_sampler=sampler)
# 每轮重新iter(loader)：打乱索引；collate仅为当前batch补零。
for host_batch in loader:
    batch = batch_to_device(host_batch, "cuda:0")
    # 此时该batch在GPU上为FP32，进入现有BF16 autocast模型。
```

默认`num_workers=0`、`persistent_workers=False`，不传prefetch_factor；当前实现不开放多worker。CUDA时只锁页collate后的FP16 batch；CPU运行不锁页。传入GPU后只将当前batch转FP32，保留既有BF16/autocast和FP32损失策略。Checkpoint保存的shuffle排列、游标和生成器继续用于恢复。

以140000条、平均80aa计算，纯FP16嵌入约26.70 GiB；这是公式估算，不是本次实测。真实预算由manifest和数组头估算，在约90 GB RAM内仍应人工核对进程峰值和系统余量。

本次只执行 [RAM缓存smoke](reports/cache_ram_logs/smoke.log)，真实全量加载耗时、峰值RSS和训练OS级I/O均未实测；[修改与人工启动命令](reports/CACHE_RAM_OPTIMIZATION.md)。历史PT文件不自动转换或重新抽取，旧训练缓存不能直接交给新入口；阶段05 checkpoint仍可用于阶段06推理，或明确以`--init-checkpoint`只初始化模型权重。

## 级联编码器（阶段 02）

```python
import torch
from templatedf.encoder import CascadeEncoder

# 正式默认：D=1280、N=50、4 层、20 头、fusion hidden=2560。
encoder = CascadeEncoder().eval()
embeddings = torch.randn(1, 10, 1280)
input_mask = torch.ones(1, 10, dtype=torch.bool)
with torch.no_grad():
    latent = encoder(embeddings, input_mask)  # [1,50,1280]
```

编码器将原始 H0 和每次残差更新后的 H1…H4 沿特征维拼接，经逐位置 MLP 融合，
再由 50 个独立 learned queries 做 cross-attention。10 aa 不会先补为 50 aa。
长度由输入张量决定，不使用 Reduction、Repetition 或绑定输入长度的 flatten。
基础模块允许其他正长度；数据/推理入口仍遵守 10–150 aa。

`input_mask` 必须为同设备的 bool `[B,L]`，True 为真实残基，且只允许右 padding，
每条至少一个有效残基。模型不会擅自改变输入设备；调用方应将模型和输入移到同一设备，
并选择一致浮点 dtype 或使用 autocast。

Block 保留 Raygun 的 rotary Transformer（FFN=2D）→右补零卷积（D→D/2→D/4→D/2）
→线性层结构，卷积核为 7/3/7。为保证追加 padding 和合批的一致性，关闭了会随 Lmax
改变 rotary 位置的额外 bias K/V token，并修复零填充 dtype 和各级 padding 清零。
聚合采用 pre-norm cross-attention、残差和 4D FFN；没有逐残基跳连到后续解码器。
原始版权/许可保存在 `licenses/RAYGUN_LICENSE.txt`，改动记录见第三方来源说明。

```bash
# 合成嵌入检查：默认四层参数计数、所有梯度与隔离检查、正式维度一层前向。
OMP_NUM_THREADS=4 python scripts/validate_encoder.py \
  --device cuda:1 --report reports/encoder_validation.json
```

该检查不加载 ESM 权重；正式四层模型尚未做完整训练或容量验证。

## 解码器与自编码器（阶段 03）

```python
import torch
from templatedf.model import ProteinAutoencoder

# 轻量接口示例；不改变正式默认 D1280 / N50 / 4＋4 层。
model = ProteinAutoencoder(dim=64, num_heads=4,
                           encoder_blocks=2, decoder_blocks=2).eval()
embeddings = torch.randn(3, 150, 64)
input_lengths = torch.tensor([10, 50, 150])
input_mask = torch.arange(150)[None, :] < input_lengths[:, None]
with torch.no_grad():
    same_length = model(embeddings, input_mask)  # 默认 Lout=Lin
    result = model(embeddings, input_mask, output_lengths=torch.tensor([10, 80, 150]))
    latent = model.encode(embeddings, input_mask)
    decoded = model.decode(latent, torch.tensor([10, 80, 150]))
```

`forward` 返回 `latent`、`reconstructed_embeddings`、`logits`、`output_mask`、
`output_lengths`；`decode` 返回后四项。上述异长输出的形状分别为
`[3,50,64]`、`[3,150,64]`、`[3,150,20]`、`[3,150]` 和 `[3]`。
`output_mask=True` 为真实输出残基，各条有效数为 10/80/150；输出 padding 嵌入和 logits 均为零。
logits 最后一维对应本项目固定的 20 类 `ACDEFGHIKLMNPQRSTVWY`，不是 ESM token 编号。

解码器只接收 latent 和目标长度，没有原始嵌入、标签或目标 token 的输入。
动态正弦/余弦位置编码与 `log1p(L)/log1p(150)` 的逐条长度 MLP 相加形成 query；
每层进行非因果 self-attention、对 latent 的 cross-attention 和 FFN。
self-attention 屏蔽输出 padding key，所有 latent token 都有效；每个残差和输出头后清零 padding。
位置与长度特征不依赖当前 batch 的最大长度，避免混合 batch 改变单条结果。

`ProteinAutoencoder.encode/decode/forward` 都保留 `condition=None`；任何非 None 值
（包括空 dict 或 0）均抛出 `NotImplementedError`，不会被静默忽略。
AE 入口将输入/输出真实长度限制在 10–150 aa，要求右 padding bool mask；
`decode` 要求 latent 的 N、D 与模型配置一致，长度必须为整型 `[B]` 张量。
长度可从 CPU 显式移动到 latent 所在设备，返回值统一为该设备的 int64。
底层 `SequenceDecoder` 作为通用张量模块允许任意正输出长度，肽范围由 AE 入口执行。

```bash
# 合成输入，一次反向检查；另做 D1280 的一层编码器＋一层解码器前向。
OMP_NUM_THREADS=4 python scripts/validate_autoencoder.py \
  --device cuda:1 --report reports/autoencoder_validation.json
```

不同 Lout 的成功解码只说明接口可用；当前模型没有学会序列重建或模板语义，
不能据此声称变长后保留结构或功能。正式 4＋4 层训练、训练显存预算和实际效果留待后续阶段。

## 1A 缓存训练与恢复（阶段 04）

```bash
python -m templatedf.train --help
```

`--help` 不导入 torch/ESM、不读配置、不下载模型、不启动训练。
实际训练要求明确的累计 `max_steps`、独立 train/validation FASTA 及对应完整缓存。
`configs/train.yaml` 的 `max_steps` 和缓存路径默认是 null；没有预算或缺少验证集时明确报错。
用户提供的 test 文件保持测试用途，CLI 不会自动将其当作 validation。
相同 train/validation 文件被拒绝，存在完全相同序列时也报错并列出重叠信息。

准备好对应文件后，可按本次确定的预算运行（下面的路径是需要替换的示例）：

```bash
python -m templatedf.train --config configs/train.yaml \
  --train-fasta /path/to/train.fasta --validation-fasta /path/to/validation.fasta \
  --train-cache /path/to/train_cache --validation-cache /path/to/validation_cache \
  --cache-spec /path/to/extraction_report.json \
  --device cuda:0 --precision bf16 --max-steps 100 --output-dir artifacts/run_1a
```

`--cache-spec` 是可选来源约束，通常直接读取两个缓存manifest；训练不加载ESM权重。
模型D必须为缓存的1280维；缓存和host batch保持FP16，当前batch传GPU后才转FP32，BF16通过CUDA autocast实现。
不支持原生 BF16 时直接报错，不静默回退。JSONL 日志记录实际输出 dtype 与 autocast 状态。

训练每步固定 Lout=Lin，标签只进入损失，不进入模型。总损失为 CE＋lambda_emb×嵌入 MSE：
CE 只计有效 0..19 类标签，padding=-100；MSE 先沿 D 平均，再按有效残基平均。
日志分别包含 CE、MSE、有效残基准确率与每条序列准确率的平均，验证聚合按实际计数加权。
验证处于 eval/no_grad，随后恢复原 train/eval 模式和 RNG，不干扰训练随机序列。
梯度裁剪默认 1.0，日志中的 gradient_norm_before_clip 是裁剪前范数。

**lambda_emb=0 时没有 ESM 对齐监督**，仍记录 MSE 供观察；连续表示可能只服务于分类。
当前只支持 stage=1A，1B/cycle/diffusion 明确不支持。本阶段不提供 FP16 或 GradScaler；
FP32/BF16 的 checkpoint 中 scaler=null，实际使用的 ExponentialLR 状态会保存。
`scheduler_gamma=1` 表示不使用 scheduler，较小正数表示每优化步按 gamma 衰减。

```bash
# 恢复到累计 200 步：不是额外再跑 200 步。
python -m templatedf.train --config configs/train.yaml \
  --output-dir artifacts/run_1a --resume artifacts/run_1a/last.pt --max-steps 200
```

恢复时应使用相同的实际数据/缓存配置；若首次使用 CLI 路径覆盖，上述命令也需重复相同覆盖，
或使用含相同路径的配置文件。`--init-checkpoint` 与 `--resume` 互斥：前者只加载兼容模型权重，
重新初始化优化器、随机状态、样本顺序和步数；后者恢复全部训练状态。

Checkpoint 包含配置、模型、AdamW 状态、实际 scheduler/scaler 状态、全局步数、AA 顺序、
ESM 来源与数据指纹、Python/NumPy/CPU Torch/当前 CUDA 设备 RNG，以及 shuffle 排列、游标和生成器状态。
采用临时文件＋原子替换，读取使用 weights_only=True。
resume 检查模型、来源、数据与训练设置；只允许改变累计预算及验证/保存间隔。
当前缓存仅启动时读取，采样为单进程无预取，沿用shuffle游标恢复；历史CPU恢复对照属于更改前版本，本次按要求未重跑训练恢复。新checkpoint增加缓存manifest指纹，不直接将历史PT缓存训练状态跨格式resume。

输出目录保存 metrics.jsonl、summary.json、last.pt、cache_loading.json（容量估算和实际加载耗时），以及CLI的数据审核和缓存来源文件。
新训练拒绝覆盖已有日志；resume 可接续原目录或写入新目录。max_steps 达到后保存并停止。
日志以 data_kind 区分 real_cached 与 mock 来源的 synthetic，不能将合成结果视为真实肽效果。

历史训练验收脚本（已适配新格式，但本次未执行）：

```bash
OMP_NUM_THREADS=4 python scripts/validate_training.py \
  --output-root artifacts/new_04_synthetic --report reports/new_training_validation.json \
  --bf16-device cuda:0
```

该脚本固定进行 60 步小模型拟合、3 步连续/恢复对照，并用独立 mock 缓存做累计 2 步 GPU BF16 CLI 验证。
使用新的输出目录，避免覆盖已有证据；不会读取用户的真实 FASTA 进行训练。

## 配置、证据和后续范围

[`configs/train.yaml`](configs/train.yaml) 记录数据路径及公共约定：10–150 aa、1280 维、
50 latent token（不是最短肽长度）、编码器/解码器各 4 层；训练 CLI 读取该 YAML 并支持显式参数覆盖；数据审核与 ESM 抽取 CLI 仍通过参数配置。阶段05已实测正式模型 BF16 batch8，allocated峰值约4.46 GiB；未搜索最大batch。

- [阶段 06 报告](reports/06_TEMPLATE_GENERATION.md)：模板推理接口、5条真实模板演示、采样重复率和原始latent分片导出。
- [阶段 05 报告](reports/05_1A_PILOT.md)：640条真实缓存、正式模型显存基准、300步短训曲线、分桶验证和checkpoint哈希。
- [阶段 04 报告](reports/04_TRAINING_ENGINE.md)：损失分母、合成曲线、断点恢复与 BF16 缓存训练证据。
- [阶段 03 报告](reports/03_DECODER_AND_AE.md)：输出尺寸、latent 依赖、完整梯度与异常输入验证。
- [阶段 02 报告](reports/02_ENCODER.md)：多级数据流、梯度、mask/batch 隔离和正式尺寸前向。
- [阶段 01 报告](reports/01_DATA_AND_ESM.md)：全量数据计数、mock 测试及 3 条真实 ESM 样本证据。
- [阶段 00 报告](reports/00_PROJECT_SETUP.md)：历史环境与骨架验收。
- [第三方来源](THIRD_PARTY_NOTICES.md)、[阶段进度](tasks/STATUS.md)。

阶段05已获用户放行，阶段06完成后待人工审核；阶段07未开始。完整数据集缓存及追加训练须有明确预算。

## 阶段05试训产物

本次实际配置为 [ce_config.yaml](artifacts/05_pilot/ce_config.yaml)，完整checkpoint为 [last.pt](artifacts/05_pilot/ce_run/last.pt)，停在300步。验证CE为2.8858，残基准确率10.38%；训练集组成基线为9.62%，尚不足以支持高质量重建结论。`lambda_emb=0`，没有正权重ESM对齐监督。

- [训练曲线](reports/05_logs/training_curves.png)、[分桶指标CSV](reports/05_logs/validation_by_length.csv)、[运行汇总](artifacts/05_pilot/ce_run/summary.json)。
- `scripts/summarize_pilot.py` 从实测记录重新导出图表和组成基线。
- 历史 `scripts/verify_pilot_checkpoint.py` 依赖阶段05当时的数据缓存；旧PT缓存与当前NPY读取器不兼容，本次未重跑。Checkpoint本身可继续用于阶段06独立模板推理。
- `scripts/run_pilot.py` 保存本次 prepare／extract／benchmark／train 的复现入口；实际命令见阶段05报告。固定产物已存在，脚本拒绝重复抽样、基准及重启训练，以免覆盖证据或无意追加预算。

## 模板重建、候选生成与 latent 导出（阶段06）

使用 stage1A checkpoint 和与训练来源哈希一致的本地 ESM 权重。两个模型均冻结且 eval；ESM计算FP32，其特征按训练缓存dtype做量化回读后进入encoder。AE可选FP32或原生CUDA BF16，来源与实际精度保存在manifest。模板非法字符或长度直接报错，不过滤或改写。

```python
from templatedf.inference import TemplateGenerator

runtime = TemplateGenerator.from_checkpoint(
    "artifacts/05_pilot/ce_run/last.pt", device="cuda:0", precision="bf16"
)
candidates = runtime.generate_from_template(
    "ACDEFGHIKLMNPQRSTVWY", target_length=None,
    num_samples=3, temperature=0.8, seed=42, condition=None,
)
# candidates 是包含 sequence 和模板/checkpoint/长度/采样追溯字段的字典列表。
```

`target_length=None` 保持原长度，显式长度必须为10–150整数；50是latent token数量。
温度0用argmax，重复样本正常；正温度使用softmax(logits/T)并从独立CPU Generator采样，同一次调用的所有模板／样本共享递进随机状态。相同输入、checkpoint、seed、精度、分批方式和运行环境下可复现，不强制去重，不保证跨设备或不同batch布局逐位相同。非空condition明确不支持。
多模板Python入口为 `runtime.iter_generate(records, batch_size=4, ...)`，records为PeptideRecord迭代器；生成次序为模板优先、sample index其次，两者从0计数。

下面的命令是本阶段演示的可复制形式。输出目录必须尚不存在；示例使用新的 `_copy` 目录，避免覆盖已有证据：

```bash
PY=/home/gh/miniconda3/envs/peptide_vae/bin/python
$PY -m templatedf.reconstruct \
  --checkpoint artifacts/05_pilot/ce_run/last.pt \
  --fasta artifacts/06_demo/templates.fasta --limit 5 \
  --device cuda:0 --precision bf16 --batch-size 2 \
  --output-dir artifacts/06_reconstruction_copy

$PY -m templatedf.generate \
  --checkpoint artifacts/05_pilot/ce_run/last.pt \
  --fasta artifacts/06_demo/templates.fasta --limit 5 \
  --target-length 80 --num-samples 3 --temperature 0.8 --seed 626 \
  --device cuda:0 --precision bf16 --batch-size 2 \
  --output-dir artifacts/06_candidates_copy

$PY -m templatedf.export_latents \
  --checkpoint artifacts/05_pilot/ce_run/last.pt \
  --fasta artifacts/06_demo/templates.fasta --limit 5 \
  --device cuda:0 --precision bf16 --batch-size 2 --shard-size 2 \
  --output-dir artifacts/06_latents_copy
```

三个CLI均要求`--limit N`或显式`--all`；帮助页不加载ESM/模型。可用`--esm-checkpoint`和`--regression-checkpoint`重定位本地权重，内容哈希必须匹配AE checkpoint，不联网下载。reconstruct固定原长度、argmax、每模板1条；generate不指定target-length时同样保持原长度。

候选目录包含`candidates.fasta`、`metadata.jsonl`和完成后才发布的`manifest.json`。FASTA ID与元信息的candidate_id连接；元信息包括模板ID/hash、模板索引、原长/目标长、checkpoint路径/hash/step、seed、温度、sample index与候选序列。manifest记录配置、ESM来源、文件hash及`1-unique/count`重复率（模板内与全局分别统计）。不完整目录没有完成manifest；不会覆盖已有目录。

latent目录包含manifest和分片PT。每条保存**原始eval输出 `[N,D]`**，保留实际dtype、序列ID/hash/长度、checkpoint标识、模型/训练配置、AA顺序与ESM来源；不估计均值方差、不做第二阶段标准化。导出内存只保留当前batch和当前分片，记录总数与每片数量必须相符。

```python
import torch
from templatedf.inference_io import iter_latent_records

# 使用同一个runtime；流式逐片验哈希、形状、来源及条目计数。
for item in iter_latent_records("artifacts/06_demo/latents", expected_identity=runtime.identity):
    output = runtime.decode_latents(
        item["latent"].unsqueeze(0), torch.tensor([item["sequence_length"]])
    )
# 完整遍历后验证总记录数；不要将整个大型导出list()到内存。
```

更换checkpoint、encoder、ESM来源或精度时重新导出。当前300步试训模型仅用于接口验证，变长输出未经过1B训练，也没有结构／功能保持证据。
