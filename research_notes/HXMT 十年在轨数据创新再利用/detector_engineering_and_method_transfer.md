# Insight-HXMT 十年在轨数据再利用：(A) 探测器/工程科学 与 (B) 跨学科方法迁移 — 研究笔记

> **检索说明（请先读）**
> - 检索日期 2026-11-17；文献范围到 2026-11。共完成 36 次 WebSearch + 23 次 WebFetch；**最后一轮 9 个查新检索因本轮会话的 WebSearch 配额（200 次/轮，多个 agent 共享）耗尽而未执行**，受影响的条目在各节 Gaps 中逐条列出。
> - 作者署名只在本次检索确认过的条目上给出；其余条目只给论文标题与链接。
> - 标签约定：**[confirmed in literature]** = 本次检索到原文/摘要并给出 URL；**[inference]** = 由已检索事实推出；**[speculation]** = 物理上合理但无文献支撑；**【未检索核实】** = 仅凭记忆、本次未取回原文，引用前必须复核。
> - 多数论文细节来自 arXiv 摘要页或 ar5iv 全文经小模型摘录，**具体数值在写入正式报告前应对照 PDF 复核**。
> - 本地仓库 `/home/user/NeuBacH`（用户的 NeuBacH 本底模型代码）被用作"公开 Level-1 产品里到底有哪些 HK 列"的一手证据。
>
> **入选想法一览（共 14 个；评分 1–5 = 新颖性 / 可行性 / 潜在影响）**
>
> | 编号 | 想法 | 新 | 可 | 影 | HXMT 是否比别的任务更合适 |
> |---|---|---|---|---|---|
> | A1 | 96 片 SCD + 1728 路 Si-PIN 的在轨损伤曲线 → 经验 NIEL 损伤律（屏蔽/温度"自然实验"） | 3 | 4 | 3 | SCD：是（在轨最大 SCD 阵列）；Si-PIN：一般 |
> | A2 | 把每次 SAA 穿越当作"脉冲束流活化实验"：(E, τ) 活化谱图 → Geant4/FLUKA 基准 + 余辉检验 | 3 | 4 | 4 | 是（大面积 NaI/CsI + PSD + 盲探测器 + 20 MeV 质子监测器） |
> | A3 | 43°/550 km 轨道"本底遗产图集" + 向中国空间站轨道的外推仿真器 | 2 | 5 | 3 | 是（43° 倾角的 X 射线天文台数据稀缺） |
> | A4 | 回溯式俘获质子监测：PM + ACD + 活化"质子剂量计"验证 AP9/IRENE、SAA 漂移与太阳周调制 | 2 | 5 | 3 | 一般（积分计数器，无粒子鉴别；但轨道位置有价值） |
> | A5 | 被反符合丢弃的事例 = 粗粒度"描迹器"：ACD 命中图样 + Pulse_Width 的带电粒子分类 | 4 | 3 | 2 | 否（做不了 Timepix 式 LET 谱；只能做粗分类） |
> | A6 | 用飞行数据做"反事实仪器设计"：ACD 效用审计、盲探测器比例、各 HK 通道的信息价值 | 4 | 4 | 4 | 是（ACD 标志保留在事例里、三种视场 + 盲探测器） |
> | A7 | 吸湿闪烁体在真空中的"自愈"：NaI/CsI 光产额与 PSD 十年动力学作为材料学数据集 | 4 | 4 | 2 | 是（18 个模块、AGC 高压史 + 手动 LGM 高压史） |
> | B1 | 粒子物理范式：盲探测器作控制区的 ABCD 闭合检验 + 盲分析 + 带 look-elsewhere 修正的谱线扫描 + CWoLa 事例分类 | 3 | 4 | 3 | 方法上合适；暗物质谱线灵敏度不占优 |
> | B2 | 地震台阵处理：18 块 ACD/CsI 当"台阵"，STA/LTA + beam-forming 建十年电子沉降/TEB 目录并测投掷角 | 3 | 4 | 4 | 与 GECAM 互补（43° 轨道、三层能段的"射程望远镜"） |
> | B3 | 气候学 EOF 模态分解 + detection-and-attribution（最优指纹法）定量归因本底十年演化 | 4 | 4 | 3 | 是（数据时空覆盖规则、物理模态少且可模拟） |
> | B4 | 地球遮挡数据当信号：准直器扫过地球盘面，测大气硬 X 射线反照的中心—临边角分布与沉降轫致辐射 | 4 | 3 | 3 | 是（1.1° 准直视场，宽视场仪器只能测盘面积分） |
> | B5 | 气象卫星 SNO 式"同时交会"交叉定标：HXMT–Fermi–GECAM–SVOM–EP–CSES 组成分布式 LEO 辐射观测网 | 4 | 3 | 3 | 是网络的一个节点；单独不占优 |
> | B6 | 中子堆方法：地磁截止刚度当磁谱仪求"本底产额函数"、东西效应、Forbush 下降历元叠加、磁暴截止压低 | 3 | 4 | 3 | 宇宙线物理不占优；用于本底可迁移性则独特 |
> | B7 | 神经本底模型残差流 → 公开"未解释本底偏移"目录（time-slide 显著性）+ 从模型中提取衰变核 + 域适应 | 3 | 4 | 4 | 是（用户已有模型；HXMT 上未见同类工作） |

---

## Q0. 任务给定参数核对与公开数据可得性

### Takeaway
任务描述中的仪器参数大体正确，但有三处需要修正：(1) LE 定标能段是 1–10 keV（2020-06-25 之后建议只用 2–10 keV），ME 虽标称 5–30 keV 但科学产品只发布 >10 keV；(2) **eXTP 已不再是低倾角近地轨道，而是大椭圆轨道**，因此"HXMT 本底图集直接服务 eXTP"的前提已过时；(3) SAA 穿越次数不是"数千次"而是约 3×10⁴ 次量级。公开 Level-1 数据里确实包含逐探测器的高压、温度、粒子监测器、反符合板计数，以及 HE 事例级的 Pulse_Width 和 18 位 ACD 标志。

### Cited Findings
- 发射日期 2017-06-15，轨道高度 550 km、倾角 43° [confirmed in literature] — [Li X.B. et al. 2023, LE 5 年性能](https://arxiv.org/abs/2302.10714)
- LE：96 片 SCD；定标能段 1–10 keV；增益不确定度 <20 eV（2–9 keV）、分辨率不确定度 <15 eV；每月用 Cas A 与空天区定标；空天区谱中用 Ni、Cu、Zn 荧光线；工作温度约 −55 至 −38 °C（平均 −47.9 °C），每轨摆动约 5 °C，被动辐射制冷；每片 CCD 每 32 ms 记录一次 forced-trigger 噪声事例；只用单事例（split 事例在相邻读出周期）；大多数 CCD 阈值已抬到约 450 道，1–1.7 keV 的效率在 2020-06-25 后不稳定，建议此后用 2–10 keV；Table 1 列出 7 片坏 CCD（2017-06 至 2020-06），原因未说明 [confirmed in literature] — [LE 5 年性能 (ar5iv 全文)](https://ar5iv.labs.arxiv.org/html/2302.10714)
- ME：1728 个 Si-PIN 像素、54 片 ASIC、总面积 952 cm²；标称 5–30 keV，但只发布 >10 keV 数据；5 年后 298 像素因噪声过大关闭、91 像素能谱不可靠、20 像素仅作定标，1319 像素（742 cm²，约 78%）正常；每事例死时间 255 μs [confirmed in literature] — [Tan Y. et al. 2023, ME 5 年性能](https://arxiv.org/abs/2303.00339)，[ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2303.00339)
- HE：18 个 NaI(Tl)/CsI(Na) phoswich；NaI 20–250 keV；CsI 正常模式 (NOM) 40–800 keV、低增益模式 (LGM) 约 0.2–3 MeV；SAA 模式下主探测器高压关闭 [confirmed in literature] — [Li X.F. et al. 2023, HE 5 年性能](https://arxiv.org/abs/2302.03859)，[ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2302.03859)
- 粒子监测器：3 个 HPM，塑料闪烁体 + 小 PMT，窗阈值电子 1 MeV / 质子 20 MeV（模拟给出电子 >1.5 MeV）；计数率连续 3 s 超阈值即触发 HE 关高压；HPM 数据也用来画 SAA 区域 [confirmed in literature] — [Lu X.F. et al., Design and Calibration of the High Energy Particle Monitor onboard the Insight-HXMT](https://arxiv.org/abs/1911.01594)，[The High Energy X-ray telescope (HE) onboard the Insight-HXMT astronomy satellite](https://arxiv.org/abs/1910.04955)
- HE 事例文件含 `Pulse_Width`、`ACD`（18 个布尔标志，对应 18 块反符合探测器）、`Event_Type` 列；常规分析要求全部 ACD=0，即**被反符合的事例是保留在公开事例文件中的** [confirmed in literature] — [Zenodo: Supporting Information for "First Results of TGF observations by Insight-HXMT"](https://zenodo.org/records/10976454)
- HE 脉冲宽度窗：NOM NaI 54–70 道、NOM CsI 80–120 道、LGM CsI 78–120 道 [confirmed in literature] — [HE 5 年性能 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.03859)
- 公开 L1P 产品包含 `HE-HV`（18 路 PHODet 高压）、`HE-TH`（18 路 PHODet 与 3 路 PM 温度）、`HE-PM`（Cnt_PM_0..2）、`HE-Cnts`（Cnt_VetoDet_0..17 反符合板计数）、`LE-TH`、`EHK`（COR、T_SAA、TN_SAA、SUNSHINE、ELV、DYE_ELV 等）、`Att`、`Orbit`；标准 HE GTI 条件为 `ELV>10&&COR>8&&SAA_FLAG==0&&TN_SAA>300&&T_SAA>300&&ANG_DIST<=0.04`，用户的训练管线已放宽到 `ELV>0&&COR>0&&T_SAA>0`；LE 盲探测器 DetID 为小视场 13/45/77、带定标源的大视场 21/53/85 [confirmed：本地代码] — 本地文件 `/home/user/NeuBacH/hxmt_prepare_dataset.py`、`/home/user/NeuBacH/hxmt_filtering.py`
- eXTP 现行方案：大椭圆轨道，近地点约 5000 km、远地点 ≥100,000 km、倾角约 28.5°，计划 2030 年发射 [confirmed in literature] — [The enhanced X-ray Timing and Polarimetry mission – eXTP for launch in 2030](https://arxiv.org/abs/2506.08101)；另一篇给出远地点 165,500 km，与前者不一致 — [eXTP WFM 论文](https://arxiv.org/abs/2411.03050)
- POLAR-2 已获批 2027 年发射到中国空间站 [confirmed in literature] — [arXiv:2509.06601](https://arxiv.org/abs/2509.06601)；HERD 计划 2027 年发射 — [HEASARC HERD 页面](https://heasarc.gsfc.nasa.gov/docs/heasarc/missions/herd.html)
- 用户的 NeuBacH（Spec-BERT v1.0.0）：用秒级遥测（ELV、COR、卫星位置、反符合计数、PM 计数等）预测 HE NaI 25–200 keV 本底能谱；Zenodo 页未给精度指标，也未链接论文 [confirmed in literature] — [Zenodo 17476734](https://zenodo.org/records/17476734)
- HXMT 团队关于长期演化的已发表工作：RDTM 2023 专刊的四篇"前 5 年"论文（LE、ME、HE、本底）[confirmed in literature] — [LE](https://arxiv.org/abs/2302.10714)、[ME](https://arxiv.org/abs/2303.00339)、[HE](https://arxiv.org/abs/2302.03859)、[本底](https://arxiv.org/abs/2302.14459)

### Inferences
- 2017-06-15 到 2026-11 为 9.4 年（约 3430 天）。轨道周期约 95 min、每天约 15 圈，43° 倾角下每天约 8–9 圈穿过 SAA，累计 **约 3×10⁴ 次 SAA 穿越**，不是"数千次" [inference]。
- 既然 ACD 标志、Pulse_Width、逐模块 HV/温度/反符合板计数都在公开 L1 里，A2、A5、A6、B2、B6 所需的 HE 数据原则上都公开 [inference]。
- LE 的 forced-trigger 事例由 HXMTDAS 在定标步骤里使用（每秒平均后扣除），因此应存在于 1L 事例文件中 [inference，基于 LE 5 年论文的描述]。

### Gaps
- HE 总面积约 5000 cm²、LE 384 cm²、HE 15 小视场 + 2 大视场 + 1 盲模块、18 块 ACD 的 6 顶 + 12 侧布局：【未检索核实】。
- 空间环境监测器 (SEM) 的数据内容及是否公开：未找到资料。
- ME、LE 事例文件里是否有多重性/grade/溢出标志列，ME 是否记录超量程事例：未核实（只确认 ME 记录 pseudo-events，LE 记录 forced-trigger 与 split）。
- `Event_Type` 的取值含义：Zenodo SI 说 0=正常增益、1=低增益，而本地 `hescreen eventtype=1` 用于常规 NaI 分析，两者可能不是同一个量，需查 HXMT 用户手册。
- HXMT 在 2023–2026 年的运行状态、是否有 >5 年的定标/本底更新论文：未检索到（5 年论文只说"预期还能健康运行数年"）。
- Li X.B. et al. 2020（JHEAp，在轨定标）与 Xiao et al. 2020（HE 死时间）：【未检索核实】。
- GECAM（约 600 km/29°）、SVOM（约 625 km/30°）、EP（约 600 km/29°）、中国空间站（41.5°、约 390 km）的轨道参数：【未检索核实】，检索配额耗尽。

---

## A-Q1. 在轨辐照损伤与老化本身作为数据集（想法 A1、A7）

### Takeaway
HXMT 团队已把 LE/ME/HE 的增益、分辨率、噪声演化做成了定标用的经验模型，但都停留在"现象学 + 定性归因"，没有把它们变成以注量为自变量的损伤律，也没有与地面辐照预言对比。LE 的 96 片同型号 SCD 是迄今在轨最大的 SCD 样本，是这里最有价值的部分；HE 则给出一个意外结果——闪烁体不是变差而是变好。

### Cited Findings
- LE 增益下降被归因于辐照损伤导致的电荷转移效率降低，"空间辐射使噪声水平升高"；谱线展宽用时间 × 温度的二维模型描述，残差 <15 eV；FWHM 逐年增大但最近两年增速放缓；噪声峰在前 200 天漂移 <50 道、1500 天后约 250 道 [confirmed in literature] — [LE 5 年性能 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.10714)
- LE 团队发射前估计 4 年任务期的质子剂量为 10 MeV 等效注量 3×10⁸ p/cm²，并于 2012 年在 CIAE HI-13 加速器上做了室温 10 MeV 质子辐照，结论是辐照同时恶化暗电流与 CTI、降低工作温度可改善性能 [confirmed in literature] — [Proton irradiation effect on SCDs](https://arxiv.org/abs/1404.4931)；辐照后 SCD 的 CTI 测量 — [arXiv:1308.1698](https://arxiv.org/abs/1308.1698)
- Gow et al. 将 CCD234/236 辐照到 3.0×10⁸ p/cm²（10 MeV 等效）并报告比上一代改善约 2 倍 [confirmed in literature] — [Proton radiation damage study of the next generation of swept charge devices](https://oro.open.ac.uk/19851)
- CCD236 同时用于 Chandrayaan-2 CLASS 和 HXMT [confirmed in literature：出自 SCD CTI 系列论文之一，检索摘要未指明具体哪篇] — [Charge transfer inefficiency in the pre- and post-irradiated Swept Charge Device CCD236](https://oro.open.ac.uk/51482)，[arXiv:1308.1698](https://arxiv.org/abs/1308.1698)
- 低温（约 −35 °C）辐照后不同缺陷退火程度不同，作者建议要准确预言在轨损伤需同时做室温与低温辐照 [confirmed in literature] — [Proton irradiation of a swept charge device at cryogenic temperature and the subsequent annealing](https://oro.open.ac.uk/43518)
- CLASS 的在轨噪声数据只见于会议海报（2019 与 2021 年噪声差别不大）[confirmed in literature] — [L. Jones poster, PSD12](https://indico.global/event/5677/contributions/46993/attachments/23096/38950/L_Jones_Poster_PSD12_item_103.pdf)
- ME：17.8 keV 处 FWHM 从 3 keV 增到 3.14 keV；标称 FWHM 在第 110–1814 天增长 2.35%，其中 1.43% 来自增益上升（归因于读出电子学增益上升），真实退化约 0.91%；像素损失归因于"污染物沉积与辐照损伤"导致漏电流上升 [confirmed in literature] — [ME 5 年性能 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2303.00339)
- HE：NaI 在 59.5 keV 的分辨率稳定且略有改善；CsI 增益持续上升，LGM 高压在 2020-06 与 2021-10 被调低；NOM 下 AGC 把 59.5 keV 峰锁在第 50 道，多数模块高压五年来下降；脉冲宽度峰变窄（NaI 0.9–2.8%/yr，CsI 1–2.7%/yr），PSD 品质因数每年改善 1.2–2.8%，在轨 FoM 2–2.7；作者"推测"原因是发射前潮解在真空中恢复，使荧光产额与收集效率提高；DetID 2 行为反常且未被理解 [confirmed in literature] — [HE 5 年性能 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.03859)
- Gaia 的在轨经验：外推到任务末期的 CTI 效应比发射前预言低约一个量级 [confirmed in literature] — [Crowley et al. 2016, On-orbit performance of the Gaia CCDs at L2](https://arxiv.org/abs/1609.04240)

### Inferences

#### 想法 A1：SCD 与 Si-PIN 的在轨损伤曲线 → 经验 NIEL 损伤律
1. **一句话**：把 LE forced-trigger 噪声、增益、线宽和 ME 像素"死亡"记录，从定标副产品改写成以位移损伤剂量为自变量的损伤律，并与地面辐照预言逐项对比。
2. **原理与来源**：器件物理的 NIEL 标度假设（体损伤 ∝ 非电离能损）；榜样是 Gaia/Chandra/Swift/XMM 的在轨 CTI 史与 Gow et al. 的 SCD 地面辐照系列。关键可检验命题：地面室温辐照给出的"3×10⁸ p/cm² 对应多少 CTI/暗电流"是否与在轨一致（Gaia 的答案是差一个量级；低温辐照论文预言室温试验会偏）。
3. **HXMT 特性与诚实评价**：(a) 96 片同型号 CCD236，分属小视场、大视场、盲三类，准直器与遮挡不同 → 质子注量不同，构成**器件间剂量—响应的内部对照**；(b) 每轨 5 °C 温度摆动 + 季节变化 → 可用 Arrhenius 关系把暗电流（体损伤）与 CTI（陷阱）分开；(c) PM（>20 MeV 质子）给出逐次 SAA 的实测注量史，不必依赖 AP8。对 SCD，HXMT 明显优于 C1XS（寿命短）和 CLASS（月球轨道、只有会议海报）。对 Si-PIN，论文自己把像素损失归于"污染 + 辐照"两个混杂因素，HXMT 并不比专门的辐照试验更干净。
4. **数据**：LE 1L 事例（含 forced-trigger）、`LE-TH`、CALDB 增益/分辨率历史；ME 的 ²⁴¹Am 定标像素事例与坏像素表；`HE-PM`。HK 与 CALDB 公开；forced-trigger 是否在公开 1L 中为 [inference]；ME 逐像素关闭时间表是否公开未知。
5. **查新**：HXMT 5 年论文只给现象学模型，未见 NIEL 标度或与地面辐照对比的工作；CLASS 在轨结果只有海报。Chandra ACIS/Swift XRT/XMM EPIC 的 CTI 史本次未能检索（配额耗尽）。
6. **可行性**：信号很强（噪声峰 1500 天漂 250 道）。主要系统误差是绝对注量（AP8/AP9 在 550 km 的不确定度约 2 倍量级【未检索核实】），因此**相对量**（盲 vs 开口、不同温度）才是稳健产出；简单线性外推 9.4 年约 7×10⁸ p/cm²（10 MeV 等效）[inference]。阈值在 2020 年被整体抬高，会截断低能端，需要在分析中显式建模。ME 部分建议做成生存分析（Kaplan–Meier/Weibull 风险率 vs 模块位置与温度）。
7. **评分**：新颖 3 / 可行 4 / 影响 3 —— 数据现成且样本独一无二，但 eXTP、EP 都不用 SCD，直接受益者主要是行星 XRF 仪器。

#### 想法 A7：吸湿闪烁体在真空中的"自愈"动力学
1. **一句话**：把 HE 的 AGC 高压史、LGM 手动高压史、CsI 本底线位置和脉冲宽度演化，当作 18 块 NaI(Tl)/CsI(Na) 晶体在真空中十年光产额与发光动力学变化的材料学数据集，检验"潮解恢复"假说。
2. **原理与来源**：HE 5 年论文自己提出但未检验的假说（"we suppose"）。可检验预言：扩散限制的脱水应给出 1−exp(−t/τ) 或 √t 型饱和曲线，τ 随模块的封装与地面存放史不同；PMT 老化通常使增益**下降**、辐照色心使光产额**下降**，符号都与观测相反，因此净趋势可给出辐照致光输出损失的上限 [inference]。
3. **HXMT 特性与诚实评价**：18 个模块、NaI 用 ²⁴¹Am 闭环（高压即观测量）、CsI 开环（峰位即观测量）、同时有 PSD 脉冲宽度 → 可以区分"光产额变了"与"发光时间常数变了"。这种组合在 Fermi-GBM 等任务上没有（GBM 无 phoswich PSD）。
4. **数据**：`HE-HV`、`HE-TH`、HE 事例的 Channel 与 Pulse_Width、CALDB 增益表。公开。
5. **查新**：除 HE 5 年论文的定性描述外未找到专门分析；其他任务 NaI/CsI 长期增益史的对比文献本次未检索。
6. **可行性**：数据量充足，分析简单。障碍是 AGC 把多种效应折叠进一个高压数，需要用 LGM 段与 191 keV 等本底线交叉约束；DetID 2 的反常可作为对照或单独个案。精度：59.5 keV 峰位稳定在 0.1% 量级，足以拟合百分之几每年的趋势。
7. **评分**：新颖 4 / 可行 4 / 影响 2 —— 新但小众，对使用吸湿晶体的后续任务（NaI、LaBr₃、CsI 量能器）有参考价值。

### Gaps
- LE 逐片 CCD 的屏蔽差异（质子注量差多少）需要质量模型，未公开。
- 没有找到 HXMT 团队发表的在轨 CTI 数值或"每单位注量 CTI 增量"。
- Chandra ACIS、Swift XRT、XMM EPIC 的 CTI 史与太阳质子事件对应关系：检索未执行。
- NaI/CsI 在 LEO 十年累计剂量（估计 0.1–1 krad 量级）与辐照致光输出损失阈值：【未检索核实】。

---

## A-Q2. 活化物理（想法 A2）

### Takeaway
HXMT 团队已有发射前模拟、两年在轨对比和一个盲探测器参数化模型，但公开文献里没有"按同位素分解"的活化产额测量，也没有给出本底线的同位素认证。把约 3×10⁴ 次 SAA 穿越当作重复的脉冲辐照实验，在 (能量, 衰变时间常数) 平面上做反演，是统计上绰绰有余、系统误差可控的基准测量。

### Cited Findings
- 发射前 Geant4 质量模型：HE 在 20–250 keV 平均本底约 540 cts/s（在轨 100 天后），SAA 延迟成分约占 47%；每次穿越后延迟本底在最初几分钟快速下降，累积成分在前约 100 天快速上升、约 1 年后接近稳态 [confirmed in literature] — [Xie F. et al. 2015, Simulation of the In-flight Background for HXMT/HE](https://arxiv.org/abs/1511.02997)
- 在轨对比：HE 模拟本底率在两年后与观测偏差约 5%，总谱和约 67 keV 线的相对丰度在模拟与观测中都长期上升；ME 偏差约 15% 且无明显长期上升；LE 观测到的约 7.5 与 8.0 keV 线在模拟中不明显，作者归因于 LE 结构建模不确定 [confirmed in literature] — [Zhang J. et al. 2020, Comparison of simulated backgrounds with in-orbit observations](https://arxiv.org/abs/2009.14489)
- HE 本底由轨道粒子贡献、以长时标活化为主；本底模型把长期本底图与盲探测器数据结合 [confirmed in literature] — [Liao J.Y. et al. 2020, HE 本底模型](https://arxiv.org/abs/2005.01661)
- 已有一个针对 HE 盲探测器的物理参数化模型，用遗传算法 + MCMC 拟合 [confirmed in literature] — [You Y. et al. 2021, ApJS 256, 47](https://ui.adsabs.harvard.edu/abs/2021ApJS..256...47Y/abstract)
- HE NaI 用于定标的在轨碘活化线为 31、56、67、191 keV（67 keV 鼓包第一年增长后趋于平衡）；CsI 用于定标的本底线约 67、191、478、692、1182、1372、1729、2767 keV；**论文没有指明对应同位素** [confirmed in literature] — [HE 5 年性能 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.03859)
- 5 年本底：HE 本底第一年急升、之后趋于饱和；在同一地理位置，升轨段（刚过 SAA）本底高于降轨段，因为短寿命活化尚未衰变；ME 有一个在 SAA 等高通量区之后累积的弱延迟成分 [confirmed in literature] — [Liao J.Y. et al. 2023, Five-year in-orbit background (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.14459)
- 其他任务的对照：Hitomi HXI 的质子活化 Geant4 框架与在轨数据"符合很好" — [Odaka et al. 2018](https://arxiv.org/abs/1804.00827)；Suzaku HXD 的 NXB 模型再现性 PIN <3%、GSO <1% — [Fukazawa et al. 2009](https://arxiv.org/abs/0901.0419)；Fermi-GBM 物理本底模型 — [Biltzinger et al. 2020](https://arxiv.org/abs/2005.11219)；GBM 的 SAA 活化在 BGO 中尤其明显 — [The Fermi Gamma-Ray Burst Monitor](https://arxiv.org/abs/0908.0450)，并会在轨道本底扣除法中造成系统偏移 — [Uncovering low-level Fermi/GBM emission using orbital background subtraction](https://arxiv.org/abs/1111.3779)
- 验证 Geant4 放射性衰变实现的测量"很少或只覆盖有限同位素"，INTEGRAL 数据提示长期活化可占仪器本底的 20% [confirmed in literature] — [Progress and Validation of Geant4 Based Radioactive Decay Simulation (Simbol-X/IXO)](https://arxiv.org/abs/0912.0233)

### Inferences

#### 想法 A2：SAA 穿越 = 脉冲束流活化实验，(E, τ) 活化谱图
1. **一句话**：以 PM 实测的逐次 SAA 质子注量为"束流记录"，把 HE NaI/CsI 逐道计数率写成注量史与一组指数核的卷积，反演出每个能道的衰变时间常数谱，从而按 (线能量, 半衰期) 认证同位素并测出每入射质子的产额。
2. **原理与来源**：核物理的活化分析 + 时间分辨衰变谱学；Odaka et al. 的"产生概率 → 延迟发射"半解析框架。模型：R(E,t) = prompt(E, COR) + Σ_k A_k(E) Σ_j Φ_j exp[−(t−t_j)/τ_k]，其中 Φ_j 取自 PM；τ 谱用非负约束的 Laplace 反演求解。可检验产出：Geant4 不同物理列表（BERT/BIC/INCL）与 FLUKA 对 p+I、p+Cs、p+Na 在约 50–400 MeV 的同位素产额谁对。
3. **HXMT 特性与诚实评价**：(a) 43° 倾角比 Fermi（约 25.6°【未检索核实】）穿 SAA 更深；(b) phoswich 的 Pulse_Width 可把 NaI 与 CsI 的活化分开，CsI 低增益模式覆盖到约 3 MeV；(c) 盲模块给出无天空贡献的纯仪器谱；(d) SAA 内高压关闭，开机后是"干净的衰变段"；(e) 18 个模块可互检。就 NaI/CsI 活化而言 HXMT 确实比 GBM 更合适；但 GBM 有 17 年连续公开数据，适合作为第二数据集而不是竞争者。
4. **数据**：HE 1L 事例（含 T_SAA<300 s 的被弃数据）、`HE-PM`、`EHK`、`HE-HV`；全部公开。用户管线已保留 T_SAA>0 的数据。
5. **查新**：已有 Xie 2015、Zhang J. 2020、Liao 2020、You 2021；**未找到**按同位素分解的在轨产额测量或本底线同位素认证。SPI（Weidenspointner 等）与 Suzaku HXD（Kokubun 等）的同位素级工作本次未取回。
6. **可行性**：统计量不是问题——若出 SAA 后 30 min 内的短寿命超出为 100 cts/s 量级（[inference]，由 540 cts/s × 47% 推），单次穿越约 2×10⁵ 计数，叠加 10⁴ 次为 10⁹ 量级。限制来自系统误差：(i) prompt 成分随 COR 变化必须同时拟合；(ii) 时间常数相差不到约 2–3 倍的成分难以分辨，需靠线能量辅助；(iii) NaI 增益稳定在 0.1–1%，CsI 需逐月定标；(iv) PM 不分电子/质子，SAA 边缘的电子污染会偏置 Φ_j。附带检验：**闪烁体余辉/磷光**应表现为幂律（t⁻¹ 型）而非指数衰减、集中在最低能道且 Pulse_Width 落在正常窗外，可与活化区分 [speculation，未检索到 HXMT 相关文献]。
7. **评分**：新颖 3 / 可行 4 / 影响 4 —— 团队已做过总量级对比，但同位素级基准对 CsI/NaI 类 MeV 任务（立方星伽马监测器、量能器）的本底预言是稀缺输入。

### Gaps
- 31/56/67/191 keV 及 CsI 各线的同位素归属：文献未给，本笔记不臆测。
- You et al. 2021 模型里是否已包含显式的多指数衰变核：只读到 ADS 摘要级信息。
- HE 开机后最初几十秒数据是否稳定可用（高压爬升过程）：未知。
- COSI、AMEGO-X、POLAR-2 等的活化模拟现状：检索未覆盖。

---

## A-Q3. 本底作为设计数据库与对未来任务的教训（想法 A3、A6）

### Takeaway
"五年本底"论文给出了地理分布与逐年演化的定性描述，但没有发布面向任务设计的多维图集，也没有回答"反符合到底值多少""需要几个盲探测器"这类设计问题。43° 轨道的受益对象是中国空间站载荷（POLAR-2、HERD），而不是已改为大椭圆轨道的 eXTP。

### Cited Findings
- 5 年本底论文覆盖光变、能谱、地理分布与长期演化，结论是本底模型对最新观测仍有效；系统误差：ME 10–40 keV 约 1.6%（10–15 keV 约 2%），HE 25–250 keV 各年均 <3%，LE 2–10 keV 相对前两年无显著变化；PM 计数率五年基本不变，磁赤道附近约 1.7–1.8 cts/s；**文中未提太阳周调制与 SAA 漂移** [confirmed in literature] — [Five-year in-orbit background (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.14459)
- LE 高能段本底受地磁场调制，ME 本底与截止刚度反相关；ME 在高纬有粒子事件峰被 GTI 剔除；LE 会被低能带电粒子和可见光淹没甚至存储饱和，LE "flare" 的流量大致正比于视场大小 [confirmed in literature] — [同上](https://ar5iv.labs.arxiv.org/html/2302.14459)
- ME 本底模型按 5°×5° 地理网格建立，8.9–44.0 keV 系统误差约 1.3% [confirmed in literature] — [Guo C.C. et al. 2020](https://arxiv.org/abs/2003.06260)
- 风格上的先例：用 BeppoSAX 粒子监测器 1996–2002 年数据刻画约 4° 倾角 LEO 的辐射环境（强度图、SAA 穿越时间与注量、西漂、随高度变化）[confirmed in literature] — [Campana et al. 2014, The radiation environment in a Low Earth Orbit: the case of BeppoSAX](https://arxiv.org/abs/1405.0360)
- NICER 上已有 transformer 本底模型 BERTground，输入为航天器与望远镜元数据，作者称多数指标优于现役模型 — [BERTground (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/30335)；但独立比较发现 NICER 的 ML 与 Space Weather 估计器缺乏内禀的离群过滤 — [AGN STORM 2. III](https://arxiv.org/abs/2302.12896)
- GRB 221009A 期间 HE/CsI 在超高计数率时段数据饱和，PM 不受影响；HE 正常事例死时间 4 μs、溢出事例 ≥69 μs — [Insight-HXMT observations of the extremely bright GRB 221009A](https://arxiv.org/abs/2504.18952)，[Insight-HXMT and GECAM-C observations of the brightest-of-all-time GRB 221009A](https://arxiv.org/abs/2303.01203)；5 年论文则给出每个 NaI 事例最大死时间约 8 μs、累计死时间比例约 2–6% — [HE 5 年性能 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.03859)

### Inferences

#### 想法 A3：43°/550 km "本底遗产图集" + 轨道外推仿真器
1. **一句话**：把 LE/ME/HE-NaI/HE-CsI/PM/ACD 的本底谱整理成以 (L 或 COR、经纬度、高度、出 SAA 时间、累计 SAA 剂量、太阳周相位、指向相对地球/地磁场的角度) 为坐标的公开数据立方，并附一个可查询的仿真器。
2. **原理与来源**：Campana et al. 2014 用 BeppoSAX 数据为 LOFT 做的环境刻画；RXTE/Suzaku/NuSTAR 的本底遗产研究。可检验点：用 A2 的活化核 + B6 的产额函数，把 HXMT 数据外推到 41.5° 轨道，再用 POLAR-2 入轨后的首批数据做盲检验。
3. **HXMT 特性与诚实评价**：43° 倾角在 X 射线天文台中罕见（多数 <31° 或 51.6°【未检索核实】），与中国空间站接近；但空间站更低，SAA 通量随高度变化很陡，高度外推是主要不确定度。对 eXTP（大椭圆轨道）基本不适用。
4. **数据**：用户训练集的超集；全部来自公开 L1 + EHK。
5. **查新**：5 年本底论文只有逐年比较与地理图；未找到公开的多维图集产品。
6. **可行性**：几乎是 NeuBacH 数据准备流程的副产品。真正的工作量在"可迁移性"的物理论证，而不是出图。
7. **评分**：新颖 2 / 可行 5 / 影响 3 —— 不新但有用，且发表门槛低。

#### 想法 A6：用飞行数据做"反事实仪器设计"
1. **一句话**：利用"被反符合事例仍在事例文件里""有盲探测器和三种视场"这两个事实，定量回答三个设计问题：主动屏蔽把本底降低了多少、需要多少盲探测器、哪些 HK 通道对本底预测真正有信息。
2. **原理与来源**：因果推断中的反事实/消融实验；LHC 的 trigger 效率"tag-and-probe"思想。三个具体分析：(a) **ACD 效用审计**——同一批事例分别按"ACD 全 0"与"忽略 ACD"成谱，得到本底抑制因子随能量与 COR 的函数，并扣除 ACD 引入的死时间代价；(b) **盲探测器比例**——对 NeuBacH 做输入消融（去掉盲探测器谱/去掉 ACD 计数/去掉 PM），比较残差的系统误差地板，再用子采样模拟"只有 1/2、1/4 盲面积"的情形；(c) **材料线清单**——LE 的 Ni/Cu/Zn、ME 的 Ag、HE 的活化线，逐条给出强度与来源，标出模拟未重现的线。
3. **HXMT 特性与诚实评价**：ACD 标志逐事例保留、盲探测器与开口探测器同构，这在准直型望远镜中很少见；HXMT 很适合。局限：结论严格只对 HXMT 的几何与轨道成立，外推到别的设计要靠模拟。
4. **数据**：HE 1L 事例（ACD 列）、`HE-Cnts`、`HE-DTime`【未检索核实其公开文件名】、NeuBacH 训练集。公开。
5. **查新**：未找到 HXMT 的 ACD 效用定量评估或盲探测器比例优化的论文；NuSTAR 等任务关于主动屏蔽效用的讨论本次未检索。
6. **可行性**：(a) 是一两周的工作量，统计误差可忽略；系统误差在于 ACD 触发的事例里既有被否决的本底也有偶然符合丢失的真光子，需要用偶然符合率修正。(b) 需要重训模型若干次。预期精度：抑制因子可测到百分之几；信息价值排序是稳健的定性结论。
7. **评分**：新颖 4 / 可行 4 / 影响 4 —— 直接可写成"lessons learned"，对后续准直型/监测型载荷的设计权衡有具体数字。

### Gaps
- 5 年本底论文未给出各年本底增幅的百分数和空天区总曝光。
- ACD 板在 SAA 模式下是否加电未知，影响 A4/B2。
- 主动屏蔽效用在其他任务上的定量文献未检索。
- 死时间数值在两篇论文中不一致（4 μs vs 约 8 μs），可能是定义不同，需查 Xiao et al. 2020。

---

## A-Q4. 带电粒子事例形态、回溯剂量计与辐射带模型验证（想法 A4、A5）

### Takeaway
HXMT 做不了 Timepix/HST 那种径迹形态学：SCD 不是成像器件，主探测器对最小电离粒子早已饱和。能做的是两件事：用 PM（及可能的 ACD）做 9 年积分通量的模型验证，以及用 ACD 命中图样对被丢弃的事例做粗粒度分类。团队已用 PM 研究过 2017–2019 年的 SAA，但没有覆盖整个太阳周，也未见与 AP9 的比较。

### Cited Findings
- 团队用三个 PM 追踪了 2017-06 至 2019-06 的 SAA，构建粒子通量图并用两种方法得到一致的漂移率 [confirmed in literature] — [Zhao H.S. et al., The observation of the South Atlantic Anomaly with the particle monitors onboard Insight-HXMT（aminer 条目）](https://www.aminer.org/pub/5fae64bcd4150a363cdbd06e/the-observation-of-the-south-atlantic-anomaly-with-the-particle-monitors-onboard)
- HXMT 本底模拟采用 SPENVIS 的 AP-8 太阳极小轨道平均谱作为 SAA 质子输入 [confirmed in literature：检索摘要未指明出自哪一篇模拟论文] — [Xie F. et al. 2015](https://arxiv.org/abs/1511.02997) 或 [Zhang J. et al. 2020](https://arxiv.org/abs/2009.14489)
- PM 计数率在赤道附近 <3 cts/s、高纬 >10 cts/s，SAA 内高得多 [confirmed in literature] — [HE 本底模型](https://arxiv.org/abs/2005.01661)
- NOAA-15 SEM-2 1998–2022 年数据显示 SAA 整体北漂（纬度与太阳活动反相关）、24 年西漂 7° — [EGU23-12704](https://meetingorganizer.copernicus.org/EGU23/EGU23-12704.html)；2026 年发表的动态俘获质子模型（基于 >70 MeV 的 SEM-2 数据）可描述西北漂移、太阳周变化与迟滞效应 — [J. Space Weather Space Clim. 2026](https://www.swsc-journal.org/10.1051/swsc/2025057)
- 用科学 CCD 当粒子探测器的先例：HST 五台 CCD 的 75,908 幅暗场中表征了 >12 亿条宇宙线径迹，看到 SAA 与太阳调制 — [Miles et al. 2021](https://arxiv.org/abs/2006.00909)，[HSTCosmicrays](https://arxiv.org/abs/2011.11604)
- Timepix 类像素探测器靠径迹形态（点、团、直线径迹）+ 沉积能量 + dE/dx 区分粒子种类，SATRAM 自 2013 年在约 820 km 太阳同步轨道运行 — [Timepix in space 综述预印本](https://www.preprints.org/manuscript/202311.1087)，[LUCID](https://arxiv.org/abs/1810.12876)，[Data Processing Engine](https://arxiv.org/abs/2310.15723)
- 用 ML 识别成像探测器中宇宙线诱发事例可把未被拒绝的本底再降约 30%（后续工作称最高 41.5%）— [Wilkins et al. 2022](https://arxiv.org/abs/2208.07906)，[2024](https://arxiv.org/abs/2407.16768)

### Inferences

#### 想法 A4：回溯式俘获质子监测与模型验证
1. **一句话**：把 PM 的 SAA 研究从 2 年扩展到 9.4 年，并加上"HE 短寿命活化幅度"这一只对质子敏感的第二剂量计，验证 AP8/AP9/IRENE 与 2026 年的动态质子模型在 550 km、43° 处的预言。
2. **原理与来源**：Zhao et al. 的 HXMT SAA 工作、Campana et al. 2014、SEM-2 长期漂移研究。可检验预言：SAA 质子通量与 F10.7 反相关并有滞后；西漂/北漂速率与 SEM-2 结果一致；电子不产生核活化，因此"PM 计数/活化幅度"之比随位置的变化可分离电子与质子贡献 [inference]。
3. **HXMT 特性与诚实评价**：PM 只是三个积分计数器，无能谱、无粒子鉴别，逊于 POES/Proba-V/CSES 的专用仪器。价值在于轨道位置（专用监测器多在 800 km 极轨或太阳同步轨道【未检索核实】）与恰好跨越第 24 周极小到第 25 周极大的时间基线，以及活化这一独立通道。
4. **数据**：`HE-PM`（含 SAA 内数据）、`EHK`、A2 的活化幅度。公开。
5. **查新**：HXMT 只有 2017–2019 年的 PM 研究；未检索到 HXMT 与 AP9 的定量比较（检索返回的唯一 AP9 比较是 CSES-01 HEPD 的会议幻灯片标题）。
6. **可行性**：极高。障碍：PM 计数到通量的转换需要响应函数；HXMT 轨道高度十年间是否衰减未知，而 SAA 通量对高度敏感，必须用 `SAT_ALT` 显式建模。精度：年均通量图的统计误差 <1%，系统误差由响应与各向异性（俘获质子东西效应）主导。
7. **评分**：新颖 2 / 可行 5 / 影响 3。

#### 想法 A5：被否决事例的粗粒度"描迹器"
1. **一句话**：对标准流程丢弃的 ACD≠0 事例，用 18 位命中图样（哪些顶板/侧板同时响）+ 被击中模块 + Pulse_Width + 是否溢出，重建贯穿粒子的粗方向并做粒子类别的粗分类。
2. **原理与来源**：HST/Timepix 的"科学探测器当粒子探测器"思路，但改用**符合拓扑**而非径迹形态；神经科学的 spike sorting（对脉冲形状做无监督聚类）可用于 Pulse_Width × Channel 平面上的异常族群。
3. **HXMT 特性与诚实评价**：**不比其他任务更合适做 LET 谱**——没有 dE/dx 分辨率，SCD 无像素径迹。能做的只是方向性计数与类别粗分，其主要用途是为 B6（东西效应、截止刚度谱仪）和 B2（沉降事件方向）提供输入，而不是独立的剂量学产品。
4. **数据**：HE 1L 事例的 ACD 与 Pulse_Width 列（公开，已确认）；ME 多像素符合需要事例级时间标记（公开，但是否有多重性标志未知）。
5. **查新**：未找到利用 HXMT ACD 命中图样的工作；GECAM 已有"基于探测器计数分布"的投掷角方法（见 B2）。
6. **可行性**：中等。未知量：ACD 标志的符合时间窗、板的几何与阈值。若贯穿粒子通量为 1 cm⁻² s⁻¹ 量级【未检索核实】，5000 cm² 上每秒有数千事例，统计量极大；系统误差在于簇射与次级粒子造成的多板同时触发。
7. **评分**：新颖 4 / 可行 3 / 影响 2。

### Gaps
- Zhao et al. 的原始 arXiv/ADS 链接与具体漂移率数值未取回（只有 aminer 条目）。
- AP9/IRENE 在 550 km 的不确定度、俘获质子东西效应的测量文献：检索未执行，【未检索核实】。
- HXMT 轨道高度十年演化未知。
- ME 是否记录超量程/多像素事例标志未知；SEM 数据可得性未知。

---

## A-Q5. Housekeeping 当科学、微流星体、极端计数率（无独立入选想法）

### Takeaway
这一方向没有找到 HXMT 的已发表工作，但也没有足以支撑独立课题的信号量：单粒子翻转/闩锁记录不在公开数据中，微流星体撞击的期望次数只有 1 次量级。极端计数率的经验教训已在 GRB 221009A 论文中部分总结。

### Cited Findings
- 焦平面微流星体事件此前只见于掠入射聚焦望远镜：Swift XRT 2005 年 5 月一次亮闪光后留下坏列，XMM-Newton 有五次类似事件、其中一次导致一片 CCD 失效 — [Meteoroid and space debris impacts in grazing-incidence telescopes](https://arxiv.org/abs/0803.1983)，[Health and cleanliness of the XMM-Newton science payload since launch](https://arxiv.org/abs/astro-ph/0508241)
- eROSITA 在 2019-12 至 2022-01 记录到至少 7 次疑似微流星体事件，只有 1 次明确伴随强光学闪光；尘埃模型预言任务期内约 2–5 次约 1 μm 颗粒撞击 — [SRG/eROSITA micrometeoroid hits and effects (SPIE)](https://journals.spiedigitallibrary.org/conference-proceedings-of-spie/12181/1218155/SRGeROSITA-micrometeoroid-hits-and-effects/10.1117/12.2630890.full)，[EPSC2022-563](https://meetingorganizer.copernicus.org/EPSC2022/EPSC2022-563.html)
- ME 5 年论文未报告 ASIC 闩锁、单粒子翻转或高压跳闸；LE 5 年论文未提微流星体、漏光或滤光膜损伤 — [ME (ar5iv)](https://ar5iv.labs.arxiv.org/html/2303.00339)，[LE (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.10714)
- HXMT 对弱暴的 targeted coherent search 专门改进以滤除光变曲线尖峰造成的假触发 — [Cai et al. 2021](https://arxiv.org/abs/2109.12270)

### Inferences
- **微流星体期望数量级** [inference，输入量未检索核实]：取 LEO 中 >1 μm 颗粒通量约 10³–10⁴ m⁻² yr⁻¹，LE 小视场（约 1.6°×6°，占半球立体角约 5×10⁻⁴）几何面积约 0.03 m²，9 年期望约 0.1–1 次。结论：HXMT 是无聚焦镜的准直望远镜，不会像 XMM/Swift/eROSITA 那样把尘埃"聚"到焦面，不值得立项。
- 可做的低成本检查：对 7 片坏 CCD，看失效前后 forced-trigger 噪声是否与 `SUNSHINE`/`DYE_ELV`（光照）相关——相关则是滤光膜穿孔漏光，只与温度相关则是辐照损伤 [inference]。
- SAA 边缘计数率在几十秒内平滑变化两个量级，可当作"天然的率扫描"来验证死时间与堆积模型 [speculation]；价值有限，并入 A6。

### Gaps
- HXMT 的 SEU/闩锁/复位日志：属工程遥测，未见公开；星敏/陀螺异常同理。
- LEO 微流星体/碎片通量的权威数值未检索。
- Sco X-1、磁星巨耀发等极端计数率下 ME/LE 的行为：未检索。

---

## B-Q1. 粒子物理分析范式（想法 B1）

### Takeaway
LHC 的整套"数据驱动本底 + 控制区闭合 + 盲分析 + 全局显著性"流程可以原样搬到 HXMT 的盲探测器/多视场/地球遮挡数据上，最大的价值是给神经本底模型提供一个**不依赖训练集**的系统误差标尺。作为暗物质谱线搜寻，HXMT 的能量分辨率与活化线使它不占优势。

### Cited Findings
- CWoLa：在两个信号比例不同的混合样本之间训练分类器，无需逐事例标签也无需知道混合比例，最优 CWoLa 分类器等同于最优全监督分类器 [confirmed in literature] — [Metodiev, Nachman & Thaler 2017, JHEP 10 (2017) 174](https://arxiv.org/abs/1708.02949)
- look-elsewhere 效应的 trial factor 可快速估计，且在固定质量处近似随显著性线性增长 [confirmed in literature] — [Gross & Vitells 2010, Eur. Phys. J. C 70, 525](https://arxiv.org/abs/1005.1891)
- HXMT 的本底模型本身就建立在盲探测器与其他探测器计数率的相关性上 [confirmed in literature] — [HE 本底模型](https://arxiv.org/abs/2005.01661)，[LE 本底模型](https://arxiv.org/abs/2004.01432)
- 已有用 HXMT 数据限制原初黑洞丰度的工作，作者指出在"原始数据"和"天体物理本底模型"两种情形下 HXMT 无优势，只有采用幂律加指数截断谱时硬 X 射线灵敏度才有帮助 [confirmed in literature] — [Yang, Pan & Zhang 2026, Commun. Theor. Phys. 78, 045402](https://arxiv.org/abs/2412.09297)
- 检索未发现 HXMT 的惰性中微子/类轴子谱线搜寻；该领域的空天区搜寻来自 XMM-Newton、Hitomi/XRISM、eROSITA [confirmed：检索结果为空] — 例如 [eROSITA early data](https://arxiv.org/abs/2401.16747)

### Inferences

#### 想法 B1：ABCD 闭合检验 + 盲分析 + 谱线扫描 + CWoLa
1. **一句话**：以 (探测器：开口/盲) × (指向：空天区/地球遮挡) 构成 ABCD 四区，用 N_A ≈ N_B·N_C/N_D 做与神经网络无关的本底预言，把它与 NeuBacH 的差作为闭合检验；在冻结系统误差之后才"开盲"看残差谱并做带 trial factor 的谱线扫描。
2. **原理与来源**：LHC 的 ABCD 法、控制区/验证区、sideband、盲分析；Gross–Vitells；CWoLa。四个具体步骤：(i) 在不同 COR 带的空天区上做 pull 分布，得到系统误差随曝光的标度；(ii) 利用盲/小/大三档视场做"视场阶梯"回归——弥散天空成分 ∝ 立体角、粒子本底与视场无关，从而无模型地分离 CXB/银河脊成分；(iii) 对残差做滑动窗口 profile-likelihood 谱线扫描；(iv) 用"Crab 指向 vs 空天区"两个混合样本训练 CWoLa 分类器（特征：Pulse_Width、Channel、ACD 图样、距同模块上一个大能量沉积的时间），检验能否在标准脉宽窗之外再提高 S/√B。
3. **HXMT 特性与诚实评价**：盲探测器 + 多档视场 + 大量地球遮挡数据正好提供 ABCD 所需的两个近似独立的轴，方法学上很合适。**作为暗物质谱线搜寻不占优**：NaI 分辨率差且谱中布满活化线，ME 在 17.8 keV 处 FWHM 约 3 keV；NuSTAR/SPI/XRISM 更合适。CWoLa 的增益预计有限，因为 PSD 的 FoM 已达 2–2.7，可能只在 20–30 keV 低能端和余辉尖峰剔除上有改进 [inference]。
4. **数据**：空天区与地球遮挡时段的 1L 事例、盲探测器事例；公开。
5. **查新**：未找到在 HXMT 上显式使用 ABCD/盲分析/全局显著性的工作；PBH 限制已有人做；HXMT 是否已发表 CXB 测量本次未能检索。
6. **可行性**：高。ABCD 的前提是两轴近似独立，而盲/开口探测器对地球反照与大气辐射的响应不同会破坏独立性——这本身就是可测的"非闭合"，并直接给出 B4 的信号。谱线灵敏度的量级估计缺输入（空天区总曝光、各能段本底率），无法在此给出。
7. **评分**：新颖 3 / 可行 4 / 影响 3 —— 主要价值是本底模型的独立验证框架，而非新物理限制。

### Gaps
- CMS data scouting、LHCb turbo stream、CERN Open Data 的具体文献：检索只返回了 CWoLa，未取回其余。
- "放宽 GTI 的数据流"（data scouting 的类比）对稳定源的增益估计只有 [inference]：若被弃时段多出 25% 时间但本底翻倍，信噪比仅提高约 6%；对随机发生的暴发类事件，可恢复的事件数近似正比于恢复的时间。HXMT 现有磁星暴/GRB 搜寻是否已使用非 GTI 数据未知。
- HXMT 空天区累计曝光未知。

---

## B-Q2. 地震学方法（想法 B2）

### Takeaway
"模板匹配找亚阈值暴"在 HXMT 上已有等价实现（targeted coherent search），不再新颖。真正可迁移且尚无人在 HXMT 上做的是**台阵处理**：把 18 块朝向不同的 ACD/CsI 当作地震台阵，用 STA/LTA 触发 + beam-forming 给带电粒子沉降事件定方向（投掷角）。GECAM 在 2025 年刚发表了方法和首批事件，HXMT 是天然的第二台站。

### Cited Findings
- HXMT 的 targeted coherent search 利用暴的位置、能谱与探测器响应，比盲搜更能发现弱暴；在 HE 前两年数据中找回 40 个 Swift/BAT 或 Fermi/GBM 探测到但盲搜漏掉的 GRB；200 keV–3 MeV 灵敏度约 1.5×10⁻⁸ erg/cm² [confirmed in literature] — [Cai et al. 2021, MNRAS](https://arxiv.org/abs/2109.12270)
- GECAM 提出基于探测器计数分布测量带电粒子投掷角的方法，并用于一次 GECAM-C 电子沉降事件，得到约 90° 投掷角；摘要未提 Insight-HXMT [confirmed in literature] — [Wang C. et al. 2025, Pitch Angle Measurement Method based on Detector Counts Distribution](https://arxiv.org/abs/2505.06167)
- GECAM 在 2024-03-21 磁暴期间于连续三轨、同一区域探测到三次带电粒子沉降事件，其中两次有显著周期性，带电粒子持续超过 3.5 小时，未发现与闪电活动相关 [confirmed in literature] — [GECAM Discovery of Peculiar Oscillating Particle Precipitation Events](https://arxiv.org/abs/2505.06192)
- GECAM 首批 TGF 与 TEB 结果 — [The First GECAM Observation Results on Terrestrial Gamma-ray Flashes and Terrestrial Electron Beams](https://arxiv.org/abs/2306.10255)；GECAM 2024–2026 综述的参考文献列有上述投掷角方法（Experimental Astronomy 2025）与振荡沉降事件（SCPMA 2026）两篇 — [Xiong 2026, Overview of GECAM Observations from 2024 to 2026, Chin. J. Space Sci.](https://www.cjss.ac.cn/en/article/doi/10.11728/cjss2026.04.2026-yg19)
- HXMT 已发布 282 个 TGF 的数据集 — [Zenodo 10976454](https://zenodo.org/records/10976454)；HXMT 的 TGF 也用于 TGF 与窄双极性事件关系的研究 — [GRL 2021](https://par.nsf.gov/servlets/purl/10342078)
- SAMPEX 数据直接观测到闪电诱发的 MeV 电子从内辐射带沉降 — [Nature Communications 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11461791/)
- 环境噪声互相关：对台站对之间数天的噪声记录做互相关可提取台间 Green 函数 [confirmed in literature] — [Shapiro & Campillo 2004, GRL](https://data-ww3.ifremer.fr/BIB/Shapiro_Campillo_GRL2004.pdf)
- 检索未发现 HXMT 关于电子沉降或磁暴期间粒子事件的论文 [confirmed：两次检索结果为空]

### Inferences

#### 想法 B2：HE/ACD"台阵"上的沉降事件目录与投掷角
1. **一句话**：对 18 路 ACD 计数、3 路 PM、CsI 与 ME/LE 的"粒子事件"时段运行 STA/LTA 触发，再按各板法向与当地磁场方向做 beam-forming，建立 2017–2026 年 43° 轨道上的电子沉降/TEB/磁暴粒子带目录，并给出每个事件的投掷角估计。
2. **原理与来源**：地震学的 STA/LTA 触发与台阵 beam-forming（把各台记录按假定入射方向延时叠加，这里换成按假定投掷角分布对各板计数做加权拟合）；直接先例是 GECAM 的计数分布法。可检验预言：(i) 2024-03-21 GECAM 事件发生时若 HXMT 经过共轭或邻近 L 壳层，应看到同源信号；(ii) 目录事件率应随 Kp/Dst 和 L 值按已知的沉降带分布；(iii) LE、ME、ACD、PM 的阈值逐级升高，构成粗能谱（"射程望远镜"）。
3. **HXMT 特性与诚实评价**：5 年本底论文已承认 ME 高纬"粒子事件峰"与 LE "flare" 的存在并把它们剔除，这正是被丢弃的信号。相对 GECAM：HXMT 没有专门的带电粒子探测器阵列和 4π 布局，方向重建更粗；优势是时间基线（2017 年起，比 GECAM 早 3 年多）、43° 倾角覆盖更高 L 值，以及 keV–MeV 多层探测器。结论：**互补而非更优**。
4. **数据**：`HE-Cnts`（1 s）、`HE-PM`、HE/ME/LE 1L 事例（含 GTI 外时段）、`Att`、`EHK`；公开。ACD 板几何与各层电子阈值需要仪器论文或团队内部信息。
5. **查新**：GECAM 2025 的方法与事件；HXMT 只有 TGF 目录，未见沉降事件/TEB 目录。Fermi-GBM 用时间信息间接推断 TEB 投掷角的工作被 GECAM 论文引用，本次未取回原文。
6. **可行性**：高。1 s 的 HK 计数足以做分钟级磁暴粒子带与秒级 LEP；毫秒级 TEB 需用事例数据。主要系统误差：惯性定向下各板相对磁场的角度随轨道连续变化，必须逐秒计算；SAA 边缘的俘获粒子要与沉降区分（用 L 与 B 判别）。用 NeuBacH 残差代替 LTA 可显著降低假触发。
7. **评分**：新颖 3 / 可行 4 / 影响 4 —— 时机好（GECAM 刚打开这个方向），且直接把"被剔除的粒子事件"变成目录产品。

### Gaps
- Gibbons & Ringdal 2006（波形相关检测）等模板匹配原始文献：检索未执行，【未检索核实】。
- 星间"环境噪声互相关"式的亚阈值符合搜寻（HXMT × GBM × GECAM）是否已有流水线：未检索到，凭记忆存在 GECAM/HXMT 联合分析系统，需核实。
- ACD 在 SAA 模式下是否工作、各板阈值与几何：未知。
- LE "flare" 的物理成因（低能电子、质子还是弥散 X 射线）文献未说明。

---

## B-Q3. 气象/气候学方法（想法 B3）

### Takeaway
辐射带电子的"再分析"在空间物理界已成熟，HXMT 的积分计数难以作为同化主力；但气候学的另外两件工具——EOF 模态分解与 detection-and-attribution——可以直接用在十年本底立方上，把 5 年论文里"活化累积、辐照损伤、地磁调制"这些定性归因变成带置信区间的定量归因，并补上该论文完全没讨论的太阳周调制与 SAA 漂移。

### Cited Findings
- 5 年本底论文对各望远镜本底上升的解释是定性的（LE：辐照损伤 + 抬高 split 阈值；ME：SAA 后累积的弱延迟成分 + 像素辐照损伤；HE：晶体活化，长寿命同位素驱动缓慢累积），未给百分比，也未讨论太阳周调制与 SAA 漂移 [confirmed in literature] — [Five-year in-orbit background (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.14459)
- 辐射带电子再分析：用 split-operator Kalman 滤波把五颗卫星数据同化进 VERB 三维代码，再分析揭示了 1991 年 3 月磁暴期间此前未报道的四区结构 — [Kellerman et al. 2014, JGR](https://rbm.epss.ucla.edu/reference/kellerman-et-al-2014)；四颗卫星的再分析及用 innovation 向量定位缺失源汇 — [Cervantes et al.](https://ui.adsabs.harvard.edu/abs/2019EGUGA..21.8547C/abstract)
- 数据同化可用来把 LEO 卫星电子测量与 POES 做相互定标 — [Castillo et al. 2024, Space Weather](https://gfzpublic.gfz-potsdam.de/pubman/item/item_5025472_1/component/file_5025513/5025472.pdf)
- SEM-2 质子通量 1998–2022 年的研究显示 SAA 的纬度与太阳活动反相关 — [Studying the South Atlantic Anomaly temporal evolution from 1998 to 2022 using the SEM-2 proton flux (EGU23-12704)](https://meetingorganizer.copernicus.org/EGU23/EGU23-12704.html)；动态俘获质子模型包含太阳周变化与迟滞效应 — [J. Space Weather Space Clim. 2026](https://www.swsc-journal.org/10.1051/swsc/2025057)

### Inferences

#### 想法 B3：EOF 模态分解 + 最优指纹归因
1. **一句话**：把本底计数率整理成 (经度, 纬度, 能段, 探测器, 月份) 的数据立方做 EOF/SVD，再把观测到的时空变化回归到由模拟或解析模型给出的若干"指纹"上，得到每个物理成因的标度因子 β 及其置信区间。
2. **原理与来源**：气候学的 EOF 分析与 detection-and-attribution（最优指纹法）。预期模态及其可检验特征：(1) 平均地磁图样（随 COR）；(2) SAA 下游的活化"尾迹"（升轨/降轨不对称，已被观测到）；(3) 太阳周调制——时间系数应跟随中子堆计数，幅度在低截止刚度处大、高截止刚度处小；(4) SAA 漂移——空间图样应等于 SAA 图的经纬度导数；(5) 长寿命活化累积——饱和型时间系数；(6) 探测器老化——与天空/地理无关、只与探测器编号有关。
3. **HXMT 特性与诚实评价**：惯性定向 + 规则轨道使地理覆盖均匀；盲探测器让模态 (6) 可与其他模态分离。很适合。与"直接看神经网络"的区别在于 EOF 给出的是可与物理指纹逐一对照的正交基，而不是黑箱。
4. **数据**：与 A3 相同的数据立方；中子堆与 F10.7 等外部指数。公开。
5. **查新**：未找到对 X 射线/伽马探测器本底做 EOF 或最优指纹归因的工作（未做专门检索，[inference]）；辐射带再分析已成熟。
6. **可行性**：高。障碍：EOF 模态是数学正交的，不保证与物理成因一一对应，需要旋转 EOF 或直接做指纹回归；增益漂移必须先校正否则会伪装成谱形变化。量级估计：PM 在磁赤道处五年"基本不变"与高截止刚度处调制仅百分之几的预期一致 [inference]；低截止刚度处第 24 周极小到第 25 周极大的宇宙线调制预期为 10–20% 量级【未检索核实】，以 HXMT 的统计量可测到远好于 1%。
7. **评分**：新颖 4 / 可行 4 / 影响 3。

### Gaps
- 把 HXMT + GBM + GECAM + POES/CSES 同化成"本底再分析产品"：对单个分析者工作量过大，且 HXMT 的 PM 需要先有通量响应；降级为 B5 的远期延伸。
- Hasselmann/Allen–Stott 等最优指纹原始文献：【未检索核实】。
- ensemble 预报本底：未找到先例，也未形成具体可检验方案，未入选。

---

## B-Q4. 成像与层析（想法 B4）

### Takeaway
HXMT 的 X 射线掩地**吸收**层析（反演中高层大气密度）已被做了至少四篇，不再是新想法。尚未见到的是它的**发射**对应物：窄准直视场随轨道扫过地球盘面，测大气硬 X 射线反照的中心—临边角分布及其随截止刚度、磁暴的变化——这正是所有 LEO 任务本底模型里靠模拟给定的输入量。

### Cited Findings
- 用 Crab 掩地数据反演大气密度：115 组掩地数据得到 55–100 km 密度比 NRLMSISE-00 低 10–25% — [ApJS 2022（DOAJ 条目）](https://doaj.org/article/cafb0de60cb54e51948813f066fb5302)；LE 单次指向数据的 Bayesian 反演（105–200 km）— [Atmos. Meas. Tech. 2022（DOAJ 条目）](https://doaj.org/article/2d81bdb495c448128e60333be0fba3be)；2022–2024 年 83 组掩地数据把反演扩展到 130 km，并通过与空天区掩地比较说明 CXB 贡献可忽略 — [arXiv:2502.19678（ApJ 2025）](https://arxiv.org/abs/2502.19678)
- HE 的在轨定标同时使用空天区与地球遮挡数据 [confirmed in literature] — [HE 5 年性能](https://arxiv.org/abs/2302.03859)
- HXMT 的银道面扫描巡天方法与四年监测结果已发表 — [Methodology and Performance of the Two-Year Galactic Plane Scanning Survey of Insight-HXMT](https://arxiv.org/abs/2003.07741)，[The Long-term Monitoring Results of Insight-HXMT in the First 4 Yr Galactic Plane Scanning Survey](https://arxiv.org/abs/2303.00360)
- GECAM 的计数分布法给出带电粒子投掷角（见 B2）— [arXiv:2505.06167](https://arxiv.org/abs/2505.06167)

### Inferences

#### 想法 B4：地球盘面的准直扫描——大气硬 X 射线反照的角分布与沉降轫致辐射
1. **一句话**：利用被 GTI 剔除的 ELV<0 数据，把 HE（1.1°×5.7° 小视场）和 ME/LE 随轨道扫过地球盘面的过程当作一维扫描成像，测量大气反照硬 X 射线的中心—临边亮度分布及其对截止刚度、太阳周与磁暴的依赖。
2. **原理与来源**：医学/工业 CT 的"多角度线积分 → 重建"；HXMT 自身的扫描成像传统。几何（[inference]）：550 km 高度处地平线距离约 2700 km，地球盘面半径约 67°；1.1° 对应临边处约 50 km。可检验预言：(i) 宇宙线诱发的大气发射应有临边增亮，且强度随视场足点处的截止刚度降低而升高；(ii) 磁暴期间，当视场足点落在高磁纬时，应出现辐射带电子沉降的轫致辐射超出；(iii) 开口探测器减盲探测器的差在 ELV 扫过 0° 时应呈台阶 + 临边峰。
3. **HXMT 特性与诚实评价**：这是窄准直视场的真正优势——宽视场仪器只能测盘面积分。局限：50 km 的切向分辨率**分辨不了**发射层的垂直廓线（发射高度约几十公里），只能得到盘面角分布；43° 倾角下临边最多看到约 66° 地理纬度，能否看到极光带取决于经度。
4. **数据**：ELV<0 时段的 HE/ME/LE 事例与盲探测器、`Att`、`EHK`；公开。
5. **查新**：HXMT 的吸收型掩地层析已做；**未检索到**用 HXMT 做地球反照发射角分布的工作，但这一条未能做专门检索，新颖性需复核。Swift/BAT、INTEGRAL 的地球反照测量文献【未检索核实】。
6. **可行性**：中等。量级估计 [inference，输入不确定]：若小视场内大气信号为几 cts/s、总本底约 500 cts/s，单次扫过（约 16 s）信噪比约 1，叠加 10⁴ 次可达约 100。主要系统误差：地球遮挡本身改变了 CXB 与卫星受照，本底模型在 ELV<0 时的外推精度是关键——这也是对 NeuBacH 的一个明确检验。压缩感知/深度先验重建在这里不必要：未知量只是角分布的少数几个参数。
7. **评分**：新颖 4 / 可行 3 / 影响 3。

### Gaps
- "把卫星扫过辐射带当 CT，重建俘获粒子投掷角分布"：物理上可行，但 HE 在 SAA 内关高压，只剩 PM（各向同性塑闪）和可能的 ACD，角信息不足；并入 B2/A4，未单列。
- HE 模块间 Compton 成像：见"拒绝"一节。
- 银道面扫描的压缩感知/深度先验重建：团队已有 DDM 与四年巡天，属相邻课题，新增空间受 1° 级准直 PSF 与源混淆限制，未入选。
- 地球反照 X/γ 的已有测量与模型：检索未执行。

---

## B-Q5. 分布式传感网络与交叉定标（想法 B5）

### Takeaway
把多颗高能天文卫星看成一个非专用的 LEO 辐射观测网，关键的方法缺口不是"数据同化"而是更基础的一步：像气象卫星那样，在轨道交会点上做环境响应的相互定标。这一步同时为把 HXMT 训练的本底模型迁移到别的卫星提供成对样本。

### Cited Findings
- 数据同化可在物理模型约束下对不同 LEO 卫星的电子测量做相互定标 [confirmed in literature] — [Castillo et al. 2024](https://gfzpublic.gfz-potsdam.de/pubman/item/item_5025472_1/component/file_5025513/5025472.pdf)
- GECAM 已发展为多星座（GECAM-A/B、GECAM-C、GECAM-D），覆盖约 10 keV–5 MeV [confirmed in literature] — [Xiong 2026](https://www.cjss.ac.cn/en/article/doi/10.11728/cjss2026.04.2026-yg19)
- Fermi-GBM 的数据驱动本底估计器（前馈神经网络，12 个 NaI、多能段）的作者明确说该技术为 GBM 定制、不能直接用于其他实验 [confirmed in literature] — [Crupi et al. 2023, Exp. Astron.](https://arxiv.org/abs/2303.15936)，[Crupi 博士论文 DeepGRB](https://arxiv.org/abs/2401.15632)
- GBM 物理本底模型把本底分解为若干物理源并正确处理探测器响应与航天器几何，在两个月试运行中找到 300 多个未触发的长时标暂现信号 — [Biltzinger et al. 2020](https://arxiv.org/abs/2005.11219)，[Kunzweiler et al. 2022](https://arxiv.org/abs/2205.13649)

### Inferences

#### 想法 B5：SNO 式交会定标与"LEO 辐射观测网"
1. **一句话**：找出 HXMT 与 Fermi、GECAM、SVOM、EP、CSES 在 (L, B, 高度, 时间) 上同时接近的交会时刻，在这些点上建立各仪器本底相关量之间的传递函数，并长期监测其漂移。
2. **原理与来源**：气象卫星辐射定标的 simultaneous nadir overpass (SNO) 方法；GNSS/重力测量中"用非专用传感器网络"的思路。可检验命题：同一 L、B、高度、几分钟之内，两星的粒子诱发计数率之比应只依赖探测器响应，与时间无关；比值的长期漂移 = 某一方的仪器老化；磁暴期间比值的短时偏离 = 空间不均匀性。
3. **HXMT 特性与诚实评价**：HXMT 只是网络的一个节点，单独并不占优；它的贡献是 43° 倾角（把网络的 L 覆盖向外推）和 2017 年起的时间基线。
4. **数据**：HXMT 公开 HK；Fermi-GBM 公开连续数据；GECAM/SVOM/EP/CSES 的公开程度各异（未核实）。
5. **查新**：GRB/Crab 的仪器间交叉定标很常见；未找到"粒子环境响应"的 SNO 式相互定标（未做专门检索，[inference]）。Castillo et al. 用同化做电子测量互定标是最接近的先例。
6. **可行性**：中等。两星高度相近、倾角不同，每圈有两次轨道面交叉，但同时满足位置与时间容差的交会频率需用 TLE 实算（本次未算）。障碍：各星探测器对电子/质子/γ 的响应不同，传递函数必须分 L 段、分粒子成分；HXMT 以外的数据权限。
7. **评分**：新颖 4 / 可行 3 / 影响 3 —— 其产出（成对样本）是 B7 域适应的前提。

### Gaps
- 各星轨道根数与交会频率：未计算；GECAM/SVOM/EP 轨道参数【未检索核实】。
- GRID 立方星、CSES 数据的可得性未查。
- GSICS SNO 方法的原始文献：【未检索核实】。

---

## B-Q6. 宇宙线观测方法（想法 B6）

### Takeaway
中子堆界的三件老工具——纬度巡测求产额函数、东西效应、Forbush 下降的历元叠加——都能在 HXMT 上实现，且统计量充足。HXMT 做不出比 AMS-02 或中子堆更好的宇宙线物理，但能得到一个别处拿不到的量：**每个探测器的本底对原初宇宙线刚度的产额函数**，这是把本底预言迁移到别的轨道和太阳周相位的物理基础。

### Cited Findings
- Fermi-LAT 没有磁铁，利用地磁场使正负电荷的"地球阴影"方向相反来分离正负电子 [confirmed in literature] — [Ackermann et al. 2012, PRL 108, 011103](https://arxiv.org/abs/1109.0521)
- 单个探测器的计数率即可做 Forbush 下降监测：EPHIN 的反符合探测器可在小时尺度上测到 <1% 的计数率变化 — [The Electron Proton Helium INstrument as an Example for a Space Weather Radiation Instrument](https://arxiv.org/abs/2010.00864)；LISA Pathfinder 辐射监测器记录到约 18% 的 Forbush 下降 — [Measuring the Galactic Cosmic Ray Flux with the LISA Pathfinder Radiation Monitor](https://arxiv.org/abs/1711.07427)
- 2024-05-10 的 Forbush 下降在 70–1000 GV 刚度范围被 ALPAQUITA 空气簇射阵列测到，幅度约 4.3% — [arXiv:2601.08154](https://arxiv.org/abs/2601.08154)
- HXMT 的 ME 本底与截止刚度反相关、LE 高能段本底受地磁场调制；PM 在赤道附近 <3 cts/s、高纬 >10 cts/s [confirmed in literature] — [5 年本底 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2302.14459)，[HE 本底模型](https://arxiv.org/abs/2005.01661)
- BeppoSAX PM 的计数率依赖当地磁刚度 — [Campana et al. 2014](https://arxiv.org/abs/1405.0360)
- 检索未发现用 INTEGRAL SPI-ACS 或 Fermi-GBM 做 Forbush 下降研究的论文，也未发现 HXMT 的相关工作 [confirmed：检索结果为空]

### Inferences

#### 想法 B6：截止刚度谱仪、东西效应与 Forbush 历元叠加
1. **一句话**：把 SAA 之外的瞬发本底（PM、ACD、CsI 高能道）对垂直截止刚度 R_c 求导得到各探测器的产额函数，用板的朝向测东西不对称，用历元叠加测 Forbush 下降和磁暴期间的截止刚度压低。
2. **原理与来源**：中子堆的纬度巡测（计数率对 R_c 的导数 = 响应加权的原初微分谱）、历元叠加、东西效应。四个可检验预言：(i) 用两段太阳周相位的 dN/dR_c 之比反推调制势，应与中子堆一致；(ii) 朝西的侧板在同一位置计数率高于朝东的侧板，不对称度随 R_c 增大；(iii) 以 Forbush 下降起始为零点叠加高纬计数率残差，应得到幅度随 R_c 减小的下降—恢复曲线；(iv) 大磁暴期间同一地理位置的计数率**升高**，升高量与 Dst 及 R_c 的关系符合截止刚度压低的计算——这为 B7 的一类"未解释本底偏移"提供物理解释。
3. **HXMT 特性与诚实评价**：43° 倾角使 R_c 覆盖约 2–15 GV【未检索核实，量级】，比 <30° 倾角的卫星宽；18 块朝向各异的 ACD 提供方向信息。**宇宙线物理本身不占优**；价值在于产额函数——它把"本底随地理位置的经验图"换成"本底随原初谱的物理响应"，因而可迁移。
4. **数据**：`HE-PM`、`HE-Cnts`、CsI 事例、`EHK`（COR）、`Att`；中子堆与 Dst 指数。公开。
5. **查新**：未找到 HXMT 或同类伽马监测器的 Forbush/东西效应/产额函数工作；EPHIN 与 LISA Pathfinder 是单探测器 Forbush 监测的先例。
6. **可行性**：高。量级估计 [inference]：三个 PM 在高纬合计约 30–50 cts/s，每圈高纬驻留约 16 min → 每圈约 4×10⁴ 计数（统计精度约 0.5%），每天约 6×10⁵ 计数（约 0.13%），足以在日尺度上高显著性测到百分之几的 Forbush 下降；ACD 计数率更高、精度更好。系统误差：必须先扣除轨道调制（NeuBacH 残差正好提供）；EHK 里的 COR 来自静态地磁模型，磁暴期需用外部模型重算；SAA 活化尾迹要用 T_SAA 剔除。
7. **评分**：新颖 3 / 可行 4 / 影响 3。

### Gaps
- 东西效应的经典文献与 LEO 俘获质子东西不对称的测量：检索未执行，【未检索核实】。
- ACD 板的朝向几何未公开核实。
- 磁暴期截止刚度压低的定量模型文献未检索。

---

## B-Q7. 时间序列方法与"有物理抓手"的机器学习（想法 B7）

### Takeaway
HXMT 上除用户自己的 NeuBacH 外未检索到神经网络本底模型；最接近的外部先例是 NICER 的 BERTground 和 Fermi-GBM 的前馈网络估计器。在此基础上最有价值的三步都带明确的物理检验：把残差流变成带严格显著性的公开目录、从训练好的模型里读出活化衰变核并与同位素半衰期对照、用交会点成对样本做跨卫星域适应。因果分析应选 transfer entropy 而非 convergent cross mapping。

### Cited Findings
- 检索"Insight-HXMT machine learning background"只返回传统的盲探测器相关模型与遗传算法 + MCMC 参数化模型，未见神经网络本底论文 [confirmed：检索结果] — [HE](https://arxiv.org/abs/2005.01661)、[ME](https://arxiv.org/abs/2003.06260)、[LE](https://arxiv.org/abs/2004.01432)、[You et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJS..256...47Y/abstract)
- NICER：transformer 模型 BERTground 由航天器/望远镜元数据预测本底谱 — [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30335)；独立比较指出 ML 估计器缺少内禀离群过滤 — [arXiv:2302.12896](https://arxiv.org/abs/2302.12896)
- Fermi-GBM：神经网络本底 + FOCuS 触发算法（DeepGRB）既能确认目录事件也能发现目录外事件；作者称方法为 GBM 定制 — [Crupi et al. 2023](https://arxiv.org/abs/2303.15936)，[博士论文](https://arxiv.org/abs/2401.15632)
- 信息论方法（互信息、条件互信息、transfer entropy）已系统用于太阳风驱动与辐射带电子的因果分析 — [Applications of Information Theory in Solar and Space Physics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7514618/)；2026 年的比较指出 CCM 对单向耦合更可靠，而只有 TE 能稳定捕捉双向耦合 — [EGU26-11122](https://meetingorganizer.copernicus.org/EGU26/EGU26-11122.html)；CCM 在空间气候中的应用例（宇宙线与全球温度）— [PNAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC4371914)
- 自激点过程/ETAS：地震余震模型已被用于快速射电暴（ETAS + 旋转矢量模型解释等待时间双峰）— [Luo et al. 2025, ApJ](https://arxiv.org/abs/2502.16626)；综述 — [Reinhart, A Review of Self-Exciting Spatio-Temporal Point Processes](https://arxiv.org/abs/1708.02647)；检索未发现专门针对磁星 X 射线暴的 Hawkes/ETAS 论文
- HXMT 已有 SGR J1935+2154 暴发相位分布研究 — [arXiv:2301.07333](https://arxiv.org/abs/2301.07333)

### Inferences

#### 想法 B7：残差流目录、衰变核提取与域适应
1. **一句话**：把 NeuBacH 的残差当作异常流，建立公开的"未解释本底偏移 (UBE)"目录并用 time-slide 法给出与外部事件目录关联的显著性；同时对模型做受控探针，读出它学到的 SAA 后衰变核；再用 B5 的交会样本把模型迁移到其他卫星。
2. **原理与来源**：(a) 引力波搜寻的 time-slide 本底估计（把一个目录的时间整体平移许多次，得到偶然符合的经验分布）+ 粒子物理的 trial factor；(b) 可解释性中的 partial-dependence/受控干预；(c) 迁移学习。具体检验：
   - **目录与归因**：对残差做变点检测与聚类，按 (持续时间, 能谱硬度, 哪些探测器同时偏移, L, 地方时) 分类；与 GOES 耀斑、Dst/Kp、闪电 (WWLLN)、TGF/GRB 目录、地震目录分别做 time-slide 关联。对"地震前粒子暴"这类有争议的主张，预先登记窗口与阈值后做一次盲检验，零结果本身可发表。
   - **衰变核**：固定其余输入、只扫描 T_SAA，得到模型的响应曲线并拟合多指数，与 A2 的显式反演对照。若模型输入里没有剂量史，它原则上无法表示天量级的活化记忆；检验办法是看残差在 1 天及更长滞后的自相关，并加入"过去 1 h/6 h/1 d/10 d 累计 PM 剂量"特征看改善多少。
   - **域适应**：在 HXMT–GBM/GECAM 交会点上学习两星特征空间的映射，再微调。
3. **HXMT 特性与诚实评价**：用户已有模型且 HXMT 上无竞争工作，这是现实优势。风险：NICER 的经验表明 ML 本底估计器若无离群过滤会产生不可信的扣除结果，目录的可信度取决于残差的系统误差地板（由 B1 的 ABCD 闭合给出）。
4. **数据**：NeuBacH 训练与推断数据（公开 L1）；外部目录。WWLLN 需申请，其余公开。
5. **查新**：HXMT 无同类工作；GBM 有 DeepGRB 与物理本底模型的暂现源搜寻；"从本底神经网络中提取活化时间常数"未找到先例（未做专门检索，[inference]）。
6. **可行性**：高。障碍：真实天体暂现源会混入残差，需先用已知目录屏蔽；因果分析应使用 TE/条件互信息而不是 CCM（系统是强随机驱动的）。灵敏度：32 s 分箱下 HE 总本底约 10⁴ 计数量级，泊松精度约 1%，因此百分之几、持续分钟级的偏移可逐个检出 [inference]。
7. **评分**：新颖 3 / 可行 4 / 影响 4 —— 是用户现有工作的自然延伸，且每一步都有可证伪的物理对照。

### Gaps
- 2024–2026 年 GECAM/GBM/HXMT 上是否出现新的深度学习本底论文：最后一轮检索未执行。
- NuSTAR、Swift/BAT 的 ML 本底模型：检索未返回结果。
- NeuBacH 的定量精度与残差统计：Zenodo 页未给。
- Sugihara et al. 2012（CCM 原始文献）、simulation-based inference 与符号回归的代表文献：【未检索核实】，未展开。
- "housekeeping 时间序列的基础模型/自监督表征"：未找到与具体物理检验挂钩的方案，按用户标准不入选。

---

## 考虑过但拒绝或降级的想法

### Takeaway
以下想法或者已被做过，或者 HXMT 并不比其他任务合适，或者期望信号量太小，或者缺少可检验的物理抓手。列出是为了避免报告撰写时重复提出。

### Cited Findings
- **X 射线掩地反演大气密度**：检索到至少四篇 HXMT 论文（其中三篇有链接）— [arXiv:2502.19678（ApJ 2025）](https://arxiv.org/abs/2502.19678)，[ApJS 2022](https://doaj.org/article/cafb0de60cb54e51948813f066fb5302)，[Atmos. Meas. Tech. 2022](https://doaj.org/article/2d81bdb495c448128e60333be0fba3be)。拒绝理由：已完成。
- **TGF 目录**：HXMT 已发布 282 个 TGF — [Zenodo 10976454](https://zenodo.org/records/10976454)。拒绝理由：已完成（TEB/电子沉降部分保留在 B2）。
- **模板匹配/相干搜索找亚阈值暴**：HXMT 已实现 targeted coherent search 并找回 40 个 GRB — [Cai et al. 2021](https://arxiv.org/abs/2109.12270)。拒绝理由：等价方法已存在；且磁星暴波形不像重复地震那样可复用，光变模板增益有限 [inference]。
- **微流星体/碎片撞击统计**：此类事件见于掠入射聚焦望远镜 — [Meteoroid and space debris impacts in grazing-incidence telescopes](https://arxiv.org/abs/0803.1983)，[eROSITA](https://journals.spiedigitallibrary.org/conference-proceedings-of-spie/12181/1218155/SRGeROSITA-micrometeoroid-hits-and-effects/10.1117/12.2630890.full)。拒绝理由：HXMT 无聚焦镜，期望约 0.1–1 次（见 A-Q5）。
- **Timepix/HST 式径迹形态 LET 谱仪** — [Miles et al. 2021](https://arxiv.org/abs/2006.00909)，[Timepix 综述](https://www.preprints.org/manuscript/202311.1087)。拒绝理由：SCD 是非成像的扫式读出，Si-PIN 像素为厘米级，主探测器对最小电离粒子饱和，没有 dE/dx 信息；保留为 A5 的粗分类版本。
- **Hawkes/ETAS 拟合 HXMT 暴发序列**：ETAS 已用于 FRB — [Luo et al. 2025](https://arxiv.org/abs/2502.16626)。降级理由：HXMT 是有地球遮挡与 SAA 间断的定点观测，占空比缺口会偏置等待时间与分支比推断；连续监测的 GECAM/GBM 或射电数据更合适 [inference]。
- **以暗物质谱线搜寻为主目标**：HXMT 的 PBH 限制论文自己指出多数情形下无优势 — [Yang, Pan & Zhang 2026](https://arxiv.org/abs/2412.09297)。降级理由：分辨率与活化线；只作为 B1 的副产品。
- **convergent cross mapping 做太阳风—本底因果**：比较研究显示 CCM 适合确定性单向耦合 — [EGU26-11122](https://meetingorganizer.copernicus.org/EGU26/EGU26-11122.html)。降级理由：改用 transfer entropy，作为 B7 的工具而非独立想法。
- **面向 eXTP 的 LEO 本底图集**：eXTP 已改为大椭圆轨道 — [arXiv:2506.08101](https://arxiv.org/abs/2506.08101)。拒绝理由：前提过时；图集的受益对象改为空间站载荷（A3）。

### Inferences
- **HE 模块间 Compton 成像/偏振**：18 个带被动准直器的大块 phoswich，模块间散射几何差、无硬件符合标志；POLAR、AstroSat/CZTI、GECAM 更合适 [inference，未检索]。拒绝。
- **银道面扫描的压缩感知/深度先验重建**：团队已有直接解调与四年巡天产品；改进空间受准直 PSF 与源混淆限制，且属于相邻课题范围。未入选。
- **辐射带 CT（重建俘获粒子投掷角分布）**：SAA 内 HE 关高压，剩余探测器角信息不足。并入 B2/A4。
- **SEU/闩锁/高压跳闸统计、轨道 β 角热模型**：数据不公开或无可迁移的科学结论。拒绝。
- **在轨 X 射线脉冲星导航**：凭记忆 HXMT 团队 2019 年已发表在轨演示【未检索核实】。拒绝理由：已完成。
- **CsI 全天监测器的地球掩星阶跃法监测亮源**（BATSE/GBM 的 EOT）：新颖性未能核实，且 GBM 视场更开阔、数据更连续。暂不入选，建议先查新。
- **"用基础模型重新分析全部 HK"**：没有具体可检验的物理命题，按用户标准拒绝。

### Gaps
- 上述"凭记忆已完成"的条目（脉冲星导航、EOT）需要补做查新。
- 最后一轮未执行的 9 个检索主题：HE 死时间论文 (Xiao et al. 2020)、Zhao et al. SAA 论文的原始链接与漂移率、俘获质子东西效应测量、HXMT 的 CXB 测量、Gibbons & Ringdal 2006、Chandra/Swift/XMM/Gaia 的在轨 CTI 对比、2024–2026 年伽马监测器深度学习本底、各任务轨道参数、Sugihara et al. 2012。
