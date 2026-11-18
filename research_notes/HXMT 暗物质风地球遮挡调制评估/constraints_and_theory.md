# HXMT 暗物质风地球遮挡调制评估：现有实验/天体物理约束与理论靶标（参数空间哪里还开放？）

阅读说明（非正文章节）：
- 标注：【文献证实】= 直接取自本轮实际取回的原文/公开数据；【推断】= 本笔记作者由文献数字做的推算；【猜测】= 尚无文献支撑的设想。
- 截面约定（全文严格区分）：**contact**（记 σ_c）= 对所有原子核相同的"几何/接触"截面，文献中又叫 Model I、"no A⁴ scaling"、"per-nucleus"、"nucleon-independent"、"geometric"；**per_nucleon**（记 σ_n）= 自旋无关每核子截面，σ_A = A²(μ_A/μ_n)² σ_n，重暗物质时 → A⁴σ_n，文献中又叫 Model II。f ≡ ρ_χ/ρ_DM 为该成分占暗物质的比例，除特别说明外文献限制均假设 f = 1。
- 单位换算：1 g = 5.61×10²³ GeV；1 cm²/g = 1.78×10⁻²⁴ cm²/GeV。
- "图读"表示从论文图中数字化读出，精度约 ±0.3 dex（个别标 ±0.5 dex）。
- 配套机读表：同目录 `existing_constraints.csv`（850 行，列为 label,convention,f,m_GeV,sigma_min_cm2,sigma_max_cm2,source_url,quality）。说明：(i) quality 三类分别为 table 520 行、figure_readoff 218 行、estimate 112 行；table 中 474 行直接取自 Picker et al. 2026 公开数据仓库（Zenodo 10.5281/zenodo.21892125）的 f_DM(m,σ) 数组与 DD 多边形，其余为论文表格或正文给出的数值；f 取 1、0.1、10⁻²、10⁻³、10⁻⁴、10⁻⁶、10⁻⁸ 七档（f < 1 的行只对有公开 f 依赖数据或可做简单通量缩放的限制给出）；(ii) "上不封顶"的 σ/m 型限制（gas cloud、CMB、MW satellites、Lyman-α）的 sigma_max 统一写成 1e-5 cm²，仅为作图上限，**不是物理边界**；(iii) quality=estimate 的行（IMP 7/8 的 contact 换算、XQC/Chicago 的 f<1 缩放、宇宙学限制的质量外推）是本笔记的量级推断；(iv) "Direct-detection union polygon"（CRESST 地表 + CDMS-I + DAMA + Chicago 等）只给了 f = 1，没有做 f 缩放；(v) label 中第一个" ("之前的短名相同的行是同一实验的不同版本（如 "Skylab plastic etch" 的 Picker 2026 与 Bhoonah 2021 两版、"Gas cloud heating" 的两版，二者相差约 5 倍），短名含 "superseded" 的是已被取代的旧版本（Starkman 时代的 Skylab、旧 mica），含 "IMP 7/8" 的 contact 行是未经同行评议结果的量级换算，作图时宜与其余限制区分线型。

---

## 1. 理论靶标：哪些模型预言会被大气或地球挡住的暗物质（或稀有亚成分）？Digman et al. 2019 对点状截面上限与标度约定怎么说？

### Takeaway
点状暗物质的接触截面不可能超过原子核几何截面量级（σ_χN ≳ 10⁻²⁵ cm² 即不可能是点状），所以"被大气挡住的重暗物质"只能是复合/延展对象，且此时最自然的参数化是与 A 无关的几何截面。文献中确有大量重复合模型（dark nuclei、nuggets、blobs、loosely bound composites），但**没有任何一篇把"m ~ 10³–10⁸ GeV、几何截面 10⁻²²–10⁻¹⁹ cm²、f ≪ 1"作为明确预言**；有明确丰度预言的强相互作用亚成分只有 Colored DM 的 hybrid 强子（12.5 TeV，f ~ 10⁻⁴）和 dark-photon 耦合的 GeV–百 GeV 费米子（f ≪ 1），而标准 macro/nuclearite/AQN/Q-ball/Planck 残骸的通量对 0.5 m² 探测器太低。

### Cited Findings
**(a) Digman et al. 2019 的结论与约定问题**
- 【文献证实】摘要原文：常用标度关系 σ_χA ∝ A²μ_A²σ_χN（重暗物质 ∝ A⁴σ_χN）在 σ_χA ~ 10⁻³²–10⁻²⁷ cm² 就普遍失效；"pointlike dark matter cannot have σ_χN ≳ 10⁻²⁵ cm²"；实现大截面"最可行的途径是 composite dark matter，但这通过 form factor 引入显著的模型依赖"；"All prior limits on dark matter with cross sections σ_χN > 10⁻³² cm² with m_χ ≳ 1 GeV must therefore be reevaluated and reinterpreted" — [Digman, Cappiello, Beacom, Hirata & Peter 2019, arXiv:1907.10618](https://arxiv.org/abs/1907.10618)
- 【文献证实】其结论分三档：σ_χN ≪ 10⁻³² cm² 标度成立；10⁻³²–10⁻²⁵ cm² 时 A⁴ 标度随重核截面在几何截面处饱和而逐步失效（图 6：A>131 约在 10⁻³² 失效，σ_χN ≳ 4×10⁻²⁸ 后任何核间标度都是模型依赖的）；> 10⁻²⁵ cm² 不可能是点状，接触相互作用不超过 σ_χA ≃ 4πr_A²，简单轻媒介子模型被强烈排除。勘误补充：吸引势下的共振可超过几何极限（轻核、m_χ ≲ m_A 时可超 3–5 个量级），但 m_χ ≳ m_A 时比氖重的核不可能超过几何极限一个量级 — [Digman et al. 2019](https://arxiv.org/abs/1907.10618)
- 【文献证实】对复合暗物质："the cross section need not scale with A at all"；"disregarding scaling with A may be the most correct way of plotting constraints"；并指出"IMP+IMAX+SKYLAB"区域在其图 1 中画成虚线，因为这些结果"commonly cited but are not based on detailed analyses in peer-reviewed papers" — [Digman et al. 2019](https://arxiv.org/abs/1907.10618)
- 【文献证实】Cappiello et al. 2021 采用的约定：复合暗物质对核"不透明"，散射截面等于暗物质几何尺寸、与靶核无关；此时"DM-nucleon cross section is equal to any DM-nucleus cross section"，A² 相干因子与 μ² 因子都不再出现 — [Cappiello, Collar & Beacom 2021, arXiv:2008.10646](https://arxiv.org/abs/2008.10646)
- 【文献证实】2026 年综述把两种参数化并列为多次散射分析的两个极限（Eq. 14 per-nucleon、Eq. 15 geometric σ_XN = σ_X），并指出"later work has shown that a certain class of loosely bound composites can have effective A⁴ enhanced scattering, even for very large nuclear cross-sections" — [Bramante 2026, arXiv:2602.23708](https://arxiv.org/abs/2602.23708)
- 【文献证实】同一"几何截面"在不同文献中差一个因子：Picker et al. 2026 与 Bhoonah et al. 用 σ = πR²；DEAP-3600 论文的饱和条件写作 σ_Tχ ≤ σ_geo = 4πR_D² — [Picker et al. 2026, arXiv:2609.05626](https://arxiv.org/abs/2609.05626)；[DEAP-3600 2022, arXiv:2108.09405](https://arxiv.org/abs/2108.09405)

**(b) 重复合暗物质（中性、接触/几何相互作用）**
- 【文献证实】Nuclear dark matter：携带守恒"暗核子数"的束缚态，合理的 asymmetric DM 模型中可合成核子数 ≳ 10⁸ 的暗核，数分布呈普适形式；与 SM 核的弹性散射可获得最高 A²（A 为组分数）的相干增强，并带有延展态特有的 form factor — [Hardy, Lasenby, March-Russell & West 2015, arXiv:1411.3739](https://arxiv.org/abs/1411.3739)；[同作者 2015, arXiv:1504.05419](https://arxiv.org/abs/1504.05419)
- 【文献证实】Dark nucleosynthesis 的其它实现：禁闭非阿贝尔规范群 + 轻媒介子可高效合成质量远高于禁闭标度的暗核 — [Krnjaic & Sigurdson 2015, arXiv:1406.1171](https://arxiv.org/abs/1406.1171)；格点显示双色双味 QCD 存在稳定核态 — [Detmold, McCullough & Pochinsky 2014, arXiv:1406.2276](https://arxiv.org/abs/1406.2276)；费米子 + 标量媒介子的 N ≫ 1 nugget — [Wise & Zhang 2014, arXiv:1407.4121](https://arxiv.org/abs/1407.4121)
- 【文献证实】ADM nugget 的三种结构区（非相对论、相对论、饱和），饱和区束缚能可达组分质量的 O(1) — [Gresham, Lou & Zurek 2017, arXiv:1707.02313](https://arxiv.org/abs/1707.02313)；早期宇宙合成的质量函数 — [Gresham, Lou & Zurek 2018, arXiv:1707.02316](https://arxiv.org/abs/1707.02316)；含 ≥10⁴ 暗核子的束缚态的直接探测：keV 阈值大曝光实验对延展态迅速失去灵敏度 — [Coskuner, Grabowska, Knapen & Zurek 2019, arXiv:1812.07573](https://arxiv.org/abs/1812.07573)
- 【文献证实】饱和复合体的半径 R(N) ≃ (9πN/4 m̄_X³)^{1/3}，内部数密度 n_sat ≃ m̄_X³/(3π²)；与 SM 核的相互作用分四个区，其中束缚场势 |⟨φ⟩| ≳ m_N 时核被反射、达到几何极限；loosely bound 复合体 R_D ~ N_D^{1/3} Λ_D⁻¹，束缚能小意味着"formation at later cosmic times" — [Bramante 2026](https://arxiv.org/abs/2602.23708)
- 【文献证实】Loosely bound composite DM：对核的截面可以很大且随核子数按 ~A⁴ 标度；某些耦合下单次原子反冲极软（≪ keV）而单次穿越探测器的总沉积能很大（≫ keV） — [Acevedo, Boukhtouchen, Bramante, Cappiello, Mohlabeng & Tyagi 2024, arXiv:2408.03983](https://arxiv.org/abs/2408.03983)；束缚极松时穿越地球会解体成级联 — [Boukhtouchen, Bramante, Cappiello & Diamond 2025, arXiv:2512.16043](https://arxiv.org/abs/2512.16043)
- 【文献证实】Dark quark nuggets：禁闭标度 10 keV–100 TeV 对应质量 10²³ g–10⁻⁷ g、半径 10⁸ cm–10⁻¹⁵ cm；"the free dark baryons, as a subcomponent of dark matter, can have direct detection signals" — [Bai, Long & Lu 2019, arXiv:1810.04360](https://arxiv.org/abs/1810.04360)
- 【文献证实】Dark blobs：强自相互作用使暗物质聚成数密度很低的大复合体，可利用其 ~220 km/s 的慢速穿越特征，在 MACRO、XENON、LUX、CDMS 等现有实验中搜索 — [Grabowska, Melia & Rajendran 2018, arXiv:1807.03788](https://arxiv.org/abs/1807.03788)
- 【文献证实】Macro：核密度对象仍开放的质量窗口为 55–10¹⁷ g 与 2×10²⁰–4×10²⁴ g — [Jacobs, Starkman & Lynn 2015, arXiv:1410.2236](https://arxiv.org/abs/1410.2236)；Axion quark nugget 为近核密度的宏观（反）夸克块 — [Zhitnitsky 2021, arXiv:2105.08719](https://arxiv.org/abs/2105.08719)；SUSY Q-ball 为稳定非拓扑孤子 — [Kusenko, Kuzmin, Shaposhnikov & Tinyakov 1998, arXiv:hep-ph/9712212](https://arxiv.org/abs/hep-ph/9712212)
- 【文献证实】扩展质量函数：复合体的合成一般给出宽质量分布；"if some subcomponent of the dark matter does not end up in large composite objects, it may be strongly constrained"，分布尾部往往给出最强约束 — [Picker, Buchanan, Diamond & Bramante 2026, arXiv:2609.05626](https://arxiv.org/abs/2609.05626)

**(c) 强相互作用/带色遗迹（亚成分）**
- 【文献证实】Colored Dark Matter：暗物质是两个稳定色八重态费米子 Q 组成的 Q-onlyum，M_Q ≈ 12.5 TeV；伴生的 hybrid 强子（Qg、Qqq̄′）是 SIMP，f_SIMP ≡ ρ_SIMP/ρ_DM "below 10⁻³, possibly a few orders of magnitude smaller"（正文估计 ~10⁻⁴，表 2 取 10⁻⁵）；σ_p ≈ π/Λ²_QCD ≈ 1.6×10⁻²⁶ cm²，并假设 σ_A ≈ σ_p A²(m_A/m_p)²；能损 dE/dx = −E Σ n_A σ_A 2m_A/M_Q；作者据此称"SIMPs with M_Q ≈ 10 TeV thermalize in the Earth atmosphere"，并称气球实验是否排除 ρ_SIMP = ρ_DM 尚存疑，"Our predicted abundance ρ_SIMP ∼ 10⁻⁴ ρ_DM is allowed" — [De Luca, Mitridate, Redi, Smirnov & Strumia 2018, arXiv:1801.01135](https://arxiv.org/abs/1801.01135)
- 【文献证实】带色遗迹的丰度因禁闭后的湮灭再降低几个量级 — [Gross, Mitridate, Redi, Smirnov & Strumia 2019, arXiv:1811.08418](https://arxiv.org/abs/1811.08418)
- 【文献证实】Sexaquark（uuddss，m_S ≲ 2 GeV，~2m_p）可稳定并作为暗物质，经 ω/φ 交换的 Yukawa 势与重子作用，截面可偏离 Born 近似好几个量级 — [Farrar 2017, arXiv:1708.08951](https://arxiv.org/abs/1708.08951)；[Farrar, Wang & Xu 2020, arXiv:2007.10378](https://arxiv.org/abs/2007.10378)
- 【文献证实】强相互作用亚成分的具体模型：耦合 sub-GeV dark photon 的隐费米子，m_χ ~ 1–100 GeV、m_A′ ~ 10–100 MeV，可自然给出 f_χ ≪ 1 与大的每核子截面；其图 1 给出 f_χ = 1、10⁻³、10⁻⁶、10⁻⁹ 时的现有限制 — [McKeen, Moore, Morrissey, Pospelov & Ramani 2022, arXiv:2202.08840](https://arxiv.org/abs/2202.08840)
- 【文献证实】Mirror Twin Higgs 的 nuclear atomic DM：孪生原子作为亚成分（要求 ≲ 10–20%） — [Asadi, Chacko, Curtin, Mellors & Neil 2026, arXiv:2611.15852](https://arxiv.org/abs/2611.15852)

**(d) 带电遗迹与长程/毫电荷情形**
- 【文献证实】CHAMP：质量与电荷满足 100(q_X/e)² ≲ m_X ≲ 10⁸(q_X/e) TeV 时，磁场阻止晕中粒子进入银盘，盘内的被 Fermi 机制加速并在 0.1–1 Gyr 内抛出，因而基于地面未观测到的旧限制在此范围失效 — [Chuzhoy & Kolb 2009, arXiv:0809.0436](https://arxiv.org/abs/0809.0436)
- 【文献证实】CHAMP cosmic rays：m ≲ 10¹⁰ q GeV 时银盘中有被超新星激波加速的 X 流量（谱 ∝ p⁻²·⁵，延伸至 (βp)_max ~ 5×10⁴ q GeV）；m ≳ 10¹⁰ q GeV 时无法被加速到逃逸速度以上；q > 10⁻⁹ 且 m < 10⁵ GeV 的 X 不能是全部暗物质，更大 q 或 m 时切伦科夫与电离探测器对 f_X 给出严格上限 — [Dunsky, Hall & Harigaya 2019, arXiv:1812.11116](https://arxiv.org/abs/1812.11116)
- 【文献证实】带电 Planck 质量黑洞残骸：探测主要受通量限制；慢速带电重粒子的阻止本领量级为 (dE/dx)/ρ ~ 30 MeV/(g/cm²)；MACRO 磁单极搜索给出通量上限 5.5×10⁻⁴ m⁻² yr⁻¹（90% CL），显著低于其估计的残骸通量 — [Lehmann, Johnson, Profumo & Schwemberger 2019, arXiv:1906.06348](https://arxiv.org/abs/1906.06348)
- 【文献证实】毫电荷/超轻 dark photon 媒介的 sub-GeV 暗物质在大截面处只允许作为亚成分（CMB 限制 f ≲ 0.4%）；毫电荷粒子还可能被银河磁场、超新星与太阳风磁场从银盘/地球附近排开，使高截面区难以用气球或卫星探测，而有质量的超轻 dark photon（m_A′ ≳ 10⁻²⁰ eV）可屏蔽这些磁场效应 — [Emken, Essig, Kouvaris & Sholapurkar 2019, arXiv:1905.06348](https://arxiv.org/abs/1905.06348)

**(e) β ~ 10⁻³ 时在普通物质中的能损**
- 【文献证实】重中性复合体（m_χ ≫ m_A）：dE/dx = (2E/m_χ) Σ_A (μ²_χA/m_A) n_A σ_χA，积分得 E(x) = E(0) exp[−(2x/m_χ) Σ n_A (μ²_χA/m_A) σ_χA]；contact 情形把 σ_χA 换成 σ_c — [Bhoonah, Bramante, Courtman & Song 2021, arXiv:2012.13406](https://arxiv.org/abs/2012.13406)
- 【文献证实】等价写法：dE/dx ~ ρσ_eff v²；单次碰撞反冲能 E_R ≈ m_N v² ~ 100 keV (m_N/100 GeV)(v/10⁻³c)²；每次散射偏转角 sinθ_max ~ m_N/m_X，m_X ~ 10¹⁰ GeV 时累计偏转 < 10⁻³ rad；液氙/液氩中 τ ~ 1 对应 σ_XN ~ 2×10⁻²³ cm²（1 m 路径） — [Bramante 2026](https://arxiv.org/abs/2602.23708)
- 【文献证实】contact 截面下阻止本领只取决于质量柱密度而与元素组成无关（重核数密度少 1/A、每次碰撞能损多 A 倍），所以"a given mass column density of lead does not have significantly higher stopping power than the same mass column density of concrete" — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)

### Inferences
- 【推断】把理论对象放进 (m, σ_c) 平面：核密度线 σ ≈ 6×10⁻²⁵ cm² (m/100 GeV)^{2/3}（由 Cappiello et al. 2021 图 4 的"Nuclear Density"点线图读）只在 m ≲ 3×10⁴ GeV 时高于大气遮蔽线 σ/m ≈ 10⁻²⁷ cm²/GeV；因此核密度 nugget 在更高质量下**不会**被大气挡住。HXMT 可及的"大气遮蔽且能穿过卫星蒙皮"区（m ~ 10³–10⁸ GeV、σ_c ~ 10⁻²²–10⁻¹⁹ cm²，对应半径 R = √(σ/π) ≈ 56 fm–1.8 pm）要求内部密度约 10¹²–10¹³ g/cm³，即比核物质稀 10–10³ 倍；由 ρ ∝ Λ_D⁴ 粗估相当于暗禁闭标度 Λ_D ~ 50–150 MeV、核子数 10⁴–10⁹ 的"暗核"，或饱和 nugget 中有效组分质量 m̄_X ~ 0.1 GeV（例：N = 10⁶ 时 m ≈ 10⁵ GeV、R ≈ 380 fm、σ ≈ 4.5×10⁻²¹ cm²、σ/m ≈ 4.5×10⁻²⁶ cm²/GeV）。前提是与 SM 核的耦合强到几何饱和。
- 【推断】各类靶标与 HXMT 的关系：

| 模型 | 质量 | 预期 f | 相互作用与能损 | 是否落入 HXMT 可及区 |
|---|---|---|---|---|
| Dark nuclei / ADM nugget（低于核密度） | 10³–10⁸ GeV（可更高） | 文献按 f = 1 讨论；质量函数低质量尾可给 f ≪ 1 | 几何 σ = πR²，dE/dx = ρσv² | 是（见第 4 节窗口 W1），但 f = 1 已被排除 |
| Loosely bound composites | TeV 组分、N 很大 | f = 1 | 可保持 A⁴ 标度；单次反冲极软 | per-nucleon 约定下 HXMT 的 I/Cs 靶极有利（第 4 节） |
| Dark quark nuggets | ≥ 10⁻⁷ g ≈ 6×10¹⁶ GeV | 1 | 几何 | 通量太低（调制搜索需 m ≲ 10¹² GeV） |
| Macro / nuclearite / AQN | 开放窗口 ≥ 55 g ≈ 3×10²⁵ GeV | 1 | ρσv² 或湮灭 | 否，比调制搜索可及质量（≲ 10¹² GeV）高 ≥ 13 个量级 |
| SUSY Q-ball、带电 Planck 残骸 | Q-ball 质量范围未从原文核对（凭记忆约 10¹² GeV 以上）；Planck 残骸 ~10¹⁹ GeV | ≤ 1 | 类 nuclearite / 电离 | 否，MACRO 的通量上限（5.5×10⁻⁴ m⁻² yr⁻¹）比 HXMT 约 4.5 m² yr 曝光所能达到的强约 3 个量级 |
| Colored-DM hybrid SIMP | 12.5 TeV | ~10⁻⁴（10⁻³–10⁻⁵ 或更低） | 强子截面 | 见下条：按几何饱和算并不被大气挡住 |
| Dark-photon 耦合费米子（McKeen） | 1–100 GeV | ≪ 1 | 每核子大截面，大气中热化 | HE 阈值够不着；LE/ME 硅探测器或可（窗口 W2） |
| Sexaquark | ~2 GeV | 可为 1 | 强子型 | 否，动能 ~1 keV，远低于 HXMT 阈值 |
| CHAMP（单位电荷） | 10⁵–10¹¹ GeV 被排出银盘 | 受限很强 | 30 MeV/(g/cm²) 等效 σ ≈ 5×10⁻²⁰ cm² | 能损恰在 W1 范围且低于 Skylab 蚀刻阈值，但不存在冷的"风"（猜测级） |

- 【推断】Colored DM 的 hybrid 强子"在大气中热化"的说法依赖未饱和的相干标度（⟨A⁴⟩^{1/4} ≈ 16.6 时 σ_A ≈ 1.2×10⁻²¹ cm²，远超核几何截面）。若按 Digman et al. 的结论让截面在 4πR_A² ≈ 1×10⁻²⁴ cm²（N、O）处饱和，则整层大气只有约 46 次碰撞，每次损失 2m_A/M_Q ≈ 2×10⁻³，12.5 TeV 粒子到达地面时仍保有约 90% 动能；它会在几十米岩石内被挡住，但浅层低本底探测器可见。本笔记估算（同样假设几何饱和）：饱和截面下被大气挡住的质量上限约 300 GeV（E_final/E_0 ≈ 2×10⁻²），≥ 1 TeV 基本穿透。
- 【推断】单位电荷慢重粒子的阻止本领 30 MeV/(g/cm²) 相当于 σ_eff = (dE/dx)/(ρv²) ≈ 5×10⁻²⁰ cm²（β = 10⁻³），比 Skylab Lexan 的蚀刻阈值 ~0.4–0.5 GeV cm²/g 低一个量级以上，即 Skylab 对慢速单电荷 CHAMP 不灵敏。

### Gaps
- Witten 1984、De Rújula & Glashow 1984（nuclearite 能损原始公式）、Starkman et al. 1990 全文、Khlopov 的 O-helium/dark atom 系列未取回原文；相关数字转引自上列文献。
- 未找到任何文献明确预言"m ~ 10³–10⁸ GeV、几何截面 10⁻²²–10⁻¹⁹ cm²、f ≪ 1"的遗迹丰度；"质量函数尾部"只有 Picker et al. 2026 的一般性陈述，没有针对该区的定量计算。
- 复合体在与核碰撞时是否保持束缚（Picker et al. 2026 明确把"不被打散"作为假设）在该密度范围内没有专门研究。

---

## 2. 约束图：各实验/天体物理限制在 (m, σ) 平面上的数值边界（两种约定，f = 1）

### Takeaway
在 contact 约定、f = 1 下，把公开数据与图读结果合并后，从 10 GeV 到 10¹³ GeV、σ_c ≥ 10⁻²⁶ cm² 的范围内**没有任何未被排除的截面区间**：大气遮蔽线（σ/m ≈ (1–7)×10⁻²⁷ cm²/GeV）以上被 Lyman-α、MW satellites、gas clouds 覆盖（单靠 Planck CMB 的 1.6×10⁻²⁶ cm²/GeV 不够），其中还叠有 XQC（m ≤ 10⁵ GeV）与 Skylab（m ≥ 10⁶ GeV、σ_c ≥ 1.5×10⁻¹⁹ cm²）。在 per-nucleon 约定下大气遮蔽线低约 5 个量级（σ_n/m ≈ 4.5×10⁻³² cm²/GeV），其上到宇宙学限制之间靠 XQC（同行评议）、IMP 7/8/IMAX/Pioneer 11（非同行评议的旧 recast）、A⁴ 增强的 gas cloud 与 Skylab 覆盖。

### Cited Findings
**深地多次散射/单次散射**
- 【文献证实】DEAP-3600：3.3 t 液氩，SNOLAB 地下 2 km，813 ± 8 live-days 盲分析，无候选事例（预测 > 2.3 事例的模型以 90% CL 排除）；排除 m = 8.3×10⁶–1.2×10¹⁹ GeV、⁴⁰Ar 散射截面 1.0×10⁻²³–2.4×10⁻¹⁸ cm²；给出 Model I（不透明、无 A 标度）与 Model II（A⁴）两种解释；Model II 实线区约 σ_n ≈ 4×10⁻³⁰–8×10⁻²⁸ cm²，外推区到 ≈ 9×10⁻²⁵ cm²（图读） — [DEAP-3600 2022, arXiv:2108.09405](https://arxiv.org/abs/2108.09405)
- 【文献证实】XENON1T MIMP 搜索：219.4 天，预期本底 0.05，无事例；SI 作用排除 m = 1×10¹²–2×10¹⁷ GeV（图中至 3×10¹⁷）；A⁴ 约定 σ_n ≈ 6×10⁻³²–1.7×10⁻²⁹ cm²，无 A⁴（per-nucleus）约定 ≈ 1.6×10⁻²³–5×10⁻²¹ cm²（图读） — [XENON 2023, arXiv:2304.10931](https://arxiv.org/abs/2304.10931)
- 【文献证实】LZ：SR1 重分析，曝光 0.9 tonne×year，无事例（2.44 事例排除）；多次散射分析最大探测 σ_n = 10⁻²⁹ cm²（A⁴ 约定）；最大质量 3.9×10¹⁷ GeV；overburden 边界只对 m ≲ 10⁵ GeV 相关；同时给出"no scaling"的 per-nucleus 限制（图读上沿 ≈ 3×10⁻²¹ cm²） — [LZ 2024, arXiv:2402.08865](https://arxiv.org/abs/2402.08865)
- 【文献证实】PandaX-4T 的多次散射分析"are ongoing"；XENON1T、LZ、DEAP-3600 三者均未见事例 — [Bramante 2026](https://arxiv.org/abs/2602.23708)
- 【文献证实】CDEX-10：205.4 kg·day，0.16–4.16 keVee，排除 m = 10⁶–10¹¹ GeV 的 SI UHDM-核子散射（两种截面标度） — [CDEX 2026, arXiv:2510.21458](https://arxiv.org/abs/2510.21458)
- 【文献证实】MAJORANA Demonstrator 的 lightly-ionizing particle 通量限制可排除质量至 10¹⁵ GeV 的 MIMP；XENON1T 常规分析在高质量区需计入通量限制与多次散射 — [Clark et al. 2020, arXiv:2009.07909](https://arxiv.org/abs/2009.07909)
- 【文献证实】DAMA 1999 是第一个多次散射搜索，质量至 10¹⁵ GeV — [Bramante 2026](https://arxiv.org/abs/2602.23708)；其 contact recast 的上沿由 Gran Sasso 岩石 overburden 决定、下沿按碘散射换算 — [Bhoonah, Bramante, Schon & Song 2021, arXiv:2010.07240](https://arxiv.org/abs/2010.07240)
- 【文献证实】Simpzilla：EDELWEISS 与 CDMS 结果排除了最自然的强相互作用超重粒子，除非 m > 10¹⁵ GeV 或相互作用弱 — [Albuquerque & Baudis 2003, arXiv:astro-ph/0301188](https://arxiv.org/abs/astro-ph/0301188)
- 【文献证实】Kavanagh 2018（per-nucleon）：CRESST 2017 地表运行（Al₂O₃，4.6×10⁻⁵ kg·day，19.7–600 eV，511 事例）的上限 ceiling 为 σ_p^SI = 4.5×10⁻²⁷ cm²（m = 10⁵ GeV），按 m 线性外推并截断于 10⁸ GeV；CDMS-I（SUF，d = 10.6 m，阈值 10 keV）的 ceiling 为 1.55×10⁻²⁸ cm²（10⁵ GeV），延伸到 O(10¹⁵) GeV；这些"close a number of gaps ... between underground experiments and high-altitude experiments in the mass range between 10⁷ and 10¹³ GeV"；地壳中 N_scat ≈ 500 (σ_p^SI/10⁻²⁸ cm²)(D/m) — [Kavanagh 2018, arXiv:1712.04901](https://arxiv.org/abs/1712.04901)
- 【文献证实】Bramante et al.：穿越 overburden 时散射次数有上限（由路径上散射中心总数决定），把部分限制延伸到任意大截面；大体积液闪探测器可探 σ_n ≳ 10⁻²⁸ cm²、质量至 10²¹ GeV（预期） — [Bramante, Broerman, Lang & Raj 2018, arXiv:1803.08044](https://arxiv.org/abs/1803.08044)；[Bramante et al. 2019, arXiv:1812.09325](https://arxiv.org/abs/1812.09325)

**浅层与地表**
- 【文献证实】Chicago 探测器：两个 1.5 L EJ-301 液闪模块上下相距 50 cm，6.25 m.w.e. 浅地下室；两次运行 69 天（全铅屏蔽）与 58.2 天（移去顶部铅砖）；每模块阈值 10 keVee（质子反冲淬灭 20%，约需 50 次质子散射）；ROI −3.5 μs < Δt < −0.8 μs（140–625 km/s）；候选 0 与 1 个，90% CL 上限 2.9 与 4.4 事例；接受立体角为 4π 的 1.3×10⁻³；右边界 m ≃ ρv(AΩt) ≃ 4×10¹² GeV，斜边 σ/m ≃ 10⁻²⁷ cm²/GeV（大气 ~10 m.w.e. + 混凝土），底边 σ ≃ 50/(n_H L) ≃ 10⁻²² cm² — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)
- 【文献证实】同文图 4 数字化（图读）：排除区 m = 2.7×10⁴–1.3×10¹² GeV；底边 σ_c ≈ 5×10⁻²⁴（10⁵ GeV）→ 8×10⁻²⁴（10⁸）→ 1.6×10⁻²³（10¹⁰）→ 3×10⁻²²（10¹² GeV）；斜边 σ_c/m ≈ (2–7)×10⁻²⁷ cm²/GeV — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)
- 【文献证实】CRESST 2017 地表运行：0.5 g 蓝宝石，阈值 19.7 ± 0.9 eV，无显著屏蔽 — [CRESST 2017, arXiv:1707.06749](https://arxiv.org/abs/1707.06749)；contact recast 在 σ_th = 3.5×10⁻²³ cm²（晶体内可散射两次以上）处截断；CDMS-I 浅站 contact recast 截断于 4.2×10⁻²² cm²，要求 99.4 live days 内 27 个粒子穿过，质量限于 < 4×10¹³ GeV — [Bhoonah et al. 2021](https://arxiv.org/abs/2010.07240)
- 【文献证实】EDELWEISS-Surf：33.4 g 锗，地表，60 eV 分析阈值，并做了计入地球屏蔽的 SIMP 分析 — [EDELWEISS 2019, arXiv:1901.03588](https://arxiv.org/abs/1901.03588)；Collar 的非相对论宇宙线成分搜索给出 < 100 MeV 暗物质核反冲的首个限制 — [Collar 2018, arXiv:1805.02646](https://arxiv.org/abs/1805.02646)；液氮杜瓦蒸发率实验约束在地球中热化、形成"暗物质大气"（地表数密度可达 10¹⁴ cm⁻³）的 GeV 暗物质 — [Xu & Farrar 2021, arXiv:2112.00707](https://arxiv.org/abs/2112.00707)

**大气层以上**
- 【文献证实】XQC 探空火箭：34 个量能器（HgTe 0.96 μm 沉积在 14 μm Si 上，0.5 mm × 2.0 mm），165–225 km（平均 201.7 km），等效全阵列观测 100.7 s；可用能谱 0.03–2.5 keV，另有 60 个 > 4 keV 的过饱和事例；遭遇的暗物质粒子数 N_dm = f (ρ/m) × (2.5 ± 0.3)×10¹⁰ cm³；Monte Carlo 结果"rules out a wide range of nucleon-scattering cross sections centered around one barn for dark matter particles with masses between 0.01 and 10⁵ GeV"；图 8 给出 f_d = 1、0.5、0.25、0.1 时排除区的收缩；除 XQC 与 EDEL+CDMS 外，图 9 中其它区域都假设相干散射，"likely too broad" — [Erickcek, Steinhardt, McCammon & McGuire 2007, arXiv:0704.0794](https://arxiv.org/abs/0704.0794)
- 【文献证实】XQC per-nucleon 边界（图 9 图读）：10 GeV：2.5×10⁻²⁹–3×10⁻²²；10² GeV：5×10⁻³⁰–1.3×10⁻²²；10⁴ GeV：6×10⁻²⁸–6×10⁻²¹；10⁵ GeV：1.3×10⁻²⁶–6×10⁻²⁰ cm² — [Erickcek et al. 2007](https://arxiv.org/abs/0704.0794)
- 【文献证实】XQC 的 contact recast（上沿按氮 A ≃ 14、下沿按硅 A ≃ 28 由 per-nucleon 结果换算）；图 2 图读（±0.5 dex）：10 GeV：1×10⁻²⁴–2.5×10⁻¹⁸；10² GeV：3×10⁻²⁴–4×10⁻¹⁸；10⁴ GeV：4×10⁻²²–2.5×10⁻¹⁶；右边界 10⁵ GeV — [Bhoonah et al. 2021](https://arxiv.org/abs/2010.07240)
- 【文献证实】XQC 与 DAMIC 联合关闭了 0.3–100 GeV、~μb 的窗口 — [Mahdawi & Farrar 2017, arXiv:1709.00430](https://arxiv.org/abs/1709.00430)；但 sub-keV 核反冲的热化效率未测，可能在 0.2–2 GeV 留下"洞"；并强调 overburden 给出的上边界"extremely sharp but quite insensitive to the statistics" — [Mahdawi & Farrar 2018, arXiv:1804.03073](https://arxiv.org/abs/1804.03073)
- 【文献证实】RRS 气球：大气顶附近的硅半导体探测器，排除 ~2 GeV–10⁵ GeV 的强相互作用粒子作为晕的主要成分 — [Rich, Rocchia & Spiro 1987, Phys. Lett. B 194, 173](https://doi.org/10.1016/0370-2693(87)90788-X)；后人按其报告的柱深假设飞行高度 50 km — [Hooper & McDermott 2018, arXiv:1802.03025](https://arxiv.org/abs/1802.03025)
- 【文献证实】IMP 7/8、Pioneer 11、IMAX、Skylab 的旧 recast：Pioneer 11 硅探测器前有 1.7 mg/cm² 钛屏蔽（Starkman et al. 未计入）；Skylab 区域曾被画反；IMP7/IMP8 星载硅探测器的数据来自 Mewaldt 等的 private communication；IMAX 是气球上用塑料闪烁体的 SIMP 搜索（飞行时间技术，McGuire 1994 学位论文） — [McGuire & Steinhardt 2001, arXiv:astro-ph/0105567](https://arxiv.org/abs/astro-ph/0105567)；IMP7 阈值 160 keV（XQC 为 25 eV），IMP7/8 与 IMAX 的完整分析当时"in preparation" — [Wandelt et al. 2000, arXiv:astro-ph/0006344](https://arxiv.org/abs/astro-ph/0006344)
- 【文献证实】这些区域在 Erickcek et al. 图 9 中的 per-nucleon 边界（图读，均假设相干标度）：IMP 7/8：m = 5×10²–10¹⁰ GeV，下沿 1.6×10⁻²⁶ cm²（m ≤ 3×10⁷ GeV）升至 5×10⁻²⁴（10¹⁰ GeV），上沿 ≈ 2×10⁻²⁴ (m/500 GeV)^{1.18}；Pioneer 11：m = 10⁴–2×10⁸ GeV，下沿 ≈ 10⁻²³；IMAX：m = 2×10⁵–10¹¹ GeV，下沿 1.6×10⁻²⁵；Skylab（旧）：m = 2.5×10⁶–1.6×10¹⁷ GeV，下沿 1.3×10⁻²² — [Erickcek et al. 2007](https://arxiv.org/abs/0704.0794)
- 【文献证实】Skylab 的现代 recast：1.17 m² Lexan 阵列（36 模块 × 32 片 × 250 μm），在轨 2.19×10⁷ s，overburden 为 0.37 cm 铝壁，天顶角 ≤ 60°，约 150 个贯穿事例被保守地全当作候选，阈值 dE/dx ≈ 0.5 GeV/cm；采用 Maxwell-Boltzmann 速度分布并计入本底后"lessening the previously reported Skylab mass reach by orders of magnitude, while somewhat improving low cross section sensitivity" — [Bhoonah et al. 2021, arXiv:2012.13406](https://arxiv.org/abs/2012.13406)
- 【文献证实】同文图 2（contact，图读）：Skylab 排除 m = 8.5×10⁵–3.8×10¹⁵ GeV，下沿 σ_c ≈ 1.5×10⁻¹⁹ cm²，上沿 σ_c/m ≈ (4–10)×10⁻²⁴ cm²/GeV；Ohya 排除 m = 2.6×10⁹–4.8×10¹⁹ GeV，下沿 ≈ 9×10⁻²⁰ cm²，上沿 σ_c/m ≈ 1.3×10⁻²⁷。图 1（per-nucleon）：Skylab m = 1.6×10⁷–3.7×10¹⁵ GeV、下沿 σ_n ≈ 5.4×10⁻²⁴ cm²；Ohya m = 7×10¹⁰–4.6×10¹⁹ GeV、下沿 ≈ 2.5×10⁻²⁴ cm² — [Bhoonah et al. 2021](https://arxiv.org/abs/2012.13406)
- 【文献证实】Ohya：2442 m² CR-39，地下运行 767 天（2.1 yr），overburden 10⁴ g/cm² 岩石（约 37 m），取 10% 各向同性通量效率对应天顶截止角 18.4°，阈值 0.3 GeV/cm，20 个候选孔 — [Bhoonah et al. 2021](https://arxiv.org/abs/2012.13406)
- 【文献证实】更早的表述：Skylab 对弹性散射 macro 要求 ρ⁻¹dE/dx ≥ 400 MeV cm²/g，即 v ≃ 250 km/s 时 σ_X ≥ 10⁻¹⁸ cm²（"σ_X ≳ 10⁻¹⁵ cm²"只是非弹性散射 macro 的蚀刻判据），需穿透 0.25 cm 故 σ_X/M_X ≲ 3 cm²/g，曝光 ~2 m² yr sr，排除 M ≲ 10¹⁷ GeV；星系团气体加热给出 σ_X/M_X < 0.06 cm²/g — [Jacobs et al. 2015](https://arxiv.org/abs/1410.2236)
- 【文献证实】Picker et al. 2026 的统一处理（表 II）：Skylab 面积 1.17 m²、0.70 yr、接受角 60°、overburden 4.0 g/cm²、R = 1 nm 时阈值速度 1.46 km/s；Ohya 2442 m²、2.1 yr、18.4°、1.15×10⁴ g/cm²、1.08 km/s；Mica 0.24 m²、5×10⁸ yr、60°、1.07×10⁷ g/cm²、下阈值 8.85 km/s、上阈值 1.42×10⁴ km/s、截止截面 3.14×10⁻⁶ cm²；阈值速度 v_min ∝ R⁻¹；三者 N_obs 取 2.3；Skylab 与 Ohya 采用含 LMC 的模拟速度分布 — [Picker et al. 2026](https://arxiv.org/abs/2609.05626)
- 【文献证实】其公开数据给出的 f = 1 排除区（contact）：Skylab m = 9.3×10⁵–2.2×10¹⁷ GeV、σ_c ≥ 1.7×10⁻¹⁹ cm²（10⁶ GeV 处仅 3.2–5.9×10⁻¹⁹）、上沿 2.9×10⁻¹⁶（10⁸）、5.5×10⁻¹⁴（10¹⁰）、5.7×10⁻¹²（10¹² GeV）；Ohya m = 2×10⁹–3.7×10²⁰ GeV、σ_c ≥ 9.3×10⁻²⁰；DEAP m = 8.4×10⁶–1.1×10¹⁹ GeV、1.06×10⁻²³–2.4×10⁻¹⁸ — [macrolimits v1.0.0, Zenodo](https://doi.org/10.5281/zenodo.21892125)

**古矿物**
- 【文献证实】Mica：Price & Salamon 的通量上限 Φ < 10⁻¹⁸ cm⁻² s⁻¹ sr⁻¹，对应 M_X ≲ 10²⁶ GeV；阈值 ρ⁻¹dE/dx > 2.4 GeV cm² g⁻¹ 即 σ ≃ 4×10⁻¹⁸ cm²；按 3 km 地壳、损失 50% 动能得 σ < 10⁻¹⁵ cm² (M_X/10¹⁵ GeV) — [Bhoonah et al. 2021](https://arxiv.org/abs/2010.07240)；Acevedo et al. 用 1200 cm²、5×10⁸ yr 的云母并保守取埋深 20 km 重新推导 — [Acevedo, Bramante & Goodman 2021, arXiv:2105.06473](https://arxiv.org/abs/2105.06473)
- 【文献证实】2026 年的批评：旧 mica recast 假设 α 反冲级损伤可保留 Gyr（金云母中 α 反冲径迹封闭温度仅 26–37 °C，~10⁶ yr 即退火）且忽略了原实验只选"optically clean"样品的筛选，因而"those limits are not reliable in the high–scattering cross-section regime" — [Boukhtouchen et al. 2026, arXiv:2606.02579](https://arxiv.org/abs/2606.02579)；Picker et al. 相应提高了 mica 的阈值速度并加了上截止 — [Picker et al. 2026](https://arxiv.org/abs/2609.05626)
- 【文献证实】石英电子显微读出是提案而非已有限制 — [Ebadi et al. 2021, arXiv:2105.03998](https://arxiv.org/abs/2105.03998)

**天体物理与宇宙学（直接约束 σ_p，f = 1）**
- 【文献证实】CMB（Planck 2015，95%）：速度无关 SI 散射排除 σ > 1.7×10⁻²⁵ cm²（1 GeV）、1.6×10⁻²³ cm²（1 TeV）；高质量端动量转移率 ∝ σ_p/m_χ，"can be directly extended to higher masses by appropriately scaling our reported limit at 1 TeV"；计入氦的相干散射使高质量端加强约 6 倍 — [Gluscevic & Boddy 2018, arXiv:1712.07133](https://arxiv.org/abs/1712.07133)
- 【文献证实】CMB + SDSS Lyman-α：n = 0 时 σ_0/m_χ < 0.12 cm²/g（仅 CMB）与 3.3×10⁻³ cm²/g（CMB + Lyman-α），对 m_χ ≫ m_H 成立且保守地忽略氦 — [Dvorkin, Blum & Kamionkowski 2014, arXiv:1311.2937](https://arxiv.org/abs/1311.2937)；m = 10 GeV 时 σ_0 < 1.9×10⁻²⁴（TTTEEE）与 5.8×10⁻²⁶ cm²（加 Lyman-α） — [Xu, Dvorkin & Chael 2018, arXiv:1802.06788](https://arxiv.org/abs/1802.06788)
- 【文献证实】只要同时允许氦与氢散射且速度幂次非负，m ≫ m_p 时氦散射主导 CMB 约束 — [Boddy, Krnjaic & Moltner 2022, arXiv:2204.04225](https://arxiv.org/abs/2204.04225)
- 【文献证实】MW satellites：σ_0 < 6×10⁻²⁷ cm²（10 GeV），m ≫ 1 GeV 时按 m 标度并"can be extrapolated to higher masses" — [Nadler, Gluscevic, Boddy & Wechsler 2019, arXiv:1904.10000](https://arxiv.org/abs/1904.10000)；DES+PS1 更新为 1.7×10⁻²⁷ cm²（10 GeV，95%） — [DES/Nadler et al. 2021, arXiv:2008.00022](https://arxiv.org/abs/2008.00022)；同数据的数值分析给出 1.8×10⁻²⁷（1 GeV）、1.2×10⁻²⁶（10 GeV）、1.3×10⁻²⁵ cm²（100 GeV） — [Maamari et al. 2021, arXiv:2010.02936](https://arxiv.org/abs/2010.02936)
- 【文献证实】Lyman-α forest（m = 10 keV–100 GeV，95%）：log σ[cm²] < −27.91（1 GeV）、< −26.28（100 GeV） — [Rogers, Dvorkin & Peiris 2022, arXiv:2111.10386](https://arxiv.org/abs/2111.10386)
- 【文献证实】Cappiello et al. 图 4 把 Nadler et al. 的限制外推到 10¹⁸ GeV，图读为 σ/m ≈ 1.5×10⁻²⁸ cm²/GeV，并写明"Our exclusion region is already partially covered by the cosmological constraints" — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)
- 【文献证实】Gas clouds：银心冷气体云的暗物质加热不得超过冷却率，对 contact 几何截面给出限制，图 2 红线图读为 σ_c/m ≈ 7.5×10⁻²⁸ cm²/GeV（10²–10¹⁶ GeV）；同图 CMB 线图读 1.6×10⁻²⁶ cm²/GeV — [Bhoonah et al. 2021](https://arxiv.org/abs/2010.07240)；Picker et al. 的公开数据给出 f = 1 时 σ_c/m ≈ (1.3–2.8)×10⁻²⁸ cm²/GeV（云 G357.8-4.7-55），并称其"competitive with both the CMB and Milky Way satellite constraints" — [Picker et al. 2026](https://arxiv.org/abs/2609.05626)
- 【文献证实】对最初 gas cloud 限制的质疑：所用云处于银心高速热外流中，不能假设在辐射冷却时标上稳定，且云参数有误 — [Farrar, Lockman, McClure-Griffiths & Wadekar 2020, arXiv:1903.12191](https://arxiv.org/abs/1903.12191)；独立的 Leo T 矮星系气体冷却给出 v_rel ~ 17 km/s 下的暗物质-重子限制 — [Wadekar & Farrar 2021, arXiv:1903.12190](https://arxiv.org/abs/1903.12190)
- 【文献证实】Picker et al. 对宇宙学限制的态度：流体近似在其图的质量范围内成立，但这些限制"none ... were obtained with the intention of extending their limits to arbitrarily high masses"，故出于谨慎未画入图 1，而以 gas cloud 作为主要 ceiling — [Picker et al. 2026](https://arxiv.org/abs/2609.05626)
- 【文献证实】银盘稳定性：σ_χN < 5×10⁻²⁴ (m_χ/GeV) cm²（Starkman et al.），更细致的研究要求 < 5×10⁻²⁵ (m_χ/GeV) cm²，均只考虑氢 — [Mack, Beacom & Bertone 2007, arXiv:0705.4298](https://arxiv.org/abs/0705.4298)
- 【文献证实】地球热流：若暗物质被地球高效俘获并自湮灭，可在 ~1–10¹⁰ GeV 关闭天体物理限制与地下探测器之间的窗口 — [Mack et al. 2007](https://arxiv.org/abs/0705.4298)；但 s 波自湮灭截面 ≲ 10⁻⁴⁴ cm² 时地球与火星加热限制被解除 — [Bramante, Buchanan, Goodman & Lodhi 2020, arXiv:1909.11683](https://arxiv.org/abs/1909.11683)
- 【文献证实】天体"存活"限制（per-nucleon）：不湮灭的重暗物质在太阳、木星、地球、月球核心积累并塌缩成黑洞；作者称这些天体的存在"exclude σ_χn = 10⁻²⁴ cm² for m_χ = 10⁷ GeV, not ruled by any other probes"；上沿由漂移时间 ≤ 1 Gyr 决定，地球为 σ_χn ≤ 10⁻²¹ cm² (m_χ/10⁷ GeV)、太阳为 10⁻¹⁷ cm² (m_χ/10⁷ GeV)；m_χ ~ 10¹⁰ GeV 附近因新生黑洞 Hawking 蒸发而截断；并明确写道 IMP7/8 与 IMAX 的限制"are not based on detailed analyses in peer-reviewed papers, and therefore, are not shown" — [Ray 2023, arXiv:2301.03625](https://arxiv.org/abs/2301.03625)
- 【文献证实】宇宙线传播（"reverse direct detection"）对轻暗物质-质子截面给出与宇宙学可比的独立限制 — [Cappiello, Ng & Beacom 2019, arXiv:1810.07705](https://arxiv.org/abs/1810.07705)
- 【文献证实】更高质量的宏观限制（土星环、小行星、火流星、白矮星）位于 m ≳ 10¹⁴ GeV、σ ≳ 10⁻⁹ cm²，与 HXMT 区域无关 — [Picker et al. 2026](https://arxiv.org/abs/2609.05626)
- 【文献证实】Snowmass 2021 白皮书图 2 并列 Model I（无 A 标度）与 Model II（A⁴）两幅约束图，其中 XQC 与 IMP 被归为"space-based experiments"；对全文检索 satellite/balloon/space-based 只命中该图注，未见把星载或气球探测列为未来方向，总结强调的是地下多次散射、机械传感器与古矿物 — [Carney et al. 2022, arXiv:2203.06508](https://arxiv.org/abs/2203.06508)

**基准质量处的排除区间表（f = 1）**

表 A：contact 约定，σ_c 单位 cm²。"P26"= Picker et al. 2026 公开数据（table）；其余为图读。

| m (GeV) | 被排除的 σ_c 区间与来源 |
|---|---|
| 10 | 8.0×10⁻⁴⁰–4.7×10⁻²⁹（XENON1T 单散射 recast，P26）；3.4×10⁻²⁸–3.4×10⁻²⁵（CRESST 地表等的 DD 并集，P26）；1×10⁻²⁴–2.5×10⁻¹⁸（XQC contact recast）；> 1.7×10⁻²⁷（MW satellites，Nadler 2021 表值）；> 3.7×10⁻²⁷（gas，P26）；> 5.8×10⁻²⁶（CMB + Lyman-α） |
| 10² | 7.5×10⁻³⁹–1.6×10⁻²⁷（XENON1T）；2.1×10⁻³⁴–8.1×10⁻²⁵（DD 并集）；3×10⁻²⁴–4×10⁻¹⁸（XQC）；> 5.3×10⁻²⁷（Lyman-α，Rogers 2022 表值）；> 1.4×10⁻²⁶（gas，P26）；> 1.3×10⁻²⁵（MW satellites，Maamari 表值）；> 1.6×10⁻²⁴（CMB） |
| 10⁴ | 2.5×10⁻³⁶–1.2×10⁻²⁵（XENON1T）；2.8×10⁻³²–2.5×10⁻²³（DD 并集）；4×10⁻²²–2.5×10⁻¹⁶（XQC）；> 1.3×10⁻²⁴（gas，P26；按 Bhoonah 原图为 > 7×10⁻²⁴）；> 1.5×10⁻²⁴（MW satellites 外推）；> 1.6×10⁻²²（CMB） |
| 10⁶ | 2.5×10⁻³⁴–9.2×10⁻²⁵（XENON1T）；2.8×10⁻³⁰–1.0×10⁻²¹（DD 并集，含 CDMS-I 至 4.2×10⁻²²、Chicago 6×10⁻²⁴–3.4×10⁻²¹）；3.2–5.9×10⁻¹⁹（Skylab，P26）；> 1.45×10⁻²²（gas，P26；原图 > 7.6×10⁻²²）；> 1.5×10⁻²²（MW satellites）；> 1.6×10⁻²⁰（CMB） |
| 10⁸ | 2.4×10⁻³²–9.2×10⁻²⁵（XENON1T）；2.7×10⁻²⁸–1.5×10⁻¹⁹（DD 并集，含 Chicago 8×10⁻²⁴–6.6×10⁻¹⁹）；1.06×10⁻²³–1.8×10⁻²¹（DEAP，P26）；1.7×10⁻¹⁹–2.9×10⁻¹⁶（Skylab，P26）；> 1.8×10⁻²⁰（gas，P26）；> 1.4×10⁻¹⁸（CMB） |
| 10¹⁰ | 2.5×10⁻³⁰–9.2×10⁻²⁵（XENON1T）；2.7×10⁻²⁶–2.3×10⁻¹⁷（DD 并集）；1.06×10⁻²³–2.1×10⁻¹⁹（DEAP）；1.6×10⁻²³–7×10⁻¹⁷（Chicago）；1.3×10⁻¹⁹–7.0×10⁻¹⁸（Ohya，P26）；1.7×10⁻¹⁹–5.5×10⁻¹⁴（Skylab，P26）；> 1.9×10⁻¹⁸（gas，P26） |
| 10¹² | 2.5×10⁻²⁸–9.2×10⁻²⁵（XENON1T 单散射）；1.6×10⁻²³–5×10⁻²¹（XENON1T 多散射）；2.9×10⁻²⁵–3.5×10⁻¹⁵（DD 并集，含 DAMA、Chicago 3.4×10⁻²²–6.9×10⁻¹⁵）；1.06×10⁻²³–2.4×10⁻¹⁸（DEAP）；9.3×10⁻²⁰–1.3×10⁻¹⁵（Ohya）；1.7×10⁻¹⁹–5.7×10⁻¹²（Skylab）；> 2.1×10⁻¹⁶（gas，P26） |
| 10¹⁴ | 2.5×10⁻²⁶–9.2×10⁻²⁵（XENON1T）；5.8×10⁻²⁴–1.6×10⁻¹⁵（DD 并集）；1.06×10⁻²³–2.4×10⁻¹⁸（DEAP）；9.3×10⁻²⁰–1.9×10⁻¹³（Ohya）；1.7×10⁻¹⁹–7.9×10⁻¹⁰（Skylab，P26）；> 2.6×10⁻¹⁴（gas） |
| 10¹⁶ | 1.09×10⁻²³–2.4×10⁻¹⁸（DEAP）；1.0×10⁻²²–1.1×10⁻¹³（DD 并集）；9.3×10⁻²⁰–2.7×10⁻¹¹（Ohya）；3.2×10⁻¹⁹–8.2×10⁻⁸（Skylab，P26 取 N = 2.3；Bhoonah 2021 取 150 事例时质量上限仅 3.8×10¹⁵ GeV）；2.4×10⁻¹⁸–6.1×10⁻¹⁶（mica，P26）；> 2.8×10⁻¹²（gas）；8.7×10⁻¹⁰–3.4×10⁻²（土星环） |

表 A 各行数据来源：[macrolimits v1.0.0](https://doi.org/10.5281/zenodo.21892125)、[Bhoonah et al. 2021](https://arxiv.org/abs/2010.07240)、[Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)、[Nadler et al. 2021](https://arxiv.org/abs/2008.00022)、[Maamari et al. 2021](https://arxiv.org/abs/2010.02936)、[Rogers et al. 2022](https://arxiv.org/abs/2111.10386)、[XENON 2023](https://arxiv.org/abs/2304.10931)。

表 B：per-nucleon 约定，σ_n 单位 cm²（均为图读；"旧"= 假设相干标度、未经同行评议详细分析的旧 recast）。常规地下单散射限制（σ_n ≲ 2×10⁻³⁵ (m/GeV) cm² 以下的区域，Kavanagh 图 5 图读）未逐行列出。

| m (GeV) | 被排除的 σ_n 区间与来源 |
|---|---|
| 10 | 2.5×10⁻²⁹–3×10⁻²²（XQC）；> 1.7×10⁻²⁷（MW satellites）；> 5.8×10⁻²⁶（CMB + Lyman-α） |
| 10² | 5×10⁻³⁰–1.3×10⁻²²（XQC）；> 5.3×10⁻²⁷（Lyman-α）；> 1.3×10⁻²⁵（MW satellites，Maamari） |
| 10⁴ | 6×10⁻²⁸–6×10⁻²¹（XQC）；1.6×10⁻²⁶–7.8×10⁻²³（IMP 7/8，旧）；> 1.6×10⁻²²（CMB，含氦） |
| 10⁶ | 2×10⁻²⁹–4.5×10⁻²⁶（CRESST 地表与 CDMS-I，Kavanagh）；1.6×10⁻²⁶–1.8×10⁻²⁰（IMP 7/8，旧）；1.6×10⁻²⁵–8×10⁻²⁵（IMAX，旧）；10⁻²³–3×10⁻²⁰（Pioneer 11，旧）；> 2.8×10⁻²⁴（gas cloud 的 A⁴ 版本） |
| 10⁸ | 10⁻³⁶–8×10⁻³⁰（LZ 多散射）；4×10⁻³⁰–8×10⁻²⁸（DEAP Model II）；2×10⁻²⁷–4.5×10⁻²⁴（CRESST 地表/CDMS-I）；5×10⁻²⁶–4×10⁻¹⁸（IMP 7/8，旧）；5.6×10⁻²⁴–3.4×10⁻²²（Skylab，Bhoonah 2021）；> 2.8×10⁻²²（gas A⁴） |
| 10¹⁰ | 1.3×10⁻³⁵–8×10⁻³⁰（LZ）；4×10⁻³⁰–8×10⁻²⁶（DEAP II 含外推）；2×10⁻²⁵–1.55×10⁻²³（CDMS-I）；5.4×10⁻²⁴–8×10⁻²⁰（Skylab）；1.6×10⁻²⁵–1.2×10⁻²⁰（IMAX，旧） |
| 10¹² | 6×10⁻³²–1.7×10⁻²⁹（XENON1T 多散射）；4×10⁻³⁰–9×10⁻²⁵（DEAP II）；2.5×10⁻²⁴–4.3×10⁻²²（Ohya）；5.4×10⁻²⁴–1.2×10⁻¹⁷（Skylab） |
| 10¹⁴ | 10⁻³¹–1.7×10⁻²⁹（XENON1T）；4×10⁻³⁰–9×10⁻²⁵（DEAP II）；2.5×10⁻²⁴–8.7×10⁻²⁰（Ohya）；5.4×10⁻²⁴–1.6×10⁻¹⁵（Skylab） |
| 10¹⁶ | 4×10⁻³²–8×10⁻³⁰（LZ）；4×10⁻³⁰–9×10⁻²⁵（DEAP II）；2.5×10⁻²⁴–1.3×10⁻¹⁷（Ohya） |

表 B 来源：[Erickcek et al. 2007](https://arxiv.org/abs/0704.0794)、[Kavanagh 2018](https://arxiv.org/abs/1712.04901)、[DEAP-3600 2022](https://arxiv.org/abs/2108.09405)、[Bhoonah et al. 2021](https://arxiv.org/abs/2012.13406)、[XENON 2023](https://arxiv.org/abs/2304.10931)、[LZ 2024](https://arxiv.org/abs/2402.08865)。

### Inferences
- 【推断】用 CSV 对每个质量做区间并集（σ_c 从 10⁻²⁶ 到 10⁻⁸ cm²）：(i) 全部限制都算时，m = 10–10¹³ GeV 没有任何开放区间，10¹⁴–10¹⁶ GeV 只在 σ_c < 1.1×10⁻²³ cm²（DEAP 下沿以下）有缝隙，与 HXMT 无关；(ii) 去掉 gas cloud 后结论不变（MW satellites 与 Lyman-α 接上）；(iii) 若宇宙学/天体限制中只相信 Planck CMB（去掉 gas cloud、MW satellites、Lyman-α 及所有外推行），则出现一条窄带：σ_c/m 介于地表探测器 ceiling（≈ 3×10⁻²⁷）与 CMB 线（1.6×10⁻²⁶ cm²/GeV）之间，例如 10⁴ GeV：2.5×10⁻²³–1.6×10⁻²²；10⁵：2×10⁻²²–1.6×10⁻²¹；10⁶：3.4×10⁻²¹–1.6×10⁻²⁰；10⁷：5×10⁻²⁰–1.5×10⁻¹⁹ cm²；10⁸ GeV 以上被 Chicago + Skylab 接上。这条窄带今天之所以算"已排除"，靠的是结构形成类限制与 gas cloud，而不是任何直接探测。
- 【推断】per-nucleon 约定下，宇宙学限制（约束的是 σ_p）在 σ_n/m ≈ 5×10⁻²⁹–1.6×10⁻²⁶ cm²/GeV，而大气遮蔽线在 4.5×10⁻³² cm²/GeV，二者之间 3–5 个量级靠 XQC（m ≤ 10⁵ GeV）、IMP 7/8 等旧 recast、A⁴ 增强的 gas cloud（≈ 2.8×10⁻³⁰ cm²/GeV，由 DEAP 图 4 Model II 图读）与 Skylab（m ≥ 1.6×10⁷ GeV）覆盖。若不采信未经同行评议的 IMP 7/8、IMAX、Pioneer 11（DEAP 2022 的 Model II 图即未画它们），则在 m ≈ 10⁵–10⁷·⁵ GeV、σ_n/m ≈ 4.5×10⁻³²–2.8×10⁻³⁰ cm²/GeV（例如 10⁶ GeV 处 4.5×10⁻²⁶–2.8×10⁻²⁴ cm²）出现一块 f = 1 的空白，对应 DEAP 图 4 下图左上角的白色三角。
- 【推断】同一 gas cloud 限制在两个来源中相差约 5 倍（Bhoonah 2021 原图 7.5×10⁻²⁸ 对 Picker 2026 数据 1.4×10⁻²⁸ cm²/GeV）；MW satellites 的两种分析在 10 GeV 相差 7 倍（1.7×10⁻²⁷ 对 1.2×10⁻²⁶ cm²）。因此"f = 1 天体物理 ceiling"的数值应带 0.5–1 dex 的系统不确定度引用。
- 【推断】Skylab 的质量上限随"本底事例数"假设变化 65 倍（150 事例 → 3.8×10¹⁵ GeV；2.3 事例 → 2.2×10¹⁷ GeV），两篇文献对同一数据的处理并不一致。
- 【推断】前期笔记把"Skylab 需要 σ ≳ 10⁻¹⁵ cm²"当作弹性散射的蚀刻阈值是误读：那是非弹性散射 macro 的判据；弹性散射阈值是 ~10⁻¹⁸ cm²，现代 recast 为 (1.5–1.7)×10⁻¹⁹ cm²。相应地"HXMT 的增量在 10⁻²²–10⁻¹⁵ cm²"应改为 10⁻²²–2×10⁻¹⁹ cm²。

### Gaps
- Shirk & Price 1978、Price & Salamon 1986、Rich et al. 1987 正文、Starkman et al. 1990 正文、Bernabei et al. 1999（DAMA SIMP）与 MACRO 慢粒子搜索原文未取回（付费墙或无 arXiv 版本）；相关数字全部来自后人的 recast。
- IMP 7/8 的 SIMP 分析没有公开的详细论文（数据来自 private communication，完整分析一直"in preparation"），其曝光量、本底事例数和窗口厚度均查不到；CSV 中它的 contact 换算是本笔记的量级估计。
- IMP 7/8、Pioneer 11、IMAX、RRS 从未被任何文献按 contact 约定重新推导；XQC 的 contact recast 只是对边界做 A 换算而非重新模拟。
- PandaX-4T 的多次散射结果未找到（综述称仍在进行）；LZ 与 XENON1T 的边界为粗略图读（±0.5 dex）。
- Gluscevic & Boddy、Nadler、Rogers 等宇宙学限制最高只算到 100 GeV–1 TeV，更高质量为 ∝ m 外推；没有文献在 macro 质量区重新做过这些分析。

---

## 3. 覆盖层物理：大气、岩石、整个地球与卫星自屏蔽的阻止条件（公式与数值）

### Takeaway
重暗物质（m_χ ≫ m_A）、contact 截面下能量按 E = E₀ exp(−2σX/m) 衰减，只依赖质量柱密度 X，ceiling 为 σ/m ≤ L/(2X)（L = ln(E₀/E_min) ≈ 1–8）：整层大气 ≈ (0.9–6)×10⁻²⁷ cm²/GeV，1–2 km 岩石 ≈ 10⁻³⁰–2×10⁻²⁹，过地心弦 ≈ 10⁻³⁴–4×10⁻³⁴，1 g/cm² 卫星蒙皮 ≈ (1–6)×10⁻²⁴ cm²/GeV。只要 σ/m ≳ 10⁻³¹ cm²/GeV，固体地球整个圆面对暗物质不透明，550 km 轨道上被挡天区约 30.5–32.5%，即地球遮挡调制在比大气遮蔽线低 4 个量级的截面上就已是"全遮挡"。

### Cited Findings
- 【文献证实】直线近似与平均能损：dE/dx = −Σ_j n_j σ_χ ⟨ΔE_j⟩，⟨ΔE_j⟩ ≃ 2E m_j/m_χ；ceiling 由"被撞核的总质量与暗物质质量相当"决定，σ_χ n_A m_A d ≃ m_χ；到达探测器的粒子在大气中已散射数百到数千次，碰撞次数的泊松涨落仅百分之几，单次偏转 θ ~ m_j/m_χ；动能随 σ 指数衰减，故以损失 10%、50% 或 90% 定义 ceiling 结果相近 — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)
- 【文献证实】速度形式：v_f = v₀ e^{−σX/m}，X 为路径上的总柱密度 — [Picker et al. 2026](https://arxiv.org/abs/2609.05626)；ceiling 的解析式 σ_XN = (M_X/2L) ln(E_0,max/E_f,min) [Σ_i μ²_XNi n_i/m_Ni]⁻¹ — [Bhoonah et al. 2021](https://arxiv.org/abs/2010.07240)
- 【文献证实】地球透明阈值：σ_χ/m_χ ≲ 10⁻³³ cm²/GeV 以下暗物质可穿过地球、出现上行事例 — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)
- 【文献证实】卫星/气球可把 overburden 柱密度降低约 10³ 倍，"Skylab diagonal edge is about this much higher than ours" — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)；Skylab 的保守 overburden 判据为 σ/M_X ≲ 1 g⁻¹ cm²（1 g cm⁻² 铝壁、保留 90% 能量），更早的判据为 ≲ 3 g⁻¹ cm² — [Bhoonah et al. 2021](https://arxiv.org/abs/2010.07240)
- 【文献证实】per-nucleon 情形的大气柱：10⁴ kg/m²，⟨A⁴⟩^{1/4} ≈ 16.6；地壳 ⟨A⁴⟩^{1/4} ≈ 31 — [De Luca et al. 2018](https://arxiv.org/abs/1801.01135)
- 【文献证实】轻暗物质（m_χ ≲ m_A）每次碰撞损失 O(1) 能量、轨迹大角度偏转，必须用 Monte Carlo 模拟 — [Emken & Kouvaris 2018, arXiv:1802.04764](https://arxiv.org/abs/1802.04764)；也可用"只计未散射粒子"的保守解析判据 — [Hooper & McDermott 2018](https://arxiv.org/abs/1802.03025)
- 【文献证实】卫星遮挡的简化处理：把穿过地球到达探测器的粒子视为完全被挡，遮挡函数 S(x,v) = 0（若 |x + λv| = r_⊕ 有 λ < 0 的解）；ISS 类轨道的分数调制 f_mod ~ 75%，北半球气球的周日调制 ~30%，质量越轻调制越深；5σ 发现条件 f_mod S_tot/√(S_tot + B) = 5 — [Emken et al. 2019](https://arxiv.org/abs/1905.06348)

### Inferences
以下均为【推断】（本笔记按上述公式的计算；括号内为与文献的交叉检验）。

- contact 约定 ceiling，σ/m = L/(2X) × 1.78×10⁻²⁴ cm²/GeV：

| 覆盖层 | X (g/cm²) | L = 1 | L = 3 | L = 7 |
|---|---|---|---|---|
| 海平面整层大气 | 1033 | 8.6×10⁻²⁸ | 2.6×10⁻²⁷ | 6.0×10⁻²⁷ |
| 大气 + 6.25 m.w.e. 混凝土（Chicago） | 1658 | 5.4×10⁻²⁸ | 1.6×10⁻²⁷ | 3.8×10⁻²⁷ |
| 高山 4300 m | ~600 | 1.5×10⁻²⁷ | 4.5×10⁻²⁷ | 1.0×10⁻²⁶ |
| 气球 30 km | ~12 | 7.4×10⁻²⁶ | 2.2×10⁻²⁵ | 5.2×10⁻²⁵ |
| 气球 40 km | ~3 | 3.0×10⁻²⁵ | 8.9×10⁻²⁵ | 2.1×10⁻²⁴ |
| 卫星蒙皮 1 g/cm² | 1 | 8.9×10⁻²⁵ | 2.7×10⁻²⁴ | 6.2×10⁻²⁴ |
| Skylab（4.0 g/cm²） | 4 | 2.2×10⁻²⁵ | 6.7×10⁻²⁵ | 1.6×10⁻²⁴ |
| 卫星 10 g/cm² | 10 | 8.9×10⁻²⁶ | 2.7×10⁻²⁵ | 6.2×10⁻²⁵ |
| Pioneer 11 钛窗 1.7 mg/cm² | 1.7×10⁻³ | 5.2×10⁻²² | 1.6×10⁻²¹ | 3.7×10⁻²¹ |
| XQC 滤光片 + 残余大气 | ~10⁻⁴ | 8.9×10⁻²¹ | 2.7×10⁻²⁰ | 6.2×10⁻²⁰ |
| Ohya | 1.15×10⁴ | 7.8×10⁻²⁹ | 2.3×10⁻²⁸ | 5.4×10⁻²⁸ |
| CDMS-I（10.6 m 岩石 + 大气） | ~3.9×10³ | 2.3×10⁻²⁸ | 6.8×10⁻²⁸ | 1.6×10⁻²⁷ |
| 1 km 岩石 | 2.7×10⁵ | 3.3×10⁻³⁰ | 9.9×10⁻³⁰ | 2.3×10⁻²⁹ |
| 2 km 岩石 | 5.4×10⁵ | 1.7×10⁻³⁰ | 5.0×10⁻³⁰ | 1.2×10⁻²⁹ |

（检验：大气 L = 1 得 8.6×10⁻²⁸，与 Cappiello 正文的"≃ 10⁻²⁷"一致；Chicago L = 7 得 3.8×10⁻²⁷，与其图 4 斜边 (2–7)×10⁻²⁷ 一致；Ohya L = 7 得 5.4×10⁻²⁸，Bhoonah 图读 1.3×10⁻²⁷；Skylab 取 L ≈ 2 ln(500/1.5) ≈ 12 得 2.6×10⁻²⁴，公开数据为 (3–9)×10⁻²⁴。）

- 整个地球（粗略五层 PREM 平均密度）：过地心弦 X ≈ 1.08×10¹⁰ g/cm²，σ/m = 8.3×10⁻³⁵（L = 1）到 4.1×10⁻³⁴（L = 5）cm²/GeV；瞄准距 6000 km 的弦 X ≈ 1.6×10⁹ g/cm² → 2.8×10⁻³³（L = 5）；离地表仅 1 km 深的掠射弦（长 226 km）X ≈ 6×10⁷ g/cm² → 7×10⁻³²（L = 5）。所以 σ/m ≳ 10⁻³¹ cm²/GeV 时固体地球圆面处处不透明，10⁻³⁴–10⁻³¹ 之间为"中心不透明、边缘半透明"的过渡区。
- 大气掠射柱（切点高度 h_t 处沿切线的总柱密度）：h_t = 0 km 时 7.2×10⁴ g/cm²；20 km：4.5×10³；30 km：9.5×10²；50 km：59；70 km：4.3；90 km：0.16；100 km：0.028 g/cm²。于是"不透明地球"的有效半径随截面增大：σ/m = 3×10⁻²⁷ cm²/GeV 时切到约 30 km，6×10⁻²⁵ 时约 70 km，10⁻²³–10⁻²² 时约 90–100 km。
- 550 km 轨道上的遮挡立体角：不透明半径取 R_⊕ 时地球角半径 67.0°，被挡天区 30.5%；R_⊕ + 70 km 时 68.5°、31.7%；R_⊕ + 110 km 时 69.5°、32.5%。
- per-nucleon（A⁴）约定：指数为 2σ_n X⟨A⁴⟩/m_χ，⟨A⁴⟩ 为按质量加权的平均（空气 7.7×10⁴，铝 5.3×10⁵，地壳岩石 9.0×10⁵，NaI 2.2×10⁸，CsI 2.9×10⁸）。e 折阈值 σ_n/m：大气 1.1×10⁻³²（L = 4 时 4.5×10⁻³²，与 Kavanagh 的 CRESST 地表 ceiling 4.5×10⁻²⁷ cm² @10⁵ GeV 完全吻合）；10.6 m 岩石 + 大气 L = 4 时 1.4×10⁻³³（Kavanagh 的 CDMS-I 为 1.55×10⁻³³）；1 g/cm² 铝 1.7×10⁻³⁰；NaI 晶体（1.285 g/cm²）3.2×10⁻³³；CsI 晶体（18 g/cm²）1.7×10⁻³⁴ cm²/GeV。含义：A⁴ 约定下轻元素屏蔽几乎透明而 I/Cs 晶体极"黑"，σ_n/m 在 ~2×10⁻³⁴ 到 ~10⁻²⁹ cm²/GeV 之间的粒子会把全部动能沉积在 HE 晶体内。
- 银盘气体拖曳：太阳圈处气体面密度 ~13 M_⊙/pc² = 2.7×10⁻³ g/cm²，晕轨道 10 Gyr 内约穿盘 100 次（计入斜入射 ×2），累计柱 ~0.5 g/cm²，能量 e 折对应 σ/m ≈ 1.6×10⁻²⁴ cm²/GeV；这与 Starkman 的银盘稳定性限制（5×10⁻²⁴）同量级。因此 σ/m ≳ 10⁻²⁴ cm²/GeV 的成分即使 f 很小也不再保持晕速度分布（会被拖成随盘共转），"来自天鹅座的风"不复存在——这与卫星自屏蔽 ceiling 恰好重合，构成可探区域的自然上界。
- 能量沉积下限（contact，薄靶极限 E_dep = σ X_det v²）：NaI 3.5 mm（1.285 g/cm²）中 E_dep[keV_nr] = σ × 7.2×10²³（v = 300 km/s），取平均淬灭 0.1、阈值 20 keVee 得 σ ≳ 2.8×10⁻²² cm²（230 km/s 时 4.7×10⁻²²，500 km/s 时 1.0×10⁻²²）；CsI 40 mm（18 g/cm²）、阈值 40 keVee、淬灭 0.08 得 σ ≳ 4.9×10⁻²³ cm²（范围 1.8–8.4×10⁻²³）；6.4 mm 塑闪反符合板取 100 keVee 阈值约 3×10⁻²¹ cm²。
- 能量上的质量下限：动能 ½mv² = 50 keV × (m/100 GeV)（300 km/s），即使全部沉积并取淬灭 0.1，m ≲ 400 GeV 也到不了 20 keVee；多次散射信号实际需要 m ≳ 10³ GeV。

### Gaps
- HE 探测器前方（铍窗、准直器、支撑结构）与侧后方（CsI、光电倍增管、平台）的真实质量分布未核对，上表以 1–10 g/cm² 参数化；真实 ceiling 需要质量模型。
- NaI/CsI 对低速重粒子连续多次核反冲径迹的发光淬灭没有实验数据（Cappiello et al. 也指出碰撞间距小到能量沉积区重叠时逐次求和不再适用）；上面的 0.08–0.1 是假设值。
- 地球柱密度用的是五层粗略平均而非完整 PREM；大气密度用标准大气的分段指数插值（垂直柱积分得 1016 g/cm²，与 1033 相差 2%）。

---

## 4. 决定性问题：是否存在同时满足"地球不透明、能过阈、事例数够、无现有限制"的区域？

### Takeaway
对 f = 1、contact 约定：**不存在**这样的开放窗口——HXMT 能探测的区域（σ_c ≳ 5×10⁻²³–3×10⁻²² cm²、σ_c/m ≲ 几×10⁻²⁴ cm²/GeV、m ≲ 10¹² GeV）已被 DEAP-3600、Chicago、CDMS-I、Skylab、Ohya、XQC 以及宇宙学/gas cloud 限制完全覆盖；无本底飞行时间搜索（≥ 10 事例）同样没有新区域。HXMT 真正可能新探的是 f ≪ 1 的亚成分：m ≈ 10³–6×10⁷ GeV、σ_c ≈ 3×10⁻²²–1.7×10⁻¹⁹ cm²、大气遮蔽线以上的一块（下称 W1），以及（需要仪器组核实的）LE/ME 硅探测器对 10–300 GeV 强子型亚成分的灵敏度（W2）。

### Cited Findings
- 【文献证实】条件 (i) 地球不透明：σ/m ≳ 10⁻³³ cm²/GeV 即满足 — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)
- 【文献证实】条件 (iii) 的统计判据：f_mod S_tot/√(S_tot + B) = 5，卫星轨道 f_mod ~ 75% — [Emken et al. 2019](https://arxiv.org/abs/1905.06348)
- 【转引，本轮未核对原文】HE 的参数（取自协调者任务书与前期调研笔记）：18 个模块各 283.5 cm²，NaI 厚 3.5 mm、CsI 厚 40 mm，17 个非 blind 模块反符合后 30–300 keV 本底 538.6 counts/s — [Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/abs/2009.14489)
- 【文献证实】条件 (iv) 所依据的现有限制：见第 2 节表 A/表 B 及 [macrolimits 数据](https://doi.org/10.5281/zenodo.21892125)
- 【文献证实】2026 年综述的总体判断："for masses well below a microgram (~10¹⁸ GeV), terrestrial detectors have largely excluded the parameter space where dark matter has a cross-section large enough to multiscatter" — [Bramante 2026](https://arxiv.org/abs/2602.23708)
- 【文献证实】地面探测器已能探到宇宙学限制以上的截面，"reducing the need for rocket- or space-based experiments" — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)
- 【文献证实】Skylab 曝光 1.17 m² × 2.19×10⁷ s — [Bhoonah et al. 2021](https://arxiv.org/abs/2012.13406)；XQC 曝光 34 个 0.5 mm × 2.0 mm 吸收体 × 100.7 s — [Erickcek et al. 2007](https://arxiv.org/abs/0704.0794)

### Inferences
以下均为【推断】（量级估计）。

- **HXMT/HE 的事例数**：取 ρ = 0.3 GeV/cm³、平均速度 3×10⁷ cm/s，平面探测器单侧通量 n v̄/4 = 2.25×10⁶ f/m_GeV cm⁻² s⁻¹；HE 面积 5103 cm²、活时间 9.4 yr × 0.6 ≈ 1.8×10⁸ s，A·T ≈ 9×10¹¹ cm² s，穿过晶体的粒子数 N_cross ≈ 2×10¹⁸ f/m_GeV。本底 B ≈ 500 c/s × 1.8×10⁸ s ≈ 9×10¹⁰，5σ 调制需 S ≈ 2×10⁶（f_mod = 0.75）到 5×10⁶（f_mod = 0.3）。于是统计极限为 **m_max ≈ (4–10)×10¹¹ f GeV，或 f_min ≈ (1–2.4)×10⁻¹² × m_GeV**（理想情况，未计系统误差）。
- **与既有大气层以上实验的曝光对比**：Skylab 的 A·T·f_g = 4.8×10¹⁰ cm² s，N ≈ 4.3×10¹⁷ f/m（取 2.3 事例得 m_max = 1.9×10¹⁷ GeV，取 150 事例得 2.9×10¹⁵ GeV，分别与 Picker 2026 和 Bhoonah 2021 吻合）；即 HXMT/HE 的几何曝光只有 Skylab 的约 5 倍，而 Skylab 是无本底径迹探测器。XQC 的 N ≈ 7.7×10⁷ f/m，HXMT 是它的 2.6×10¹⁰ 倍。Chicago 的 N ≈ 1.3×10¹³ f/m，但无本底，所以其 f 灵敏度 m/(1.3×10¹² GeV) 与 HXMT 的理想调制灵敏度 10⁻¹² m 相同——在大气遮蔽线以下 HXMT 对它没有优势。
- **f = 1、contact：无窗口。** HXMT 可及区在各质量处的现状：10⁴ GeV（σ_c 3×10⁻²²–5×10⁻²⁰）被 gas cloud/MW satellites/Lyman-α（> 1.3×10⁻²⁴）与 XQC（> 4×10⁻²²）覆盖；10⁶ GeV 被 Chicago（至 3.4×10⁻²¹）、gas 等（> 1.5×10⁻²²）覆盖；10⁸–10¹² GeV 被 DEAP（1.06×10⁻²³–2.4×10⁻¹⁸）、Chicago、Skylab（≥ 1.7×10⁻¹⁹）、Ohya 覆盖。唯一的保留意见是第 2 节所述"只信 CMB"时的窄带（σ_c/m ≈ 3×10⁻²⁷–1.6×10⁻²⁶ cm²/GeV、m ≈ 10³–10⁷ GeV）：若 f = 1 的暗物质落在此带，HE 晶体中的事例率将是 N_cross/T ≈ 10¹⁰/m_GeV s⁻¹（10⁶ GeV 时 ~10⁴ s⁻¹），远超实测总计数率，无需调制分析即可排除——这是一个"平凡但从未发表"的直接排除，价值在于不依赖结构形成与气体云模型。
- **f = 1、per-nucleon：条件性窗口 A。** 若不采信 IMP 7/8、IMAX、Pioneer 11 的旧 recast，则 m ≈ 10⁵–10⁷·⁵ GeV、σ_n/m ≈ 4.5×10⁻³²–2.8×10⁻³⁰ cm²/GeV 为空白。A⁴ 约定下这类粒子穿过铝蒙皮几乎无损、在 NaI/CsI 中全吸收，每事例沉积 0.5 MeV × (m/10³ GeV)（未淬灭），f = 1 时事例率同样是 10³–10⁵ s⁻¹ 量级，可被 HXMT 的总计数率直接排除。理论动机弱于 contact 情形（需要在超核尺度截面上仍保持 A⁴ 相干的模型，如 loosely bound composites）。另需注意这块空白对两类暗物质其实已另有间接排除：会自湮灭的被地球热流限制覆盖（Mack et al. 2007，~1–10¹⁰ GeV），不湮灭且能塌缩成黑洞的被天体"存活"限制覆盖（Ray 2023 正是把 10⁷ GeV、10⁻²⁴ cm² 作为"其它探针未排除"的例子）；只有既不湮灭、又因有内部结构而不塌缩的复合体才真正留在窗口 A 中。
- **无本底飞行时间搜索（≥ 10 事例）：无窗口。** 取多模块共线几何效率 ε ≈ 0.03–0.3，10 事例对应 m ≈ (0.6–6)×10¹⁶ f GeV。该质量区 σ_c ≥ 10⁻²³ cm² 已被 DEAP（至 1.2×10¹⁹ GeV）、XENON1T/LZ 多次散射、Ohya（至 4×10²⁰ GeV）、Skylab、mica 覆盖；即使对亚成分，这些无本底实验的 f 灵敏度（例如 10¹³ GeV 处 DEAP ~10⁻⁶、Ohya ~2.5×10⁻⁸、Skylab ~5×10⁻⁵，取自公开数据）也优于 HXMT 飞行时间法的 ~2×10⁻⁴–2×10⁻³。
- **亚成分窗口 W1（HE，contact）**：
  - 质量：下限 ~10³ GeV（动能过阈），上限 ~6×10⁷ GeV（更高质量处"大气遮蔽线以上"的全部区域 σ_c > 3×10⁻²⁷ m 已落入 Skylab 的 σ_c ≥ 1.7×10⁻¹⁹ cm² 灵敏区，而 Skylab 对 f 的灵敏度到 10⁻⁹ 量级）。
  - 截面：max[3×10⁻²² cm²（NaI；CsI 可到 5×10⁻²³）, ~3×10⁻²⁷ cm² × m_GeV（大气遮蔽线；其下地面探测器同样可及）] ≤ σ_c ≤ min[1.7×10⁻¹⁹ cm²（Skylab 下沿，m > 10⁶ GeV 时）, ~3×10⁻²⁴ cm² × m_GeV（卫星自屏蔽与银盘拖曳）]。例：10⁴ GeV：3×10⁻²²–3×10⁻²⁰；10⁵ GeV：3×10⁻²²–1.7×10⁻¹⁹；10⁶ GeV：3×10⁻²¹–1.7×10⁻¹⁹；10⁷ GeV：3×10⁻²⁰–1.7×10⁻¹⁹ cm²。
  - 丰度：f 介于 HXMT 统计极限 ~10⁻¹² m_GeV 与现有上限之间；现有上限见第 5 节（gas cloud：f < 1.4×10⁻²⁸ m_GeV/σ_c；XQC：f ≲ (10⁻⁶–10⁻⁵) m_GeV，仅 m ≤ 10⁵ GeV；IMP 7/8 若成立：f ≲ m/10¹⁰ GeV，仅 σ_c ≳ 10⁻²⁰ cm²）。
- **亚成分窗口 W2（LE/ME 硅探测器及 HE 单次散射，强子型截面）**：核几何饱和截面（4πR_A² ≈ 1.0×10⁻²⁴ cm²（N）、1.7×10⁻²⁴（Si）、4.6×10⁻²⁴（I））下，m ≲ 300 GeV 的粒子在大气中热化（46 次碰撞，每次平均损失 2m m_A/(m+m_A)²），只有大气层以上的探测器能看到；硅反冲能上限（v = 776 km/s）为 27 keV（10 GeV）、100 keV（30 GeV）、220 keV（100 GeV）。若 ME 为 ~220 g 硅（952 cm² × 1 mm，凭记忆、未核实）、LE 为 ~4.5 g 硅（384 cm² × ~50 μm，凭记忆、未核实），活时间 10⁸ s，则曝光分别约 2.6×10⁵ 与 5×10³ g·day（CRESST 地表运行为 0.046 g·day，JWST NIRSpec 单探测器 0.05 g），统计上可达 f ~ 10⁻⁸–10⁻⁹（ME）与 10⁻⁷–10⁻⁸（LE），而现有 XQC 限制在 f = 10⁻³ 时已消失。HE 的 NaI 对 m ≳ 50–100 GeV 的高速尾部单次反冲（Na 至 ~200 keV、I 至 ~330 keV）也可能过 20 keVee 阈值。此项完全依赖仪器参数、阈值与淬灭，需仪器方向的研究者核实。
- 【猜测】"晚形成"漏洞：CMB、Lyman-α、MW satellites 限制探测的是 z ≳ 10³ 时期的暗物质-重子耦合；若复合体在此之后才组装成大截面状态（综述指出 loosely bound 复合体的小束缚能意味着较晚形成；Picker et al. 明确以"macro 在相关宇宙学时期之前已形成"为前提），则 f = 1 在大气遮蔽带内只受 gas cloud 与 XQC/Skylab 等今日探针约束，卫星直接搜索的价值会相应提高。未找到文献对此做定量讨论。

### Gaps
- W1 的 f 上限中 IMP 7/8 一项来自未发表 recast 的 A⁴→contact 换算，可靠性存疑；若不采信，W1 在 σ_c ≳ 10⁻²⁰ cm² 的部分宽得多。
- 真实灵敏度受地磁/SAA/轨道进动本底调制的模板精度限制，可能比统计极限差 10²–10³ 倍（前期笔记的估计），本节数字均为理想统计极限。
- HE 对超量程沉积（W1 中 m ≳ 10⁵ GeV 的事例沉积 ≫ 1 MeV）的记录方式未知，可能决定这些事例是否进入可用计数率。
- W2 所需的 LE/ME 灵敏厚度、阈值、本底率均未在本轮核实。
- 未检索到 CGRO/BATSE、Fermi/GBM、RXTE/HEXTE、Swift/BAT 等同类低轨闪烁体/半导体探测器做过此类分析；它们的面积与在轨时间与 HXMT 相当或更大（凭记忆、未核实），HXMT 并非唯一可做者。

---

## 5. 亚成分情形：各约束如何随 f 标度？对稀有强相互作用遗迹已有哪些限制？HXMT 能到多小的 f？

### Takeaway
通量受限型限制（直接探测的右边界、XQC、Skylab、Chicago、DEAP）随 f 线性变弱，overburden ceiling 几乎不变，gas cloud 型加热限制按 σ/m ∝ 1/f 变弱，CMB 限制在 f ≳ 2% 时 ∝ 1/f、f ≲ 0.4% 时完全消失；因此 f ≲ 10⁻²–10⁻³ 时"大气遮蔽线以上"重新打开，而现有大气层以上实验缩放后很弱（XQC 只能排除 f ≳ (10⁻⁶–10⁻⁵) m_GeV）。HXMT/HE 在 m = 10⁴、10⁶ GeV 的不透明区理想统计可达 f ~ 10⁻⁸、10⁻⁶，比现有限制低 2–6 个量级；m = 10² GeV 时 HE 因动能不足无灵敏度。

### Cited Findings
- 【文献证实】CMB：原文称 f_χ ≳ 2% 时"the limit roughly scales with f_χ, independent of the DM mass"（即截面上限随 f 减小成比例放宽，其图 8 给出 f_χ = 1、0.1、0.01 三条曲线）；更小的 f 允许更大截面，当强耦合亚成分的能量密度接近重子密度的测量不确定度时与重子不可区分，限制"entirely relax"，发生在 f_χ ≲ 0.4%（与由氦丰度得到的 ~0.6% 一致），f_χ ≲ 0.2% 时完全消失；该结果以 n = −4、1 MeV 为例导出，但作者指出任何在复合时期把亚成分与重子紧耦合的相互作用都有此特征 — [Boddy, Gluscevic, Poulin, Kovetz, Kamionkowski & Barkana 2018, arXiv:1808.00001](https://arxiv.org/abs/1808.00001)
- 【文献证实】f 缩放规则（per-nucleon，m = 0.1–100 GeV）：下排除边界 σ^lo → σ^lo/f_χ；上边界由 overburden 的阈值条件决定，"the local DM density does not enter at all"，故不缩放。其图 1 显示：f_χ = 10⁻³ 时 Lyman-α 与 XQC 区域已消失，只剩 CRESST 地表、EDELWEISS 地表的小块、杜瓦加热、钽同质异能素（右上角）、深地与 CDMS 浅站；f_χ = 10⁻⁶ 时 σ_n ≳ 10⁻³⁰ cm² 以上只剩 CRESST 地表的一小条；f_χ = 10⁻⁹ 时该区基本全空 — [McKeen et al. 2022](https://arxiv.org/abs/2202.08840)
- 【文献证实】宏观暗物质的 f 依赖：多次散射型限制按 1/m 水平缩放（f_DM 等值线近似位于恒定质量）；单次散射型的 f_DM 等值线位于恒定 m/σ；gas cloud 加热 ∝ σ/m，其对 f_DM 的标度与低质量直接探测相同，允许的 f_DM 沿对角线无界增长，直到 macro 稀少到加热不再连续 — [Picker et al. 2026](https://arxiv.org/abs/2609.05626)
- 【文献证实】公开数据中各实验可排除的最小 f（contact）：Skylab 在 10⁸ GeV 为 4×10⁻¹⁰、10¹⁰ GeV 为 4.6×10⁻⁸、10¹² GeV 为 4×10⁻⁶、10¹⁴ GeV 为 4×10⁻⁴（即 f_min ≈ m/2.3×10¹⁷ GeV，仅 σ_c ≥ 1.7×10⁻¹⁹ cm²）；DEAP 在 10¹⁰ GeV 为 9×10⁻¹⁰、10¹³ GeV 为 9×10⁻⁷；Ohya 在 10¹³ GeV 为 2.5×10⁻⁸；gas cloud 在 f = 10⁻²、10⁻⁴、10⁻⁶ 时的下沿分别为 σ_c/m ≈ 1.5×10⁻²⁶、1.5×10⁻²⁴、1.5×10⁻²² cm²/GeV — [macrolimits v1.0.0](https://doi.org/10.5281/zenodo.21892125)
- 【文献证实】XQC 对亚成分：排除区的上边界与左边界对密度不敏感，右、下边界随密度降低收缩（图 8，f_d 至 0.1） — [Erickcek et al. 2007](https://arxiv.org/abs/0704.0794)
- 【文献证实】地球俘获的热化暗物质：m = 0.6–6 m_p 的粒子被地球俘获并热化后大量积累，由此得到严格截面限制 — [Neufeld, Farrar & McKee 2018, arXiv:1805.08794](https://arxiv.org/abs/1805.08794)；对 27 种原子核的截面给出实验限制 — [Neufeld & Brach-Neufeld 2019, arXiv:1904.01590](https://arxiv.org/abs/1904.01590)
- 【文献证实】核同质异能素：暗物质碰撞可使长寿命同质异能素退激发，用于检验强相互作用（亚）成分 — [Pospelov, Rajendran & Ramani 2020, arXiv:1907.00011](https://arxiv.org/abs/1907.00011)；¹⁸⁰ᵐTa 实测 T_1/2 > 1.3×10¹⁴ a（90%），给出对强相互作用遗迹的新约束 — [Lehnert et al. 2020, arXiv:1911.07865](https://arxiv.org/abs/1911.07865)
- 【文献证实】加速地球束缚暗物质：强相互作用成分在地表附近的密度可远高于银河系本底，可用地下加速器束流或高温热源把它上散射到探测阈值以上；m ≳ 10 GeV 的密度增强主要来自下沉中的"traffic jam"种群 — [McKeen et al. 2022](https://arxiv.org/abs/2202.08840)
- 【文献证实】毫电荷粒子：离子阱对 m_Q = 10 GeV、电荷 10⁻³e 的本地数密度限制到 n_Q ≲ 1 cm⁻³，比此前强约 10⁸ 倍 — [Budker, Graham, Ramani, Schmidt-Kaler, Smorra & Ulmer 2022, arXiv:2108.05283](https://arxiv.org/abs/2108.05283)；强耦合遗迹的地表密度增强可达 10¹⁴，并可延伸到 sub-GeV — [Pospelov & Ramani 2021, arXiv:2012.03957](https://arxiv.org/abs/2012.03957)；[Berlin, Liu, Pospelov & Ramani 2023, arXiv:2302.06619](https://arxiv.org/abs/2302.06619)
- 【文献证实】反常重同位素搜索（仅适用于会与核束缚或带电的遗迹）：富集海水质谱对 GeV–TeV 带正电 SIMP 给出 N_SIMP/N_N < 10⁻²⁷；陨石的 Rutherford 背散射给出 N_SIMP/N_n ≲ 3×10⁻¹⁴ (10 TeV/M_SIMP)，覆盖 100 GeV–10⁷ GeV 且不依赖电荷；若 SIMP 在地心湮灭，地热限制要求 ρ_SIMP < 10⁻³ ρ_DM — [De Luca et al. 2018](https://arxiv.org/abs/1801.01135)
- 【文献证实】Au、Fe 核中束缚 SIMP 的搜索（样品包括 NASA LDEF 卫星上暴露过的金）：反常同位素丰度上限低至 ~10⁻¹²，质量至 1.67 TeV（Au）与 0.65 TeV（Fe） — [Javorsek et al. 2001, Phys. Rev. Lett. 87, 231804](https://doi.org/10.1103/PhysRevLett.87.231804)；稳定重粒子的非对撞机搜索综述 — [Burdin et al. 2015, arXiv:1410.1374](https://arxiv.org/abs/1410.1374)
- 【文献证实】天体"存活"限制只适用于不湮灭且能塌缩成黑洞的重暗物质 — [Ray 2023](https://arxiv.org/abs/2301.03625)

### Inferences
以下均为【推断】，**量级估计**。

- 各类限制的 f 标度小结：

| 限制 | f 标度 | 备注 |
|---|---|---|
| Overburden ceiling（任何实验的上沿） | 基本不变 | 指数截断 |
| 无本底多次散射/径迹（Skylab、Ohya、DEAP、Chicago、mica） | 质量上限 ∝ f | 底边因泊松涨落略升 |
| 有本底的单次散射（XQC、CRESST 地表、CDMS-I） | 下沿 σ ∝ 1/f | 上下沿相遇即消失 |
| Gas cloud 加热 | σ/m 下沿 ∝ 1/f | 今日探针，不依赖宇宙学历史 |
| CMB | f ≳ 2% 时 ∝ 1/f；f ≲ 0.4% 消失 | |
| Lyman-α、MW satellites | 未见文献给出 f 依赖 | 类比混合暗物质，预期百分之几以下失效（猜测） |
| 银盘稳定性 | 对 f ≪ 1 不成约束 | 但 σ/m ≳ 10⁻²⁴ 的成分会被拖成共转 |

- 现有大气层以上实验缩放到亚成分：XQC 穿过吸收体的粒子数 ≈ 7.7×10⁷ f/m_GeV，> 4 keV 通道观测到 60 个事例，吸收体不透明（σ_c ≳ 1.4×10⁻²⁰ cm²）时得 f ≲ 10⁻⁶ m_GeV（10² GeV：~10⁻⁴；10⁴ GeV：~10⁻²；10⁵ GeV：~0.1），截面更小时再按 1/σ 变弱；若改用 Erickcek et al. 发表的 f = 1 右边界 10⁵ GeV 作为通量极限（CSV 中 f < 1 的 XQC 行即如此处理），则为 f ≲ 10⁻⁵ m_GeV，再弱 10 倍；IMP 7/8 若成立为 f ≲ m/10¹⁰ GeV（σ_c ≳ 10⁻²⁰ cm²）；RRS 的探测器很小，缩放后更弱；Skylab 对 f 很灵敏但只在 σ_c ≥ 1.7×10⁻¹⁹ cm²。
- CSV 并集给出的亚成分开放区（contact；括号内为不采信 IMP 7/8 估计时的上端）：

| m (GeV) | f = 10⁻² | f = 10⁻⁴ | f = 10⁻⁶ |
|---|---|---|---|
| 10⁴ | 1.2×10⁻²⁵–1.5×10⁻²² | 1.2×10⁻²⁵–9.7×10⁻²¹（1.6×10⁻²⁰） | 1.2×10⁻²⁵–9.7×10⁻²¹（1.5×10⁻¹⁸） |
| 10⁵ | 2.1×10⁻²²–1.6×10⁻²¹ | 2.1×10⁻²²–9.7×10⁻²¹（1.7×10⁻¹⁹） | 2.1×10⁻²²–1.9×10⁻¹⁷ |
| 10⁶ | 3.4×10⁻²¹–9.7×10⁻²¹（1.6×10⁻²⁰） | 3.4×10⁻²¹–9.7×10⁻²¹（3.2×10⁻¹⁹） | 3.4×10⁻²¹–3.2×10⁻¹⁹ |
| 10⁷ | 无（5.1×10⁻²⁰–1.7×10⁻¹⁹） | 5.1×10⁻²⁰–1.7×10⁻¹⁹ | 7.7×10⁻²³–1.7×10⁻¹⁹ |

  注：10⁴ GeV 行的低截面端含大气遮蔽线以下、未做 f 缩放的 CDMS-I/CRESST 区域；f = 10⁻² 时大气遮蔽带只有约半个量级宽（上端被 gas cloud 的 1.5×10⁻²⁶ cm²/GeV 压住），f ≤ 10⁻⁴ 时才展宽到 2 个量级以上。
- **HXMT/HE 在不透明区的 f 灵敏度（理想统计，f_min ≈ (1–2.4)×10⁻¹² m_GeV）**：

| m | HXMT f_min（统计） | 现有上限 | 是否低于现有限制 |
|---|---|---|---|
| 10² GeV | 无灵敏度（动能 50 keV，淬灭后 < 20 keVee） | XQC ~10⁻⁴–10⁻³（contact 不透明区） | 否（HE）；LE/ME 见 W2 |
| 10⁴ GeV | ~(1–2.4)×10⁻⁸ | XQC ~10⁻²–0.1；gas cloud 10⁻⁴–10⁻²（σ_c/m = 10⁻²⁴–10⁻²⁶）；IMP 7/8 估计 ~10⁻⁶（σ_c ≳ 10⁻²⁰） | 是，低 2–6 个量级 |
| 10⁶ GeV | ~(1–2.4)×10⁻⁶ | gas cloud 10⁻³–0.1（σ_c = 10⁻¹⁹–10⁻²¹）；IMP 7/8 估计 ~10⁻⁴；Skylab 仅 3–6×10⁻¹⁹ cm² 窄带内到 ~2×10⁻⁹ | 是，低 2–5 个量级（σ_c < 3×10⁻¹⁹ cm²） |

  若系统误差使灵敏度劣化 10²–10³ 倍，则 10⁴ GeV 处仍可达 10⁻⁶–10⁻⁵，相对 gas cloud/XQC 仍有 1–4 个量级的增益，但相对 IMP 7/8 估计值的优势消失。
- 对"既不带电也不与核束缚"的中性不透明复合体，重同位素搜索与地球束缚种群实验（杜瓦、钽、离子阱、加速器上散射）没有直接约束力：前者要求束缚，后者针对 GeV–百 GeV 的热化种群。
- Colored-DM hybrid（12.5 TeV）一类重强子型遗迹：按几何饱和截面它能到达浅层探测器，本笔记粗估 CDMS-I 浅站数据（99.4 live days、27 个候选事例；锗靶质量按 0.5 kg 量级取，凭记忆、未核实）对它的灵敏度可达 f ~ 10⁻⁹–10⁻⁸，远强于 HXMT/HE 单次散射的 ~10⁻⁶；所以 HXMT 对 ≳ 3 TeV 的强子型遗迹没有独特价值。此估算未在文献中找到对应结果，需专门核对。

### Gaps
- Lyman-α 与 MW satellites 限制对 f 的依赖没有找到文献；表中"百分之几以下失效"只是类比。
- 中性重复合体（10³–10⁸ GeV）被大气/地壳挡住后的地球束缚种群及其可观测效应没有文献。
- IMP 7/8 的曝光与本底未知，"f ≲ m/10¹⁰ GeV"只是把其 f = 1 质量上限当作通量极限的反推。
- XQC 对亚成分的 contact 限制没有文献给出，这里的 (10⁻⁶–10⁻⁵) m_GeV 是由事例数、几何面积和已发表的 f = 1 质量上限做的估算。

---

## 6. 2023–2026 年的新进展：新的火箭/气球/卫星限制与提案、复合暗物质 recast、Snowmass 白皮书的判断

### Takeaway
"用地球遮挡造成的轨道调制在太空搜索强相互作用亚成分"这一思路在 sub-GeV 电子反冲领域已成型：Emken et al. 2019 提出，JWST NIRSpec 暗场图像 2024 年给出首个空间限制（f = 0.4% 乃至 ~0.01%），DarkNESS 立方星明确计划对天鹅座做轨道调制搜索（截至检索到的最新资料尚未发射）。重质量、核反冲/多次散射、闪烁体卫星这一组合在检索范围内没有任何已发表的分析或提案；同期的复合暗物质工作集中在地下多次散射（XENON1T、LZ、CDEX-10）、古矿物重估和带 f 依赖的约束汇编。

### Cited Findings
- 【文献证实】卫星/气球轨道调制的原始提案：把屏蔽最薄的一面指向天鹅座，地球遮挡使信号率随轨道强烈调制；"a satellite-borne experiment could look for orbital modulations of higher frequencies"；以 0.1 gram-month 曝光、10⁹ 本底事例为例给出 5σ 调制发现范围 — [Emken et al. 2019](https://arxiv.org/abs/1905.06348)
- 【文献证实】JWST：用 NIRSpec HgCdTe 探测器（单探测器质量 0.05037 g）的暗场定标图像，对经超轻 dark photon 作用的 sub-GeV 暗物质-电子散射给出限制；"For a 0.4% subcomponent ... we disfavor all previously allowed parameter space at high cross sections, and constrain some parameter regions for subcomponent fractions as low as ∼0.01%"；其图 1 在 f = 0.4% 下并列了 XQC、RRS、分子云、加速器、CMB N_eff、BBN 与地面直接探测 — [Du, Essig, Rauscher & Xu 2024, arXiv:2412.13131](https://arxiv.org/abs/2412.13131)
- 【文献证实】DarkNESS：搭载 skipper-CCD 的纳卫星（活性质量约 2 g），目标之一是强相互作用 sub-GeV 暗物质的电子反冲；观测指向天鹅座，"As the NanoSatellite orbits Earth, planetary shadowing is expected to result in a modulated signature in the instrument's low-energy event rate"；设想约 450 次、每次 10 分钟的观测得到 0.1 gram-month 曝光，假设 f_χ = 0.01% 与 10⁹ 个本底事例；屏蔽仅 ~50 nm 铝（保守取 10 μm 硅）；已通过 CDR，经 Firefly Aerospace DREAM 2.0 获得发射机会 — [DarkNESS 2025, arXiv:2505.16981](https://arxiv.org/abs/2505.16981)；早先预期 2025 年末发射 — [DarkNESS 2024, arXiv:2412.12084](https://arxiv.org/abs/2412.12084)
- 【文献证实】纳卫星数据库（2026 年 1 月更新）仍把 DarkNESS 列为未发射、预计 2026 年由 Firefly Alpha 发射 — [nanosats.eu](https://www.nanosats.eu/sat/darkness)
- 【文献证实】Micro-X 探空火箭的暗物质构型针对的是 keV 惰性中微子的 X 射线线 — [Micro-X 2019, arXiv:1908.09010](https://arxiv.org/abs/1908.09010)
- 【文献证实】带 f 依赖的宏观暗物质约束汇编与公开代码/数据；重新计算了白矮星、小行星、土星环限制，统一处理 Skylab、Ohya、Mica — [Picker et al. 2026](https://arxiv.org/abs/2609.05626)
- 【文献证实】LMC 引起的高速暗物质流改变 Ohya 与 Skylab 的重暗物质限制 — [Bozorgnia, Bramante & Buchanan 2026, arXiv:2511.21841](https://arxiv.org/abs/2511.21841)
- 【文献证实】Mica 的熔融径迹建模与 X 射线荧光读出，并指出旧 mica 排除限"not reliable in the high–scattering cross-section regime" — [Boukhtouchen et al. 2026](https://arxiv.org/abs/2606.02579)
- 【文献证实】综述：地下探测器、符合阵列与旧数据 recast 合起来"constrain dark matter across nearly twenty orders of magnitude in mass and six in cross section"；下一步是低阈值、更大活性体积和大体积中微子探测器的专门分析 — [Bramante 2026](https://arxiv.org/abs/2602.23708)
- 【文献证实】其它新探针：磁悬浮粒子首次搜索 10⁶–10¹⁵ GeV、经轻媒介子作用的超重暗物质 — [Uitenbroek et al. 2026, arXiv:2608.20464](https://arxiv.org/abs/2608.20464)；单次散射搜索的截面 ceiling 与"neutrino roof" — [Raj & Mondal 2024, arXiv:2406.17015](https://arxiv.org/abs/2406.17015)；DUNE、CYGNUS 等千吨级探测器的多次散射前景 — [Aggarwal & Raj 2024, arXiv:2410.22168](https://arxiv.org/abs/2410.22168)；反应堆中子上散射地球束缚暗物质 — [Ema, Pospelov & Ray 2024, arXiv:2402.03431](https://arxiv.org/abs/2402.03431)
- 【文献证实】地球遮挡调制在地下实验中的应用仍在发展：DAMA/LIBRA-phase1 的地球遮挡周日效应分析 — [Bernabei et al. 2015, arXiv:1505.05336](https://arxiv.org/abs/1505.05336)；液氩/液氙探测器的 Earth-shielding 调制形状检验并应用于 DarkSide-50 数据 — [Kozynets, Leane & Smirnov 2026, arXiv:2607.20620](https://arxiv.org/abs/2607.20620)
- 【文献证实】Snowmass 2021 超重暗物质白皮书：把直接探测分为弱接触作用与长程力两类，图 2 给出两种标度下的现状；总结强调多次散射分析、未来 DARWIN/Argo、机械传感器（Windchime）与古岩石，全文未出现星载闪烁体方案 — [Carney et al. 2022](https://arxiv.org/abs/2203.06508)

### Inferences
- 【推断】对引用 Cappiello et al. 2021（INSPIRE 共 45 篇）、Erickcek et al. 2007（129 篇，筛 2023 年后）与 Emken et al. 2019（214 篇，筛 2022 年后并按 satellite/space/orbit/balloon 等词过滤）的文献做了逐条浏览，空间直接探测只有 JWST 与 DarkNESS 两项，且都是 sub-GeV 电子反冲；没有发现用天文卫星闪烁体或半导体探测器做重复合暗物质调制搜索的论文。因此 F10 的"思想"并非首创（Emken 2019 → DarkNESS），但"用已有十年卫星数据、针对 TeV 以上核反冲/多次散射"的实现在文献中是空白。
- 【推断】JWST 与 DarkNESS 的做法说明社区已接受"f ≲ 0.4% 亚成分 + 空间探测器"作为正当的搜索框架，HXMT 的结果若以同样方式（在若干固定 f 下画 (m, σ) 排除图，并与缩放后的 XQC/RRS/Skylab 对比）呈现，容易被理解和引用。
- 【推断】近年的趋势是给所有约束标注 f 依赖并公开数据（Picker et al. 2026），HXMT 的结果可以直接以相同格式并入。

### Gaps
- DarkNESS 截至 2026 年 11 月是否已发射、有无在轨结果，未能确认（最新可得资料为 2026 年 1 月的数据库条目）。
- 未检索到 skipper-CCD 气球飞行的强相互作用暗物质结果，也未检索到 Micro-X 的强相互作用暗物质限制。
- 检索以 INSPIRE 引用链与少量网络搜索为主，可能漏掉未引用上述三篇论文的工作（例如会议摘要、中文文献）。

---

## 7. 明确裁决：H1–H3 以及早先"m ~ 10¹³ GeV、σ ≳ 10⁻²² cm² 可探"的说法

### Takeaway
H1 在 contact 约定下成立（f = 1 时大气遮蔽区已被 Lyman-α、MW satellites、gas cloud 排除，地面与深地探测器把其下方接满），在 per-nucleon 约定下不成立于宇宙学而靠旧的空间实验 recast 关闭；H2 成立（真正开放的是 f ≪ 1 的亚成分，现有大气层以上限制缩放后很弱）；H3 成立（Digman et al. 2019 之后 contact 是复合暗物质的默认约定，两种约定必须分开对照）。早先的"10¹³ GeV、10⁻²² cm²"说法在统计上高估约一个量级，更关键的是它瞄准的区域早已被灵敏度高得多的实验排除，应当撤回并改写为"10³–10⁸ GeV 的稀有亚成分"。

### Cited Findings
- 【文献证实】（H1）地面探测器 ceiling 为 σ/m ≃ 10⁻²⁷ cm²/GeV，且其排除区"already partially covered by the cosmological constraints" — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)；f = 1 的结构形成与气体云限制为 5.2×10⁻²⁹（Lyman-α，100 GeV 处换算）— [Rogers et al. 2022](https://arxiv.org/abs/2111.10386)、1.7×10⁻²⁸（MW satellites，10 GeV 处换算）— [Nadler et al. 2021](https://arxiv.org/abs/2008.00022)、(1.3–2.8)×10⁻²⁸ cm²/GeV（gas cloud）— [Picker et al. 2026](https://arxiv.org/abs/2609.05626)；仅 CMB 为 1.6×10⁻²⁶ cm²/GeV — [Gluscevic & Boddy 2018](https://arxiv.org/abs/1712.07133)
- 【文献证实】（H1，per-nucleon）地表探测器 ceiling 为 4.5×10⁻³² cm²/GeV（CRESST 2017 地表运行） — [Kavanagh 2018](https://arxiv.org/abs/1712.04901)；其上的"high-altitude"区域来自 RRS、XQC、IMP 7/8、IMAX、Skylab，其中 IMP/IMAX/Skylab 的旧结果"not based on detailed analyses in peer-reviewed papers" — [Digman et al. 2019](https://arxiv.org/abs/1907.10618)
- 【文献证实】（H2）CMB 限制在 f ≲ 0.4% 时消失 — [Boddy et al. 2018](https://arxiv.org/abs/1808.00001)；空间探测器针对该亚成分的搜索已有先例 — [Du et al. 2024](https://arxiv.org/abs/2412.13131)；f = 10⁻³–10⁻⁹ 时 0.1–100 GeV 强相互作用成分的现有限制大片消失 — [McKeen et al. 2022](https://arxiv.org/abs/2202.08840)
- 【文献证实】（H3）点状暗物质不可能有 σ_χN ≳ 10⁻²⁵ cm²，大截面只能来自复合暗物质，且"disregarding scaling with A may be the most correct way of plotting constraints" — [Digman et al. 2019](https://arxiv.org/abs/1907.10618)；此后 DEAP、XENON1T、LZ 均同时给出两种标度的结果 — [DEAP-3600 2022](https://arxiv.org/abs/2108.09405)；[XENON 2023](https://arxiv.org/abs/2304.10931)；[LZ 2024](https://arxiv.org/abs/2402.08865)
- 【文献证实】（早先说法）10⁸–10¹³ GeV、σ_c ≥ 10⁻²³ cm² 的现有排除：DEAP 1.0×10⁻²³–2.4×10⁻¹⁸ cm²（m = 8.3×10⁶–1.2×10¹⁹ GeV） — [DEAP-3600 2022](https://arxiv.org/abs/2108.09405)；Chicago 至 4×10¹² GeV — [Cappiello et al. 2021](https://arxiv.org/abs/2008.10646)；Skylab 与 Ohya — [Bhoonah et al. 2021](https://arxiv.org/abs/2012.13406)；这些实验在 10¹³ GeV 处可排除的最小 f 为 DEAP 9×10⁻⁷、Ohya 2.5×10⁻⁸、Skylab 4.7×10⁻⁵ — [macrolimits v1.0.0](https://doi.org/10.5281/zenodo.21892125)

### Inferences
- 【推断】**H1 裁决：contact 约定下成立，但有三点限定。** (1) 起决定作用的是 Lyman-α、MW satellites 与 gas cloud，而不是 CMB：Planck CMB 单独的限制（1.6×10⁻²⁶ cm²/GeV）比大气遮蔽线（(1–7)×10⁻²⁷）高，留有一条约半个到一个量级宽的带。(2) 其中两项宇宙学限制（Lyman-α、MW satellites）是从 ≤ 100 GeV 按 ∝ m 外推到 10⁸ GeV 的，不同分析之间相差 0.5–1 dex，最新汇编出于谨慎未把宇宙学限制画入图中；gas cloud 限制是直接对高质量 contact 截面计算的，但最初版本受到过质疑，且两个版本之间相差约 5 倍。(3) 在 per-nucleon（A⁴）约定下 H1 的"宇宙学版本"不成立——宇宙学限制比大气遮蔽线高 3–5 个量级，该区域是靠 XQC 与未经同行评议的 IMP 7/8 等旧 recast、A⁴ 增强的 gas cloud 关闭的。此外，"地面浅层探测器 + 深地多次散射搜索把遮蔽线以下接满"这一半也成立：Chicago、CDMS-I、DEAP、XENON1T、LZ 合起来在 f = 1 下没有留缝。
- 【推断】**H2 裁决：成立，并可量化。** f ≲ 10⁻²–10⁻³ 时大气遮蔽线以上重新打开（上界由 gas cloud 的 σ/m ≲ 1.4×10⁻²⁸/f cm²/GeV 给出，f ≲ 0.4% 时无 CMB 限制）；XQC 缩放后只能排除到 f ~ (10⁻⁶–10⁻⁵) m_GeV，RRS 更弱，Skylab 只管 σ_c ≥ 1.7×10⁻¹⁹ cm²，IMP 7/8 的可靠性存疑。HXMT/HE 的理想统计灵敏度 f ~ 10⁻¹² m_GeV 在 10³–6×10⁷ GeV、3×10⁻²²–1.7×10⁻¹⁹ cm² 内比这些限制低 2–6 个量级。但需要说明："卫星的巨大曝光"这一论据要打折——HE 的几何曝光只有 Skylab 的约 5 倍、与无本底的 Chicago 探测器在 f 灵敏度上持平；HXMT 的独特之处不在曝光，而在于它位于大气层以上、能量阈值比蚀刻塑料低三个量级（3×10⁻²² 对 1.7×10⁻¹⁹ cm²）。
- 【推断】**H3 裁决：成立。** 追踪约定时还需注意：(a) contact 图的纵轴常仍标作 σ_χN 或 σ_nχ；(b) 几何截面有 πR² 与 4πR² 两种写法；(c) 旧文献（Starkman 1990、McGuire-Steinhardt 2001、Erickcek 2007 图 9、Mack 2007、Kavanagh 2018 的 high-altitude 区）全部是相干 A² 标度，不能直接搬到 contact 图上，换算因子在重暗物质时为 A⁴（Si：6×10⁵；I：2.6×10⁸），且上、下沿要分别用 overburden 核与探测器核；(d) 宇宙学限制约束的是 σ_p，在两种约定下数值相同，但与其它限制的相对位置完全不同——这正是 H1 结论随约定翻转的原因。
- 【推断】**对早先"m ~ 10¹³ GeV、σ ≳ 10⁻²² cm²"说法的裁决：不成立，应撤回。** (1) 统计上：按单侧通量与 5σ 调制判据，f = 1 的质量上限是 (4–10)×10¹¹ GeV，不是 10¹³ GeV（早先估计漏了平面探测器的 1/4 几何因子并用了 1σ 级的判据）。(2) 物理上：10⁸–10¹³ GeV、σ_c ≥ 10⁻²³ cm² 对 f = 1 已被 DEAP-3600、Chicago、Skylab、Ohya、gas cloud 完全排除，而且这些无本底实验对亚成分的灵敏度（10¹³ GeV 处 10⁻⁵–10⁻⁸）远高于 HXMT 在该质量的统计极限（f ≳ 1）。(3) 早先所说"Skylab 蚀刻阈值 10⁻¹⁵ cm²"应为 ~2×10⁻¹⁹ cm²，"增量区间 10⁻²²–10⁻¹⁵"相应缩为 10⁻²²–2×10⁻¹⁹。(4) 正确的表述：HXMT/HE 的潜在新贡献集中在 m ≈ 10³–6×10⁷ GeV、σ_c ≈ 3×10⁻²²（CsI 可到 5×10⁻²³）–1.7×10⁻¹⁹ cm²、σ_c/m ≈ 3×10⁻²⁷–3×10⁻²⁴ cm²/GeV 的**稀有亚成分**，可达 f ~ 10⁻⁹–10⁻⁵（理想统计）；对 f = 1 只能提供对天体物理限制的直接复核（靠总计数率即可），以及在 per-nucleon 约定下取代未经同行评议的 IMP 7/8 结果。
- 【推断】科学意义的坦率评估：W1 是一块真实存在、目前只有气体云加热和一枚 100 秒探空火箭约束的参数空间，零结果可以在若干固定 f 的 (m, σ) 图上画出新的排除区，并且是首个来自天文卫星闪烁体的此类限制；但没有任何已发表模型明确预言该处的遗迹，理论动机属于"质量函数尾部/暗核低质量端"这类一般性论证。若 IMP 7/8 的旧 recast 被认为可信，W1 在 σ_c ≳ 10⁻²⁰ cm² 的部分增益缩小到 2–3 个量级。

### Gaps
- H1 的结论在宇宙学一侧依赖把 ≤ 1 TeV 的限制外推 5–7 个量级质量（没有文献在该质量区直接计算过，也没有文献讨论复合体晚形成时这些限制是否仍适用），在今日宇宙一侧只依赖 gas cloud 加热这一种探针。
- 没有找到 f ≪ 1 时 Lyman-α/MW satellites 限制的定量结果，H2 中"f ≲ 10⁻²–10⁻³ 重新打开"的上端由 gas cloud 的 1/f 标度决定，而该限制本身有约 5 倍的版本差异。
- 本笔记所有 HXMT 灵敏度数字都是理想统计极限，未含系统误差、真实屏蔽质量模型、淬灭与电子学饱和；这些由其他方向的调研负责。
