# Insight-HXMT（慧眼）十年存档数据在基础物理 / 粒子物理中的"变废为宝"再利用（调研笔记，截至 2026-11）

> 阅读说明
> - 标签：**【文献证实】**= confirmed in literature（本次已取回原文或官方文档核对）；**【推断】**= inference（由已证实数字推算/逻辑推导）；**【推测】**= speculation（尚无文献或数据支撑的设想）。
> - 想法编号 F1–F11；每个想法按任务要求的 7 项记录。打分 1–5（5 最高）。
> - **检索局限（重要）**：本会话的 WebSearch 配额在调研中途被用尽（全会话共享上限）。此后的核对改用三条途径：(a) 用 arXiv 编号直接取回题录/全文；(b) 逐条扫描官方 "Insight-HXMT Publications" 清单（AO-09 White Book，更新至 2026-05）的标题；(c) 之前已完成的约 30 次 web 检索。因此"查新"结论的含义是：**在 (b)+(c) 中未见**，不等于绝对不存在（清单只按标题扫描；2026-05 之后、或未被官方清单收录的外部论文可能遗漏）。凡凭记忆、未取回原文的文献，一律放在各节 Gaps 中并注明。

---

## 0. 卫星参数核对、可用数据产品与"查新"基线

### Takeaway
任务给出的参数基本正确，需修正/补充的要点：HE 几何面积 ≈5100 cm²（18×283.5 cm²）；CsI 正常模式能段文献有 40–600 keV 与 40–800 keV 两种说法；HE 事例包里**逐事例保留了 18 路反符合（HVT）标签、脉宽、2 μs 级时间戳**，被反符合打标的事例（空白天区约占 30%）并不丢弃——这是把"噪声"变信号的关键入口。官方论文清单（至 2026-05）中，与基础物理沾边的只有 3 篇"理论组借用已发表 CXB 谱"的工作（PBH、fermionic DM、dark photon），**没有**任何 HXMT 团队的暗物质线搜寻、轴子、LIV、奇异粒子、衰变率类论文。

### Cited Findings
- 【文献证实】轨道 ~550 km、倾角 ~43°；HE 18 个 NaI(Tl)/CsI(Na) phoswich，单体直径 190 mm，总几何面积约 5100 cm²；能量分辨 ~15%@60 keV；定时精度 <10 μs，单探测器死时间 <10 μs；15 个小视场 1.1°×5.7°、2 个大视场 5.7°×5.7°、1 个遮挡（blind） — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/abs/1910.04955)
- 【文献证实】CsI(Na) 沉积能量范围：正常模式 40–600 keV、低增益模式 0.2–3 MeV；NaI 厚 3.5 mm、CsI 厚 40 mm — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)；[Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/pdf/2009.14489)。而 GRB 星表写的是正常模式 40–800 keV — [Song et al. 2022, arXiv:2203.00878](https://arxiv.org/abs/2203.00878)（两处不一致，使用前需按具体时段的增益核对）。
- 【文献证实】HE"good event"数据包内容：能量 8 bit、脉宽 8 bit、**18 路 HVT（反符合）信号标签 18 bit**、通道号 5 bit、事例类型 2 bit、亚秒时间 19 bit（500 kHz 时钟，相对 GPS 的定时精度 ±2 μs）；18 路 HVT 计数率与 3 个 HPM 计数率每秒记录一次 — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)
- 【文献证实】反符合由 18 块塑料闪烁体组成：6 块顶部（每块约 2250 cm²，覆盖 3 个 HED）+ 12 块侧面（每块约 1000 cm²）；单块对带电粒子探测效率要求 >95% — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)
- 【文献证实】高能质子穿过时 NaI 与 ACD 在 1 μs 内同时有信号，这类事例**在 Event File 中被标记**（空白天区约 30%），常规分析中剔除 — [Liao et al. 2020, arXiv:2005.01661](https://arxiv.org/pdf/2005.01661)
- 【文献证实】每个 HE 探测器的准直器内装有 ~200 Bq 的 ²⁴¹Am 源（嵌在 BC-448M 塑闪里，MPPC 读出）；5.5 MeV α 与 59.5 keV X 射线在 2 μs 窗内符合的事例被标为 calibration event，与普通事例一样存盘但带不同 flag；AGC 通过调 PMT 高压把 59.5 keV 峰锁在第 50 道 — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)；[Li X.B. et al. 2020, arXiv:2003.06998](https://arxiv.org/abs/2003.06998)
- 【文献证实】HE blind 探测器的准直器用钽片封盖，<200 keV 的 X 射线进不来；由于 CXB 在 HE 本底中占比很低，blind 与非 blind 探测器的本底谱一致 — [Liao et al. 2020, arXiv:2005.01661](https://arxiv.org/pdf/2005.01661)
- 【文献证实】ME：1728 个 Si-PIN 像素、总面积 952 cm²，每机箱 15 小视场（1°×4°）+2 大视场（4°×4°）+1 blind；LE：96 片 SCD（CCD236）、总面积 384 cm²，每机箱 20 小视场（1.6°×6°）+6 大视场（4°×6°）+2 遮挡 +4 超宽视场（50–60°×2–6°） — [Zhang S. et al. 2019, arXiv:1910.04434](https://arxiv.org/pdf/1910.04434)；[Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/pdf/2009.14489)
- 【文献证实】总览表给出的能量分辨：LE 2.5%@6 keV、ME 14%@20 keV、HE 19%@60 keV；时间分辨：HE 25 μs、ME 280 μs、LE 1 ms；灵敏度 0.5 mCrab（3σ，10⁵ s，仅统计） — [Zhang S. et al. 2019, arXiv:1910.04434](https://arxiv.org/pdf/1910.04434)（注意：HE 仪器论文写的是时间戳 ±2 μs、定时精度 <10 μs，与总览表的"25 μs"口径不同）
- 【文献证实】观测约束：**太阳规避角 >70°**、月球规避角 >6°；几乎每轨有约 30 min 地球遮挡；每天 8–9 轨穿 SAA、每次约 15 min — [Zhang S. et al. 2019, arXiv:1910.04434](https://arxiv.org/pdf/1910.04434)
- 【文献证实】在轨本底水平（寿命约 1–2 年时）：HE 17 个非 blind 模块 30–300 keV 合计 538.6±0.1 counts/s，即 (3.0–4.1)×10⁻⁴ counts s⁻¹ keV⁻¹ cm⁻²；ME 小视场 10–40 keV 为 37.8–40.2 counts/s，即 (1.6–1.7)×10⁻³；LE 小视场 1–12 keV 为 12.1–12.7 counts/s，即 (4.1–4.3)×10⁻³ counts s⁻¹ keV⁻¹ cm⁻²；LE 在 ~7.5 keV 以下由 CXB 主导 — [Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/pdf/2009.14489)
- 【文献证实】HE 本底模型系统误差：26–100 keV 能谱平均 1.5%（8 ks 曝光），30–150 keV 光变 <3%；五年复核仍 <3% — [Liao et al. 2020, arXiv:2005.01661](https://arxiv.org/pdf/2005.01661)；[Liao et al. 2023, arXiv:2302.14459](https://arxiv.org/abs/2302.14459)
- 【文献证实】LE 响应系统误差在 1–7 keV <1%（Si K 边 1.839 keV 处至 1.5%） — [Li X.B. et al. 2020, arXiv:2003.06998](https://arxiv.org/abs/2003.06998)
- 【文献证实】空白天区例行观测约每月 10 次、每次 3 ks — [AO-09 White Book（2026-05-09），发布页](http://hxmten.ihep.ac.cn/notice/987.jhtml)。HE 本底论文称到 2019-12-12 已有 200 多次 blank sky 观测、有效曝光约 3.5 Ms — [Liao et al. 2020, arXiv:2005.01661](https://arxiv.org/pdf/2005.01661)。五年本底论文的观测表中，HE 还把 PSR B0540−69、GW 170817、Cas A 天区当作等效空白天区使用（这些源对 HE 太弱） — [Liao et al. 2023, arXiv:2302.14459](https://arxiv.org/abs/2302.14459)
- 【文献证实】数据政策：非 ToO 观测保护期一年；Level-1 数据在 http://archive.hxmt.cn/proposal 下载；截至 2026-05-01 已公开 10573 个数据产品和 728 个 GRB 数据；HXMTDAS V2.08 于 2026-04 发布 — [AO-09 White Book 发布页](http://hxmten.ihep.ac.cn/notice/987.jhtml)（White Book 正文由本会话其他调研员抓取，本笔记据其文本核对）
- 【文献证实】官方论文清单（White Book 第 12 章，更新至 2026-05，来源页 [hxmten.ihep.ac.cn/papers.jhtml](http://hxmten.ihep.ac.cn/papers.jhtml)）按标题扫描：命中的基础物理相关条目只有 "Constraining primordial black hole abundance with Insight-HXMT"（Yang et al. 2026）、"Revisiting the fermionic dark matter absorption on electron target"（Ge et al. 2022）、"Heavy dark photon handbook: Cosmological and astrophysical bounds"（Caputo et al. 2026）、"GRB 200716C: Evidence for a Short Burst Being Lensed"（Wang et al. 2021）；**无**标题含 Lorentz / LIV / axion / sterile / quantum gravity / nuclearite / decay rate 的条目。
- 【文献证实】Ge et al. 2022 使用的是 LE 本底模型论文中公布的 (l,b)=(219.3°, −50.0°) 空白天区 1–12 keV CXB 数据点，并称其对 fermionic DM 衰变给出很强约束 — [Ge, He, Ma & Sheng 2022, arXiv:2201.11497](https://arxiv.org/abs/2201.11497)
- 【文献证实】HE 数据处理中 `hepical` 默认删除 CsI 事例并剔除"spike events"；spike 通常只出现在单个探测器、持续 <200 μs、记录能量约 25–95 keV，被认为是高能宇宙线与探测器相互作用所致 — [Cai et al. 2021, arXiv:2109.12270](https://arxiv.org/pdf/2109.12270)；官方清单另有专文 "The removal method and generation mechanism of spikes in Insight-HXMT/HE telescope"（Wu Baiyang et al. 2022, Exp. Astron. 53, 1037；仅见于[官方论文清单](http://hxmten.ihep.ac.cn/papers.jhtml)，原文未取回）

### Inferences
- 【推断】几何 grasp（面积×立体角，未乘准直器透过率与探测效率）：HE ≈ 15×283.5×6.27 + 2×283.5×32.5 ≈ 4.5×10⁴ cm² deg² ≈ 13.7 cm² sr；ME ≈ 4.9×10³ cm² deg² ≈ 1.5 cm² sr；LE（小+大视场）≈ 4.0×10³ cm² deg² ≈ 1.2 cm² sr。对比 NuSTAR 0-bounce 每个 FPM 仅 ≈55 cm² deg²（[Roach et al. 2023, arXiv:2207.04572](https://arxiv.org/abs/2207.04572)），HXMT 三台望远镜对**弥漫信号**的 grasp 分别约为 NuSTAR 双模块的 400、44、37 倍。这是所有"弥漫/各向同性信号"类想法的物理基础。
- 【推断】HVT 塑闪总面积 6×2250+12×1000 ≈ 2.55×10⁴ cm²（2.55 m²），是一个被长期忽视的大面积带电粒子计数阵列。
- 【推断】按"每月 10×3 ks"外推，2017-07 至 2026-11 的专门空白天区曝光约 3–4 Ms（LE/ME 口径）；HE 口径若按前 2.5 年 3.5 Ms 的速率并计入弱源场，约 10 Ms 量级（两种口径相差数倍，实际数值需从观测日志统计）；若把所有定点观测中的地球遮挡段（每轨约 30 min）也算作"off"样本，可用时间再高一个量级。

### Gaps
- White Book 与官方论文清单的文本来自同一会话中其他调研员的抓取（共享工作目录），我未独立打开这两个网页；URL 取自 White Book 正文与官网通知页列表。
- "space-environment monitor（SEM）"确有其物（White Book 写明是监测粒子环境的辅助载荷），但其能道、几何因子、数据是否公开，本次未查到。
- HE 的 overflow（超量程）事例如何编码、calibration event 的计数率、spike 事例是否在 Level-1 中完整保留——这三项对 F7/F8/F9 至关重要，文献中未找到，需直接看 1L 事例文件或问仪器组。
- 2025 年"昆明会议"报告（宋黎明）称累计公开一级数据约 142 TB、公开率 >92%，但我没有该报告的可引用 URL，仅作线索。

---

## 1. 衰变暗物质线搜寻（sterile neutrino 等）：HXMT 能否在 1–250 keV、尤其 10–100 keV 给出有竞争力的限制？

### Takeaway
没有检索到任何用 HXMT 原始存档做的暗物质发射线搜寻（只有理论组借用已发表 CXB 数据点）。HXMT 的统计量极其充裕（grasp 是 NuSTAR 杂散光方法的几十到几百倍），**瓶颈完全在系统误差**：若本底/响应只能控制到 1%，HE 的线灵敏度大致只相当于 Fermi-GBM 2015 的水平（寿命 ~10²⁷ s）；若利用 HXMT 独有的"同轴大/小/盲三种视场 + 地球遮挡 on/off"差分把系统误差压到 0.1%，则 20–30 keV 光子（m_s ≈ 40–60 keV，NuSTAR 杂散光分析止于 20 keV、INTEGRAL/SPI 起于 30 keV 的接缝处）和 LE 的 3.5 keV 线检验有望达到现有最好限制的同一量级。

### Cited Findings
- 【文献证实】Fermi-GBM 先例：用**单个** NaI 探测器（det-7）、4 年数据经筛选后 4.6×10⁶ s 活时间，在 10–25 keV（m_s = 20–50 keV）搜线，无信号；m_s = 25–40 keV 的限制比此前提高约一个量级；能量分辨约 10%；作者明确说该分析是**系统误差主导**、统计误差可忽略（每道 √N/N < 10⁻³） — [Ng, Horiuchi, Gaskins, Smith & Preece 2015, arXiv:1504.04027](https://arxiv.org/abs/1504.04027)；会议版 [Horiuchi et al. 2015, arXiv:1502.03399](https://arxiv.org/abs/1502.03399)
- 【文献证实】NuSTAR 杂散光（0-bounce）先例：0-bounce grasp ≈55 cm² deg²/FPM，在 E≳10 keV 基本不随能量变；用 ~7 Ms/detector 空白场 + ~20 Ms/detector 叠加数据，在 3–20 keV 给出 m_s = 6–40 keV 的 sin²2θ 限制，并强烈不支持 ~7 keV sterile neutrino 解释 3.5 keV 线 — [Roach et al. 2023, arXiv:2207.04572](https://arxiv.org/abs/2207.04572)
- 【文献证实】NuSTAR 11 年杂散光：|b|>3° 数据总曝光 234 Ms，能段 3–20 keV（m_s = 6–40 keV），限制已触到 νMSM 理论预言下沿；40 keV 处的边缘超出被归因于已知仪器效应 — [Krivonos, Barinov, Mukhin & Gorbunov 2024, arXiv:2405.17861](https://arxiv.org/abs/2405.17861)
- 【文献证实】其他 NuSTAR 工作：银心 [Perez et al. 2017, arXiv:1609.00667](https://arxiv.org/abs/1609.00667)；深场 [Neronov, Malyshev & Eckert 2016, arXiv:1607.07328](https://arxiv.org/abs/1607.07328)；M31 [Ng et al. 2019, arXiv:1901.01262](https://arxiv.org/abs/1901.01262)；银河核球 [Roach et al. 2020, arXiv:1908.09037](https://arxiv.org/abs/1908.09037)
- 【文献证实】INTEGRAL/SPI：16 年数据、光子能量 30 keV–8 MeV，配暗物质空间模板，对 m_DM ≈ 60 keV–16 MeV 的两体衰变线给出当时最强限制；其模板的 D-factor 为 1.289×10²⁹ keV/cm² — [Calore, Dekker, Serpico & Siegert, arXiv:2209.06299](https://arxiv.org/abs/2209.06299)；更早 [Boyarsky et al. 2008, arXiv:0710.4922](https://arxiv.org/abs/0710.4922)
- 【文献证实】3.5 keV 争论中的"非聚焦大 grasp"先例 HaloSat：视场 ~100 deg²，给出银晕 3.5 keV 线流量 F ≤ 0.077 ph cm⁻² s⁻¹ sr⁻¹、sin²2θ ≤ 4.25×10⁻¹¹（90% CL）；作者强调其响应系统误差与大型聚焦望远镜不同 — [Silich et al. 2021, arXiv:2105.12252](https://arxiv.org/abs/2105.12252)
- 【文献证实】聚焦望远镜的空白天区限制：XMM-Newton [Dessert, Rodd & Safdi 2020, arXiv:1812.06976](https://arxiv.org/abs/1812.06976)、[Foster et al. 2021, arXiv:2102.02207](https://arxiv.org/abs/2102.02207)；Hitomi 并预言 XRISM 在 1–20 keV 质量段领先 [Dessert, Ning, Rodd & Safdi 2023, arXiv:2305.17160](https://arxiv.org/abs/2305.17160)；SRG/ART-XC 全天 [Zakharov et al. 2024, arXiv:2303.12673](https://arxiv.org/abs/2303.12673)；综述 [Boyarsky et al. 2019, arXiv:1807.07938](https://arxiv.org/abs/1807.07938)
- 【文献证实】HXMT 自己的弥漫背景测量只用了很小一部分数据：LE 有效曝光 117 ks；HE "blank-sky" 377 ks、"Earth-occulted" 187 ks（2021 年前的公开数据），HE 本底取自地球遮挡段，得到 1.5–120 keV 的 CXB 谱 9.57 E^−1.402 exp(−E/55 keV) ph cm⁻² s⁻¹ keV⁻¹ sr⁻¹；ME 因本底不稳未用；作者发现累计曝光超过 ~90 ks 后用地球遮挡估本底会出现明显偏差，需按 blind 探测器筛选 — [Huang et al. 2022, MNRAS 513, 4074, arXiv:2306.10114](https://arxiv.org/abs/2306.10114)
- 【文献证实】HXMT 的仪器线清单：HE/NaI 有 31、56、67、191 keV 碘活化线；ME 有 Ag K 线（~22.5 keV）；LE 有 7.5、8.0、8.6 keV（Ni/Cu/Zn 荧光） — [Li X.B. et al. 2020, arXiv:2003.06998](https://arxiv.org/abs/2003.06998)；[Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/pdf/2009.14489)
- 【文献证实】理论组已用 HXMT 公布的 CXB 数据做暗物质/PBH 连续谱约束（不是线搜寻） — [Ge et al. 2022, arXiv:2201.11497](https://arxiv.org/abs/2201.11497)；[Yang, Pan & Zhang 2026, arXiv:2412.09297](https://arxiv.org/abs/2412.09297)

### Inferences
#### 想法 F1：三望远镜"差分视场"暗物质衰变线搜寻（1–250 keV，主攻 m_s ≈ 40–60 keV 接缝与 3.5 keV 线的独立检验）
1. **一句话**：把十年空白天区、银道面扫描和所有定点观测的地球遮挡 on/off 数据叠起来，利用同轴的大/小/盲视场探测器做差分，搜寻强度正比于视场立体角、且随银心角变化的窄线。
2. **原理与灵感**：暗物质两体衰变给出 E=m/2 的单能线，强度 I = Γ·D/(4π m)【文献证实，公式见 Ng 2015、Roach 2023】。灵感直接来自 Fermi-GBM 全天 NaI 线搜寻（Ng et al. 2015）、NuSTAR 0-bounce（Perez/Roach/Krivonos）和 HaloSat"非聚焦大 grasp 检验 3.5 keV"。
3. **利用的 HXMT 特征与适配性**：(i) 巨大 grasp（见第 0 节）；(ii) 同一机箱内同型探测器分大/小/盲三种视场——弥漫线在"大−小""小−盲"差谱中应按 5.2:1:0 的立体角比例出现，而内禀活化线、粒子本底与视场无关【推断】，这是 GBM、NuSTAR、SPI 都没有的内建对照；(iii) 无反射镜，LE 在 2–4 keV 没有 Au M 边等镜面响应结构【推断】；(iv) 每轨地球遮挡提供天然 on/off。**坦率结论**：10–20 keV 光子段 HXMT/ME 的系统误差（Ag 线、噪声像素、响应）大概率不如 NuSTAR 234 Ms 杂散光；>30 keV 不如 SPI（2 keV 级分辨 + 编码成像 + 空间模板）；HXMT 的真正机会在 (a) 20–30 keV 接缝、(b) 作为系统误差完全独立的第三种方法、(c) LE 对 3.5 keV 线的独立检验。
4. **数据产品**：LE/ME/HE 的 1L 事例文件 + EHK + 高压/温度/死时间文件，空白天区与全部定点观测（含地球遮挡段）、小天区扫描数据；超过一年保护期者公开（第 0 节）。需要自建"全视场分组"的响应与本底，不能直接用 HXMTDAS 默认流程（默认只为点源设计）。
5. **查新**：早期 web 检索与官方论文清单标题扫描均未见 HXMT 暗物质线搜寻；仅见上述借用 CXB 数据点的理论工作【文献证实，见第 0 节】。其他卫星的先例见 Cited Findings。
6. **可行性与量级估计**（【推断】，输入全部列出）：
   - 取高银纬 D ≈ 2×10²² GeV cm⁻²（保守；SPI 模板值 1.3×10²³ GeV cm⁻²），sterile neutrino Γ = 1.38×10⁻²² s⁻¹·sin²2θ·(m/keV)⁵，得 I/sin²2θ ≈ 2.2×10⁵ (m/keV)⁴ ph cm⁻² s⁻¹ sr⁻¹；名义 3.5 keV 线（m=7.1 keV, sin²2θ=7×10⁻¹¹）对应 I ≈ 0.04 ph cm⁻² s⁻¹ sr⁻¹。
   - 2σ 统计极限 I_stat = 2√(B/T)/(Gε)，B = b·A·ΔE；系统极限 I_sys = f·B/(Gε)。
   - **LE @3.5 keV**：b≈6×10⁻³、A=240 cm²（小视场）、ΔE=0.15 keV、Gε≈0.56 cm² sr、T=3 Ms → B≈0.22 c/s，I_stat≈1×10⁻³，I_sys(f=1%)≈4×10⁻³ ph cm⁻² s⁻¹ sr⁻¹ → sin²2θ ≈ (2–7)×10⁻¹²；比 HaloSat 的 0.077 好 20–80 倍，名义 3.5 keV 线应以 ≥10σ 被看到或排除。
   - **ME @15 keV**：b≈1.65×10⁻³、A≈790 cm²、ΔE≈3 keV、Gε≈0.68、T=3 Ms → B≈3.9 c/s，I_stat≈3×10⁻³（sin²2θ≈2×10⁻¹⁴，τ≈2×10²⁸ s），I_sys(1%)≈6×10⁻²（sin²2θ≈3×10⁻¹³，τ≈1×10²⁷ s）。
   - **HE @25/40/60/100 keV**：b≈(6,5,4,3.5)×10⁻⁴、A=4820 cm²、ΔE≈(5.5,7.3,9,12) keV、Gε≈10 cm² sr、T=10 Ms → B≈16–20 c/s；I_stat≈3×10⁻⁴（τ≈(3–12)×10²⁸ s），I_sys(1%)≈2×10⁻²（τ≈(0.4–2)×10²⁷ s，sin²2θ≈10⁻¹⁴→6×10⁻¹⁷）；若差分把 f 压到 0.1%，τ≈(0.4–2)×10²⁸ s。
   - **主要障碍**：模块间活化线强度与增益差异（差分不能完全抵消）；HE 能量分辨差（15–19%）且 31/56/67/191 keV 线附近基本不可用；ME 噪声像素与 Ag 线；LE 辐照损伤致分辨变差、阈值上调、漏光与低能带电粒子"flare"；地球遮挡段的本底与非遮挡段并不相同（Huang et al. 2022 已明确指出）；银道面扫描数据有银河脊与点源连续谱（对线搜寻影响小，但会抬高 B）。
7. **打分**：新颖性 4（方法有先例，但 HXMT 从未做过，差分视场是新手段）｜可行性 4（LE/ME）/3（HE）｜影响 4（哪怕只做到系统极限，也是 20–60 keV 质量段少有的独立限制；LE 对 3.5 keV 的检验有独立话语权）。

### Gaps
- NuSTAR、SPI、XMM 在各质量点的具体 sin²2θ/寿命数值都在论文图中，本次未逐点提取，F1 与它们的定量比较（尤其 m_s=40–60 keV 接缝处谁更强）需读图核对。
- 未能检索 2025–2026 年是否已有 GECAM、Einstein Probe、XRISM 的新暗物质线限制（搜索配额用尽）。
- HXMT 对弥漫源的有效 grasp 需用准直器真实透过率与离轴响应积分，本笔记只用了 FWHM 乘积近似。
- LE 的 3.5 keV 附近是否存在未公布的仪器结构（如探测器材料荧光）未查到。

---

## 2. 轴子 / ALP：地磁场转换（GECOSAX）、14.4 keV 线、谱不规则性、SN/GRB/磁星信号——准直型仪器能做哪些？

### Takeaway
标准的 GECOSAX（夜侧穿过地球朝太阳方向看）和 14.4 keV ⁵⁷Fe 太阳轴子线对 HXMT **不可行**：太阳规避角 >70°，太阳方向永远进不了任何准直视场。ALP 谱不规则性、磁星、Betelgeuse 类搜索也因灵敏度/分辨率不如 NuSTAR 等而应放弃。唯一真正可做且便宜的是**检验 XMM-Newton 报告的 2–6 keV 近地 X 射线背景季节调制（11σ）及其 Axion Quark Nugget 解释**——后者明确预言信号延伸到 ~100 keV、且随高度降低而增强，HXMT 位于 550 km 且恰好覆盖 1–250 keV。

### Cited Findings
- 【文献证实】GECOSAX 原始提议：太阳轴子在地球磁层转换成 X 射线，可在夜侧（地球挡住太阳 X 射线）探测；有效面积 10⁴ cm² 的 LEO 探测器一年可探到 g_aγ ~10⁻¹¹ GeV⁻¹（m_a<10⁻⁴ eV） — [Davoudiasl & Huber 2006, arXiv:hep-ph/0509293](https://arxiv.org/abs/hep-ph/0509293)
- 【文献证实】可行性细算：用真实轨道、非均匀地磁场与大气吸收，并采用 Suzaku 实测本底；面积 10³ cm²、~10⁶ s 数据可得 g_aγ < (4.7–6.6)×10⁻¹¹ GeV⁻¹（2σ）；1–10 keV 轴子通量 3.58×10¹¹ cm⁻² s⁻¹（g_aγ=10⁻¹⁰ GeV⁻¹） — [Davoudiasl & Huber 2008, arXiv:0804.3543](https://arxiv.org/abs/0804.3543)
- 【文献证实】GECOSAX 观测几何是"望远镜隔着固体地球指向太阳"，信号峰在 ~3 keV、平均 ~4.2 keV — [Fraser et al. 2014, arXiv:1403.2436](https://arxiv.org/abs/1403.2436)
- 【文献证实】XMM-Newton 季节调制：2000–2012 年去点源的空白天区叠加数据在 2–6 keV 出现季节变化，EPIC-pn 显著性 11σ（MOS1 4σ、MOS2 5σ）；pn 最大差值计数率 0.19±0.0025 s⁻¹，对应 ~4.6×10⁻¹² erg cm⁻² s⁻¹ deg⁻²；作者解释为太阳轴子在地磁场中转换（需假设光子并非严格前向），给出 g_ae·g_aγ = 2.2×10⁻²² GeV⁻¹，并提到对 14.4 keV ⁵⁷Fe 线的初步搜寻 — [Fraser et al. 2014, arXiv:1403.2436](https://arxiv.org/abs/1403.2436)
- 【文献证实】反驳：不质疑观测到的季节调制本身，但认为与太阳轴子无关（XMM 从不指向太阳、视场极小等），并称"该季节调制 X 射线超出的物理解释仍是挑战" — [Roncadelli & Tavecchio 2015, arXiv:1411.3297](https://arxiv.org/abs/1411.3297)
- 【文献证实】AQN 解释：Axion Quark Nugget 穿出地球时发射高能光子；预言季节变化幅度 20–25%；谱延伸至 ~100 keV；**明确建议用 Fermi-GBM 或 NuSTAR 多年存档搜寻至 100 keV 的近地季节变化**；并指出信号与望远镜高度有关（高度越高背景应越低）且与视线取向有关 — [Ge, Rachmat, Siddiqui, Van Waerbeke & Zhitnitsky 2022, arXiv:2004.00632](https://arxiv.org/abs/2004.00632)
- 【文献证实】HXMT 太阳规避角 >70° — [Zhang S. et al. 2019, arXiv:1910.04434](https://arxiv.org/pdf/1910.04434)；LE 数据筛选常用 SUN_ANG>30°、并可选只取夜侧（DYE_ELV=200） — [Huang et al. 2022, arXiv:2306.10114](https://arxiv.org/abs/2306.10114)
- 【文献证实】HXMT 团队对 LE 空白天区的定性描述：五年内 LE 本底光变行为变化很小，低能段计数率稳定、高能段受地磁调制 — [Liao et al. 2023, arXiv:2302.14459](https://arxiv.org/abs/2302.14459)
- 【文献证实】其他 ALP 先例（HXMT 均不占优）：NuSTAR 观测 Betelgeuse [Xiao et al. 2021, arXiv:2009.09059](https://arxiv.org/abs/2009.09059)；NuSTAR 作 helioscope [Ruz et al. 2024, arXiv:2407.03828](https://arxiv.org/abs/2407.03828)；磁星硬 X 射线 [Fortin & Sinha 2018, arXiv:1804.01992](https://arxiv.org/abs/1804.01992)、[Lloyd et al. 2021, arXiv:2001.10849](https://arxiv.org/abs/2001.10849)；超星团 [Dessert, Foster & Safdi 2020, arXiv:2008.03305](https://arxiv.org/abs/2008.03305)；CAST 的 14.4 keV 搜索 [CAST 2009, arXiv:0906.4488](https://arxiv.org/abs/0906.4488)；SN 1987A γ [Payez et al. 2015, arXiv:1410.3747](https://arxiv.org/abs/1410.3747)、[Hoof & Schulz 2023, arXiv:2212.09764](https://arxiv.org/abs/2212.09764)；SN 2023ixf 与 Fermi-LAT [Ravensburg et al. 2024, arXiv:2306.16397](https://arxiv.org/abs/2306.16397)、[Meyer & Petrushevska 2020, arXiv:2006.06722](https://arxiv.org/abs/2006.06722)；RHESSI 宁静太阳 [Hannah et al. 2010, arXiv:1009.2918](https://arxiv.org/abs/1009.2918)

### Inferences
#### 想法 F2：用 LE/ME/HE 的空白天区 + 盲探测器 + 地球遮挡数据，检验"近地 X 射线背景季节调制"（XMM 2–6 keV 信号与 AQN 的 ≤100 keV 预言）
1. **一句话**：把十年例行空白天区观测（每月约 10 次）的 2–6 keV（LE）、10–30 keV（ME）、30–100 keV（HE）扣除粒子本底后的残余强度做成时间序列，搜寻年周期/季节调制，并比较"朝外看"与"朝夜侧地球看"两种几何。
2. **原理与灵感**：XMM 的 11σ 季节调制至今没有公认解释；AQN 模型给出可证伪的预言（20–25% 调制、谱到 100 keV、低轨更强）并点名建议用 GBM/NuSTAR 存档检验。
3. **HXMT 特征与适配性**：(i) LEO 550 km，与 XMM（工作高度 >46000 km）形成高度对照；(ii) 1–250 keV 连续覆盖正好横跨 XMM 信号段与 AQN 预言段；(iii) blind 探测器给出同时刻的粒子本底；(iv) 同一批空白天区被反复观测，可消除宇宙方差；(v) 每轨都有夜侧地球遮挡段——AQN 图像下"向下看"应最亮【推断】。**坦率结论**：对 2–6 keV，HXMT/LE 是比 NuSTAR/GBM 更合适的仪器（真正覆盖该能段且 grasp 大）；对 30–100 keV，GBM 的统计量更高但没有 blind 探测器对照。
4. **数据产品**：全部空白天区观测的 LE/ME/HE 1L 事例 + EHK（含 COR、ELV、DYE_ELV、SUN_ANG、地理位置）；公开。
5. **查新**：未检索到用 HXMT 做此检验的论文（官方清单无相关标题）；是否已有人用 GBM/NuSTAR 存档按 AQN 论文建议做过检验，**本次未能检索**（配额用尽）。
6. **可行性与量级**（【推断】）：XMM 差值 4.6×10⁻¹² erg cm⁻² s⁻¹ deg⁻² 按平均光子能量 ~3.5 keV 折算约 2.7 ph cm⁻² s⁻¹ sr⁻¹，约为 2–6 keV CXB（按 Huang 2022 谱积分约 6.5 ph cm⁻² s⁻¹ sr⁻¹）的 40%。若同样强度各向同性地出现在 HXMT 处，LE 小视场（Gε≈0.56 cm² sr）将多出约 1.5 counts/s，而 2–6 keV 总率约 5–6 counts/s；单次 3 ks 观测即 >30σ。十年数据对季节幅度的统计灵敏度可到 CXB 的 ≲1%，实际受限于 LE 增益/温度/阈值的年周期漂移、漏光、低能电子 flare、太阳活动周（2019 极小→2024–25 极大）与轨道面进动的混叠——需用 blind 探测器、COR 分箱与"同一天区不同季节"配对来控制。HXMT 团队已定性报告"LE 低能段计数率稳定"，这已暗示 LEO 处不存在 40% 量级的各向同性调制，但没有定量上限。
7. **打分**：新颖性 4｜可行性 4｜影响 3（零结果即可对 AQN 的"近地各向同性发射"给出强约束，并把 XMM 异常限定为高轨/特定视线效应；若看到调制则影响很大）。

#### 被否决：标准 GECOSAX 与 14.4 keV ⁵⁷Fe 线（详见第 10 节）
- 【推断】信号来自太阳核心方向（隔着地球），而 HXMT 指向与太阳夹角恒 >70°；HE/ME/LE 主视场半宽 ≤3°，LE 超宽视场沿长轴也只有 ±25–30°，均够不到。CsI 全向监测器虽无准直，但阈值 ≥40 keV，远高于 1–10 keV 的 Primakoff 谱。

### Gaps
- AQN 模型对"550 km 处朝外看/朝下看"的定量预言未在原文中找到现成数字，需按其公式重算。
- XMM 调制是否已被后续 XMM 或其他卫星数据复核/否定，本次未检索到（配额用尽）。
- CAST 最新的 g_aγ 上限数值凭记忆约为 0.66×10⁻¹⁰ GeV⁻¹（未取回原文），仅用于判断 GECOSAX 的相对意义。

---

## 3. Lorentz 不变性破坏（LIV）及相关检验：HXMT 已做了什么、还剩什么新角度？

### Takeaway
官方论文清单中没有任何 HXMT 的 LIV 论文；GRB 221009A 的两篇 HXMT 观测论文通篇不含 "Lorentz" 一词。对"能量依赖时延"这条路，HXMT 的 keV–MeV 能段天然处于劣势：Crab 脉冲相位法比 MAGIC 弱十几个量级，GRB 221009A 的 keV–MeV 光变也无法与 LHAASO 的 TeV 光子竞争。没有偏振计就做不了真空双折射——**除非**把 18 个 CsI 晶体之间的 Compton 双击当作"意外的偏振计"，这是一个高风险高回报的推测性想法。

### Cited Findings
- 【文献证实】HXMT 与 GECAM-C 是少数没有被 GRB 221009A 饱和的仪器，给出了前 ~1800 s 的精确测量 — [An et al. 2023, arXiv:2303.01203](https://arxiv.org/abs/2303.01203)；[Zhang W.-L. et al. 2025, arXiv:2504.18952](https://arxiv.org/abs/2504.18952)（两文全文检索 "Lorentz" 均为 0 次命中）
- 【文献证实】GRB 221009A 的 LIV 限制来自 LHAASO 的 0.2–7 TeV 光子：n=1 时 E_QG > 5.9 m_pl（亚光速）/6.2 m_pl（超光速），n=2 时 5.8×10⁻⁸/4.6×10⁻⁸ m_pl；该文只把 HXMT/GECAM 的 keV–MeV 光变作为背景引用 — [Piran & Ofengeim, arXiv:2308.03031](https://arxiv.org/abs/2308.03031)；另见 [LHAASO Collaboration 2024, arXiv:2402.06009](https://arxiv.org/abs/2402.06009)、[Yang, Bi & Yin 2024, arXiv:2312.09079](https://arxiv.org/abs/2312.09079)
- 【文献证实】GRB 内禀谱延迟会把 LIV 限制改变至数倍 — [Vardanyan et al. 2023, arXiv:2212.02436](https://arxiv.org/abs/2212.02436)；keV–MeV 段的"spectral lag transition"方法 — [Wei et al. 2017, arXiv:1612.09425](https://arxiv.org/abs/1612.09425)；多波段 GRB 新限制 — [Chen et al. 2026, arXiv:2412.07625](https://arxiv.org/abs/2412.07625)；综述 — [Wei & Wu 2021, arXiv:2102.03724](https://arxiv.org/abs/2102.03724)
- 【文献证实】脉冲星法先例：Crab 脉冲的 X/γ 相位比较 — [Kaaret 1999, arXiv:astro-ph/9903464](https://arxiv.org/abs/astro-ph/9903464)；MAGIC 用到 TeV 的 Crab 脉冲 — [MAGIC Collaboration 2017, arXiv:1709.00346](https://arxiv.org/abs/1709.00346)；用 Crab 多波段计时检验弱等效原理 — [Zhang & Gong 2017, arXiv:1612.00717](https://arxiv.org/abs/1612.00717)
- 【文献证实】HXMT 的 Crab 脉冲轮廓研究（11–250 keV；与 NICER 联合）是常规天体物理，未做 LIV — [Tuo et al. 2019, arXiv:1906.03633](https://arxiv.org/abs/1906.03633)；[Yan et al. 2022, arXiv:2204.05828](https://arxiv.org/abs/2204.05828)
- 【文献证实】真空双折射/CPT 检验依赖 GRB 偏振测量：GAP [Toma et al. 2012, arXiv:1208.5288](https://arxiv.org/abs/1208.5288)；INTEGRAL/IBIS [Götz et al. 2014, arXiv:1408.4121](https://arxiv.org/abs/1408.4121)；SME 框架下的 GRB 约束 [Kostelecký & Mewes 2013, arXiv:1301.5367](https://arxiv.org/abs/1301.5367)
- 【文献证实】"非专用偏振计靠多探测器 Compton 双击测偏振"的先例：AstroSat/CZTI [Chattopadhyay et al. 2019, arXiv:1707.06595](https://arxiv.org/abs/1707.06595)；RHESSI [Coburn & Boggs 2003, arXiv:astro-ph/0305377](https://arxiv.org/abs/astro-ph/0305377)；GRB 221009A 只有 IXPE 的余辉偏振测量 [Negro et al. 2023, arXiv:2301.01798](https://arxiv.org/abs/2301.01798)
- 【文献证实】HE 事例时间戳量化为 2 μs（500 kHz），18 个探测器各自独立记录 — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)

### Inferences
#### 想法 F3：把 HE 的 18 个 CsI 晶体当作"意外的 Compton 偏振计"，尝试测 GRB 221009A 等极亮暴的瞬时偏振 → 真空双折射 LIV 限制
1. **一句话**：在 GRB 模式数据里找出相邻 CsI 模块间 ≤1 个时间 tick（2 μs）内的双击事例，由散射方位角分布提取偏振度/偏振角，再用"偏振未被抹平"约束双折射型 LIV。
2. **原理与灵感**：线偏振光子的 Compton 散射方位角呈 cos2φ 调制；若存在 d=5 的 CPT-odd LIV，偏振面旋转 ∝E²·距离，宽能段内仍能测到偏振即给出极强限制（Toma et al. 2012；Kostelecký & Mewes 2013）。CZTI、RHESSI 都是"非设计用途"的双击偏振测量先例。
3. **HXMT 特征与适配性**：HE 是 GRB 221009A 期间少数未饱和、且在 0.2–3 MeV 有 ~千 cm² 面积的仪器；18 模块呈近六角密排，天然提供 6 个方位向邻居；μs 时间戳可做离线符合。**坦率结论**：HE 不是为此设计的——晶体大（19 cm）、厚（4 cm CsI）、模块间有外壳与准直器等被动物质，调制因子和双击比例都未知；这是"可能根本做不出来"的想法，但目前没有任何仪器给出该暴的瞬时 MeV 偏振。
4. **数据产品**：GRB 模式（低增益）HE 事例文件，需保留 CsI 事例（默认流程会删除）；公开的 728 个 GRB 数据中包含亮暴。
5. **查新**：官方清单标题中无 HXMT GRB 偏振或 LIV 论文；未见任何"HE 模块间符合偏振"的文献【在已做检索范围内】。
6. **可行性**（【推测】为主）：若双击比例 ~1%、亮暴 CsI 总计数 ~10⁷，则双击 ~10⁵ 个；取 μ₁₀₀~0.15，MDP ≈ 4.29/(μ√N) ≈ 9%。偶然符合：单模块 10⁴ c/s 时每对模块 r²τ ≈ 10⁸×4×10⁻⁶ ≈ 400 s⁻¹，与真双击同量级，峰值段必须统计扣除。**缺失信息**：Geant4 质量模型下的模块间散射比例与调制因子；离轴入射与卫星平台散射造成的伪调制；高计数率下时间戳与死时间行为；GRB 221009A 期间 HE 数据是否有丢包。若能测到偏振，由 E² 标度，MeV 能段比 GAP（70–300 keV）的双折射限制可望再紧 1–2 个量级【推断】。
7. **打分**：新颖性 5｜可行性 1–2｜影响 4（成功则同时是 BOAT 瞬时偏振的唯一测量与最强双折射限制之一）。

#### 低优先级补充：SME 各向异性系数与"谱延迟转折"
- 【推断】在 Standard-Model Extension 框架中，非双折射的色散系数依赖源方向，每个 GRB 约束一组球谐组合；HXMT 的 GRB 样本（四年星表 + 公开的 728 个 GRB 数据）可增加天区方向，但与 Fermi-GBM 样本相比无本质优势（新颖性 2、可行性 4、影响 1–2）。

#### 定量否决：Crab 脉冲相位 LIV
- 【推断】n=1 时 Δt = (ΔE/E_QG)(d/c)；Crab d≈2 kpc（d/c≈2×10¹¹ s），ΔE=250 keV，即便相位对齐做到 10 μs，也只得 E_QG ≳ 5×10¹² GeV ≈ 4×10⁻⁷ E_Pl，比 LHAASO 的 ~6 E_Pl 低约 7 个量级。十年、千万光子也改变不了这个标度。

### Gaps
- 未能检索 2023–2026 年间是否有外部作者用 HXMT/GECAM 的 GRB 221009A 或 GRB 230307A keV–MeV 光变做 LIV（配额用尽；官方清单标题中无）。
- MAGIC Crab 的 E_QG 数值（凭记忆 n=1 约 5.5×10¹⁷ GeV）未从原文提取。
- RHESSI GRB 021206 偏振结果后来受到质疑（凭记忆），相关反驳文献未取回。

---

## 4. 原初黑洞（PBH）：蒸发暴、femto/pico-lensing、弥漫背景贡献——HXMT 的现实角色？

### Takeaway
PBH 是基础物理方向里 HXMT **已被占位最多**的一块：弥漫背景约束（Yang et al. 2026）、蒸发暴的"波前曲率"几何测距（Xiao & Zhang 2026，HXMT/HE 作为交叉验证仪器）都已发表。剩余空间有限：把 CXB 谱从 120 keV 推到 250 keV（NaI）乃至 0.2–3 MeV（CsI）会直接改善 10¹⁷ g 附近的约束，但信本比只有 10⁻³ 量级，难度很大；lensing 视差受制于仪器间流量互定标，基本不可行。

### Cited Findings
- 【文献证实】用 Huang et al. 2022 的 LE+HE 弥漫背景数据（1.5–120 keV）约束 2×10¹⁶–5×10¹⁷ g 的 PBH：直接用数据点或加天体物理本底模型时 HXMT 无优势；用截断幂律外推后对 >10¹⁷ g 有所加强，排除"全部暗物质"的质量上限到 4×10¹⁷ g；作者写明"若有 120 keV 以上数据将更有利"，并计划改用模板分析 — [Yang, Pan & Zhang 2026, Commun. Theor. Phys. 78, 045402, arXiv:2412.09297](https://arxiv.org/abs/2412.09297)
- 【文献证实】同类先例：INTEGRAL [Laha, Muñoz & Slatyer 2020, arXiv:2004.00627](https://arxiv.org/abs/2004.00627)；各向同性 X 射线背景 [Iguaz, Serpico & Siegert 2021, arXiv:2104.03145](https://arxiv.org/abs/2104.03145)；[Ballesteros, Coronado-Blázquez & Gaggero 2020, arXiv:1906.10113](https://arxiv.org/abs/1906.10113)
- 【文献证实】几何测距法：近处 PBH 蒸发暴的波前是弯曲的，多星时延偏离平面波；用 Swift 定位的短暴样本未见偏离，最强事例约束到 1.2 AU；据检索摘要，GBM–HXMT/HE 等近地组合被用作交叉验证 — [Xiao & Zhang 2026, arXiv:2603.16508](https://arxiv.org/abs/2603.16508)；更早的 IPN 版本 [Ukwatta et al. 2016, arXiv:1512.01264](https://arxiv.org/abs/1512.01264)
- 【文献证实】"反向暴"模板搜索：39 个 Swift 短暴全部是快升慢降，与 PBH 末期蒸发的单调上升光变不符，给出本地爆发率 R ≲ 10⁵ pc⁻³ yr⁻¹ — [Profumo & Wen 2025, arXiv:2512.08778](https://arxiv.org/abs/2512.08778)
- 【文献证实】GeV–TeV 的爆发率上限：Fermi-LAT 7.2×10³ pc⁻³ yr⁻¹ [Fermi-LAT 2018, arXiv:1802.00100](https://arxiv.org/abs/1802.00100)；H.E.S.S. [Glicenstein et al. 2013, arXiv:1307.4898](https://arxiv.org/abs/1307.4898)、[H.E.S.S. 2023, arXiv:2303.12855](https://arxiv.org/abs/2303.12855)；HAWC [Albert et al. 2020, arXiv:1911.04356](https://arxiv.org/abs/1911.04356)；末期蒸发观测特征 [Ukwatta et al. 2016, arXiv:1510.04372](https://arxiv.org/abs/1510.04372)
- 【文献证实】GRB lensing：femtolensing 旧限制 [Barnacka, Glicenstein & Moderski 2012, arXiv:1204.2056](https://arxiv.org/abs/1204.2056) 及其重新审视 [Katz et al. 2018, arXiv:1807.11495](https://arxiv.org/abs/1807.11495)；lensing parallax 需要 r⊕–AU 量级基线 [Jung & Kim 2020, arXiv:1908.00078](https://arxiv.org/abs/1908.00078)；可行性分析 [Gawade, More & Bhalerao 2024, arXiv:2308.01775](https://arxiv.org/abs/2308.01775)；HXMT 参与的毫透镜候选 "GRB 200716C: Evidence for a Short Burst Being Lensed"（Wang et al. 2021, ApJ 918, 34；见[官方论文清单](http://hxmten.ihep.ac.cn/papers.jhtml)）
- 【文献证实】地球遮挡法测 CXB 的先例：INTEGRAL [Churazov et al. 2007, arXiv:astro-ph/0608250](https://arxiv.org/abs/astro-ph/0608250)、[Türler et al. 2010, arXiv:1001.2110](https://arxiv.org/abs/1001.2110)；Swift/BAT [Ajello et al. 2008, arXiv:0808.3377](https://arxiv.org/abs/0808.3377)；地球大气硬 X 射线发射与 CXB 反照的模拟 [Sazonov et al. 2007, arXiv:astro-ph/0608253](https://arxiv.org/abs/astro-ph/0608253)、[Churazov et al. 2008, arXiv:astro-ph/0608252](https://arxiv.org/abs/astro-ph/0608252)

### Inferences
#### 想法 F4：把 HXMT 的弥漫背景谱延伸到 120–250 keV（NaI）并尝试 0.2–3 MeV（CsI 地球遮挡调制），配银心/反银心模板，收紧 10¹⁷ g PBH 与 MeV 暗物质约束
1. **一句话**：用十年全部定点观测的"入/出地影"成对数据与 blind 探测器，把 HE 的 CXB 测量推过 120 keV，并探索 CsI 全向数据随地球方位的轨道调制。
2. **原理/灵感**：Hawking 辐射光子谱峰随质量升高移向 100 keV–MeV；Yang et al. 自己指出缺 >120 keV 数据。灵感来自 INTEGRAL、BAT 的地球遮挡 CXB 测量。
3. **HXMT 特征与适配性**：blind 探测器 + 大/小视场差分；CsI 92 kg（18×283.5 cm²×4 cm×4.51 g cm⁻³）【推断】。**坦率结论**：不如 INTEGRAL/SPI（分辨率、成像、已有 16 年模板分析）；HXMT 只在"独立系统误差"意义上有价值。
4. **数据产品**：全部定点观测的 HE 事例（含地球遮挡段，默认 GTI 会剔除）+ EHK；CsI 事例需关闭默认删除。公开。
5. **查新**：>120 keV 的 HXMT CXB 与 CsI 弥漫背景测量均未见（官方清单标题无）。
6. **可行性**（【推断】）：按 Huang 2022 拟合，200 keV 处 CXB ≈1.5×10⁻⁴ ph cm⁻² s⁻¹ keV⁻¹ sr⁻¹；HE 在 150–250 keV 的 CXB 计数率 ≈1.5×10⁻⁴×100×13.7×0.6 ≈ 0.1 c/s，而本底 ≈3.5×10⁻⁴×4820×100 ≈ 170 c/s，信本比 ~10⁻³，需要把 on/off 本底差控制到 10⁻⁴——比现有本底模型精度（1.5–3%）高两个量级，只能指望"大视场−小视场"同刻差分；Huang et al. 已发现地球遮挡法在累计 >90 ks 后出现偏差。CsI 无准直，地球始终挡住约 30% 天空（550 km 处地球张角约 3.8 sr【推断】），调制信号与大气 γ 反照（随 COR 变化）强简并。
7. **打分**：新颖性 2–3｜可行性 2｜影响 3。

#### 想法 F5：HE/CsI 毫秒–亚毫秒"反向暴"无触发搜索 + 联合 GECAM/GBM 的近场曲率检验
1. **一句话**：在十年 CsI 连续事例流中搜索单调上升、谱极硬、无余辉的亚秒暴，并对候选用多星时延检验是否来自有限距离。
2. **原理/灵感**：Profumo & Wen 2025 的"backwards burst"模板 + Xiao & Zhang 2026 的波前曲率。
3. **适配性**：CsI 面积大、μs 定时、低增益到 3 MeV；但标准模型下 PBH 末期辐射主峰在 GeV–TeV，keV–MeV 探测距离很短。**坦率结论**：单星不占优，价值在与 GECAM/GBM 组网；该方向已被 HXMT 团队成员占位。
4. **数据**：HE 事例（保留 CsI）、GECAM/GBM 公开数据。
5. **查新**：几何法已发表（2603.16508）；HXMT 无触发"反向暴"专门搜索未见。
6. **可行性**（【推断】）：技术上可沿用 Cai et al. 2021 的相干搜索管线；主要障碍是 spike 与 TGF/带电粒子事件的误触发；无法给出有意义的爆发率灵敏度估计，缺"HXMT/CsI 对 PBH 末期谱的探测距离"这一输入（需把含全部标准模型自由度的蒸发谱折叠进 CsI 响应）。
7. **打分**：新颖性 2｜可行性 3｜影响 2。

### Gaps
- Xiao & Zhang 2026 中 HXMT 的具体使用方式来自检索摘要与论文摘要，未逐节核对。
- 是否已有 LHAASO 的 PBH 爆发率新上限（凭记忆 2025 年有相关预印本）未能核对。
- lensing parallax 所需的仪器间流量互定标精度（HXMT/CsI 相对 GBM）没有找到公开数字，故判为不可行主要基于【推断】。

---

## 5. 奇异事件：strangelets / nuclearites、Q-balls、磁单极、反核、重离子径迹、反符合数据中的"无法解释"事例

### Takeaway
HE 的事例数据同时带有 2 μs 时间戳、18 路反符合标签和脉宽，等于一个 0.5 m² 的 18 模块 + 2.55 m² 塑闪的**飞行时间阵列**；星系速度（~10⁻³c）的大质量粒子穿越 1 m 阵列需 3–10 μs，而宇宙线只需 ~3 ns——这与芝加哥地面复合暗物质探测器的判选原理完全相同。被流程丢弃的"spike 事例"（巨能量沉积后的余辉）正是这类搜索的天然触发标签。但 HE 没有电荷/能量分辨能力，相对论性 strangelet、磁单极、反核**做不了**。

### Cited Findings
- 【文献证实】nuclearite/strangelet 综述与现有限制：综述 [Madsen 2006, arXiv:astro-ph/0612740](https://arxiv.org/abs/astro-ph/0612740)；MACRO [MACRO Collaboration 2000, arXiv:hep-ex/9904031](https://arxiv.org/abs/hep-ex/9904031)；高山径迹探测器 SLIM [Balestra et al. 2006, arXiv:hep-ex/0506075](https://arxiv.org/abs/hep-ex/0506075)；ANTARES 九年数据对质量至 10¹⁷ GeV/c² 的 nuclearite 给出约 5×10⁻¹⁷ cm⁻² s⁻¹ sr⁻¹ 量级的通量上限 [ANTARES 2023, arXiv:2208.11689](https://arxiv.org/abs/2208.11689)
- 【文献证实】空间搜索的必要性与现状：拟议中的 SQM-ISS 用闪烁体 + 压电元件 + 飞行时间系统测量以 ~250 km/s 穿过探测器的重粒子 — [SQM-ISS 2024, PMC11359484](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11359484/)；DAMPE 搜索分数电荷粒子 [DAMPE 2022, arXiv:2209.04260](https://arxiv.org/abs/2209.04260)
- 【文献证实】Macro dark matter 约束：暗物质通量 (4π)⁻¹ n v ≈ 2×10¹⁷ (GeV/M) m⁻² yr⁻¹ sr⁻¹；Skylab 塑料径迹探测器曝光量约 2 m² yr sr，排除 M ≲ 10¹⁷ GeV（约 2×10⁻⁷ g）的 macro，但要求可蚀刻径迹即 σ ≳ 10⁻¹⁵ cm²；古云母限制给出 M ≳ 55 g，对质量 ≳10⁻¹⁰ g、σ ≳ 6×10⁻¹⁸ cm² 的对象适用 — [Jacobs, Starkman & Lynn 2015, arXiv:1410.2236](https://arxiv.org/abs/1410.2236)
- 【文献证实】复合暗物质的地面飞行时间搜索：两个相距 50 cm 的液闪模块做符合，v~10⁻³c 的暗物质飞行时间 ~2 μs、宇宙线 ~2 ns；最大可探质量 m ≈ ρv(AΩt) ≈ 4×10¹² GeV；上边界由大气（~10 m.w.e.）衰减决定，约 σ/m ≈ 10⁻²⁷ cm²/GeV；下边界 σ ≈ 10⁻²² cm²；作者指出点状暗物质的大截面已被理论封死，但**复合暗物质仍有大片参数空间开放**，图中其他限制只有 Skylab、云母与一条宇宙学限制 — [Cappiello, Collar & Beacom 2021, arXiv:2008.10646](https://arxiv.org/abs/2008.10646)
- 【文献证实】HE 单体外形 243 mm×243 mm×285 mm；事例时间戳量化 2 μs；每事例带 18 路 HVT 标签 — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)
- 【文献证实】HE 的 spike：多发生在单个探测器、持续 <200 μs、能量 25–95 keV，被归因于高能宇宙线；常规流程将其删除 — [Cai et al. 2021, arXiv:2109.12270](https://arxiv.org/pdf/2109.12270)

### Inferences
#### 想法 F6：把 HE 当作在轨飞行时间阵列，搜索"慢速、共线、多模块、按序触发"的巨能量沉积事例（nuclearite / 轻 strangelet / 复合暗物质 / Q-ball）
1. **一句话**：在不做反符合剔除、不删 spike 的原始事例流里，寻找 ≥4 个共线模块按恒定速度（100–700 km/s）依次出现饱和信号或 spike 的事例。
2. **原理/灵感**：nuclearite 能损 dE/dx = ρσv²；取最小截面 σ≈3×10⁻¹⁶ cm²、v=250 km/s，在 CsI 中约 0.9 erg/cm ≈ 550 GeV/cm【推断】，即使发光效率 10⁻⁴ 也远超量程。判选思想与 Cappiello et al. 2021 的 μs 飞行时间完全一致，SQM-ISS 是为同一目标设计的专用载荷。
3. **适配性**：18 模块 + 18 路 HVT + 2 μs tick + 十年无人值守曝光是现成的；轨道上没有大气遮蔽，能探到无法到达地面/高山的轻质量、大截面对象。**坦率结论**：HE 没有 dE/dx、电荷或速度的专门测量，时间分辨（2 μs，对应 250 km/s 下 0.5 m）只够分辨跨越 ≥3–4 个模块的径迹；对标准 nuclearite（σ≥10⁻¹⁵ cm²、M≲10¹⁷ GeV）Skylab 的曝光量已更大。HXMT 的增量在 Skylab 蚀刻阈值以下的截面（10⁻²²–10⁻¹⁵ cm²）与实时电子学判选。
4. **数据**：HE 1L 事例（必须是 `hepical` 之前、含 spike、含 ACD 标记与 CsI 事例的版本）；公开，但需确认 overflow 事例的编码。
5. **查新**：早期检索与官方清单均未见 HXMT 或其他 X 射线天文卫星做过 nuclearite/慢重粒子搜索；空间先例为 Skylab（见上）与拟议的 SQM-ISS。
6. **可行性与量级**（【推断】）：共线 ≥4 模块、近阵列平面入射的接受度约 10²–10³ cm² sr；活时间按 10 年×0.6 ≈ 1.9×10⁸ s；曝光 2×10¹⁰–2×10¹¹ cm² sr s → 90% CL 通量上限 ~10⁻¹¹–10⁻¹⁰ cm⁻² s⁻¹ sr⁻¹。与暗物质通量 6×10⁵/M_GeV cm⁻² s⁻¹ sr⁻¹（ρ=0.3 GeV cm⁻³、v=250 km/s）比较，可覆盖到 M ~10¹⁶ GeV 量级。偶然符合本底：若单模块"大信号"率 r~100 s⁻¹，五重共线且等时间间隔（速度自洽因子 ~10⁻⁵）的偶然率约 10×r⁵τ⁴×10⁻⁵ ~10⁻¹⁰ s⁻¹（τ=100 μs），十年约 0.02 个，可视为无本底；三重符合则不行（~0.5 s⁻¹）。**主要未知**：PMT 对巨脉冲的饱和与恢复、巨沉积后的余辉是否淹没后续时间信息、overflow 事例是否入盘、总览表"25 μs 时间分辨"与"2 μs 时间戳"哪个是有效精度。
7. **打分**：新颖性 4–5｜可行性 2｜影响 3（零结果也能在复合暗物质参数图上圈出一块"地面探测器因大气遮蔽看不到"的区域）。

### Gaps
- De Rújula & Glashow 1984（Nature 312, 734，nuclearite 能损公式）、Shirk & Price 1978（Skylab）、Price & Salamon 1986（云母）、PAMELA strangelet 搜索（凭记忆为 PRL 115, 111101）均**凭记忆引用、未取回原文**；本节相关数字取自 Jacobs et al. 2015 与 Cappiello et al. 2021 的转述。
- IMP 7/8、XQC、RRS 气球等对"强相互作用暗物质"的旧限制在 10⁻²²–10⁻¹⁵ cm²、10¹⁰–10¹⁷ GeV 区域是否已覆盖 F6/F7 的目标区，未能完整核对（见第 9 节）。
- HE 对 MeV–GeV 级沉积的实际电子学响应无公开文献。
- 磁单极、反核湮灭、Q-ball 的专门特征在 HE 数据里没有可区分的观测量（无磁谱仪、无 dE/dx），未找到任何可行路径。

---

## 6. 用 X 射线卫星做宇宙线物理：太阳调制、Forbush 下降、地磁截止刚度谱仪、East–West 效应

### Takeaway
这一块与"空间天气"调研方向重叠（同目录下 `space_weather_earth_solar_system.md` 已有"每 95 分钟扫一遍 2–15 GV 的宇宙线监测器"等想法），此处只记基础物理视角的判断：2.55 m² 的 HVT 塑闪每秒计数、十年连续，统计精度远超地面中子堆，但它是**积分计数器**（无种类、无刚度、含再入反照粒子），物理上无法与 AMS-02/DAMPE/CALET 的直接测量竞争；其价值在高时间分辨与"地磁截止当谱仪"的差分技巧，属于稳妥但影响有限的方向。

### Cited Findings
- 【文献证实】HPM（3 个）对 >20 MeV 质子、>1.5 MeV 电子灵敏（设计阈值 20 MeV/1 MeV），每秒记录计数率；磁赤道附近 <3 cts/s、高纬 >10 cts/s；五年间磁赤道低计数区平均值稳定在 1.7–1.8 cts/s — [Lu et al. 2020, arXiv:1911.01594](https://arxiv.org/abs/1911.01594)；[Liao et al. 2020, arXiv:2005.01661](https://arxiv.org/pdf/2005.01661)；[Liao et al. 2023, arXiv:2302.14459](https://arxiv.org/abs/2302.14459)
- 【文献证实】HE 本底与地磁截止刚度（COR）显著反相关，谱形随 COR 变化；ME 本底可随位置变化达 50%；LE 高能段受地磁调制 — [Liao et al. 2020, arXiv:2005.01661](https://arxiv.org/pdf/2005.01661)；[Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/pdf/2009.14489)；[Liao et al. 2020, arXiv:2004.01432](https://arxiv.org/abs/2004.01432)
- 【文献证实】HXMT 数据已被用于 2017 年 9 月太阳爆发引起的宇宙线事件研究（"A Comparative Study of Two Contrasting Cosmic-Ray Events Caused by Solar Eruptions from NOAA AR 12673 in 2017 September", Yu et al. 2024, ApJ 960, 85）和 SAA 漂移研究（Zhao et al. 2024, JHEAp 44, 187）——见[官方论文清单](http://hxmten.ihep.ac.cn/papers.jhtml)（仅核对标题）
- 【文献证实】其他非宇宙线专用平台的先例：LISA Pathfinder 的 GCR 复现性下降与 Forbush 下降 [Armano et al. 2018, arXiv:1802.09374](https://arxiv.org/abs/1802.09374)；DAMPE 的电子/正电子 Forbush 下降 [Alemanno et al. 2021, arXiv:2110.00123](https://arxiv.org/abs/2110.00123)；Fermi-GBM 物理本底模型 [Biltzinger et al. 2020, arXiv:2005.11219](https://arxiv.org/abs/2005.11219)；Suzaku HXD 本底与 COR 的关系 [Fukazawa et al. 2009, arXiv:0901.0419](https://arxiv.org/abs/0901.0419)；Forbush 下降期间截止刚度变化的影响 [Zhao & Feng 2026, arXiv:2604.06857](https://arxiv.org/abs/2604.06857)；X 射线探测器在轨本底综述 [Campana 2022, arXiv:2205.09359](https://arxiv.org/abs/2205.09359)

### Inferences
#### 想法 F7：HVT 侧面板的 East–West 不对称 + 截止刚度扫描，给出 2–15 GV 积分谱及其在第 24→25 太阳周的调制
1. **一句话**：用 12 块朝向不同的侧面 HVT 的每秒计数率，按"朝东/朝西"与垂直截止刚度分箱，得到积分刚度谱、电荷不对称和 Forbush 下降的刚度依赖。
2. **原理/灵感**：带正电原初粒子在截止附近优先从西边来（East–West 效应）；地磁场本身是天然磁谱仪。
3. **适配性**：面积大、十年连续、卫星惯性指向使各面板相对地磁东西向不断变化。**坦率结论**：不如 AMS-02 等直接测量；与兄弟方向的"想法 7"重叠，建议合并执行。
4. **数据**：HE 工程/状态数据中的 18 路 HVT 每秒计数率、HPM 计数率、姿态与轨道；是否随 Level-1 公开**未知**。
5. **查新**：官方清单中有 2017-09 事件个例与 SAA 研究，未见全任务期刚度谱/East–West 分析。
6. **可行性**（【推断】）：统计上每 100 s 可达 ~0.1% 精度（按 2.55 m²×~1 粒子 cm⁻² s⁻¹）；系统误差来自 HVT 阈值/增益十年漂移、反照粒子污染、SAA 后活化。无法给出绝对通量精度，缺 HVT 的几何因子与阈值标定。
7. **打分**：新颖性 2–3｜可行性 4–5｜影响 2。

### Gaps
- HVT 每秒计数率是否在公开的 Level-1 产品里（文件名、列名）未查到。
- AMS-02 逐日质子通量（凭记忆 PRL 127, 271102）未取回，未做定量对比。
- RXTE/HEXTE、BeppoSAX、INTEGRAL/SPI-ACS 作宇宙线监测的具体文献本次未检索到（配额用尽）。

---

## 7. 在轨核物理：活化线测截面/半衰期、长寿命核素累积、核衰变率时间变化（Jenkins–Fischbach）检验、511 keV

### Takeaway
两条线索都从未在 X 射线天文卫星上做过：(a) HE 自带 18 个 α 标记的 ²⁴¹Am 源，十年累计 10¹⁰ 量级的标定事例，可做"衰变率年调制/耀斑相关"的空间检验——但 ²⁴¹Am 是 α 衰变，而争议主张多针对 β/EC 衰变，且该主张已被多项实验否定，所以物理回报有限；(b) 每天 8–9 次穿 SAA 相当于重复了上万次的"质子束照射—衰变曲线"实验，NaI/CsI 中 I、Te、Xe、Na 系核素的产额与半衰期可作为地下 NaI 暗物质实验宇生本底模型的独立校验——稳妥、可发表，但属于核数据而非新物理。

### Cited Findings
- 【文献证实】HE/NaI 本底线 31、56、67、191 keV 来自宇宙线与 SAA 质子对碘的活化；~67 keV 线相对强度在第一年快速上升后趋于平衡；在轨前五年所有 HE/NaI 探测器的增益变化 <1%（用 191 keV 线监测） — [Li X.B. et al. 2020, arXiv:2003.06998](https://arxiv.org/abs/2003.06998)；[Li X.F. et al. 2023, arXiv:2302.03859](https://arxiv.org/abs/2302.03859)
- 【文献证实】CsI 谱中用于增益标定的本底线约在 67、191、478、692、1182、1372、1729、2767 keV — [Li X.F. et al. 2023, arXiv:2302.03859](https://arxiv.org/abs/2302.03859)
- 【文献证实】HE 本底每次过 SAA 后迅速升高再逐渐衰减；同一地理位置升轨与降轨的本底强度和谱形明显不同（活化衰减程度不同）；长期演化用"二次函数/两段折线"经验描述；卫星约 10% 时间在 SAA 内 — [Liao et al. 2020, arXiv:2005.01661](https://arxiv.org/pdf/2005.01661)；[Liao et al. 2023, arXiv:2302.14459](https://arxiv.org/abs/2302.14459)；[Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/pdf/2009.14489)
- 【文献证实】Geant4 模拟与在轨观测的 HE 总本底率在两年后相差约 5%（538.6 对 563.2 counts/s），早期（前三个月）偏差达 20–30% — [Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/pdf/2009.14489)
- 【文献证实】同类"在轨活化验证模拟"先例：Hitomi HXI/SGD — [Odaka et al. 2018, arXiv:1804.00827](https://arxiv.org/abs/1804.00827)；INTEGRAL/SPI 本底线谱学 — [Diehl et al. 2018, arXiv:1710.10139](https://arxiv.org/abs/1710.10139)
- 【文献证实】地下 NaI(Tl) 的宇生核素研究（与 HE 同一靶材料）：[Amaré et al. 2015, arXiv:1411.0106](https://arxiv.org/abs/1411.0106)；COSINE-100 [Barbosa de Souza et al. 2020, arXiv:1905.12861](https://arxiv.org/abs/1905.12861)
- 【文献证实】衰变率争议：主张衰变率与日地距离相关 [Jenkins et al. 2009, arXiv:0808.3283](https://arxiv.org/abs/0808.3283)、耀斑期间扰动 [Jenkins & Fischbach 2009, arXiv:0808.3156](https://arxiv.org/abs/0808.3156)、综述 [Fischbach et al. 2010, arXiv:1007.3318](https://arxiv.org/abs/1007.3318)、相位分析 [Sturrock et al. 2011, arXiv:1106.2374](https://arxiv.org/abs/1106.2374)；否定证据 [Norman et al. 2009, arXiv:0810.3265](https://arxiv.org/abs/0810.3265)、Gran Sasso [Bellotti et al. 2012, arXiv:1202.3662](https://arxiv.org/abs/1202.3662)、[Bellotti et al. 2013, arXiv:1311.7043](https://arxiv.org/abs/1311.7043)、PTB [Kossert & Nähle 2014, arXiv:1407.2493](https://arxiv.org/abs/1407.2493)
- 【文献证实】唯一的航天器检验：Cassini 的 ²³⁸Pu RTG（α 衰变）输出功率未见随日心距变化 — [Cooper 2009, arXiv:0809.4248](https://arxiv.org/abs/0809.4248)
- 【文献证实】²⁴¹Am 源 ~200 Bq/探测器，α–X 符合窗 2 μs，标定事例带独立 flag 入盘；phoswich 工作温度主动控制在 18±2 °C — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)；[The HXMT Data Reduction Guide for hxmtsoftv2.06（HXMT Science Data Center, 2023-12-27）](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)（手册文本由本会话其他调研员抓取，本笔记据其文本核对）

### Inferences
#### 想法 F8：18 个 α 标记 ²⁴¹Am 源 + 活化 EC 线：十年在轨"核衰变率恒定性"检验
1. **一句话**：把一直只用于增益锁定的 calibration event 计数率（18 路独立）做成十年时间序列，搜寻年周期、太阳自转周期与 X 级耀斑相关的变化；并用每次出 SAA 后 191 keV 等线的衰减曲线反复测 EC 核素半衰期。
2. **原理/灵感**：Jenkins–Fischbach 主张衰变率随日地距离有 ~10⁻³ 量级年调制【量级凭记忆】；Cooper 用 Cassini RTG 做了唯一的航天器检验。
3. **适配性**：18 路冗余、符合标记几乎无本底、十年连续；另有上万次 SAA 后的 EC 衰变曲线。**坦率结论**：卫星与地球同处 1 AU、日地距离变化与地面实验相同（±1.7%），不像 Cassini 那样有大的日心距杠杆；²⁴¹Am 是 α 衰变，Cassini 已对 α 衰变给出零结果；主张本身已被主流否定。因此这是"几乎零成本的副产品"，而非发现型课题。
4. **数据**：HE 事例文件中的 calibration-flag 事例、HE-HV、HE-TH、死时间文件；公开（需确认标定事例是否全量下传）。
5. **查新**：未见任何 X 射线天文卫星做过在轨衰变率恒定性分析（早期检索 + 官方清单）；RXTE/HEXTE 等也带 ²⁴¹Am 增益源，是否有人做过同类分析**未能检索**。
6. **可行性**（【推断】）：若每探测器标定事例率 10–50 s⁻¹、活时间 (1–2.5)×10⁸ s，则单探测器 10⁹–10¹⁰ 事例，年调制幅度统计灵敏度 ~(1–5)×10⁻⁵，18 路合并 ~10⁻⁵ 以下——比被主张的 10⁻³ 低两个量级。真正的限制是系统误差：MPPC 增益与 α 标记效率的温度依赖（年/β 角周期）、死时间随本底率变化、偶然符合（~本底率×200 Bq×2 μs，随轨道位置变化）、AGC 调压本身。18 路之间的相互比对与"同 COR、同 SAA 后时间"配对可压低这些效应。**缺失信息**：标定事例实际计数率及其下传完整性。
7. **打分**：新颖性 4｜可行性 3｜影响 1–2。

#### 想法 F9：把 SAA 当质子束——NaI/CsI 在轨活化产额与半衰期的"核数据实验"，服务地下 NaI 暗物质实验的宇生本底模型
1. **一句话**：利用上万次"过 SAA → 衰减"循环和十年累积曲线，逐条分解 HE NaI（6.6 kg）与 CsI（92 kg）本底线的核素、产额与半衰期，并与 Geant4/截面库对比。
2. **原理/灵感**：Odaka et al. 2018 用 Hitomi 做过 CdTe/BGO 的在轨活化验证；DAMA/COSINE/ANAIS 类实验的低能本底模型依赖 I、Te、Na 系宇生核素的产额。
3. **适配性**：NaI(Tl) 正是地下暗物质实验的靶材料；blind 探测器保证谱中无天体信号；升/降轨差异提供不同"冷却时间"的自然对照。**坦率结论**：辐照谱是 SAA 俘获质子 + 宇宙线混合，不是单能束，只能约束"谱折叠后的产额"；HE 阈值 >20 keV，看不到地下实验最关心的 ³H、²²Na 低能成分。
4. **数据**：HE 1L 事例（含 CsI）、HPM 计数（作为注量代理）、EHK；公开。
5. **查新**：HXMT 团队已做本底模拟与线的经验使用，但未见逐核素产额/半衰期测量论文（官方清单标题）；Fermi-GBM、GECAM 是否做过同类工作未能检索。
6. **可行性**（【推断】）：统计量充裕（191 keV 线每天上万计数量级）；障碍是 15–19% 的能量分辨导致线重叠、不同核素半衰期相近时的简并、SAA 注量的逐次变化（需用 HPM 归一）。可给出的结果是"相对产额 ±10–20%、短寿命核素半衰期 ~1%"量级——此为粗估，缺逐线计数率。
7. **打分**：新颖性 3｜可行性 4｜影响 2。

### Gaps
- 各线对应的具体核素（如 191 keV ≈ ¹²³I 的 159 keV γ + K 壳结合能、67 keV ≈ ¹²⁵I）是我凭核数据常识的指认，HXMT 论文只写"碘活化"，未取回逐线指认的文献。
- Pommé et al. 2016（"Evidence against solar influence on nuclear decay constants", Phys. Lett. B 761, 281）凭记忆引用，未取回。
- Jenkins et al. 主张的调制幅度数值未从原文提取。
- 511 keV 线：CsI 谱的线表里没有列 511 keV（列的是 478 keV 等），大气/活化 511 keV 的监测能力未查到数据；该题目兄弟方向已覆盖。

---

## 8. 以地球、月球、太阳为遮挡体或靶

### Takeaway
地球是 HXMT 唯一可用的"遮挡体/靶"：每轨约 30 min 的地球遮挡段是纯本底 + 可能的奇异信号样本，已被用来测 CXB，但只用了不到 0.2 Ms。太阳（规避角 >70°）和月球（规避角 >6°）永远不进主视场，相关想法全部否决。地球遮挡数据的基础物理用途已并入 F1（线搜寻的 on/off）、F2（AQN"向下看"）和 F4（CXB 延伸）。

### Cited Findings
- 【文献证实】地球遮挡几乎每轨发生、持续约 30 min；太阳规避角 >70°、月球规避角 >6° — [Zhang S. et al. 2019, arXiv:1910.04434](https://arxiv.org/pdf/1910.04434)
- 【文献证实】HE 本底数据库构建时**刻意不用**地球遮挡段数据（地球反射的 X 射线可进入视场） — [Liao et al. 2020, arXiv:2005.01661](https://arxiv.org/pdf/2005.01661)
- 【文献证实】Huang et al. 2022 用 166 段地球遮挡数据（HE "Earth-occulted" 累计 187 ks，判据 ELV<−6°）作为 HE 本底，并需修正地球反照与大气发射；LE 的地球遮挡法受滤光膜漏光等系统效应影响，故改用 blind 探测器 — [Huang et al. 2022, arXiv:2306.10114](https://arxiv.org/abs/2306.10114)
- 【文献证实】Swift/BAT 同时测了 CXB 与地球反照谱 — [Ajello et al. 2008, arXiv:0808.3377](https://arxiv.org/abs/0808.3377)；地球大气硬 X 射线发射模拟 — [Sazonov et al. 2007, arXiv:astro-ph/0608253](https://arxiv.org/abs/astro-ph/0608253)
- 【文献证实】HXMT 已有的地球遮挡应用是测高层大气密度（官方清单中 2022–2025 年多篇，如 "Measurement of the vertical atmospheric density profile from the X-ray Earth occultation of the Crab Nebula with Insight-HXMT"）——属地球科学，见[官方论文清单](http://hxmten.ihep.ac.cn/papers.jhtml)

### Inferences
- 【推断】十年定点观测里地球遮挡段总时长应达数十 Ms（每轨 30 min/96 min ≈ 30%），而已发表工作只用了 0.19 Ms；这是存档中最大的一块"被 GTI 剔除"的数据。其基础物理用途：(i) F1 中"线在遮挡时消失"的判据——可把线搜寻扩展到任意有亮源的天区；(ii) F2 中夜侧地球方向的异常发射搜索；(iii) F4 的本底样本。
- 【推断】月球：主视场内永不出现；LE 超宽视场偶尔可能扫到，但月面硬 X 射线/γ 反照在 <250 keV 极弱，无基础物理价值。
- 【推断】太阳：只能被 CsI 从侧/背向看到（耀斑），属太阳物理（兄弟方向已覆盖）；太阳轴子类搜索全部不可行。

### Gaps
- 地球遮挡段数据在公开 Level-1 中是否完整保留（应当保留，因为 GTI 是用户端生成的）未逐文件确认。
- 夜侧/昼侧地球反照在 1–250 keV 的 HXMT 实测谱未见发表（兄弟方向"想法 8"拟做）。

---

## 9. 其他检验：DAMA 式年/恒星时调制（空间 NaI 阵列）、强相互作用暗物质、基本常数变化、超轻暗物质/引力波记忆、dark photon/毫电荷粒子

### Takeaway
对标准 WIMP，用 HE 的 NaI 做 DAMA 式调制分析**没有意义**：阈值 20 keV（DAMA 信号在 1–6 keVee）、本底率高出约 6–7 个量级、还有地磁/太阳周/SAA/轨道进动的强调制。但把问题换成"因截面太大而到不了地下的暗物质"，结论反转：HE 是在轨时间最长、质量最大的无遮蔽闪烁体靶（NaI 6.6 kg + CsI 92 kg × 十年），并且轨道上存在一个地面没有的干净判据——**地球对暗物质风的遮挡**造成的、相位锁定在惯性系（天鹅座方向）的轨道周期调制。

### Cited Findings
- 【文献证实】DAMA/LIBRA-phase2 的模型无关年调制结果 — [Bernabei et al. 2018, arXiv:1805.10486](https://arxiv.org/abs/1805.10486)；同靶材料的检验 ANAIS-112 三年结果 — [Amaré et al. 2021, arXiv:2103.01175](https://arxiv.org/abs/2103.01175)
- 【文献证实】"强相互作用暗物质"的空间/高空限制史：自相互作用暗物质与 IMP/Skylab 等限制 [Wandelt et al. 2000, arXiv:astro-ph/0006344](https://arxiv.org/abs/astro-ph/0006344)；XQC 探空火箭量能器 [McCammon et al. 2002, arXiv:astro-ph/0205012](https://arxiv.org/abs/astro-ph/0205012)、[Erickcek et al. 2007, arXiv:0704.0794](https://arxiv.org/abs/0704.0794)；排除窗口的讨论 [Zaharijas & Farrar 2005, arXiv:astro-ph/0406531](https://arxiv.org/abs/astro-ph/0406531)、[Mack & Manohar 2012, arXiv:1211.1951](https://arxiv.org/abs/1211.1951)、[Mahdawi & Farrar 2017, arXiv:1709.00430](https://arxiv.org/abs/1709.00430)、[Mahdawi & Farrar 2018, arXiv:1804.03073](https://arxiv.org/abs/1804.03073)
- 【文献证实】地下与地表探测器对强相互作用暗物质"失明"的临界截面 — [Emken & Kouvaris 2018, arXiv:1802.04764](https://arxiv.org/abs/1802.04764)；大截面下常用标度关系失效、点状暗物质截面不可能远超核几何截面，旧限制需重新解释 — [Digman et al. 2019, arXiv:1907.10618](https://arxiv.org/abs/1907.10618)；复合暗物质仍有大片开放参数空间 — [Cappiello, Collar & Beacom 2021, arXiv:2008.10646](https://arxiv.org/abs/2008.10646)
- 【文献证实】地球遮挡/散射造成的日调制是"显著阻止本领暗物质"的判据 — [Kouvaris & Shoemaker 2014, arXiv:1405.1729](https://arxiv.org/abs/1405.1729)；[Kavanagh, Catena & Kouvaris 2017, arXiv:1611.05453](https://arxiv.org/abs/1611.05453)
- 【文献证实】HE 17 个非 blind 模块反符合后 30–300 keV 本底 538.6 counts/s；NaI 厚 3.5 mm、CsI 厚 40 mm、单体面积 283.5 cm² — [Zhang J. et al. 2020, arXiv:2009.14489](https://arxiv.org/pdf/2009.14489)；[Zhang S. et al. 2019, arXiv:1910.04434](https://arxiv.org/pdf/1910.04434)

### Inferences
#### 想法 F10：HE 作为"在轨直接探测靶"——用地球遮挡暗物质风的惯性系轨道调制，搜索大截面复合/强相互作用暗物质
1. **一句话**：在全部 NaI+CsI 事例率（含 blind 探测器、含被反符合打标的事例）中，搜索幅度随"卫星天顶方向与暗物质风来向（天鹅座）夹角"变化、相位锁定在惯性系的轨道周期调制。
2. **原理/灵感**：截面足够大的暗物质被地球完全挡住；550 km 处地球遮住约 30% 天空，卫星每 ~95 min 绕一圈，暗物质风方向交替被挡 → 事例率出现 O(1) 相对幅度的调制。地面版本即 Kouvaris & Shoemaker 的"daily modulation"；探空火箭 XQC 是空间直接探测的先例。
3. **适配性**：十年曝光（NaI 6.6 kg、CsI 92 kg【推断，由几何尺寸算得】），仅隔几 g cm⁻² 卫星物质而非 10 m.w.e. 大气；地磁调制锁定在地理坐标、太阳相关效应锁定在日地系，而信号锁定在恒星系，十年间轨道面进动与地球自转使三者相位充分打散。**坦率结论**：阈值高（≥20 keVee），只对重（≳100 GeV–10¹³ GeV）且截面 ≳10⁻²³–10⁻²² cm² 的复合暗物质敏感；对标准 WIMP 完全无用。
4. **数据**：HE 1L 事例（不做 ACD 剔除、保留 CsI）+ 轨道/姿态；公开。
5. **查新**：早期检索未见任何 X 射线/γ 射线天文卫星做过此类分析；官方清单无。相近的只有 XQC（100 s 级火箭飞行）与 Skylab/IMP 的旧结果。DAMA 式分析用于空间 NaI 的文献未检索到。
6. **可行性与量级**（【推断】）：
   - DAMA 对照：HE NaI 本底 538.6 c/s ÷ 6.2 kg ≈ 87 c s⁻¹ kg⁻¹ ≈ 7.5×10⁶ counts kg⁻¹ day⁻¹（30–300 keV），而地下 NaI 实验在信号区约为 1 count kg⁻¹ keV⁻¹ day⁻¹ 量级【凭记忆】——差 6 个量级以上，且能段不重合。结论：标准 DAMA 检验不可行。
   - 调制灵敏度：T=10⁸ s 时 N≈5.4×10¹⁰，分数调制统计极限 √(2/N)≈6×10⁻⁶，即 ~3×10⁻³ c/s（17 模块合计）≈ 2×10⁻⁴ c/s/模块。
   - 对"每个穿过晶体的粒子都被记录"的大截面暗物质，单模块事例率 ≈ Φ·A ≈ (7×10⁶/m_GeV cm⁻² s⁻¹)×283.5 cm² ≈ 2×10⁹/m_GeV s⁻¹；与调制极限相比，可达 m ~10¹³ GeV（芝加哥地面探测器为 4×10¹² GeV）；仅用原始总率（~30 c/s/模块）也能排除 m ≲ 10⁷–10⁸ GeV。
   - 截面下限：碘/铯反冲 ~250 keVnr×淬灭 ~0.05–0.1 ≈ 15–25 keVee/次，4 cm CsI 中需 ≥2–3 次散射 → σ ≳ 4×10⁻²³ cm²；NaI 3.5 mm 需 σ ≳ 10⁻²² cm²。截面上限由卫星自身遮蔽（几 g cm⁻²）决定，比 10 m.w.e. 大气宽松约 300 倍，即 σ/m ~3×10⁻²⁵ cm²/GeV。
   - **主要障碍**：天鹅座方向恰有亮硬 X 射线源（Cyg X-1 等），其地球掩食会在 CsI 中产生同相位的台阶——需用掩食台阶的陡峭时间轮廓与 blind NaI（<200 keV 不透光子）来区分；地磁调制幅度达几十 %，要求模板回归精度 10⁻⁵ 量级才能达到统计极限，实际灵敏度很可能差 2–3 个量级；淬灭因子与多次散射响应需模拟。
7. **打分**：新颖性 5｜可行性 3｜影响 3（目标参数区是否仍开放需对照 Skylab/云母/XQC/IMP 仔细核实）。

### Gaps
- F10 目标区（σ~10⁻²²–10⁻¹³ cm²、m~10⁷–10¹³ GeV、大气遮蔽线以上）是否已被 Skylab、古云母、IMP 7/8、RRS、XQC 或更新的"蚀刻塑料"类分析排除，未能完整核对；Cappiello et al. 2021 的图只列了 Skylab、云母与一条宇宙学限制。
- DAMA 的本底率与调制幅度数值凭记忆，未从原文提取。
- 拟议中的专门空间暗物质小卫星（如使用 skipper-CCD 的 cubesat，凭记忆名为 DarkNESS）未能检索核实。
- Collar & Avignone 1992 关于地球遮挡日调制的早期文献凭记忆，未取回。
- 基本常数变化、超轻暗物质/引力波记忆、dark photon/毫电荷粒子：未找到任何使 HXMT 具备灵敏度的机制（见第 10 节否决理由），也未检索到相关 HXMT 文献。

---

## 10. 综合排序与被考虑后否决的想法

### Takeaway
最值得立刻动手的是 **F1（差分视场暗物质线搜寻）** 和 **F2（近地 X 射线背景季节调制检验）**：都只用公开的空白天区数据、方法有成熟先例、HXMT 从未做过、零结果也可发表。最"大胆"的是 **F10（地球遮挡暗物质风调制）**、**F6（慢重粒子飞行时间搜索）** 和 **F3（CsI 意外偏振计）**，三者都依赖常规流程丢弃的数据（反符合事例、spike、CsI 事例、模块间符合），但各有一个尚无公开答案的仪器响应问题。HXMT 的不可替代之处有三：对弥漫信号的超大 grasp、同型探测器的大/小/盲视场对照、以及逐事例保留的 μs 时间 + 18 路反符合拓扑信息。

### Cited Findings
- 【文献证实】HXMT 相对其他仪器的硬指标对比依据：NuSTAR 0-bounce grasp 55 cm² deg²/FPM — [Roach et al. 2023, arXiv:2207.04572](https://arxiv.org/abs/2207.04572)；GBM 线搜寻系统误差主导 — [Ng et al. 2015, arXiv:1504.04027](https://arxiv.org/abs/1504.04027)；HXMT 事例格式与反符合阵列 — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)
- 【文献证实】中子寿命的空间测量先例（用于说明为何 HXMT 做不了）：MESSENGER — [Wilson et al. 2020, arXiv:2006.10008](https://arxiv.org/abs/2006.10008)；Lunar Prospector — [Wilson et al. 2021, arXiv:2011.07061](https://arxiv.org/abs/2011.07061)
- 【文献证实】官方论文清单（至 2026-05）中基础物理类只有借用 CXB 数据的 3 篇理论工作与 1 篇透镜候选 — [Insight-HXMT publications](http://hxmten.ihep.ac.cn/papers.jhtml)

### Inferences
#### 想法总表（打分为暂定）

| 编号 | 想法 | 用到的"废料" | 新颖 | 可行 | 影响 | HXMT 是否比别的卫星更合适 |
|---|---|---|---|---|---|---|
| F1 | 差分视场暗物质衰变线搜寻（LE/ME/HE） | 空白天区、地球遮挡段、blind 探测器 | 4 | 4/3 | 4 | 部分是：20–30 keV 接缝与 3.5 keV 独立检验；10–20 keV 不如 NuSTAR，>30 keV 不如 SPI |
| F2 | 近地 X 射线背景季节调制（XMM/AQN）检验 | 例行空白天区本底观测 | 4 | 4 | 3 | 是（2–6 keV + LEO + blind 对照） |
| F10 | 地球遮挡暗物质风的惯性系轨道调制 | 全部本底事例、反符合事例、CsI | 5 | 3 | 3 | 是（无同类在轨大质量闪烁体长期数据被这样用过） |
| F6 | 慢速共线多模块巨沉积事例（nuclearite/复合 DM） | spike、反符合标签、overflow | 4–5 | 2 | 3 | 是，但不如专用载荷；Skylab 在大截面区曝光更大 |
| F3 | CsI 模块间 Compton 双击偏振 → 双折射 LIV | GRB 模式 CsI 事例的模块间符合 | 5 | 1–2 | 4 | 仅因它是 GRB 221009A 少数未饱和的仪器 |
| F8 | ²⁴¹Am 标定事例 + EC 活化线的衰变率恒定性 | calibration event、SAA 后数据 | 4 | 3 | 1–2 | 独特但物理回报低 |
| F9 | 在轨活化产额/半衰期核数据 | SAA 后被剔除的高本底段 | 3 | 4 | 2 | 是（靶材料即 NaI），但辐照谱不单纯 |
| F4 | CXB 延伸到 250 keV / MeV → PBH、MeV DM | 地球遮挡段、CsI | 2–3 | 2 | 3 | 否（不如 INTEGRAL/SPI） |
| F5 | CsI"反向暴"无触发搜索 + 近场曲率 | 未触发的 CsI 连续数据 | 2 | 3 | 2 | 否（需组网；已被占位） |
| F7 | HVT East–West/截止刚度宇宙线谱 | 反符合计数率 | 2–3 | 4–5 | 2 | 否（不如 AMS-02）；与空间天气方向合并 |
| — | SME 各向异性 LIV（多 GRB 方向） | GRB 样本 | 2 | 4 | 1–2 | 否 |

#### 被考虑后否决的想法及理由
- **标准 GECOSAX（夜侧朝太阳方向看地磁转换的太阳轴子）**：【推断】太阳规避角 >70°，信号方向永不进入任何准直视场；CsI 无准直但阈值 ≥40 keV。否决（几何上不可能）。
- **14.4 keV ⁵⁷Fe 太阳轴子线**：【推断】同样需要看太阳；星上也没有可用的 ⁵⁷Fe 共振吸收靶与低本底条件。否决。
- **ALP–光子谱不规则性（亮源穿过星系团/银河磁场）**：【推断】需要 eV–百 eV 级分辨和极高统计的光滑谱，HXMT 分辨率与响应系统误差（LE ~1%，HE 更差）不够。否决。
- **磁星 / Betelgeuse / 超星团的 ALP 硬 X 射线信号**：【推断】HXMT 点源灵敏度 0.5 mCrab（10⁵ s，仅统计）远逊于 NuSTAR 聚焦观测。否决。
- **近邻超新星 ALP 暴（SN 2023ixf 等）**：【推断】ALP 转换/衰变光子能量在几十 MeV，CsI 上限 3 MeV；已有 Fermi-LAT 结果。否决。
- **Crab 脉冲相位 LIV / 周期源叠加 LIV**：【推断】见第 3 节，E_QG 限制比 LHAASO 低约 7 个量级，标度决定无法靠统计弥补。否决。
- **femto/pico-lensing 视差**：【推断】HXMT–GBM–GECAM 基线只有地球尺度，且要求百分之几的仪器间流量互定标，CsI 做不到。否决。
- **PBH 穿越地球/大气的瞬变信号**：【推测】无可计算的 HXMT 可观测量。否决。
- **相对论性 strangelet、磁单极、反核湮灭、分数电荷粒子**：【推断】HE 无 dE/dx、无电荷测量、无磁谱仪，8 bit 能量在 MeV 以上即溢出。否决（DAMPE、AMS-02 才是合适平台）。
- **标准 WIMP 的 DAMA 式年/恒星时调制**：【推断】见第 9 节，阈值与本底差 6 个量级以上。否决（仅保留大截面版本 F10）。
- **由线能量测基本常数变化**：【推断】线心精度至多 10⁻³，实验室原子钟限制好十几个量级。否决。
- **X 射线脉冲星计时探测超轻暗物质 / 引力波记忆**：【推断】HXMT 的脉冲星计时精度不及射电 PTA 与 NICER，且观测稀疏。否决。
- **探测器噪声中的 dark photon / 毫电荷粒子信号**：【推断】LE（SCD）阈值 ~1 keV、非单电子灵敏，无法与 skipper-CCD 类实验相比。否决。
- **月球作宇宙线靶（月面 γ 反照）**：【推断】月球永不进主视场，<250 keV 反照极弱。否决。
- **空间中子寿命测量**：【推断】该方法需要专用中子谱仪和随高度变化的行星反照中子测量（MESSENGER、Lunar Prospector），HXMT 没有中子探测器、轨道高度固定。否决。
- **银河 511 keV 线的地球掩食测量**：【推断】统计上或许可行，但 INTEGRAL/SPI 已做得远更好，且属天体物理而非基础物理。降级。

### Gaps
- 以下文献或数字为凭记忆引用、本次未取回原文，使用前须核对：De Rújula & Glashow 1984；Shirk & Price 1978；Price & Salamon 1986；PAMELA strangelet 搜索；Pommé et al. 2016；Collar & Avignone 1992；AMS-02 逐日质子通量；CAST 最新 g_aγ 上限；MAGIC Crab 的 E_QG 数值；DAMA 本底率与调制幅度；DarkNESS 立方星。
- 因搜索配额用尽而未能完成的查新：(i) 2025–2026 年 GECAM/EP/XRISM 的暗物质线限制；(ii) 是否已有人用 GBM/NuSTAR 检验 AQN 季节调制；(iii) 外部作者用 HXMT 数据做 LIV 的论文；(iv) RXTE/HEXTE、BeppoSAX、Fermi-GBM、GECAM 上的活化截面与衰变率类先例；(v) LHAASO 等最新 PBH 爆发率上限。
- 决定 F3/F6/F10 成败的三项仪器信息无公开文献：HE 对超量程沉积的记录方式、spike 事例的原始保留情况、高计数率下模块间 μs 符合的可靠性；建议在动手前向 HE 仪器组确认或直接检查 1L 事例文件。
- 所有灵敏度估计都是量级估算（FWHM 乘积近似视场、本底取任务早期数值、D-factor 取 2×10²² GeV cm⁻²），未做响应折叠与本底十年增长修正（HE 本底第五年明显高于第一年）。
