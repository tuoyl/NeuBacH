# 先例、新颖性与影响力：慧眼（Insight-HXMT）"地球遮挡暗物质风的轨道调制"搜索（截至 2026-11）

> 说明：检索日期 2026-11-18。标注约定：【文献证实】= confirmed in literature（本人实际取回了摘要或全文）；【推断】= inference（由已取回文献推出，或本人做的几何/算术估算）；【猜测】= speculation；凡"凭记忆，未核实"者单独标出。所有引用数均为 INSPIRE-HEP API 的 `citation_count` 字段，检索时刻 2026-11-18 约 19:45–20:10 UTC。工具：INSPIRE API（文献与作者档案）、arXiv API、arXiv PDF 全文（pdftotext）、Crossref API、WebFetch、WebSearch（共用 8 次，标准模式）。

---

## 1. 地球遮挡 / 日调制的形式体系与实验实践；在轨探测器应采用什么形式体系

### Takeaway
"地球挡住暗物质风 → 探测率随恒星时调制"的想法自 1992 年就有（Collar & Avignone），2014–2019 年被发展成成熟的解析与蒙卡工具（EarthShadow、DaMaSCUS、verne），2023–2026 年已被 DAMIC-M、SENSEI 做成正式的地下搜索并发表在 PRL。**对在轨探测器，"以轨道周期为日、地球为几何遮挡体"的版本已由 Emken、Essig、Kouvaris & Sholapurkar 在 2019 年明确写出（含遮挡函数、ISS 轨道 ~90 min 周期、调制度 ~75%），并被 DarkNESS 立方星采纳为观测策略**；HXMT 想法在形式上是它的直接套用，只是换成重质量/复合暗物质的核反冲和存档闪烁体数据。

### Cited Findings

**(a) 理论谱系（地面/地下）**
- 【文献证实】Collar & Avignone 1992（Phys. Lett. B 275, 181；引用 103）首次系统预言：把暗物质在地球核素上的弹性散射、地质结构、地球取向/自转/穿越银晕的轨迹和各向同性麦氏速度分布都算进去，得到探测率和能谱的可观测日调制 — [INSPIRE 335921](https://inspirehep.net/literature/335921)
- 【文献证实】Collar & Avignone 1993（Phys. Rev. D 47, 5238；引用 88）：预言通量与速度分布的日调制，"The effects are seen to be maximal at southern latitudes" — [INSPIRE 35300](https://inspirehep.net/literature/35300)
- 【文献证实】第一次实验搜索：COSME-II 0.234 kg 锗探测器本底中搜索日调制并给出限制（Nucl. Phys. B Proc. Suppl. 28, 297, 1992）— [INSPIRE 350684](https://inspirehep.net/literature/350684)；阿根廷 Sierra Grande 实验专为"以地球为吸收体"的日调制设计 — [Di Gregorio et al. 1993, arXiv:astro-ph/9311049](https://arxiv.org/abs/astro-ph/9311049)
- 【文献证实】Hasenbalg et al. 1997（Phys. Rev. D 55, 7350；引用 58）重新估算"eclipsing of dark matter particles in the Earth"的调制，结论是"a significantly larger statistics than previously estimated is needed to achieve the same level of sensitivity in the modulated signal" — [arXiv:astro-ph/9702165](https://arxiv.org/abs/astro-ph/9702165)
- 【文献证实】Foot 2012（JCAP 04, 014；引用 32）、Foot & Vagnozzi 2015（Phys. Lett. B 748, 61；引用 132）：镜像/耗散隐藏部门暗物质被地球俘获后屏蔽探测器，南半球探测器的日调制"large (≳10%) for a wide range of parameters" — [arXiv:1110.2908](https://arxiv.org/abs/1110.2908)；[arXiv:1412.0762](https://arxiv.org/abs/1412.0762)
- 【文献证实】Kouvaris & Shoemaker 2014（Phys. Rev. D 90, 095011；引用 81）："the flux of DM may be blocked from below but not above a detector thereby producing a novel daily modulation"，建议用南半球地表/浅埋低阈值探测器 — [arXiv:1405.1729](https://arxiv.org/abs/1405.1729)
- 【文献证实】Kouvaris & Nielsen 2015（Phys. Rev. D 92, 075016；引用 22）：地球引力聚焦也产生日调制且相位不同、可区分；并建议"null result experiments should check for a daily modulation of their rejected background signal with specific phases. A potential discovery could mean that dark matter collisions have been vetoed out"（与 HXMT 使用被反符合打标事例的思路一致）— [arXiv:1505.02615](https://arxiv.org/abs/1505.02615)
- 【文献证实】Kavanagh, Catena & Kouvaris 2017（JCAP 01, 012；引用 84）：EarthShadow 解析计算（至多一次散射），同时计入被散射走与被偏转进来的粒子；地球散射"reduces the direct detection rate at certain detector locations while increasing the rate in others"，日调制"highly sensitive to the detector latitude and to the form of the DM-nucleon interaction"；文中给出的例子是平均散射概率 10% 时速度分布的百分之几量级畸变 — [arXiv:1611.05453](https://arxiv.org/abs/1611.05453)
- 【文献证实】Emken & Kouvaris 2017（JCAP 10, 031；引用 88）：DaMaSCUS 蒙卡给出任意深度与位置的密度、速度分布以及"the phase and amplitude of the diurnal modulation caused by this shadowing effect of the Earth"，并强调把不同探测器的预期调制互相关联对判断信号真伪很重要 — [arXiv:1706.02249](https://arxiv.org/abs/1706.02249)
- 【文献证实】Emken & Kouvaris 2018（Phys. Rev. D 97, 115047；引用 154）：临界截面以上地下与地表探测器全盲，"Dedicated surface or high-altitude experiments may be the only way to directly probe this part of the parameter space" — [arXiv:1802.04764](https://arxiv.org/abs/1802.04764)
- 【文献证实】Kavanagh 2018（Phys. Rev. D 97, 123013；引用 126）：对超重暗物质（m ≳ 10⁵ GeV）"the standard nuclear-stopping formalism is applicable"（连续能损、直线轨迹，verne 代码），并用 CRESST 2017 表面运行与 CDMS-I 旧数据重新分析 — [arXiv:1712.04901](https://arxiv.org/abs/1712.04901)
- 【文献证实】Lantero-Barreda et al. 2025：Verne2 半解析（直线传播 + 沿原路反射），与全蒙卡相差 10–30%，计算量减少 ~10⁴ 倍，专为日调制搜索的信号建模而写 — [arXiv:2511.10589](https://arxiv.org/abs/2511.10589)
- 【文献证实】CDEX 2022（Phys. Rev. D 105, 052005；引用 55）：为锦屏开发了 CJPL_ESS 地球屏蔽模拟包 — [arXiv:2111.11243](https://arxiv.org/abs/2111.11243)
- 【文献证实】Bertou et al. 2025（JHEP 11, 042；引用 9）：低阈值探测器的地球散射调制（周期一个恒星日）；"a search for a modulation signal can probe new regions of parameter space near the energy thresholds of current experiments, where the data are typically dominated by backgrounds" — [arXiv:2507.00344](https://arxiv.org/abs/2507.00344)
- 【文献证实】Kozynets, Leane & Smirnov 2026：对调制"形状"（含各实验室对不同穿地路径的位置依赖曝光）做统计检验，并用 DarkSide-50 数据示范 — [arXiv:2607.20620](https://arxiv.org/abs/2607.20620)
- 【文献证实】Blanco, Foster, Kahn & Lillard 2026（"Statistics of Daily Modulation in Dark Matter Direct Detection Experiments"）：本底主导时发现显著性 ∝ f_RMS·√T，"the significance continues to improve with exposure rather than saturating due to systematic uncertainties in the background rate"；允许本底平坦或以已知/未知相位做 24 h 调制；多探测器取向优化可把所需曝光减少约 5 倍 — [arXiv:2602.15947](https://arxiv.org/abs/2602.15947)
- 【文献证实】2026-11-16 的 113 页综述（Kahlhoefer, von Krosigk, Marrodán Undagoitia, Novati & Sarsa）把地球屏蔽日调制列为与年调制、方向性并列的时间依赖信号，并写道方向性不对称与地球屏蔽日调制这两种效应"are much more difficult to mimic by any known background, while some background components and systematic effects are expected to show an annual modulation" — [arXiv:2611.21052](https://arxiv.org/abs/2611.21052)
- 【文献证实】Snowmass 2021 低阈值白皮书：近期实验将探测到暗物质可在到达探测器前于地球中散射的参数区，"this leads to a diurnal modulation in the signal rate …, and could help distinguish a DM signal from backgrounds" — [Essig et al. 2022, arXiv:2203.08297](https://arxiv.org/abs/2203.08297)

**(b) 在轨/高空版本（与 HXMT 想法直接对应）**
- 【文献证实】Emken, Essig, Kouvaris & Sholapurkar 2019（JCAP 09, 070；引用 214；[arXiv:1905.06348](https://arxiv.org/abs/1905.06348)）§4.2.4 "Balloon and satellite experiments"：
  - "A characteristic signature of DM with large interaction strengths would be a strong modulation due to this shadowing effect … For a balloon-borne experiment we would expect a frequency of one per sidereal day, while a satellite-borne experiment could look for orbital modulations of higher frequencies."
  - 估算方法："we assume that the DM particles passing the Earth on their way to the detector get shielded completely, while the shielding of the experimental apparatus itself can be neglected"，即把本地速度分布乘以遮挡函数 S(x,v)（若粒子反向延长线与地球相交则 S=0，否则 S=1；原文式 4.1–4.2）。
  - 算例：ISS，高度 ~400 km，"The orbital, and therefore also the modulation period (assuming the detector's orientation does not change with respect to Cygnus) is about 90 minutes"，倾角 ~50°。
  - 调制度定义 f_mod ≡ (R_max − R_min)/(R_max + R_min)："While the geostationary balloon-borne detector in the northern hemisphere would show a diurnal modulation with f_mod ∼30%, a detector aboard a satellite or space station could expect high-frequency orbital modulations with f_mod ∼75%."
  - 发现判据（式 4.4）：f_mod·S_tot/√(S_tot + B) = 5；卫星算例取 0.1 gram-month、10⁹ 个本底事例（"chosen for purposes of illustration only"），并承认"the assumption of a flat background is optimistic"。
  - "It would be a powerful discriminant against possible systematics as well as to modulations in the background rate induced by the rotating Earth or by the detector orbiting the Earth."
  — [arXiv:1905.06348](https://arxiv.org/abs/1905.06348)
- 【文献证实】DarkNESS（6U 立方星，skipper-CCD，活性质量约 2 g）把上述方案写成观测计划："As the NanoSatellite orbits Earth, planetary shadowing is expected to result in a modulated signature in the instrument's low-energy event rate"；"observations towards Cygnus utilize Earth obstruction to search for modulation signatures of strongly interacting DM"；约 450 次 × 10 min 的 Cygnus 观测凑成 0.1 gram-month，假设 f_χ = 0.01%、10⁹ 个"constant in time"的本底事例；图 1 中"DarkNESS-modulation"线远低于"DarkNESS-rate"线 — [DarkNESS Collab. 2025, Adv. Space Res. 76, 18615, arXiv:2505.16981](https://arxiv.org/abs/2505.16981)
- 【文献证实】同一几何思想 2026 年也出现在波动型暗物质上：对与标准模型有二次耦合的超轻标量暗物质，地球周围的场分布出现偶极，"In Low Earth Orbits this dipole temporally modulates potential dark matter signals. This provides a powerful cross-check of the orbit-averaged effect"（ISS 上的原子钟/核钟）— [Brzeminski & Pierce 2026, JHEP 08, 161, arXiv:2601.16259](https://arxiv.org/abs/2601.16259)
- 【文献证实】地球何时"不透明"：Cappiello, Collar & Beacom 2021 指出在 σ_χ/m_χ ≃ 10⁻³³ cm²/GeV 这条线以下"DM particles may pass through Earth, allowing upgoing events"，即该线以上地球把重复合暗物质完全挡住 — [arXiv:2008.10646](https://arxiv.org/abs/2008.10646)；重暗物质在覆盖层中的能损公式 E_f ≃ E_i·exp[−(2/m_X)∫dz Σ n_N (μ²_XN/m_N) σ_eff] 见 Bramante 2026 综述式 (20) — [arXiv:2602.23708](https://arxiv.org/abs/2602.23708)

**(c) 实验上的日/恒星时调制搜索及其结果表述方式**
- 【文献证实】DAMA/NaI 1999（Nuovo Cim. A 112, 1541；引用 68）最早用 NaI 搜索日效应 — [INSPIRE 515251](https://inspirehep.net/literature/515251)
- 【文献证实】DAMA/LIBRA-phase1 2014（Eur. Phys. J. C 74, 2827；引用 82；1.04 ton·yr）：按太阳时和恒星时（GMST）两种分箱（1 小时），用 χ² 检验与 run test；固定周期 24 h、相位 14 h 拟合得 A_d = −(1.0 ± 1.3)×10⁻³ cpd/kg/keV，上限 < 1.2×10⁻³ cpd/kg/keV（90% C.L.，2–6 keV），而由其年调制推得的预期只有 ≃1.5×10⁻⁴ cpd/kg/keV — [arXiv:1403.4733](https://arxiv.org/abs/1403.4733)
- 【文献证实】DAMA/LIBRA-phase1 2015 "Investigating Earth shadowing effect"（Eur. Phys. J. C 75, 239；引用 61）：对"高截面、低局域密度"候选者做地球分层蒙卡，得到随恒星时变化的速度分布，把无恒星日变化转化为（ξ, σ_n, m_DM）空间的约束，例如所考虑情形下 m_DM = 60 GeV 时排除 σ_n > 0.05 pb 且 ξ > 10⁻³ — [arXiv:1505.05336](https://arxiv.org/abs/1505.05336)
- 【文献证实】LUX 2018（Phys. Rev. D 98, 062005）：20 个月数据，事例率 2.3 ± 0.2 cpd/keVee/tonne；"no observation of diurnal modulations above 0.2 cpd/keVee/tonne amplitude between 2 and 6 keVee" — [arXiv:1807.07113](https://arxiv.org/abs/1807.07113)
- 【文献证实】PandaX-II 2022（Phys. Rev. Lett. 128, 171801；引用 97）：宇宙线加速暗物质搜索中"The diurnal modulations in rate and energy spectrum are utilized to further enhance the signal sensitivity"（理论依据是 Ge, Liu, Yuan & Zhou 2021, PRL 126, 091804，引用 74）— [arXiv:2112.08957](https://arxiv.org/abs/2112.08957)；[arXiv:2005.09480](https://arxiv.org/abs/2005.09480)
- 【文献证实】DAMIC-M 2023（Phys. Rev. Lett. 132, 101006；引用 44）：39.97 g·day、8779 幅图像；先做模型无关分析（幅度 A、周期 T、相位 φ 的正弦，对 T = 1–48 h 逐周期做似然比检验，恒星日周期处与零假设一致），再做模型相关分析（以"isodetection angle"Θ 为自变量，暗电流本底取 λ_A·e^(−t/τ) + λ_eq）；"Taking advantage of the time-dependent signal we improve by ∼2 orders of magnitude on our previous limit obtained from the total rate of 1e⁻ events, using the same data set" — [arXiv:2307.07251](https://arxiv.org/abs/2307.07251)
- 【文献证实】DAMIC-M 2025（Phys. Rev. D 114, 032010；引用 11）：1.257 kg·day，"A model-independent analysis reveals no significant modulation in the 1e⁻ event rate over periods from 1 to 48 h" — [arXiv:2511.13962](https://arxiv.org/abs/2511.13962)
- 【文献证实】SENSEI 2025（Phys. Rev. Lett. 136, 211001；引用 12）：2.19 g、141 幅 6 小时图像、75 天（数字取自 Kahlhoefer 等综述）；最低噪声象限的 1e⁻ 率约 135 e⁻/g/day；模型无关调制幅度上限 a₁ ≤ 6.8 e⁻ g⁻¹ day⁻¹（90% C.L.）；对时间戳做 bootstrap 重采样得到"Poisson-consistent band"，并用"scaled-Poisson"本底模型（ĉ = 1.90）处理过离散；若偏离泊松则"it is necessary to investigate the origin of the non-Poissonian background … before interpreting the result" — [arXiv:2510.20889](https://arxiv.org/abs/2510.20889)；[arXiv:2611.21052](https://arxiv.org/abs/2611.21052)
- 【文献证实】DMSQUARE：南半球地表 skipper-CCD 日调制搜索（会议文集，引用 9）— [Ávalos et al. 2021, J. Phys. Conf. Ser. 2156, 012074](https://inspirehep.net/literature/2037307)
- 【文献证实】COSINE-100 发表过缪子通量的年/日变化测量（JCAP 02 (2021) 013），但在 INSPIRE 按合作组 + 摘要关键词（diurnal / daily modulation / sidereal / day-night）检索，未找到 COSINE-100、ANAIS-112、XENON、LZ 的专门暗物质日调制搜索论文 — [arXiv:2005.13672](https://arxiv.org/abs/2005.13672)

### Inferences
- 【推断】对 550 km 轨道，可直接沿用 Emken et al. 2019 的遮挡函数：在 HXMT 有独特灵敏度的参数区（σ/m 高于大气遮蔽线），地球远在"完全不透明"一侧（Cappiello 等给的透明界线是 σ/m ≃ 10⁻³³ cm²/GeV），所以遮挡是纯几何的，不需要 DaMaSCUS 式的穿地蒙卡；需要蒙卡的是卫星自身几 g/cm² 物质的自遮挡和探测器对多次散射的响应。
- 【推断，本人几何计算】550 km 处地球角半径 67.0°，挡住 30.5% 的天空。取标准晕模型（v₀ = 238 km/s、v_esc = 544 km/s、太阳速度 250.6 km/s）、地球完全不透明、探测器各向同性：天底指向太阳运动方向（apex）时被挡通量占 87%（按 v³ 加权为 93%），天底与 apex 成 90° 时 21%，天底指向反 apex 时 1.3%。因此当 apex 位于轨道面内时 f_mod ≈ 0.77，与 Emken 等对 ISS 给出的 ~75% 一致。
- 【推断，本人几何计算】apex 的赤道坐标约为 RA ≈ 313.5°、Dec ≈ +47.5°（v₀ 取 220–238 km/s 时变化 <0.5°）；HXMT 轨道法线赤纬 = 90° − 43° = +47.0°。于是轨道法线每个节点进动周期（J₂ 进动 −5.46°/day，惯性系周期 66.0 天）扫过 apex 一次，最近时只差约 0.5°：此时 apex 方向整圈都不被地球掩（f_mod ≈ 1%）；半个进动周期后法线与 apex 相距 85.5°，f_mod ≈ 0.77。**即信号是"95.6 min 载波 × 66.0 天包络"，包络零点锁定在轨道法线 RA ≈ 313.6° 的惯性方位**。太阳相关效应的包络周期是 55.9 天（β 角周期），两者每年恰好相差一圈；可仿照宇宙线的 anti-sidereal 构造一个 80.5 天的"无物理信号"对照包络。9.4 年约含 5.2×10⁴ 圈、52 个进动周期。apex 方向在一年内还会因地球公转速度（约 30 km/s 对 250 km/s）摆动约 ±7°，所以"零点"并不严格为零。轨道参数取任务说明给的 550 km、43°；这些数值需由数值预报与系统误差两位研究者复核（未计入卫星姿态、准直器取向、自遮挡）。
- 【推断】在轨相对地下的其他差别：(i) 轨道速度 7.6 km/s 约为太阳速度的 3%，给出与遮挡效应相位差 90° 的 Compton–Getting 式小调制；(ii) 地球引力聚焦可忽略（(v_esc/v)² ~ 10⁻³）；(iii) 调制频率是每天 15 圈，远离活化衰变、增益漂移等慢变化，但地磁截止刚度、SAA、大气反照恰好也以轨道频率及其谐波变化，这与地下实验"实验室本底无恒星时成分"的前提不同。
- 【推断】结果表述可照搬 DAMIC-M / SENSEI 的两步式：先给"以恒星系遮挡模板为形状的分数调制幅度上限"（模型无关），再给 (m, σ) 排除图（模型相关）；并像 DAMA 2015 和 JWST 那样给出随丰度分数 f_χ 变化的版本。

### Gaps
- Collar & Avignone 1992/1993 的具体调制幅度数值没有取回（两文不在 arXiv，只拿到 INSPIRE 摘要）。
- 没有找到针对"重质量、多次散射暗物质 + 在轨探测器 + 航天器自遮挡"的已发表形式体系或代码；Emken 2019 的算例是亚 GeV 暗物质-电子散射，且忽略仪器自遮挡。
- COSINE-100 / ANAIS-112 / XENON / LZ 是否做过未单独成文的日调制检验未能确认。

---

## 2. 空间（卫星、火箭、气球、空间站）暗物质直接探测：结果与提案清单

### Takeaway
已发表的"空间直接探测"限制只有很少几条：1987 年气球硅探测器（RRS）、Pioneer 11 与 IMP 7/8 的宇宙线硅探测器、Skylab 的 Lexan 径迹片、IMAX 气球塑料闪烁体、XQC 探空火箭量能器（约 100 s）、以及 2025 年的 JWST 暗图像；它们全部是**总计数率/无事例**型限制，**没有一项做过轨道相位或恒星时调制分析**。计划中的 DarkNESS 会做，但截至 2026-11 未见发射或在轨结果。

### Cited Findings

**(a) 已有结果**
- 【文献证实】RRS 气球硅探测器 — Rich, Rocchia & Spiro 1987（Phys. Lett. B 194, 173；INSPIRE 引用 92）："A silicon semiconductor detector near the top of the atmosphere is used to search for strongly interacting particles in our galactic halo. The data exclude, as the dominant component of the halo, such particles with masses between ∼2 GeV/c² and 10⁵ GeV/c²" — [INSPIRE 245683](https://inspirehep.net/literature/245683)；[doi:10.1016/0370-2693(87)90788-X](https://doi.org/10.1016/0370-2693(87)90788-X)
- 【文献证实】综合性起点 — Starkman, Gould, Esmailzadeh & Dimopoulos 1990（Phys. Rev. D 41, 3594；引用 294）汇总了气球硅探测器（Rich et al. 1987）、Pioneer 11 硅探测器（Simpson et al. 1980）、Skylab 塑料径迹片（Shirk & Price 1978）等对强相互作用大质量粒子（SIMP）的限制，发现若干开放窗口 — [INSPIRE 294530](https://inspirehep.net/literature/294530)；[McGuire & Steinhardt 2001, arXiv:astro-ph/0105567](https://arxiv.org/abs/astro-ph/0105567)
- 【文献证实】Pioneer 11：硅探测器前有 1.7 mg/cm² 的钛屏蔽，Starkman 等最初漏算，导致排除区被错误地延伸到任意大截面；McGuire 1994 更正 — [McGuire & Steinhardt 2001, arXiv:astro-ph/0105567](https://arxiv.org/abs/astro-ph/0105567)
- 【文献证实】IMP 7/8（宇宙线硅探测器卫星）：Wandelt et al. 2000 首次给出完整分析的排除区，IMP7 阈值 160 keV — [arXiv:astro-ph/0006344](https://arxiv.org/abs/astro-ph/0006344)；另有 Snowden-Ifft, Barwick & Price 1990 "A search for charged massive particles in IMP 8 data"（Astrophys. J. 364, L25）— [doi:10.1086/185866](https://doi.org/10.1086/185866)，以及同组的 CHAMP 搜索（Phys. Rev. Lett. 64, 2859；引用 15）— [INSPIRE 304176](https://inspirehep.net/literature/304176)
- 【文献证实】IMAX 气球："the IMAX SIMP search experiment using plastic scintillation detectors aboard an antimatter-search balloon-payload (McGuire et al., 1994)"，是塑料闪烁体搜索慢速大质量粒子的在高空先例 — [McGuire & Steinhardt 2001, arXiv:astro-ph/0105567](https://arxiv.org/abs/astro-ph/0105567)
- 【文献证实】Skylab：1.17 m² 的 36 个 Lexan 径迹片模块，留轨 2.19×10⁷ s，覆盖层为 0.37 cm 铝壁，天顶角 ≤60° 内有 150 个径迹；原始目的为 Z ≥ 60 宇宙线（Shirk & Price 1978, Astrophys. J. 220, 719）；Bhoonah et al. 2021（Phys. Rev. D 103, 103001；引用 37）重做后"reduce the Skylab bound's mass range by a few orders of magnitude"，并称"Both Skylab and Ohya continue to place leading limits on heavy SIMP dark matter's interactions more than three decades after they collected data" — [arXiv:2012.13406](https://arxiv.org/abs/2012.13406)；[doi:10.1086/155955](https://doi.org/10.1086/155955)
- 【文献证实】XQC 探空火箭（"inadvertent dark matter detector"）：1999 年飞行，高度 165–225 km，34 个工作量能器（HgTe 0.96 μm 吸收体 + 14 μm Si），150 s 数据折合 100.7 s 全阵列有效曝光；可探测 >20 eV（36 eV 以上全效率，一半时间阈值 120 eV），可用能段 0.03–2.5 keV 加 >4 keV 饱和率 0.6 s⁻¹；排除"a wide range of nucleon-scattering cross sections centered around one barn for dark matter particles with masses between 0.01 and 10⁵ GeV"，并给出只有一部分暗物质强相互作用时的限制 — [Erickcek, Steinhardt, McCammon & McGuire 2007, Phys. Rev. D 76, 042007, arXiv:0704.0794](https://arxiv.org/abs/0704.0794)（引用 129）；仪器论文 [McCammon et al. 2002, Astrophys. J. 576, 188, arXiv:astro-ph/0205012](https://arxiv.org/abs/astro-ph/0205012)（引用 263）
- 【文献证实】XQC 数据的后续再利用：Zaharijas & Farrar 2005（引用 109）— [arXiv:astro-ph/0406531](https://arxiv.org/abs/astro-ph/0406531)；Mahdawi & Farrar 2017（JCAP 12, 004；引用 57）与 2018（JCAP 10, 007；引用 114；速度依赖截面、毫电荷暗物质与 EDGES）— [arXiv:1709.00430](https://arxiv.org/abs/1709.00430)、[arXiv:1804.03073](https://arxiv.org/abs/1804.03073)；Li, Liu & Xue（南京大学）2023（JCAP 05, 060；引用 13；自旋/速度依赖与不同丰度分数）— [arXiv:2209.04387](https://arxiv.org/abs/2209.04387)。Erickcek 等在计算中已把 XQC 视场方向相对暗物质风的取向转到探测器坐标系，但那是单次百秒飞行，无调制可言 — [arXiv:0704.0794](https://arxiv.org/abs/0704.0794)
- 【文献证实】Micro-X 探空火箭（TES 微量能器）：2018-07-22 首飞（指向故障），2022-08-21 第二次飞行；每次"338 s exposure above 160 km"；科学目标是 SNR 与"indirect detection of galactic dark matter"（keV 线），取回的两篇论文中没有强相互作用暗物质的直接探测分析 — [Adams et al. 2020, J. Low Temp. Phys. 199, 1072, arXiv:1908.09010](https://arxiv.org/abs/1908.09010)；[Micro-X Collab. 2023, arXiv:2212.12064](https://arxiv.org/abs/2212.12064)
- 【文献证实】JWST 暗图像 — Du, Essig, Rauscher & Xu 2025（Phys. Rev. Lett. 135, 051002；引用 12）：NIRSpec NRS2 HgCdTe 探测器（2048×2048 像素，靶质量 0.05037 g），30 组"dark"定标数据，每组曝光约 3574 s；加自定义掩模去除高能事例后比较像素电荷分布；对与超轻暗光子耦合、丰度 0.4% 的亚 GeV 暗物质"disfavor all previously allowed parameter space at high cross sections"，并约束到 f_χ ∼ 0.01%；仪器自遮挡取四周约 20 mm SiC、背面 12 mm Mo（约等效 120 mm SiC，保守地视为不可穿透并把入射通量减半）、底部 ≲25 mm SiC，用 DaMaSCUS-CRUST 模拟穿过 20 mm SiC 后的速度分布；"JWST is located at Lagrange point L2"（因此没有地球遮挡调制）；分析流程先在 20 组数据上设计、再追加数据细化后用于全部 30 组（不是盲分析）；原文定位："In this letter, we show that existing space-based sensors can already probe the allowed region" — [arXiv:2412.13131](https://arxiv.org/abs/2412.13131)
- 【文献证实】近地轨道航天器的间接约束：Neufeld, Farrar & McKee 2018（Astrophys. J. 866, 111；引用 59）用被地球俘获热化的暗物质"大气"对 LEO 航天器轨道衰减、LHC 束流寿命、液氮杜瓦蒸发率、地壳热导给出 0.6–6 m_p 质量段的截面限制 — [arXiv:1805.08794](https://arxiv.org/abs/1805.08794)

**(b) 提案与在建项目**
- 【文献证实】Wandelt et al. 2000："the best way to search for strongly interacting particles is in balloons or in space with detectors with little or no shielding and low energy threshold" — [arXiv:astro-ph/0006344](https://arxiv.org/abs/astro-ph/0006344)
- 【文献证实】Emken et al. 2019 的气球（30 km，1 gram-hour，10⁶ 本底）与卫星（ISS，0.1 gram-month，10⁹ 本底）skipper-CCD 方案；气球可比当时地表限制高约两个量级的截面，卫星再高两个量级；可行模型是丰度 ≲0.4% 的超轻暗光子媒介子成分 — [arXiv:1905.06348](https://arxiv.org/abs/1905.06348)
- 【文献证实】DarkNESS：三篇仪器/任务论文（PoS TAUP2023, 062，引用 1；arXiv:2412.12084，引用 2；Adv. Space Res. 76, 18615，引用 2）加两篇辐照/X 射线标定论文（arXiv:2602.02461、arXiv:2606.26409）；6U、4 片 1.3 Mpix skipper-CCD、活性质量 ~2 g；两个科学目标：强相互作用亚 GeV 暗物质的电子反冲（轨道调制）与 1–10 keV 衰变线；已过 CDR，"Deployment is scheduled no earlier than mid-2026"，搭 Firefly Alpha（DREAM 2.0）— [arXiv:2505.16981](https://arxiv.org/abs/2505.16981)；[arXiv:2412.12084](https://arxiv.org/abs/2412.12084)；[PoS TAUP2023 062](https://inspirehep.net/literature/2765395)；[arXiv:2602.02461](https://arxiv.org/abs/2602.02461)
- 【文献证实】DarkNESS 发射状态：Nanosats Database 条目写"Status: not launched, expected in 2026"，Last modified 2026-08-01；Firefly 的 DREAM 页面（2026-11-18 取回）仍写第二轮载荷"to launch aboard Firefly's Alpha rocket no earlier than 2026"，未说明已发射；INSPIRE/arXiv 上截至 2026-11-18 没有在轨结果 — [nanosats.eu/sat/darkness](https://www.nanosats.eu/sat/darkness.html)；[fireflyspace.com/dream](https://fireflyspace.com/dream/)；[INSPIRE 检索 DarkNESS](https://inspirehep.net/literature?q=title%3ADarkNESS)
- 【文献证实】Niedermann & Sloth 2025（JCAP 05, 042；引用 2）"Radiation Exposure from the Dark"：明确写"In space, there are very few constraints on the potentially harmful effects dark matter could have"；提出三条路：低阈值空间剂量计、"A version of the XQC experiment but with a much longer flight time or a careful re-analysis of pile-up events in X-ray telescopes could exclude this possibility"、以及"a careful re-analysis of previous space-based radiation experiments"；指出 ISS-RAD、LIDAL 阈值 ~10⁴ keV/cm（对应只能探到 σ_N ≳ 10⁻²⁰ cm²/w_N），RadMap M-42 为 60 keV/300 μm；并在其图 1 中按 Chandra 的典型灵敏度画了一条"tentative constraint"虚线（σ_N ≳ 10⁻²¹ cm² 时事件会成为"pile-up event"，限制更不确定）— [arXiv:2411.10521](https://arxiv.org/abs/2411.10521)
- 【文献证实】Cappiello, Collar & Beacom 2021（Phys. Rev. D 103, 023019；引用 45）在"Extensions for future work"中写："Significantly moving the diagonal edge would require a satellite or high-altitude balloon experiment. The column density of the overburden could realistically be reduced by about ∼10³, which would increase the sensitivity by the same factor. (For example, note that the Skylab diagonal edge is about this much higher than ours.)"；但结论段同时写："for composite DM, terrestrial detectors can probe cross sections above cosmological constraints, reducing the need for rocket- or space-based experiments" — [arXiv:2008.10646](https://arxiv.org/abs/2008.10646)
- 【文献证实】Bhoonah et al. 2021 建议"a purpose-built high mass dark matter search program based on plastic track detectors, located both above and below ground, like Skylab and Ohya" — [arXiv:2012.13406](https://arxiv.org/abs/2012.13406)
- 【文献证实】SQM-ISS（闪烁体 + 压电元件叠层，测电荷态、质量与飞行时间，目标是 v ≲ 250 km/s 的慢速大质量粒子：奇异夸克物质、Q-ball、原初黑洞等，"already chosen by ESA through the Open Space Innovation Platform"）— [Bianchi et al. 2024, Sensors 24, 5090](https://inspirehep.net/literature/2816062)（引用 2）；[Casolino et al., PoS ICRC2025 514](https://inspirehep.net/literature/3067418)；[arXiv:2506.10246](https://arxiv.org/abs/2506.10246)
- 【文献证实】Cantatore et al. 2023（Symmetry 15, 1167；引用 14）提出平流层气球直接探测（主要针对暗光子等"streaming"暗物质）— [arXiv:2012.03353](https://arxiv.org/abs/2012.03353)

**(c) 相邻类别（非暗物质风散射，但属于"空间数据挪用"）**
- 【文献证实】PAMELA 奇异子（strangelet）流量上限（Phys. Rev. Lett. 115, 111101；引用 21；磁谱仪对带电粒子的搜索，刚度 1–10³ GV，电荷 1 ≤ Z ≤ 8、质量数 4 ≤ A ≤ 1.2×10⁵，2006–2009 年数据）— [INSPIRE 1392954](https://inspirehep.net/literature/1392954)；AMS-01 奇异子盲搜索 — [Baldassari 2004](https://inspirehep.net/literature/671601)；Mini-EUSO 与 TUS 从轨道上找大气中的 nuclearite/macro 径迹 — [PoS ICRC2023 1410](https://inspirehep.net/literature/2688056)、[PoS ICRC2019 545](https://inspirehep.net/literature/1820207)
- 【文献证实】DAMPE 分数电荷粒子搜索（相对论性）：Phys. Rev. D 106, 063026（2022；5 年数据；引用 16）与 2026 年十年数据的轻质量分数电荷粒子搜索 — [arXiv:2209.04260](https://arxiv.org/abs/2209.04260)；[arXiv:2602.20519](https://arxiv.org/abs/2602.20519)
- 【文献证实】其他把航天器数据挪用于暗物质的例子：GPS 卫星原子钟找畴壁暗物质（Nature Commun. 8, 1195；引用 114）— [arXiv:1704.06844](https://arxiv.org/abs/1704.06844)；LISA Pathfinder 找超轻暗物质（Phys. Rev. D 107, 063015，引用 39；Phys. Lett. B 848, 138328，引用 26）— [arXiv:2301.08736](https://arxiv.org/abs/2301.08736)、[arXiv:2310.06017](https://arxiv.org/abs/2310.06017)；Parker Solar Probe 原位测暗光子暗物质（Phys. Rev. Lett. 134, 171001；引用 28；清华 An Haipeng 等）— [arXiv:2405.12285](https://arxiv.org/abs/2405.12285)；木星探测器电子通量数据（JHEP 10 (2022) 186；引用 30）— [arXiv:2207.13709](https://arxiv.org/abs/2207.13709)；航天器轨迹测"暗物质风"作用力（JHEP 07 (2024) 136；引用 26）— [arXiv:2312.13345](https://arxiv.org/abs/2312.13345)

### Inferences
- 【推断】按"探测器质量 × 时间"排序，已发表的空间直接探测曝光都极小：XQC 约 100 s × 毫克量级吸收体（本人按 34 × 1 mm² ×（0.96 μm HgTe + 14 μm Si）粗估），JWST 0.05 g × 30 个约 1 小时曝光（约 5 kg·s），DarkNESS 计划 0.1 gram-month（约 3×10² kg·s）；Skylab 是面积 × 时间（1.17 m² × 253 天）型。HXMT/HE 的 ~100 kg × 9.4 年（约 3×10¹⁰ kg·s）在质量-时间上比 DarkNESS 计划值高约 8 个量级、比 JWST 高约 10 个量级，但阈值也高出 3–5 个量级（≥20 keVee 对 eV–几十 eV），所以两者覆盖的是完全不同的模型类（重/复合、多次散射核反冲 对 亚 GeV 电子反冲）。
- 【推断】"把非为暗物质设计的空间仪器当直接探测器"的传统已有 35 年以上（Pioneer 11、IMP 7/8、Skylab → XQC → JWST），评审人对这一思路不会陌生；陌生的是用调制而非总率。

### Gaps
- Pioneer 11（Simpson et al. 1980）、IMP 7/8（Mewaldt 等，Wandelt 2000 中标为"in preparation / private communication"）和 IMAX（McGuire 1994 博士论文）的曝光量、面积等细节未取回；ADS 页面无法通过工具访问。
- RRS 1987 的阈值与曝光只有摘要信息。
- DarkNESS 在 2026-08 至 2026-11 之间是否已发射，没有找到任何发射记录或否定声明；只能说"未见已发射的证据"。
- 没有找到 Micro-X、Hitomi/XRISM 微量能器数据被用于 SIMP 分析的文献（未找到，不等于不存在）。

---

## 3. 查新：是否已有人用天文/宇宙线卫星的探测器做过（或提出过）暗物质风直接探测或地球遮挡调制分析

### Takeaway
**"地球遮挡造成的轨道调制"这个概念不是新的**（Emken et al. 2019 提出，DarkNESS 列为观测计划，2026 年还有 LEO 原子钟版本）；**"用现成空间天文探测器的本底数据约束强相互作用暗物质"也不是新的**（XQC 2007、JWST 2025，Niedermann & Sloth 2025 还点名了 X 射线望远镜）。但在本次检索范围内，**没有找到任何已发表或已预印的工作，用某颗 X/γ 射线天文卫星或宇宙线卫星的闪烁体、半导体或反符合屏蔽，对暗物质风做轨道相位/恒星时调制分析，也没有找到任何实际完成的在轨调制搜索**。这是"未找到"，不是"确证不存在"。

### Cited Findings

**(a) 检索做了什么**
- 【文献证实】INSPIRE 引用网络：取回 13 篇关键论文的全部施引文献并合并去重，共 693 条：Erickcek 2007（129）、Emken 2019（214）、Cappiello 2021（45）、Bhoonah 2021（37）、Kavanagh 2018（126）、Emken & Kouvaris 2018（154）、Mahdawi & Farrar 2018（114）、Digman 2019（98）、Hooper & McDermott 2018（86）、Kouvaris & Shoemaker 2014（81）、Kavanagh et al. 2017（84）、Emken & Kouvaris 2017（88）、Carney 2022（93）。其中 Erickcek、Cappiello、Bhoonah 三篇的施引标题逐条读过；其余按标题关键词（satellite/orbit/space/balloon/CubeSat/rocket/telescope/sidereal/shadow/X-ray/GBM/scintillator/NaI/CsI/veto/spacecraft/mission 等）过滤 — 例：[refersto:recid:748100](https://inspirehep.net/literature?q=refersto%3Arecid%3A748100)、[refersto:recid:1735240](https://inspirehep.net/literature?q=refersto%3Arecid%3A1735240)、[refersto:recid:1813235](https://inspirehep.net/literature?q=refersto%3Arecid%3A1813235)、[refersto:recid:1838365](https://inspirehep.net/literature?q=refersto%3Arecid%3A1838365)
- 【文献证实】过滤后与空间相关的只有：DarkNESS 三篇、Du et al.（JWST）、Niedermann & Sloth、Cantatore et al.（平流层）、Zioutas et al.（平流层温度异常）；另有 Saha 2025 "Searching for dark matter with MeVCube"（经核对是间接探测：原初黑洞蒸发与衰变/湮灭光子）— [arXiv:2501.13162](https://arxiv.org/abs/2501.13162)
- 【文献证实】JWST 论文的 12 篇施引文献中没有把同法用于其他空间望远镜的后续工作（其中有 APS Physics 提要与 2026 版 PDG 综述）— [refersto:recid:2860455](https://inspirehep.net/literature?q=refersto%3Arecid%3A2860455)；DarkNESS 两篇的施引（4 条）与 Niedermann & Sloth 的施引（2 条）同样没有 — [refersto:recid:2924116](https://inspirehep.net/literature?q=refersto%3Arecid%3A2924116)、[refersto:recid:2849499](https://inspirehep.net/literature?q=refersto%3Arecid%3A2849499)
- 【文献证实】INSPIRE 摘要检索：`"orbital modulation" + "dark matter"` 0 条；`"dark matter" + "low Earth orbit" + (direct detection|scatter*|recoil*)` 0 条；`"dark matter" + "satellite-borne" + (direct detection|strongly interacting)` 3 条（仅 Emken 2019 相关）；`"dark matter" + "Earth shadow*"` 0 条；`"dark matter" + "Earth shield*" + (satellite|orbit*|space)` 0 条 — [INSPIRE 检索示例](https://inspirehep.net/literature?q=abstracts.value%3A%22orbital%20modulation%22%20and%20abstracts.value%3A%22dark%20matter%22)
- 【文献证实】INSPIRE 全文检索：`"orbital modulation(s)" + "dark matter" + ("strongly interacting"|"direct detection")` 的相关命中只有 Emken 2019 与 DarkNESS；把 {strongly interacting dark matter, SIMP} 以及 {composite / ultraheavy / macroscopic dark matter, nuclearite, multiply interacting massive} 分别与 {Fermi GBM, BATSE, Swift BAT, Insight-HXMT, GECAM, AstroSat, NuSTAR, SPI-ACS / anticoincidence shield, Suzaku, RXTE, Hitomi, DAMPE, CALET, AMS-02} 配对并加"direct detection"，命中全是综述、白皮书或无关论文；`"Earth occultation" + "dark matter" + "direct detection"`、`"space-based direct detection"` 等短语同样无相关结果 — [INSPIRE 全文检索示例](https://inspirehep.net/literature?q=fulltext%3A%22orbital%20modulation%22%20and%20fulltext%3A%22dark%20matter%22%20and%20fulltext%3A%22strongly%20interacting%22)
- 【文献证实】arXiv API（全学科）：`"dark matter" AND "low Earth orbit" AND (direct detection|strongly interacting|scattering)` 4 条（DarkNESS 两篇、Neufeld 2018、LEO 原子钟）；`"dark matter" AND "orbital modulation"` 0 条；`"dark matter" AND ("Earth shadow"|"Earth shadowing")` 2 条（DAMA 2015；CTA 正电子）；`"dark matter" AND ("Insight-HXMT"|"HXMT")` 1 条（原初黑洞）；`"dark matter" AND (GECAM|AstroSat|"Suzaku WAM"|"SPI-ACS"|"anticoincidence shield")` 5 条均无关；摘要含 `"dark matter"` 与 `"Earth occultation"` 0 条；摘要含 dark matter + (scintillator|NaI|CsI|BGO) + (satellite|in orbit|space station) + (direct detection|strongly|composite|nuclear recoil|modulation) 14 条均无关（HERD/DAMPE 仪器等）— [arXiv API 示例](https://export.arxiv.org/api/query?search_query=all:%22dark+matter%22+AND+all:%22low+Earth+orbit%22&max_results=50)
- 【文献证实】WebSearch（标准模式）用过的相关提问：Fermi GBM/BATSE + direct detection + strongly interacting + Earth shadow modulation；strongly interacting dark matter + satellite scintillator + Earth's shadow + orbital modulation + archival；dark frames/dark images + CCD + Hubble/Chandra/XMM/Gaia + strongly interacting；orbital period/phase modulation + composite/ultraheavy + space-based proposal；中文"卫星 闪烁体 暗物质 强相互作用 直接探测 地球遮挡/轨道调制 慧眼/GECAM/悟空"。返回的相关结果仍只有 Emken 2019、DarkNESS、JWST、DAMA 2015 — 代表性命中：[arXiv:1905.06348](https://arxiv.org/abs/1905.06348)、[arXiv:2505.16981](https://arxiv.org/abs/2505.16981)、[arXiv:2412.13131](https://arxiv.org/abs/2412.13131)

**(b) 找到的最接近先例（按接近程度）**
- 【文献证实】概念本身：Emken et al. 2019 §4.2.4（卫星上"orbital modulations of higher frequencies"、遮挡函数、f_mod ∼75%）— [arXiv:1905.06348](https://arxiv.org/abs/1905.06348)
- 【文献证实】列入任务计划：DarkNESS 的 Cygnus 指向"utilize Earth obstruction to search for modulation signatures of strongly interacting DM" — [arXiv:2505.16981](https://arxiv.org/abs/2505.16981)
- 【文献证实】现成空间天文探测器数据 → 强相互作用暗物质直接限制：XQC（2007）与 JWST（2025，"existing space-based sensors can already probe the allowed region"）— [arXiv:0704.0794](https://arxiv.org/abs/0704.0794)；[arXiv:2412.13131](https://arxiv.org/abs/2412.13131)
- 【文献证实】JWST 论文作者的公开表态（科技日报 2025-08-10，中国日报网转载）：杜佩之（中国科学技术大学特任研究员）团队称"此约束填补了地面实验的探测空白"，并称该方法可应用于其他空间望远镜的数据、"为未来空间暗物质探测器的设计及数据处理提供坚实的理论支撑" — [cn.chinadaily.com.cn 2025-08-11](https://cn.chinadaily.com.cn/a/202508/11/WS68b7aaa2a310f07257746795.html)
- 【文献证实】点名 X 射线望远镜与空间剂量计用于重、强相互作用暗物质：Niedermann & Sloth 2025（定性，未做数据分析，未涉及调制）— [arXiv:2411.10521](https://arxiv.org/abs/2411.10521)
- 【文献证实】呼吁卫星/气球版的闪烁体复合暗物质搜索：Cappiello, Collar & Beacom 2021（未涉及调制）— [arXiv:2008.10646](https://arxiv.org/abs/2008.10646)
- 【文献证实】建议用 Fermi-GBM / NuSTAR **存档数据**找与暗物质有关的调制（但是光子信号的**季节**调制，针对 axion quark nugget 模型，不是暗物质风散射）：Ge, Rachmat, Siddiqui, Van Waerbeke & Zhitnitsky 2022（Phys. Dark Univ. 36, 101031；引用 19）："We suggest that the multi-year archival data from the GBM or NuSTAR could be used to search for a seasonal variation in the near-Earth environment up to 100 keV"；其 19 篇施引文献中未见有人实际做了这项分析 — [arXiv:2004.00632](https://arxiv.org/abs/2004.00632)；[refersto:recid:1789425](https://inspirehep.net/literature?q=refersto%3Arecid%3A1789425)
- 【文献证实】卫星宇宙线探测器被挪用为 SIMP 探测器的老先例（总率限制）：Pioneer 11、IMP 7/8、Skylab — [arXiv:astro-ph/0105567](https://arxiv.org/abs/astro-ph/0105567)；[arXiv:astro-ph/0006344](https://arxiv.org/abs/astro-ph/0006344)
- 【文献证实】地面 NaI 的 SIMP 搜索先例（延迟符合/飞行时间法）：BPRS 1994（Astropart. Phys. 2, 13；两块 NaI 的延迟符合率上限 0.0011 events/hour）与 DAMA 1999（Phys. Rev. Lett. 83, 4918；引用 133；~100 kg NaI 的两"层"探测器）— [INSPIRE 359315](https://inspirehep.net/literature/359315)；[INSPIRE 513688](https://inspirehep.net/literature/513688)

**(c) HXMT 及同类任务**
- 【文献证实】arXiv 全学科检索 "dark matter" + HXMT 只有 1 篇：Yang et al.（用慧眼已发表的弥漫背景谱约束原初黑洞丰度，Commun. Theor. Phys. 78, 045402, 2026；引用 6）；INSPIRE 上 GECAM + dark matter 为 0 — [arXiv:2412.09297](https://arxiv.org/abs/2412.09297)

### Inferences
- 【推断】新颖性应分层表述：(1) 物理机制与"轨道调制"判据——已有，优先权属于 Emken/Essig/Kouvaris/Sholapurkar 2019；(2) 把它用于**重质量/复合暗物质的核反冲**——未见；(3) 用**现役天文卫星的存档闪烁体与反符合数据**来做——未见；(4) **实际完成**的在轨调制搜索——未见（DarkNESS 未飞，JWST 在 L2 无地球遮挡）。因此"首个在轨地球遮挡调制搜索"和"首个空间闪烁体对复合暗物质的直接限制"这两个说法在本次检索范围内站得住，但"首次提出该思想"站不住，论文必须把 Emken et al. 2019 与 DarkNESS 作为出发点引用。原综述给的"新颖性 5 分"宜下调到 3.5–4 分。
- 【推断】被抢先的风险是实的：(i) DarkNESS 一旦入轨即可宣称亚 GeV 段的"首个轨道调制搜索"；(ii) JWST 论文四位作者中两位现已在国内（杜佩之在中国科大，Hailin Xu 在李政道研究所，见第 6 节），且公开说方法可推广到其他空间望远镜；(iii) Fermi-GBM 自 2008-08-12 起连续提供数据（见第 4 节 Wilson-Hodge 2012；数据对外公开一事凭记忆，未核实），已累计 18 年，做同类分析的门槛比 HXMT 低。HXMT 的相对优势是逐事例保留的反符合图样、CsI 大质量和脉宽信息。
- 【猜测】没人做过的主要原因多半是学科间隔：提出概念的是亚 GeV 暗物质理论组（关注 eV 阈值的 CCD），熟悉 LEO 闪烁体本底的是 GRB/X 射线天文组，二者几乎不交叉引用。

### Gaps
- "未找到"的局限：INSPIRE 全文索引对 astro-ph.HE/IM、SPIE、COSPAR、AAS 摘要、中文期刊覆盖不全；合作组内部备忘、学位论文、会议口头报告无法检索；WebSearch 只用了标准模式的少量提问。
- 没有逐一核对 CGRO-BATSE、RXTE、Swift-BAT、INTEGRAL、Suzaku、NuSTAR、AstroSat、Hitomi、CALET、AMS-02 各自的官方论文清单；结论依赖关键词检索。
- Chandra/XMM/Gaia/HST 的 CCD 暗帧是否被用于暗物质散射：除 Niedermann & Sloth 的定性讨论外未找到任何分析。

---

## 4. 可移植的恒星系分析实践：宇宙线各向异性、方向性暗物质探测、在轨洛伦兹破缺检验、地球掩食技术

### Takeaway
宇宙线各向异性领域已经把"在 10¹⁰–10¹¹ 个事例里测 10⁻³–10⁻⁴ 的恒星时信号"做成了标准流程，核心是三件事：用 anti-sidereal（以及 extended-sidereal）时间估计太阳时调制泄漏到恒星时的系统误差；用数据驱动的参考图（时间打乱或整年平均）而不是模拟来求曝光；用差分法（East–West、forward–backward）消掉探测器与大气的慢变化。这些都能搬到 HXMT 上，但 HXMT 需要的分数精度（~10⁻⁵）比任何已发表的卫星各向异性分析（Fermi-LAT 质子，~10⁻³–10⁻⁴）高一到两个量级。

### Cited Findings
- 【文献证实】Milagro 2009（Astrophys. J. 698, 2121；引用 276）：7 年、9.5×10¹⁰ 个事例，测到幅度 ~0.1% 的各向异性，最强结构 (2.49 ± 0.02 stat ± 0.09 sys)×10⁻³；方法是 forward–backward asymmetry 加谐波拟合，同时在恒星时、世界时和 anti-sidereal 时（Farley & Storey 1954）三个时标下做直方图；"we can estimate the systematic errors in the sidereal signal by examining the anti-sidereal signal. Since there are no physical processes which occur in the anti-sidereal time frame …" — [arXiv:0806.2293](https://arxiv.org/abs/0806.2293)
- 【文献证实】IceCube 2016（Astrophys. J. 826, 220；引用 159）：3.18×10¹¹ 个缪子事例，各向异性幅度 10⁻³ 量级；系统误差用 anti-sidereal 与 extended-sidereal 两个非物理时标估计："No physical signal is expected with a frequency of 364.24 cycles per year, so any significant 'signal' that appears in the anti-sidereal frame stems from a modulation of the solar frame and is equivalent to the systematic effect of the solar frame on the sidereal frame"；太阳偶极（Compton–Getting）同时作为方法的正向校验 — [arXiv:1603.01227](https://arxiv.org/abs/1603.01227)
- 【文献证实】Tibet ASγ 2005/2006（Astrophys. J. Lett. 626, L29，引用 149；Science 314, 439，引用 296）：多 TeV 恒星日变化的能量与赤纬依赖；在 Cygnus 方向发现新的恒星时各向异性成分 — [arXiv:astro-ph/0505114](https://arxiv.org/abs/astro-ph/0505114)；[arXiv:astro-ph/0610671](https://arxiv.org/abs/astro-ph/0610671)
- 【文献证实】HAWC 2014/2018：4.9×10¹⁰ 个事例看到 10⁻⁴ 量级小尺度各向异性；偶极幅度 (8–14)×10⁻⁴ — [arXiv:1408.4805](https://arxiv.org/abs/1408.4805)；[arXiv:1805.01847](https://arxiv.org/abs/1805.01847)
- 【文献证实】East–West 法（Bonino et al. 2011, Astrophys. J. 738, 67；引用 49；Nagashima 1989 首倡）：对东、西两个方向的计数率之差做分析，"largely independent of experimental effects, that is, it does not require corrections for acceptance and/or for atmospheric effects" — [arXiv:1106.2651](https://arxiv.org/abs/1106.2651)
- 【文献证实】卫星上的恒星系各向异性分析 — Fermi-LAT 质子 2019（Astrophys. J. 883, 33；引用 19）：8 年、1.79×10⁸ 个 >78 GeV 质子，偶极幅度 δ = (3.9 ± 1.5)×10⁻⁴（试验前 p = 0.01），95% 上限 1.3×10⁻³；要点：(i) "the instrument's exposure cannot be calculated using simulation because the resulting uncertainties are significantly larger than the expected signal O(10⁻³)"，改用数据驱动参考图（按 1 s 指向历史 + 年平均事例率与仪器坐标分布；并与 time-scrambling / event-shuffling 比较）；(ii) 用整年时间箱使 Compton–Getting 偶极相消，并吸收量能器光产额的单调下降；(iii) 地磁 East–West 效应即使在垂直截止刚度以上也可见，靠在地平坐标（alt-az）下检查并收紧离轴角来消除；(iv) 2017 年的初步结果曾出现显著四极矩，后来查明是 PSF 尾部（从底部入射被重建为顶部入射）造成的 — [arXiv:1903.02905](https://arxiv.org/abs/1903.02905)；电子版本 [Fermi-LAT 2010, Phys. Rev. D 82, 092003, arXiv:1008.5119](https://arxiv.org/abs/1008.5119)
- 【文献证实】方向性暗物质探测：太阳运动使反冲方向分布出现指向太阳运动方向（Cygnus）的偶极，"This provides a powerful tool for demonstrating the Galactic origin of nuclear recoils and hence unambiguously detecting Dark Matter" — [Mayet et al. 2016, Phys. Rept. 627, 1, arXiv:1602.03781](https://arxiv.org/abs/1602.03781)（引用 302）；"current experiments have no practical way of confirming a detected signal's Galactic origin" — [Vahsen, O'Hare & Loomba 2021, Ann. Rev. Nucl. Part. Sci. 71, 189, arXiv:2102.04596](https://arxiv.org/abs/2102.04596)（引用 96）；CYGNUS 可行性研究 — [arXiv:2008.12587](https://arxiv.org/abs/2008.12587)
- 【文献证实】Snowmass 2021 粒子暗物质专题报告："Unlike the annual rate and energy modulation, the sidereal-daily directional modulation (equivalent to a dipole in galactic coordinates), is expected to be immune against time-dependent backgrounds originating in the laboratory" — [Cooley et al. 2022, arXiv:2209.07426](https://arxiv.org/abs/2209.07426)
- 【文献证实】在轨的恒星时/轨道特征检验 — MICROSCOPE 洛伦兹不变性检验（Phys. Rev. Lett. 123, 231102；引用 37）：寻找"a specific signature that depends on the orbital velocity and rotation of Earth"，用分布在一年中的 5 个测量段；Gravity Probe B 与 LAGEOS 也做过同类检验 — [arXiv:1912.03030](https://arxiv.org/abs/1912.03030)；[arXiv:1309.6399](https://arxiv.org/abs/1309.6399)；[arXiv:2501.10588](https://arxiv.org/abs/2501.10588)
- 【文献证实】地球掩食技术（点源）：BATSE（Harmon et al. 2002, Astrophys. J. Suppl. 138, 149；引用 44）利用源过地球边缘时计数率的台阶，"for measuring source intensity and spectra without the use of complex background models"，并系统讨论了源混淆、响应与快速本底涨落的系统误差；Fermi-GBM 三年监测 209 个源、探测到 99 个，"The Crab and Cyg X-1 are also detected in the 300-500 keV band"（Wilson-Hodge et al. 2012, Astrophys. J. Suppl. 201, 33；引用 49）；GBM 掩食成像（Rodi et al. 2014）；AstroSat/CZTI 掩食（Singhal et al. 2021）— [arXiv:astro-ph/0109069](https://arxiv.org/abs/astro-ph/0109069)；[arXiv:1201.3585](https://arxiv.org/abs/1201.3585)；[arXiv:1304.1478](https://arxiv.org/abs/1304.1478)；[arXiv:2105.09527](https://arxiv.org/abs/2105.09527)
- 【文献证实】地球掩食测弥漫成分（方法上最接近"弥漫的恒星系模板 × 地球遮挡"）：INTEGRAL 2006 年四次专门的地球掩食观测，Türler et al. 2010 "model instead their spatial distribution and derive for each of them the expected modulation of the detector counts. The spectra of the diffuse emission components are obtained by fitting the normalizations of the model lightcurves to the observed modulation in different energy bins"，同时分离 CXB、银河脊、地球反照与大气发射；Churazov et al. 2007 列举了致密源、银河脊、CXB 被大气反射、宇宙线致大气发射和极光等污染；Swift/BAT 在不同地磁纬度用掩食测 CXB 与反照谱 — [arXiv:1001.2110](https://arxiv.org/abs/1001.2110)；[arXiv:astro-ph/0608250](https://arxiv.org/abs/astro-ph/0608250)；[arXiv:0808.3377](https://arxiv.org/abs/0808.3377)
- 【文献证实】LEO 闪烁体的物理本底模型：Biltzinger et al. 2020（A&A 640, A8）给 Fermi-GBM 建了第一个物理驱动的本底模型，分别建模各本底来源并处理响应 — [arXiv:2005.11219](https://arxiv.org/abs/2005.11219)
- 【文献证实】慧眼自己的地球掩食应用目前是大气密度测量（Crab 掩食）— [Yu et al. 2022, arXiv:2204.09674](https://arxiv.org/abs/2204.09674)；[Xue et al. 2023, arXiv:2211.00293](https://arxiv.org/abs/2211.00293)；[Zhang et al. 2025, arXiv:2502.19678](https://arxiv.org/abs/2502.19678)

### Inferences
- 【推断】可直接移植的零检验清单：(1) "错误坐标系"对照——把暗物质模板的包络周期换成 80.5 天（anti 版本）或把 apex 换到天球其他方向重拟合；真信号应只在 (RA, Dec) ≈ (313.5°, +47.5°) 出峰，可像各向异性分析那样画一张"最佳 apex 方向"天图。(2) 太阳系模板（以太阳方向和 55.9 天 β 角周期为自变量）与地固模板（地磁纬度/截止刚度、SAA 后时间）同时拟合，报告它们与暗物质模板的相关系数。(3) 整年分箱，使年周期效应相消（Fermi-LAT 做法）。(4) 差分量：盲探测器对非盲、NaI 对 CsI、不同朝向的反符合板，相当于 East–West 法。(5) 时间戳 bootstrap / 事例打乱得到零分布（SENSEI、IceCube、Fermi-LAT 做法）。(6) 用已知的恒星系信号做正向校验：Crab 与 Cyg X-1 的掩食台阶、银河面弥漫辐射的轨道调制，相当于 IceCube 拿太阳偶极做校验。
- 【推断】点源掩食的"台阶拟合"不适用于暗物质风：暗物质到达方向的弥散约 ±40°，遮挡是几十分钟的平滑变化，更像 Türler 等对弥漫成分用的"空间模板 → 预期调制光变曲线 → 拟合归一化"。
- 【推断，本人计算】恒星系里与暗物质模板同相的天体物理混淆源：apex 位于银道面上（b ≈ +1.7°，l ≈ 87.5°），距 Cyg X-3 7.6°、Cyg X-2 13.0°、Cyg X-1 16.1°、Cas A 24.6°；Crab、银心、Sco X-1 则距 apex 约 88°–97°（接近正交）。GBM 已证明无准直 NaI 能在 300–500 keV 看到 Cyg X-1 的掩食台阶（上条 Wilson-Hodge 2012），所以 Cygnus 区的源和银河脊必须进本底模型；好处是它们的掩食是陡台阶、能谱是光子谱，且应当不出现在盲探测器里（"盲 NaI 对 <200 keV 光子不透明"一说取自前一轮综述，本次未核实）。apex 的黄纬为 +60.6°，太阳永远离它 60° 以上。
- 【推断】统计量上 HXMT（约 5×10¹⁰ 计数）与 Milagro（9.5×10¹⁰）和 IceCube（3.2×10¹¹）同级，但后两者的系统误差地板在 10⁻⁴ 量级（Milagro 0.09×10⁻³），而且它们的本底在恒星时上本来平坦；要达到 10⁻⁵ 没有先例可循。
- 【推断】宇宙线自身的恒星时各向异性是一个与暗物质模板同对称类的不可约本底：各向异性幅度 δ 的宇宙线天空被地球遮住 30% 后，无准直探测器的宇宙线致计数率会出现恒星系锁定的轨道调制，量级约 δ × O(0.3)。Fermi-LAT 给出的 >78 GeV 质子偶极为 (3.9 ± 1.5)×10⁻⁴、上限 1.3×10⁻³，对应的率调制可达 10⁻⁴ 量级，即可能高于 ~10⁻⁵ 的统计地板。区分手段是方向：TeV 段偶极相位在 RA ≈ 40° 附近（HAWC），与 apex（RA ≈ 313.5°）相差约 90°；但 Tibet 在多 TeV 段报告过 Cygnus 区方向的恒星时超出，GV 段的恒星时各向异性本次未取到数据。此项应交系统误差研究者评估。

### Gaps
- 没有找到任何 LEO X/γ 射线仪器把总计数率在恒星系里做到 10⁻⁵ 分数精度的已发表分析；GBM、BATSE 本底模型的残差水平未取数。
- Farley & Storey 1954 与 Nagashima 1989 原文未取回（经 Milagro、Bonino 论文转引）。

---

## 5. 警示：纯调制结论的可信度，以及信本比 ~10⁻⁵ 的调制分析要满足什么标准

### Takeaway
直接探测史上"只靠调制"的阳性结论没有一个被确认：DAMA/LIBRA 的 12.9σ 年调制被同靶材的 COSINE-100 与 ANAIS-112 以 4.7σ（合并六年数据）排除；CoGeNT 的 2.8σ 调制在更多数据和独立重分析后降到 <1σ–1.7σ；XENON100 的 2.8σ 局域显著性降到 1.8σ；XMM-Newton 的 11σ 季节调制十二年只得到 27 次引用，其太阳轴子解释被否定，信号本身至今没有公认解释；放射性衰变率的"日地距离调制"被 14 个实验室的数据否定。共同教训是：慢变本底 + 不当的扣除方式会凭空造出调制；没有独立仪器复现，再高的统计显著性也不被接受。

### Cited Findings
- 【文献证实】DAMA/LIBRA-phase2（2018；引用 325）：2–6 keV 调制幅度 (0.0103 ± 0.0008) cpd/kg/keV，12.9σ，相位 (145 ± 5) d，周期 (0.999 ± 0.001) yr — [arXiv:1805.10486](https://arxiv.org/abs/1805.10486)
- 【文献证实】ANAIS-112 三年（Phys. Rev. D 103, 102005；引用 163）：盲分析流程；最佳拟合 −0.0034 ± 0.0042（1–6 keV）、0.0003 ± 0.0037 cpd/kg/keV（2–6 keV），与 DAMA 不相容 3.3σ / 2.6σ；另做相位自由搜索、其他频率的周期信号搜索和一致性检验 — [arXiv:2103.01175](https://arxiv.org/abs/2103.01175)
- 【文献证实】ANAIS-112 六年（Phys. Rev. Lett. 135, 051001；引用 25）：与 DAMA 调制不相容达 4σ，并指出剩余的系统不确定性在于探测器对核反冲的响应 — [arXiv:2502.01542](https://arxiv.org/abs/2502.01542)
- 【文献证实】COSINE-100 全数据（Sci. Adv. 11, adv6503；引用 35）：6.4 年、61.3 kg，"improved energy calibration and time-dependent background modeling"，>3σ 排除 — [arXiv:2409.13226](https://arxiv.org/abs/2409.13226)
- 【文献证实】COSINE-100 + ANAIS-112 合并（Phys. Rev. Lett. 135, 121002；引用 29）：三年数据合并后与 DAMA 不相容 3.7σ / 2.6σ；六年数据简单合并为 0.0005 ± 0.0019（1–6 keV）与 0.0027 ± 0.0021（2–6 keV）cpd/kg/keV，分别以 4.7σ 与 3.5σ 排除 DAMA 信号 — [arXiv:2503.19559](https://arxiv.org/abs/2503.19559)
- 【文献证实】分析方法自己造出调制：Buttazzo et al. 2020（JHEP 04, 137；引用 31）"a slowly varying rate can bias or even simulate an annual modulation if data are analyzed in terms of residuals computed by subtracting approximately yearly averages starting from a fixed date, rather than a background continuous in time" — [arXiv:2002.00459](https://arxiv.org/abs/2002.00459)；Messina et al. 2020（JCAP 04, 037；引用 21）用贝叶斯比较反驳其能完全解释 DAMA，但同样主张用连续的慢变本底项 — [arXiv:2003.03340](https://arxiv.org/abs/2003.03340)；COSINE-100 2023（Sci. Rep. 13, 4676；引用 57）用类 DAMA 方法处理自己的数据，"observe a significant annual modulation, however the modulation phase is almost opposite to that of the DAMA/LIBRA data" — [arXiv:2208.05158](https://arxiv.org/abs/2208.05158)
- 【文献证实】CoGeNT：2011 年报告 ~2.8σ 年调制（Phys. Rev. Lett. 107, 141301；引用 685）— [arXiv:1106.0650](https://arxiv.org/abs/1106.0650)；合作组 3.4 年数据的最大似然分析降到 1.7σ（引用 114）— [arXiv:1401.6234](https://arxiv.org/abs/1401.6234)；独立重分析（Davis, McCabe & Boehm 2014, JCAP 08, 014；引用 63）在重新处理体/表面事例分离后结论为"less than 1σ" — [arXiv:1405.0495](https://arxiv.org/abs/1405.0495)
- 【文献证实】XENON100 电子反冲调制：2015 年年调制局域显著性 2.8σ，但"the analysis of a multiple-scatter control sample and the phase of the modulation disfavor a dark matter interpretation"；四年数据降到 1.8σ，并出现一个 431 天、全局 1.9σ 的弱调制 — [arXiv:1507.07748](https://arxiv.org/abs/1507.07748)；[arXiv:1701.00769](https://arxiv.org/abs/1701.00769)
- 【文献证实】综述："several experiments have already claimed such a discovery in the past decade. While most of them have later revoked their conclusions, and others have found potentially contradictory results, one still stands today" — [Froborg & Duffy 2020, J. Phys. G 47, 094002, arXiv:2003.04545](https://arxiv.org/abs/2003.04545)
- 【文献证实】空间 X 射线仪器上的纯调制结论：Fraser et al. 2014（MNRAS 445, 2146；引用 27）在 XMM-Newton 2–6 keV 本底中报告季节调制，显著性 11σ（pn）、4σ（MOS1）、5σ（MOS2），解释为太阳轴子在地磁场中转换 — [arXiv:1403.2436](https://arxiv.org/abs/1403.2436)；Roncadelli & Tavecchio 2015（MNRAS 450, L26）指出该解释有三处问题、"cannot have any relation with hypothetical axions produced by the Sun"，并称"a physically consistent interpretation of the observed seasonally-modulated X-ray excess still remains an exciting challenge" — [arXiv:1411.3297](https://arxiv.org/abs/1411.3297)；施引文献主要是 XMM 本底系统学研究与 AQN 模型论文 — [refersto:recid:1285332](https://inspirehep.net/literature?q=refersto%3Arecid%3A1285332)
- 【文献证实】放射性衰变率调制：Jenkins et al. 2009（引用 103）称衰变率与日地距离相关 — [arXiv:0808.3283](https://arxiv.org/abs/0808.3283)；Pommé et al. 2016（Phys. Lett. B 761, 281；引用 35）用全球 14 个实验室、长达 40 年的数据否定："Systematic deviations from a purely exponential decay curve differ from one data set to another and are attributable to instabilities in the instrumentation and measurement conditions"，最稳定的测量把年振荡幅度限制在 0.0006%–0.008%，"Oscillations in phase with Earth's orbital distance to the Sun could not be observed within a 10⁻⁶ to 10⁻⁵ range of precision" — [INSPIRE 1485140](https://inspirehep.net/literature/1485140)
- 【文献证实】卫星各向异性分析中的假阳性：Fermi-LAT 质子 2017 年初步结果的显著四极矩后被证明是 PSF 尾部事例造成 — [arXiv:1903.02905](https://arxiv.org/abs/1903.02905)
- 【文献证实】可比的"挪用数据"搜索采用盲分析：DEAP-3600 多次散射搜索"Using a blind analysis of data collected over a 813 d live time" — [arXiv:2108.09405](https://arxiv.org/abs/2108.09405)；XENON1T MIMP 搜索"conduct a blind search … Following unblinding, we observe no signal candidate events" — [arXiv:2304.10931](https://arxiv.org/abs/2304.10931)；盲分析方法综述 — [Klein & Roodman 2005, Ann. Rev. Nucl. Part. Sci. 55, 141](https://inspirehep.net/literature/705336)
- 【文献证实】Snowmass 2021 对"异常"的处理原则："A strength of direct detection is that experiments can be replicated or altered in response to such excesses. … replication is an option of last resort to fully confirm or rule out any potential signal for dark matter" — [arXiv:2209.07426](https://arxiv.org/abs/2209.07426)
- 【文献证实】时间依赖本底的风险（2026 综述）："such time-dependent backgrounds, if not correctly accounted for, could result in fake dark matter detection and/or a reduced sensitivity" — [arXiv:2611.21052](https://arxiv.org/abs/2611.21052)

### Inferences
- 【推断】从上述案例归纳，信本比 ~10⁻⁵ 的纯调制分析要被相信，至少需要：
  1. **盲化**：先在"错误 apex"或打乱轨道根数（RAAN）的数据上冻结全部选择与本底模型，再开盲（ANAIS、DEAP、XENON1T 做法）。
  2. **注入–恢复**：把模拟信号注入真实事例流，证明本底模型不会把它吸收掉，也不会凭空产生（对应 Buttazzo / COSINE-100 的教训，以及原综述指出的"以盲探测器为输入的本底模型会吃掉信号"）。
  3. **连续时间的本底模型**，不用与信号周期可公度的滑动窗口去趋势。
  4. **多个独立探测器类别给出一致结果**：NaI、CsI、塑料闪烁体反符合板三类靶（A 与阈值不同），18 个模块，幅度比应符合预言；XENON100 就是靠多次散射对照样本和相位判定自己的 2.8σ 不是暗物质。
  5. **预言的形状全部成立**：轨道相位、66 天包络及其零点方位、年变化（风速 ±6%）、能谱/多重性分布；任何一项对不上即判系统误差。
  6. **对照方向与对照时标**（第 4 节的 anti 版本和 apex 天图），并报告全局显著性（DAMIC-M 扫 1–48 h、XENON100 报 global significance）。
  7. **过离散处理**：检验计数是否泊松，必要时用 scaled-Poisson（SENSEI）。
  8. **独立仪器复现**：GECAM、Fermi-GBM、AstroSat 等轨道倾角不同，包络与相位的预言也不同；没有这一步，阳性结果的命运大概率与 XMM 季节调制相同。
- 【推断】零结果的可信度门槛低得多：只要注入–恢复证明灵敏度、并保守地把无法建模的残差全部算作可能的信号（Bhoonah 等对 Skylab 的 150 个径迹就是这样处理的），评审即可接受。
- 【猜测】即使满足前七条，暗物质界对单一卫星的阳性结果仍会默认它是未识别的轨道本底；Blanco et al. 2026 的"显著性随 √T 持续增长"只在本底调制可被参数化时成立，而 LEO 本底的轨道频率成分正是最难参数化的部分。

### Gaps
- 没有找到专门讨论"高本底、极小分数调制"暗物质分析可信度标准的方法论文献；以上标准是从个案归纳的。
- DAMA 合作组对 Buttazzo 等批评的正式答复未取回。

---

## 6. 影响力证据：期刊、引用数、路线图表述、活跃理论组

### Takeaway
"挪用已有数据 → 强相互作用/复合/超重暗物质新限制"是一条被社区认可、能进 PRD/JCAP（开创性的进 PRL）的路线，Snowmass 2021 明确写了"dedicated analyses of existing data can be very fruitful"。但它是小众方向：同类论文的引用率多在每年 6–16 次，最近的后来者（CDEX-10 2025：1 次；DarkSide-50 2026：0 次；DarkNESS 任务论文：2 次；明确呼吁重分析空间数据的 Niedermann & Sloth：2 次）几乎无人问津；P5 2023 与 APPEC 报告都没有把强相互作用暗物质或空间直接探测列为方向。

### Cited Findings

**(a) 可比论文的期刊、年份、引用数（INSPIRE，2026-11-18；"/yr"为按 arXiv 日期至今的年均，属【推断】的简单除法）**

| 论文 | 期刊（年） | 引用数 | 约 /yr | 性质 |
|---|---|---|---|---|
| Erickcek et al. 2007（XQC） | Phys. Rev. D 76, 042007 | 129 | 6.6 | 天文火箭数据 → 限制 |
| Emken & Kouvaris 2018（DaMaSCUS-CRUST） | Phys. Rev. D 97, 115047 | 154 | 17.6 | 方法 + 重算上界 |
| Kavanagh 2018（verne） | Phys. Rev. D 97, 123013 | 126 | 14.1 | 旧数据重分析 + 代码 |
| Mahdawi & Farrar 2018 | JCAP 10 (2018) 007 | 114 | 13.2 | XQC/DAMIC/CRESST 重分析 |
| Hooper & McDermott 2018 | Phys. Rev. D 97, 115006 | 86 | 9.8 | 重估 + 新信号 |
| Emken et al. 2019 | JCAP 09 (2019) 070 | 214 | 28.5 | 方法 + 气球/卫星提案 |
| Digman et al. 2019 | Phys. Rev. D 100, 063013 | 98 | 13.4 | 理论（改写版图） |
| Cappiello, Collar & Beacom 2021 | Phys. Rev. D 103, 023019 | 45 | 7.2 | 小型地表闪烁体新数据 |
| Bhoonah et al. 2021（etched plastic） | Phys. Rev. D 103, 103001 | 37 | 6.3 | Skylab/Ohya 重分析 |
| Acevedo, Bramante & Goodman 2021（mica） | JCAP 11 (2023) 085 | 45 | 8.2 | 旧云母数据重分析 |
| Ebadi et al. 2021（quartz） | Phys. Rev. D 104, 015041 | 42 | 7.6 | 新方法提案 |
| DEAP-3600 2022 | Phys. Rev. Lett. 128, 011801 | 86 | 16.4 | 首个普朗克质量多次散射 |
| XENON1T 2023（MIMP） | Phys. Rev. Lett. 130, 261002 | 35 | 9.8 | 同上，第二家 |
| LZ 2024（UHDM） | Phys. Rev. D 109, 112010 | 42 | 15.2 | 第三家 |
| CDEX-10 2025（UHDM） | Phys. Rev. D 113, 052011 | 1 | ~1 | 后来者 |
| DarkSide-50 2026（UHDM） | Phys. Rev. D 114, 052001 | 0 | 0 | 后来者 |
| Du et al. 2025（JWST） | Phys. Rev. Lett. 135, 051002 | 12 | 6.3 | 空间望远镜数据 → 限制 |
| DAMIC-M 2023（daily modulation） | Phys. Rev. Lett. 132, 101006 | 44 | 13.1 | 首个地下日调制限制 |
| SENSEI 2025（diurnal） | Phys. Rev. Lett. 136, 211001 | 12 | 11 | 日调制 |
| Carney et al. 2022（Snowmass UHDM 白皮书） | SciPost Phys. Core 6, 075 (2023) | 93 | 19.8 | 路线图 |
| Niedermann & Sloth 2025 | JCAP 05 (2025) 042 | 2 | 1 | 呼吁重分析空间数据 |
| DarkNESS 2025 | Adv. Space Res. 76, 18615 | 2 | ~1 | 任务论文 |
| Bianchi et al. 2024（SQM-ISS） | Sensors 24, 5090 | 2 | ~1 | 任务论文 |
| PAMELA 2015（SQM） | Phys. Rev. Lett. 115, 111101 | 21 | 1.9 | 空间奇异粒子上限 |
| DAMPE 2022（FCP） | Phys. Rev. D 106, 063026 | 16 | 3.8 | 空间奇异粒子上限 |

- 【文献证实】上表各行数据来源 — INSPIRE 记录：[748100](https://inspirehep.net/literature/748100)、[1654810](https://inspirehep.net/literature/1654810)、[1642997](https://inspirehep.net/literature/1642997)、[1666828](https://inspirehep.net/literature/1666828)、[1654360](https://inspirehep.net/literature/1654360)、[1735240](https://inspirehep.net/literature/1735240)、[1746261](https://inspirehep.net/literature/1746261)、[1813235](https://inspirehep.net/literature/1813235)、[1838365](https://inspirehep.net/literature/1838365)、[1863299](https://inspirehep.net/literature/1863299)、[1862616](https://inspirehep.net/literature/1862616)、[1909079](https://inspirehep.net/literature/1909079)、[2653323](https://inspirehep.net/literature/2653323)、[2758452](https://inspirehep.net/literature/2758452)、[3072958](https://inspirehep.net/literature/3072958)、[3118117](https://inspirehep.net/literature/3118117)、[2860455](https://inspirehep.net/literature/2860455)、[2677283](https://inspirehep.net/literature/2677283)、[3072967](https://inspirehep.net/literature/3072967)、[2050968](https://inspirehep.net/literature/2050968)、[2849499](https://inspirehep.net/literature/2849499)、[2924116](https://inspirehep.net/literature/2924116)、[2816062](https://inspirehep.net/literature/2816062)、[1392954](https://inspirehep.net/literature/1392954)、[2149558](https://inspirehep.net/literature/2149558)
- 【文献证实】其他参照：Collar 2018 的地表慢粒子搜索（Phys. Rev. D 98, 023005）94 次 — [1672020](https://inspirehep.net/literature/1672020)；Clark et al. 2020（Majorana/XENON1T 重铸，Phys. Rev. D 102, 123026）51 次 — [1817657](https://inspirehep.net/literature/1817657)；Bramante et al. 2018 多次散射前沿（Phys. Rev. D 98, 083516）94 次 — [1663565](https://inspirehep.net/literature/1663565)；Jacobs, Starkman & Lynn 2015 "Macro Dark Matter"（MNRAS 450, 3418）187 次 — [1320980](https://inspirehep.net/literature/1320980)；Dhakal et al. 2023 雷达流星（Phys. Rev. D 107, 043026）18 次 — [2152951](https://inspirehep.net/literature/2152951)；PandaX-II 2022 利用日调制的 CRDM 搜索（Phys. Rev. Lett.）97 次 — [1991901](https://inspirehep.net/literature/1991901)；DAMA 2014 日效应 82 次、DAMA 2015 地球遮挡 61 次 — [1286419](https://inspirehep.net/literature/1286419)、[1371825](https://inspirehep.net/literature/1371825)
- 【文献证实】JWST 结果的可见度：PRL + APS Physics 提要"JWST Looks Within for Dark Matter"（Physics 18, s105, 2025-07-31）+ 被 2026 版 PDG 综述与 2026-11 的直接探测综述引用 + 科技日报报道 — [refersto:recid:2860455](https://inspirehep.net/literature?q=refersto%3Arecid%3A2860455)；[cn.chinadaily.com.cn](https://cn.chinadaily.com.cn/a/202508/11/WS68b7aaa2a310f07257746795.html)

**(b) 综述与路线图怎么说**
- 【文献证实】Snowmass 2021 超重暗物质白皮书（Carney et al.；引用 93）：图 2 把现有限制分成宇宙学/天体物理、直接探测、中微子实验、"space-based experiments (purple)"和"terrestrial track-based observations"五类（图注列出的限制中来自空间数据的有 Skylab、XQC 与 IMP）；结论："It has already been demonstrated that dedicated analyses of existing data can be very fruitful in exploring new parameter space" — [arXiv:2203.06508](https://arxiv.org/abs/2203.06508)
- 【文献证实】Snowmass 2021 粒子暗物质专题组报告（Cooley et al.；引用 126）：总方针"delve deep … and search wide"，"A diverse, continuous portfolio of experiments at large, medium, and small scales"；超重暗物质"are best probed through their interactions with nucleons, using large detectors. Current and proposed detection techniques range from repurposing current large-volume liquid noble detectors to specialized impedance-matched mechanical sensors"；其图 10 沿用含"space-based experiments"的分类 — [arXiv:2209.07426](https://arxiv.org/abs/2209.07426)
- 【文献证实】P5 2023 报告：暗物质部分只在"IceCube, which has sensitivity to spin-dependent and ultra-heavy particle dark matter candidates"一句提到超重暗物质；新倡议"A Portfolio of Agile Projects for Dark Matter"聚焦"hidden-sector models and QCD axions"；全文关键词检索未见 strongly-interacting / composite dark matter 或空间直接探测 — [P5 2023, arXiv:2407.19176](https://arxiv.org/abs/2407.19176)
- 【文献证实】APPEC 直接探测委员会报告（Rep. Prog. Phys. 85, 056201；引用 516）建议 2："The diversified approach to probe the broadest experimentally accessible ranges of particle mass and interactions is needed to ensure the most conservative and least assumption-dependent exploration"；全文关键词检索未见 ultraheavy、strongly interacting、卫星/气球/火箭探测或 sidereal；"diurnal"只出现在所引 LUX 论文的标题里；提到过地表（above-ground）低温量能器运行的限制，但未把大截面暗物质作为方向讨论 — [Billard et al. 2022, arXiv:2104.07634](https://arxiv.org/abs/2104.07634)
- 【文献证实】2026-11 直接探测综述（§2.3.5）："Since for even larger couplings the atmosphere itself contributes to the shielding of dark matter experiments, it may be necessary to consider balloon- or satellite-based experiments, see Ref. [113]"（[113] 即 JWST 论文），随即提醒这类大耦合受天体物理与宇宙学（结构形成）约束 — [Kahlhoefer et al. 2026, arXiv:2611.21052](https://arxiv.org/abs/2611.21052)
- 【文献证实】Bramante 2026 年度评论（Ann. Rev. Nucl. Part. Sci. 76, 319；引用 10）："Underground detectors, coincidence arrays, and recasts of existing data now together constrain dark matter across nearly twenty orders of magnitude in mass and six in cross section"；"At large cross sections, attenuation suppresses the flux of dark matter reaching underground detectors, while shallow-site and high-altitude experiments remain sensitive to interactions near the geometric limit"；Chicago 符合实验"ruled out nucleon-independent cross sections σ_X ≳ 10⁻²² cm² for m_X ≲ 10¹² GeV"；蚀刻塑料"providing the leading limits in the geometric (nucleon-independent) regime"；Future Issues 第 2、4 条要求大体积探测器加入时间关联多次散射分析、并探索更多地质与天体物理方法；全文没有提到卫星闪烁体或轨道调制 — [arXiv:2602.23708](https://arxiv.org/abs/2602.23708)
- 【文献证实】与参数空间是否开放相关的两条提醒：Boukhtouchen et al. 2026 "revisit prior dark matter exclusions from etched mica searches, identifying shortcomings that compromise the robustness of these constraints" — [arXiv:2606.02579](https://arxiv.org/abs/2606.02579)；Niedermann & Sloth 指出宇宙学限制随丰度分数放松（Lyman-α 按 f³、CMB 按 f）— [arXiv:2411.10521](https://arxiv.org/abs/2411.10521)
- 【文献证实】中国：《国家空间科学中长期发展规划（2024—2050年）》（中国科学院、国家航天局、中国载人航天工程办公室，2024-10-15 发布）"极端宇宙"主题的"优先发展方向包括暗物质与极端宇宙、宇宙起源与演化、宇宙重子物质探测"；报道同时把"悟空"号暗物质粒子探测和"慧眼"并列为既有成果 — [人民日报 2024-10-16](https://paper.people.com.cn/rmrbwap/html/2024-10/16/nw.D110000renmrb_20241016_3-03.htm)

**(c) 活跃理论组（天然合作者或审稿人；单位取自 INSPIRE 作者档案，2026-11-18）**
- 【文献证实】Queen's University / McDonald Institute — Joseph Bramante（senior，2023–）及 Bhoonah、Boukhtouchen（博士生）、Picker（博士后，兼 Perimeter）等：蚀刻塑料、云母、松散束缚复合暗物质、macro 限制汇编（2024–2026 年多篇）— [arXiv:2602.23708](https://arxiv.org/abs/2602.23708)；[arXiv:2408.03983](https://arxiv.org/abs/2408.03983)；[arXiv:2512.16043](https://arxiv.org/abs/2512.16043)；[arXiv:2609.05626](https://arxiv.org/abs/2609.05626)
- 【文献证实】Ohio State CCAPP — John Beacom；Christopher Cappiello（现为 Washington University in St. Louis 博士后，2024–）；University of Chicago — Juan Collar（EJ-301 符合探测器）— [arXiv:2008.10646](https://arxiv.org/abs/2008.10646)；[INSPIRE 作者检索 Cappiello](https://inspirehep.net/authors?q=Cappiello%2C%20Christopher%20V.)
- 【文献证实】Stony Brook — Rouven Essig（SENSEI、DarkNESS、JWST、地球散射调制）；Timon Emken（INSPIRE 记录的最近职位为 Stockholm OKC 博士后 2021–2023，DaMaSCUS 作者）；Chris Kouvaris（NTU Athens）；Bradley Kavanagh（U. Cantabria, Santander，verne / Verne2）— [arXiv:2507.00344](https://arxiv.org/abs/2507.00344)；[arXiv:2511.10589](https://arxiv.org/abs/2511.10589)；[INSPIRE 作者检索 Kavanagh](https://inspirehep.net/authors?q=Kavanagh%2C%20Bradley%20J.)
- 【文献证实】NYU — Glennys Farrar（XQC 重分析、sexaquark、杜瓦实验）— [arXiv:2112.00707](https://arxiv.org/abs/2112.00707)；[arXiv:2101.00142](https://arxiv.org/abs/2101.00142)
- 【文献证实】IISc Bangalore — Nirmal Raj（单次散射"neutrino roof"、千吨探测器多次散射）；Hawaii — Jason Kumar（电磁相互作用超重暗物质）— [arXiv:2406.17015](https://arxiv.org/abs/2406.17015)；[arXiv:2410.22168](https://arxiv.org/abs/2410.22168)；[arXiv:2509.24938](https://arxiv.org/abs/2509.24938)
- 【文献证实】Rebecca Leane（SLAC）、Juri Smirnov 与 Tetiana Kozynets（Liverpool）："Dark Matter Weather" — [arXiv:2607.20620](https://arxiv.org/abs/2607.20620)；Carlos Blanco 与 Benjamin Lillard（Penn State）、Yonatan Kahn（Toronto）、Joshua Foster：日调制统计学 — [arXiv:2602.15947](https://arxiv.org/abs/2602.15947)
- 【文献证实】机械传感/悬浮探测超重暗物质 — Daniel Carney（LBNL）、Rafael Lang（Purdue）、Juehang Qin（现 Nanyang Technological U.）、Christopher Tunnell（Rice）、Amaral 等（2025–2026）— [arXiv:2503.11645](https://arxiv.org/abs/2503.11645)；[arXiv:2608.20464](https://arxiv.org/abs/2608.20464)
- 【文献证实】明确呼吁重分析空间数据的 Florian Niedermann（Nordita）与 Martin Sloth（Southern Denmark U.）— [arXiv:2411.10521](https://arxiv.org/abs/2411.10521)；SQM-ISS 的 Marco Casolino（INFN Roma Tor Vergata / RIKEN）— [INSPIRE 3067418](https://inspirehep.net/literature/3067418)
- 【文献证实】国内（姓名按论文署名的拼音）：Ningqiang Song（中国科学院理论物理研究所，2023–；蚀刻塑料与气体云论文的共同作者）— [arXiv:2012.13406](https://arxiv.org/abs/2012.13406)、[INSPIRE 作者检索](https://inspirehep.net/authors?q=Song%2C%20Ningqiang)；Peizhi Du（杜佩之，中国科大，2025–）与 Hailin Xu（李政道研究所博士后，2025–）（JWST 论文作者；2025-12-26 在李政道研究所报告该工作）— [TDLI indico](https://indico-tdli.sjtu.edu.cn/event/4612/contributions/18764/)；Shao-Feng Ge（李政道研究所/上海交大）、Qiang Yuan（紫金山天文台）、Ning Zhou（上海交大）、Jiang-Lai Liu（CRDM 日效应）— [arXiv:2005.09480](https://arxiv.org/abs/2005.09480)；Zuowei Liu（南京大学；XQC 与 CRESST 表面运行限制）— [arXiv:2209.04387](https://arxiv.org/abs/2209.04387)；Yu-Feng Zhou（中国科学院理论物理研究所）、Chen Xia、Mai Qiao（Migdal 日调制）— [arXiv:2307.12820](https://arxiv.org/abs/2307.12820)；Haipeng An（清华；Parker Solar Probe 暗光子）— [arXiv:2405.12285](https://arxiv.org/abs/2405.12285)；实验侧 CDEX（清华；地球屏蔽模拟、超重暗物质搜索）与 PandaX（上海交大；日调制 CRDM 搜索）— [arXiv:2510.21458](https://arxiv.org/abs/2510.21458)；[arXiv:2112.08957](https://arxiv.org/abs/2112.08957)
- 【文献证实】理论兴趣仍在延续的最新例子：Asadi, Chacko, Curtin, Mellors & Neil 2026-11-11 "The Dark Side of the Periodic Table: Theory and Signatures of Nuclear Atomic Dark Matter"（同时引用 Cappiello 2021 与 Bhoonah 2021）— [arXiv:2611.15852](https://arxiv.org/abs/2611.15852)

### Inferences
- 【推断】引用模式分三档：(1) 方法/代码/"改写版图"的论文（Emken 2019、Emken & Kouvaris 2018、Kavanagh 2018、Digman 2019）每年 13–29 次；(2) 给出一块新排除区的重分析（Erickcek、Cappiello、Bhoonah、Acevedo、JWST）每年 6–8 次；(3) 大合作组的"首个"多次散射结果每年 10–16 次，但第四、第五家已跌到 0–1 次。HXMT 零结果最可能落在第 2 档；若论文同时把"轨道调制 + 存档闪烁体"写成可复用的方法（给代码、给 GECAM/GBM 的推广方案），有机会进第 1 档。
- 【推断】长期引用主要来自"进了标准汇总图"：Skylab、XQC、IMP 至今仍是 Snowmass 图与 Bramante 综述图上的标注区域；HXMT 若给出新区域，会成为"space-based"类里继 Skylab、IMP、XQC（核子）与 JWST（电子）之后的又一条。
- 【推断】期刊预期：零结果首选 PRD 或 JCAP；若排除区在某个丰度分数下确属"此前全部允许"且是首个在轨调制搜索，可试投 PRL（DAMIC-M 日调制、SENSEI 日调制、JWST 都是 PRL）。慧眼合作组此前的仪器论文多发在《中国科学：物理学 力学 天文学》（凭记忆，未核实），也可作为备选。
- 【推断】审稿人几乎必然来自上面这十来个组，他们会追问：截面模型用几何截面还是 A⁴ 标度（Digman 2019 之后必须说明）；NaI/CsI/塑料对多次散射核反冲的响应与淬灭；与 Skylab、云母、Chicago、宇宙学（银河系卫星星系）限制的重叠；以及 10⁻⁵ 级系统误差如何受控。
- 【推断】路线图层面的"背书"只有原则性的（search wide、existing data fruitful、可能需要卫星实验），没有任何一份文件点名强相互作用暗物质的空间搜索为优先方向；这有利于论证"低成本补空白"，不利于论证"主流关注"。

### Gaps
- 中国暗物质专项规划（如锦屏、PandaX/CDEX 路线图）中是否提及强相互作用或空间直接探测，未检索到；《规划（2024—2050）》全文未取回，只有新闻摘要。
- 引用数只取了 INSPIRE，未与 ADS/Google Scholar 交叉；INSPIRE 对天文期刊的施引覆盖偏低，天文侧的引用可能被低估。
- Timon Emken 的现任单位未能确认（INSPIRE 作者档案只到 2023 年）；Jiang-Lai Liu、Chen Xia、Mai Qiao、Joshua Foster、Amaral 的单位未从作者档案核实。

---

## 7. 慧眼合作组或其他 X 射线任务是否发表过"直接探测式"暗物质限制；中国空间任务与强相互作用暗物质

### Takeaway
没有找到慧眼合作组或任何在轨 X/γ 射线天文卫星团队发表的直接探测式（暗物质粒子打在探测器上）限制；X 射线任务里唯一的先例是探空火箭 XQC，红外的是 JWST。中国空间任务中，DAMPE 做过相对论性分数电荷粒子搜索，但没有任何任务（DAMPE、GECAM、慧眼、空间站载荷、HERD）被讨论用于强相互作用或慢速重暗物质；国内与此最近的是地下的 CDEX、PandaX 工作和几位理论家。

### Cited Findings
- 【文献证实】arXiv 全学科与 INSPIRE 检索"dark matter" + HXMT：唯一命中是理论组借用慧眼已发表弥漫背景谱约束原初黑洞（Yang, et al., Commun. Theor. Phys. 78, 045402, 2026；引用 6），不是合作组工作，也不是直接探测 — [arXiv:2412.09297](https://arxiv.org/abs/2412.09297)
- 【文献证实】INSPIRE 检索 GECAM + (dark matter | axion | Lorentz | primordial black hole)：0 条 — [INSPIRE 检索](https://inspirehep.net/literature?q=abstracts.value%3AGECAM%20and%20abstracts.value%3A%22dark%20matter%22)
- 【文献证实】X 射线量能器的直接探测式限制只有 XQC 一脉（Wandelt 2000 → Erickcek 2007 → Mahdawi & Farrar 2017/2018 → Li, Liu & Xue 2023）— [arXiv:0704.0794](https://arxiv.org/abs/0704.0794)；[arXiv:2209.04387](https://arxiv.org/abs/2209.04387)
- 【文献证实】对 X 射线望远镜 CCD 的定性讨论只有 Niedermann & Sloth（Chandra，"pile-up event"）— [arXiv:2411.10521](https://arxiv.org/abs/2411.10521)
- 【文献证实】DAMPE："Search for relativistic fractionally charged particles in space"（Phys. Rev. D 106, 063026；5 年在轨数据，流量上限 6.2×10⁻¹⁰ cm⁻² sr⁻¹ s⁻¹）与 2026 年十年数据的 2/3 e 轻质量分数电荷粒子搜索（上限 5.0×10⁻¹¹ cm⁻² sr⁻¹ s⁻¹）；都是相对论性粒子 — [arXiv:2209.04260](https://arxiv.org/abs/2209.04260)；[arXiv:2602.20519](https://arxiv.org/abs/2602.20519)
- 【文献证实】HERD、"天宫/中国空间站"与暗物质的 INSPIRE 标题检索没有任何强相互作用暗物质或直接探测内容（命中为间接探测与巡天望远镜预报）— [INSPIRE 检索 HERD](https://inspirehep.net/literature?q=t%20HERD%20and%20t%20%22dark%20matter%22)
- 【文献证实】国内地下实验的相关工作：CDEX-10 超重暗物质限制（205.4 kg·day，m = 10⁶–10¹¹ GeV，含地球屏蔽蒙卡）— [arXiv:2510.21458](https://arxiv.org/abs/2510.21458)；CDEX 地球屏蔽研究 — [arXiv:2111.11243](https://arxiv.org/abs/2111.11243)；PandaX-II 用日调制的 CRDM 搜索 — [arXiv:2112.08957](https://arxiv.org/abs/2112.08957)；Bramante 综述称 PandaX-4T 的多次散射分析"are ongoing" — [arXiv:2602.23708](https://arxiv.org/abs/2602.23708)
- 【文献证实】国内作者把空间数据用于暗物质的先例：杜佩之等的 JWST 分析（见第 3 节）；An Haipeng 等用 Parker Solar Probe 原位数据约束暗光子暗物质（Phys. Rev. Lett. 134, 171001）— [arXiv:2405.12285](https://arxiv.org/abs/2405.12285)

### Inferences
- 【推断】如果做成，这将是慧眼数据的第一篇粒子暗物质直接探测论文，也是中国空间任务对强相互作用/复合暗物质的第一个限制；与"悟空"号的间接探测、DAMPE 的分数电荷粒子搜索形成互补，容易在国内获得关注（JWST 结果因作者在中国科大而上了科技日报）。
- 【推断】最自然的国内合作组合：慧眼 HE 仪器组（本底与响应）+ 中国科学院理论物理研究所 Ningqiang Song（蚀刻塑料/复合暗物质截面约定）或中国科大杜佩之 / 李政道研究所 Hailin Xu（空间探测器暗物质分析）+ CDEX 的地球屏蔽模拟经验。

### Gaps
- 慧眼官方论文清单本次未重新逐条核对（上一轮综述已核对过，结论为无）；合作组内部是否有人做过未发表的类似尝试无从得知。
- NuSTAR、Swift、INTEGRAL、AstroSat 等团队是否在会议上提过类似想法，未检索到。

---

## 8. 综合坦率评估：零结果与阳性结果各值多少

### Takeaway
零结果是一篇扎实但小众的论文：有明确的文献缺口可填（"In space, there are very few constraints"）、有 PRL 级先例（JWST）、有稳定的小社区会引用，现实预期是 PRD/JCAP、五年 30–80 次引用，前提是排除区确有新增（尤其是小丰度分数的重复合暗物质）。阳性结果的价值几乎完全取决于能否被另一颗卫星复现；单凭慧眼的纯调制证据，历史经验表明它会被当作未识别的轨道系统误差。想法本身不新，执行是新的；最大的非技术风险是被 DarkNESS 或 GBM 数据抢先。

### Cited Findings
- 【文献证实】缺口确实被文献点名："In space, there are very few constraints on the potentially harmful effects dark matter could have … a careful re-analysis of pile-up events in X-ray telescopes could exclude this possibility" — [Niedermann & Sloth 2025, arXiv:2411.10521](https://arxiv.org/abs/2411.10521)；"Significantly moving the diagonal edge would require a satellite or high-altitude balloon experiment" — [Cappiello, Collar & Beacom 2021, arXiv:2008.10646](https://arxiv.org/abs/2008.10646)；"it may be necessary to consider balloon- or satellite-based experiments" — [Kahlhoefer et al. 2026, arXiv:2611.21052](https://arxiv.org/abs/2611.21052)
- 【文献证实】但同一批作者也给出了降温的理由："for composite DM, terrestrial detectors can probe cross sections above cosmological constraints, reducing the need for rocket- or space-based experiments"；Chicago 排除区"is already partially covered by the cosmological constraints, though direct experimental sensitivity is always preferred" — [arXiv:2008.10646](https://arxiv.org/abs/2008.10646)
- 【文献证实】同类零结果的发表层级与反响：JWST 暗图像 PRL、23 个月 12 次引用；XQC PRD、19 年 129 次；Chicago PRD、45 次；Skylab/Ohya 重分析 PRD、37 次；CDEX-10 超重暗物质 PRD、约一年 1 次 — [2860455](https://inspirehep.net/literature/2860455)、[748100](https://inspirehep.net/literature/748100)、[1813235](https://inspirehep.net/literature/1813235)、[1838365](https://inspirehep.net/literature/1838365)、[3072958](https://inspirehep.net/literature/3072958)
- 【文献证实】调制搜索相对总率的增益有实例：DAMIC-M 同一批数据因利用时间依赖信号把限制改善约两个量级 — [arXiv:2307.07251](https://arxiv.org/abs/2307.07251)；DarkNESS 的"modulation"灵敏度线也远优于"rate"线 — [arXiv:2505.16981](https://arxiv.org/abs/2505.16981)
- 【文献证实】纯调制阳性结论的历史：DAMA 12.9σ 被同靶材实验以 4.7σ 排除；CoGeNT 2.8σ → <1σ（但原文引用 685 次）；XMM 11σ 季节调制 12 年 27 次引用 — [arXiv:2503.19559](https://arxiv.org/abs/2503.19559)；[arXiv:1405.0495](https://arxiv.org/abs/1405.0495)；[INSPIRE 912559](https://inspirehep.net/literature/912559)；[INSPIRE 1285332](https://inspirehep.net/literature/1285332)
- 【文献证实】复现是社区对任何异常的要求（Snowmass 2021）— [arXiv:2209.07426](https://arxiv.org/abs/2209.07426)

### Inferences
- 【推断】**零结果值多少**
  - 科学内容：在"空间类"限制里增加一条质量-时间曝光最大的记录；真正的新信息最可能出现在两处：(i) 丰度分数 f_χ ≪ 1 的重复合/强相互作用成分（宇宙学限制随 f 放松，而 HXMT 是通量受限型，曝光大就能压低 f）；(ii) 被质疑的旧限制（Skylab 质量范围已被下修、云母限制的稳健性 2026 年被质疑）所覆盖区域的独立复核。如果另一位研究者的约束图显示目标区在 f_χ = 1 时已被 Skylab + 宇宙学完全覆盖，论文就必须以 f_χ 版本和"方法首演"为主打，影响力相应下降一档。
  - 方法学价值：首次把 Emken 2019 的轨道调制判据在真实 LEO 数据上跑通，并量化 LEO 本底在恒星系里的地板；这对 DarkNESS、SQM-ISS 以及任何想用 GBM / GECAM / BATSE 存档的人都是直接可用的输入，是比排除曲线更可能被长期引用的部分。
  - 发表与引用预期：PRD/JCAP 为基线，PRL 为争取目标；五年 30–80 次引用（第 6 节第 2 档的 6–8 次/年到第 1 档的 13 次/年以上）。若只是"又一条被别的限制盖住的线"，可能与 CDEX-10、DarkSide-50 的超重暗物质论文一样只有个位数引用。
  - 附带价值：国内首例；可带动慧眼被 GTI 剔除数据与反符合事例的系统利用。
- 【推断】**阳性结果值多少**
  - 若出现与预言的轨道相位、66 天包络零点方位、三类探测器幅度比都吻合的恒星系调制，这在形式上是比 DAMA 更强的证据（方向性信息 + 可预言的开关包络）。
  - 但先验很低，且 LEO 上存在与恒星系锁定的真实天体物理调制（Cygnus 区亮源与银道面被地球掩食）。合理的发表方式是"报告一个未解释的恒星系调制残差 + 暗物质解释下的参数区 + 明确的复现预言"，而不是发现声明。
  - 价值兑现的必要条件是独立复现：另一颗倾角不同的卫星（GECAM、Fermi-GBM、AstroSat）按其自身轨道应看到不同包络但同一 apex；若复现，那就是暗物质的首次非引力探测；若不能复现，论文会获得不少引用（CoGeNT 模式），同时带来声誉成本（XMM、DAMA 模式）。
- 【推断】**弱点（坦率）**
  1. 思想有明确的在先发表（2019）和在建的专用任务（DarkNESS）；"新颖性"只能主张到"首次执行、首次用于重复合暗物质、首次用存档天文闪烁体"。
  2. 关注度有限：路线图不点名、近两年同方向后来者引用极低、最接近的 PRL（JWST）两年 12 次。
  3. 审稿难点集中在截面模型约定和 10⁻⁵ 系统误差：前者需要理论合作者，后者在已取回的文献里没有任何卫星分析做到过（Fermi-LAT 各向异性为 10⁻³–10⁻⁴）。
  4. 时效性：想法公开后，用 Fermi-GBM 数据（其公开性凭记忆，未核实）做同样分析的门槛很低；若决定做，宜尽快以方法论文或预印本确立优先权。
- 【猜测】最可能的实际结局是：系统误差地板把分数调制灵敏度限制在 10⁻³–10⁻⁴，仍能给出一块（主要在小 f_χ 或与旧限制重叠处的）排除区，论文发在 PRD，成为"空间闪烁体轨道调制法"的参考文献，五年内被 DarkNESS 首批结果和若干综述图引用。

### Gaps
- 零结果的真实新增面积取决于约束图、探测器响应和系统误差三位研究者的结论，本笔记无法独立判断。
- 没有找到任何关于"社区会如何看待 LEO 闪烁体上的暗物质阳性信号"的直接表述，相关判断全部来自类比。
- DarkNESS 的实际发射时间与首批结果时间表未知，"被抢先"的时间窗无法量化。
