# Insight-HXMT（慧眼）十年存档数据的非常规天体物理方法：调研笔记（截至 2026-11）

> **阅读说明**
> - 标签约定：**[confirmed in literature]** = 本次调研中已取回原文/摘要并核对；**[inference]** = 基于已核对事实的推断或本笔记自行做的量级估算（给出输入量）；**[speculation]** = 无直接文献支撑的设想。
> - **重要局限**：本次调研共享的 WebSearch 配额在完成 16 次检索后耗尽（系统提示 "web search budget is used up"）。此后只能用 WebFetch 逐篇取回已知 arXiv/GCN 页面核对，无法继续做关键词查新。因此下列方向的"HXMT 是否已做"**查新不完整**，在各节 Gaps 中逐条标出：CsI 脉冲星科学应用、slew survey、TGF 专文、sterile neutrino/3.5 keV 谱线、Crab 星云流量变化、CsI 双事件偏振、高阶统计量（bispectrum）、EP×HXMT 系统性联合、KM3-230213A 存档核查、GECAM 的 EOT 工作、HXMT 2026 年在轨状态。
> - ADS 页面（ui.adsabs.harvard.edu）WebFetch 返回 405，因此凡只给 ADS 链接的条目均来自检索结果摘要而非本人取回的原文。
> - 文中"本笔记计算"指用 astropy 8.0（内置 ERFA 月球历表）与简单 Python 脚本做的估算，脚本未入库，输入量均写在正文。

---

## 0. 总览：哪些非常规方法最值得做？（想法总表 + HXMT 参数核对）

### Takeaway
最值得优先尝试的是把 HE/CsI 这块"反符合屏蔽晶体"当作 **BATSE 级的 0.2–3 MeV 全天探测器**来用（地球掩食台阶拟合、全天脉冲折叠），以及把 **Crab 定标数据、空白天区/盲探测器数据**从"定标/本底"翻转为"信号"（脉冲归一化的星云流量、LE 大集光率弥散谱线搜寻）；月掩源是一个几乎零成本、可立即用观测日志验证的"彩票"。用户给出的参数基本正确，但有 4 处需要修正/补充（见 Inferences 末尾）。

### Cited Findings
- [confirmed in literature] 2017-06-15 发射；主带 1–250 keV，全天监测 0.2–3 MeV；观测模式为 pointing、scanning、GRB — [Zhang et al. 2020, SCPMA 63, 249502, arXiv:1910.09613](https://arxiv.org/abs/1910.09613)
- [confirmed in literature] 轨道高度 550 km、倾角 43° — [Yu et al. 2022, AMT 15, 3141](https://amt.copernicus.org/articles/15/3141/2022/)；导航论文给出半长轴 6922.88 km — [Zheng et al. 2019, arXiv:1908.01922](https://arxiv.org/pdf/1908.01922)
- [confirmed in literature] HE = 18 个 NaI(Tl)/CsI(Na) phoswich，单晶直径 190 mm、几何面积 283.5 cm²，总 ~5100 cm²；NaI 厚 3.5 mm、CsI 厚 40 mm；15 个窄视场 1.1°×5.7°、2 个宽视场 5.7°×5.7°、1 个用 2 mm 钽盖遮挡的 blocked 准直器；18 块塑闪反符合（HVT：顶 6 + 侧 12）；计时精度 <10 μs，单事件死时间 <10 μs，最大计数率 >30,000 cnts/s；"The mean effective area is about 1000 cm² for photons in ~200–3000 keV" — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/abs/1910.04955)
- [confirmed in literature] HE 每个 good event 数据包含：8 bit 能量、8 bit 脉冲宽度、18 bit HVT 标记、5 bit 通道号、2 bit 事件类型、19 bit 亚秒时间（2 μs 时钟）；18 个探测器分 3 组（每组 6 个，共用一个 PDAU），**同组探测器在事件处理期间（约 4–8 μs）的同时信号被丢弃** — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)
- [confirmed in literature] HE 三种工作模式：Normal（NaI 20–250 keV 准直观测 + CsI 40–800 keV 近全天监测）；Low Gain（CsI 约 0.2–3 MeV，"usually scheduled when the Insight-HXMT runs into the earth occultation"，即指向目标被地球遮挡时）；SAA（高压关闭，无科学数据）。NaI 事件取脉宽道 54–70，CsI 事件取 80–120（LGM 为 78–120）；累计死时间占比约 2%–6%；CsI 增益持续漂移，LGM 高压在 2017-07、2020-06、2021-10 调整过 — [Li X.-F. et al. 2023, "In-orbit performance of HE onboard Insight-HXMT in the first 5 years", arXiv:2302.03859](https://arxiv.org/abs/2302.03859)
- [confirmed in literature] 只有能量 ≳200 keV 的光子才能从任意方向穿透卫星结构在 CsI 中沉积能量；CsI 沉积能量范围 normal gain 约 40–600 keV、low gain 约 200 keV–3 MeV；**卫星没有星上实时触发，全部事例数据下传后在地面做 blind/targeted 搜索**；targeted 相干搜索灵敏度约 1.5×10⁻⁸ erg cm⁻²（200 keV–3 MeV）；已对 O2 与 O3a 的 GW 事件做搜索，无显著对应体 — [Cai et al. 2021, MNRAS 508, 3910, arXiv:2109.12270](https://arxiv.org/abs/2109.12270)
- [confirmed in literature] 前四年离线搜索共找到 660 个暴：322 个 GRB、245 个 TGF、4 个太阳耀斑、33 个带电粒子事件、**56 个未分类**；对 41 个高能中微子、48 个 GW、39 个 FRB 做了 targeted 搜索；GRB 能谱分析采用 NG 150–800 keV、LG 200–3000 keV；与 HVT 符合的事例及同时在 NaI 有沉积的事例被剔除 — [Song et al. 2022, "The First Insight-HXMT Gamma-Ray Burst Catalog: The First Four Years", arXiv:2203.00878](https://arxiv.org/abs/2203.00878)
- [confirmed in literature] CsI 响应由 Geant4 质量模型给出，并用 **不同入射方向上探测到的 Crab 脉冲辐射**检验（观测/模拟比的内禀弥散 Su<0.3，θ<90° 时接近 1，其他方向偏大）；与 GBM/BAT/Konus-Wind 联合拟合 10 个亮暴，HXMT/CsI 流量系统偏高 7.0±8.8% — [Luo et al. 2020, arXiv:2005.01367](https://arxiv.org/abs/2005.01367)
- [confirmed in literature] 本地时钟刻度：HE 2 μs、ME 6 μs、LE 1 μs；相对 NICER 的 ToA 偏移 HE 24.7 μs、ME 10.1 μs、LE 864.7 μs（LE 的延迟主要来自 SCD 读出机制）；修正后计时系统误差 12.1/8.6/15.8 μs；Jodrell Bank 历表自身系统误差约 40 μs — [Tuo et al. 2022, ApJS, arXiv:2109.04709](https://arxiv.org/abs/2109.04709)
- [confirmed in literature] 定标采用的 Crab 模型：wabs×power law，Γ=2.11，N=8.76 keV⁻¹ cm⁻² s⁻¹，N_H=3.6×10²¹ cm⁻²；定标能段 HE 28–250、ME 10–35、LE 1–10 keV；Table 1 给出 open area 4270/850/300 cm² — [Li X.-B. et al. 2020, "In-flight calibration of the Insight-Hard X-ray Modulation Telescope", arXiv:2003.06998](https://arxiv.org/pdf/2003.06998)
- [confirmed in literature] ME：窄视场 1°×4°、宽视场 4°×4°、含 blind 模块；LE：63 个模块 1.6°×6°（其中 3 个 blind）、21 个大视场模块（其中 3 个 blind）、6 个 60°×2.5° 超大视场模块；HE 的 blind 探测器因能量分辨和遮盖不同，**不能直接当作其他 HE 探测器的本底** — [Huang et al. 2022, MNRAS 513, 4074, arXiv:2306.10114](https://arxiv.org/pdf/2306.10114)
- [confirmed in literature] 太阳夹角约束 >70°；扫描速度 0.01/0.03/0.06 ° s⁻¹，单次扫描 2–3 h — [Sai et al. 2020, arXiv:2003.07741](https://arxiv.org/pdf/2003.07741)

### Inferences
**想法总表（评分 1–5；N=新颖性，F=可行性，I=潜在影响）**

| # | 想法 | 所在章节 | N | F | I | 一句话理由 |
|---|---|---|---|---|---|---|
| 1 | CsI 地球掩食台阶拟合：0.2–3 MeV 全天监测/成像 | §1 | 4 | 3 | 4 | 两次检索未见 HXMT/CsI 的 EOT；统计上十年可达 ~10–50 mCrab，但响应/模式切换系统误差重 |
| 2 | 月掩 Crab 及黄道带亮源（已算出窗口） | §2 | 5 | 2 | 3 | 2019-03 至 2020-09 月球 21 次经过 Crab 的 LEO 可掩范围；是否恰有指向数据未知 |
| 3 | 外部星表触发的 CsI 群体叠加（BBH/FRB/EP/中微子） | §3 | 2 | 5 | 3 | 管线现成、逐事件已做，群体级叠加未见；科学增量中等 |
| 4 | 射电巨脉冲 × HE 硬 X 射线增强叠加 | §3 | 4 | 3 | 4 | NICER 3.8%（软 X）与 Suzaku 21.5%（2.7σ，硬 X）之间的能量依赖只有 HE 级面积能判定；取决于射电数据 |
| 5 | 毫秒级自相关"透镜回波"：10–10⁶ M☉ 致密天体 | §4 | 4 | 4 | 3 | CsI 在 >200 keV 的高计数率+μs 时标适合 ms 延迟；样本量小于 GBM |
| 6 | "56 个未分类暴"再分类与河外磁星巨耀发候选 | §4 | 3 | 4 | 3 | 官方星表留下的现成"废料"；需 IPN/邻近星系交叉 |
| 7 | 用脉冲星脉冲流量归一化的 Crab 星云流量十年监测 | §5 | 4 | 5 | 3 | 有效面积漂移在比值中一阶抵消；单次 10 ks 精度 ~0.3% |
| 8 | CsI 全天脉冲折叠：MeV 脉冲星/磁星硬尾 | §5 | 4 | 4 | 4 | Crab 脉冲已被用于 CsI 定标但未见科学利用；十年 5σ 脉冲流量 ~1.4×10⁻¹¹ erg cm⁻² s⁻¹ |
| 9 | "三孔径差分 + 临边扫描"：CXB 归一化与地球反照/大气辐射的太阳周调制 | §7 | 3 | 4 | 3 | 已有结果只用 2 个宽视场探测器、前 ~3.5 年数据 |
| 10 | LE 大集光率弥散谱线搜寻（3.5 keV / sterile neutrino） | §7 | 4* | 3 | 4 | LE 的 AΩ 比 XMM/NuSTAR 高一个量级以上；*查新未完成 |
| 11 | 银道面扫描叠加的 20–100 keV GRXE 纬度剖面与 P(D) | §7 | 3 | 3 | 2 | LE 已做，ME/HE 未见；INTEGRAL 已有同类结果 |
| 12 | 扫描/机动数据的"三视场判别"短时标暂现源目录 | §8 | 3 | 3 | 2 | 团队自述有未报道的短时标候选体；对河外 FXT 不占优 |

**对用户所给参数的核对（需修正/补充处）**
- [inference] "CsI 正常模式 ~40–800 keV 近全天监测"需加限定：40–800 keV 是**沉积能量**范围，真正能"近全天"的只有 ≳200 keV 的入射光子（低能光子穿不透卫星结构）；团队能谱分析实际采用 150–800 keV（NG）与 200–3000 keV（LG）。
- [inference] 低增益（0.2–3 MeV）模式不是随机穿插的，而是**通常安排在指向目标被地球遮挡期间**。这意味着"地球遮挡时段"恰恰是 MeV 段数据的主要来源，也意味着任何掩食台阶分析都必须处理与目标掩食同步的高压/增益切换。
- [inference] "高时间分辨、无堆积"成立，但不等于无饱和：极亮暴会造成数据丢失（星表中的 'IRON' 样本），且同组 6 个 HE 探测器共享处理单元，同时信号会丢失。LE 虽然时钟刻度 1 μs，但 SCD 读出带来 ~0.86 ms 的延迟分布，有效时间分辨在毫秒量级 [inference]。
- [inference] "每台望远镜都有小/大/全遮挡视场"成立，但 HE 的 blind 探测器已被证明不能直接作本底模板；LE 另有 6 个 60°×2.5° 超大视场模块。LE 大视场尺寸文献不一致：Huang et al. 2022 写 6°×6°，Li et al. 2020 Table 1 写 4°×6°。
- [inference] HE 总几何面积 5100 cm²（用户写 ~5000 cm²，可接受）；Li et al. 2020 的 open area 为 4270 cm²。

### Gaps
- 无法确认 2026-11 时 HXMT 的在轨状态、AO 轮次与数据公开政策：官方英文站 http://hxmten.ihep.ac.cn/ WebFetch 返回 503，且检索配额已用尽。
- CsI 事例（含 LG 模式与非 GTI 时段）是否全部包含在公开的 Level-1 HE 事例文件中：Huang et al. 2022 使用了"all public Earth-occultation observations"，间接说明地球遮挡段在公开数据里，但未找到明确的数据政策文字。
- 以上 12 个想法中，#7、#8、#10、#12 的 HXMT 查新没有做完专门检索（配额耗尽），评分中的 N 值应视为上限。

---

## 1. 地球掩食技术（EOT）：CsI 台阶拟合全天监测/成像；准直望远镜的临边掩食

### Takeaway
两次针对性检索只找到 HXMT 用准直望远镜（LE/ME/HE-NaI）观测 Crab 掩食来反演**大气密度**的工作，没有找到用非准直 CsI 做 BATSE/GBM 式天体源掩食监测或 0.1–3 MeV 长期光变的论文。量级估算表明十年统计灵敏度在 200–600 keV 可到 ~10 mCrab（5σ，仅统计），是 GBM 在 >300 keV 只测到 Crab 与 Cyg X-1 之后的一个实质补充，但系统误差（穿透响应、模式切换）是成败关键。

### Cited Findings
- [confirmed in literature] BATSE EOT 方法学：源经过地球临边时计数率出现台阶，可作 ~20 keV–1 MeV 全天监测 — [Harmon et al. 2002, ApJS 138, 149, arXiv:astro-ph/0109069](https://arxiv.org/abs/astro-ph/0109069)
- [confirmed in literature] BATSE 9.1 年 EOT 星表：179 个天体的 deep sample 中 83 个确定探测，四个能段 20–40/40–70/70–160/160–430 keV；5σ 平均流量极限 3.5–20 mCrab，取决于随天区变化的系统误差 — [Harmon et al. 2004, arXiv:astro-ph/0404453](https://arxiv.org/abs/astro-ph/0404453)
- [confirmed in literature] Fermi-GBM 前两年：>100 keV 探测到 8 个源（Crab、Cyg X-1、Swift J1753.5−0127、1E 1740-29、Cen A、GRS 1915+105、XTE J1752−223、GX 339−4），>300 keV 只有 Crab 与 Cyg X-1 — [Case et al. 2011, ApJ 729, 105, arXiv:1009.4953](https://arxiv.org/abs/1009.4953)
- [confirmed in literature] GBM 三年：监测 209 个源、探测到 99 个；100–300 keV 9 个源，300–500 keV 仅 Crab 与 Cyg X-1；"the only all-sky monitor above 100 keV" — [Wilson-Hodge et al. 2012, ApJS 201, 33, arXiv:1201.3585](https://arxiv.org/abs/1201.3585)
- [confirmed in literature] GBM 掩食成像 IDEOM：利用轨道进动做层析，~4 年数据在 12–50/50–100/100–300 keV 分别探测到 57/23/7 个源，补入 17 个原星表外的源 — [Rodi et al. 2014, A&A, arXiv:1304.1478](https://arxiv.org/abs/1304.1478)
- [confirmed in literature] HXMT 已发表的掩食应用是大气密度反演：115 组 Crab 掩食数据（55–100 km 密度比 NRLMSISE-00 低 10%–25%）— [Xue et al. 2023, ApJS 264, 5, arXiv:2211.00293](https://web3.arxiv.org/pdf/2211.00293)；2022–2024 年 83 组掩食、三台望远镜、55–130 km — [Zhang H.-H. et al. 2025, arXiv:2502.19678](https://arxiv.org/abs/2502.19678)；LE 单次掩食的方法学 — [Yu et al. 2022, AMT 15, 3141](https://amt.copernicus.org/articles/15/3141/2022/)
- [confirmed in literature] HXMT 团队在测 CXB 时把 HE 的地球遮挡段当本底：从前两年空白天区收集了 166 段掩食进出过程，按 ELV<−6（遮挡）与 ELV>10（空白天）分段 — [Huang et al. 2022, arXiv:2306.10114](https://arxiv.org/pdf/2306.10114)
- [confirmed in literature] CsI 平均有效面积 ~1000 cm²（200–3000 keV）— [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)

### Inferences
#### 想法 1：CsI 地球掩食台阶拟合 —— 0.2–3 MeV 全天监测与层析成像
1. **一句话**：把 18 块 CsI 的全部事例按能段做成连续计数率，在已知源的地球临边升/落时刻拟合台阶，得到 Crab、Cyg X-1、亮黑洞暂现源等在 200 keV–3 MeV 的十年光变和平均谱，并用 66 天轨道进动做掩食层析成像。
2. **原理与来源**：CGRO/BATSE EOT（Harmon et al. 2002, 2004）、Fermi-GBM EOT（Case et al. 2011；Wilson-Hodge et al. 2012）与 IDEOM 成像（Rodi et al. 2014）。[confirmed in literature]
3. **利用的 HXMT 特性与适配度**：40 mm 厚 CsI、~1000 cm² 平均有效面积、全事例下传、无星上触发 [confirmed in literature]。在 >300 keV，GBM 的 NaI 很薄、BGO 只有两块，所以 HXMT/CsI 的有效面积应明显更大 [inference]；HXMT 指向观测时姿态惯性固定，源在载荷坐标系中的方向在一次观测内不变，比 Fermi 的摇摆扫天更利于台阶建模 [inference]。**不占优之处**：响应要穿过卫星结构，质量模型弥散达 ~30%，背向入射更差 [confirmed in literature]；<200 keV 基本不可用，所以它不是 GBM/BAT 的替代，只是 >200 keV 的补充 [inference]。
4. **数据**：HE Level-1 事例文件（按脉宽道 80–120 选 CsI 事例）、高压/模式状态、轨道与姿态文件 [confirmed in literature 关于选道]；地球遮挡段在公开数据中可得（Huang et al. 2022 使用过）[confirmed in literature]；是否所有 LG 段都随指向观测归档——未知。
5. **查新**：检索 "Insight-HXMT Earth occultation technique CsI" 与 "Insight-HXMT GECAM Earth occultation all-sky monitoring hard X-ray sources CsI light curves" 均只返回大气密度工作与 GBM 先例，未见 HXMT/CsI 的 EOT 星表或 100 keV–MeV 长期光变 [confirmed in literature：检索未见]。GECAM 是否做过 EOT 未能查证。
6. **可行性与量级估计**（本笔记计算）[inference]：
   - 输入：A_eff=1000 cm²；CsI 18 块合计本底 ~10⁴ cts/s（从 Song et al. 2022 Fig. 17 与 Cai et al. 2021 光变图纵轴读出）；假设其中 40% 落在 200–600 keV（即 B≈4000 cts/s，这是假设）；台阶两侧各取 100 s；Crab 流量用 HXMT 定标模型（Γ=2.11，N=8.76）外推且不加高能拐折（会高估）。
   - 200–600 keV：Crab 台阶 ≈15.5 cts/s，单台阶噪声 ≈8.9 cts/s，单台阶 S/N≈1.7；一天约 15 个可用台阶 → Crab 日 S/N≈7（日 1σ≈0.15 Crab）；十年约 5.5×10⁴ 个台阶 → 统计 5σ≈12 mCrab。
   - 600–3000 keV（假设 B≈3000 cts/s）：单台阶 S/N≈0.7；十年统计 5σ≈30–50 mCrab（若只有 LG 段可用取上限）。
   - 轨道：周期 95.6 min，升交点进动 −5.46°/day，进动周期 66 d，地球半张角 67°；赤纬 |δ|≳24° 的源在每个进动周期内有一段不发生掩食（Cyg X-1 受影响，Crab 基本不受影响）。
   - 主要障碍/系统误差：(a) LG/NG 切换与目标掩食同步，会在目标自身掩食时刻附近制造假台阶；(b) CsI 增益漂移与三次高压调整；(c) 方向相关响应的 ~30% 不确定度——可用 Crab 台阶幅度随入射方向 (θ,φ) 的经验标定来自校准；(d) 源混淆（银心方向）与本底曲率，BATSE 的经验是系统误差决定 3.5–20 mCrab 的下限；(e) 带电粒子造成的 spike。
   - 现实预期：十年平均灵敏度落在几十 mCrab（200–600 keV），足以给出 Crab、Cyg X-1 的 MeV 段十年谱与光变，并对亮暂现源（爆发期达 Crab 量级者）给出逐日的 >200 keV 光变。
7. **评分**：N4 / F3 / I4 —— >300 keV 的长期监测自 BATSE 之后几乎空白，但要先解决响应与模式切换两大系统误差。

#### 附：准直望远镜对指向目标的临边掩食（不单列为想法）
- [inference] 这条路已被团队用于大气密度（非天体物理）；用于天体物理（源尺寸/位置、拥挤场分解）不占优：大气消光边缘在切点处的标高约几 km，对应角尺度约 0.1°–0.2°，远差于 NuSTAR/INTEGRAL 成像，且 HXMT 的扫描 PSF 拟合已能分解 0.6° 以上的源对。建议否决（见 §12）。

### Gaps
- GECAM、POLAR、AstroSat/CZTI 的 EOT 先例：检索结果仅提到 "Singhal et al. 2021" 的 CZTI 掩食方法，未取回原文，无法给出 URL。
- CsI 在 200–600 keV、600–3000 keV 各能段的真实本底计数率与随地磁纬度的变化未找到数值，上面的 4000/3000 cts/s 是假设；这是灵敏度估计中最不确定的输入。
- Crab 在 >100 keV 的谱拐折未在本次取回的文献中核对，流量外推偏乐观。

---

## 2. 月掩与行星掩：准直视场内的亚角秒尺度测量

### Takeaway
检索未发现任何 HXMT 月掩结果。本笔记用历表计算发现：在 HXMT 任务期内，月球确实多次进入 Crab、1A 0535+262、GX 5−1、GX 9+1、GRS 1758−258 等亮源的"LEO 可掩范围"，其中 **Crab 的窗口是 2019-03 至 2020-09（21 次经过，5 次为地心真掩食）**；是否恰好有 HXMT 指向数据，需要用观测日志和轨道文件核对。

### Cited Findings
- [confirmed in literature] 月掩是 X 射线天文早期确定 Crab X 射线源位置/延展的手段（Bowyer et al. 1964, "Lunar Occultation of X-ray Emission from the Crab Nebula", Science 146, 912）；Crab 距黄道约 1.5°；Chandra 观测过土卫六 Titan 凌 Crab，由 X 射线阴影测得大气厚度 880 km — [Wikipedia: Crab Nebula（二手来源，原始论文未取回）](https://en.wikipedia.org/wiki/Crab_Nebula)
- [confirmed in literature] NuSTAR 在 3–78 keV 测得 Crab 星云（torus）尺寸随能量按幂律收缩，指数 γ=0.094±0.018，NW 方向（counter-jet）约为两倍 — [Madsen et al. 2015, ApJ 801, 66, arXiv:1502.07765](https://arxiv.org/abs/1502.07765)
- [confirmed in literature] 检索 "Insight-HXMT lunar occultation Moon" 只返回地球掩食/大气密度论文，没有月掩结果 — [检索结果中最接近的条目：Xue et al. 2023, arXiv:2211.00293](https://web3.arxiv.org/pdf/2211.00293)
- [confirmed in literature] HXMT 的太阳夹角约束为 >70° — [Sai et al. 2020, arXiv:2003.07741](https://arxiv.org/pdf/2003.07741)

### Inferences
#### 想法 2：在存档中"捡"月掩事件 —— Crab 星云 >80 keV 尺寸及黄道带亮源
1. **一句话**：用月球边缘作刀口，对恰好处在 HXMT 准直视场中心的亮源做一维扫描，首要目标是 Crab 星云在 NuSTAR 上限（78 keV）以上的尺寸。
2. **原理与来源**：月掩刀口法（1964 年火箭实验；1970 年代气球硬 X 射线实验——后者本次未取回文献）；物理检验对象是 Kennel–Coroniti 型同步辐射冷却导致的"星云随能量收缩"，NuSTAR 给出 γ=0.094±0.018（3–78 keV）[confirmed in literature]。
3. **HXMT 特性与适配度**：HE 20–250 keV、~10³ cts/s 的 Crab 计数率 [inference：由定标模型 0.27 ph cm⁻² s⁻¹ × open area 4270 cm² × 效率估计]、2 μs 计时、无堆积。准直望远镜没有成像能力，月掩是它获得角秒级信息的唯一途径；>80 keV 没有任何聚焦望远镜，所以**在这个能段 HXMT 确实比 NuSTAR/Chandra 更合适**，但前提是事件存在 [inference]。对点源而言月掩只给位置，如今已无科学价值。
4. **数据**：Crab（及下表各源）指向/小天区扫描的 HE/ME/LE 事例文件、精确轨道文件（用于计算卫星处的月球视差位置）；公开性同常规指向数据。
5. **查新**：见 Cited Findings，未见 HXMT 月掩工作 [confirmed in literature：检索未见]。RXTE 是否有 Crab 月掩先例——未能查证。
6. **可行性**（本笔记计算）[inference]：
   - 方法：astropy 内置历表，10 min 步长，2017-06-15 至 2026-11-17；判据为地心月–源角距 < 月球视半径（~0.26°）+ 550 km 轨道的最大视差（~1.03°）≈1.29°，表示"在轨道某些位置可能看到掩食"。这只是筛选，不是 HXMT 实际掩食预报。源坐标凭记忆输入（误差 ≲0.05°，不影响窗口判断）。

   | 源 | 窗口（首次–末次经过） | 经过次数 | 其中地心真掩食 |
   |---|---|---|---|
   | Crab | 2019-03-14 – 2020-09-10 | 21 | 5（2019-10-19、11-15、12-12、2020-01-08、02-05） |
   | 1A 0535+262 | 2021-09-28 – 2023-07-15 | 25 | 6 |
   | Sco X-1 | 2017-07-05 – 2018-08-19 | 15 | 0（最小角距 0.56°，仅靠视差） |
   | GX 9+1 | 2017-09-27 – 2019-04-24 | 22 | 5 |
   | MAXI J1810−222 | 2018-11-11 – 2020-04-13 | 20 | 4 |
   | IC 443 | 2018-11-25 – 2020-07-18 | 23 | 2 |
   | GS 1826−238 | 2019-10-05 – 2021-02-08 | 19 | 5 |
   | GX 5−1 | 2020-09-23 – 2022-02-25 | 20 | 5 |
   | GX 1+4 | 2020-10-20 – 2022-07-11 | 24 | 5 |
   | GRS 1758−258 | 2020-11-17 – 2022-09-05 | 25 | 5 |
   | GRS 1716−249 | 2021-04-02 – 2022-12-22 | 24 | 6 |
   | GX 3+1 | 2021-09-14 – 2023（2026-11 起新一轮） | 28 | 6 |
   | Sgr A* 区域 | 2023-07-29 – 2026-10-16 | 44 | 0（最小 0.32°） |
   | 1E 1740.7−2942 | 2024-08 – 2025-09 | 11 | 0 |
   | 3C 273 | 2017-07-28 – 2022-07-06 | 67 | 28 |

   - Crab 的 21 次经过中，扣除太阳夹角 <70° 的季节（约 4 月初–8 月下旬），可用的约 11 次：2019-03-14、2019-09-21、10-19、11-15、12-12、2020-01-08、02-05、03-03、03-31、2020-09-10（及临界的 2019-08-25）。
   - 每次经过持续约 4–5 h（月球 0.55°/h），约 3 个轨道；卫星轨道运动造成的月球视差速度最大约 4″/s，远快于月球自身的 0.55″/s，所以星云（~1′）的进/出掩只需 ~10–25 s。
   - 边缘定位精度 σ_t≈√((B+S/2)τ)/S：全能段取 S=1000、B=500 cts/s（B 为假设）、τ=20 s，得 0.14 s ≈0.6″；>80 keV 取 S≈90、B≈150 cts/s，得 0.7 s ≈3″。NuSTAR 的指数外推到 30→150 keV 预言尺寸缩小约 14%（~70″ 的 FWHM 上约 10″），单次进/出掩在 >80 keV 约 3σ 可辨，多次叠加更好。
   - 主要障碍：(a) 完全取决于是否恰有指向——若 Crab 定标观测占任务时间的 ~5%–10%（假设），期望重合数只有 ~1–2 次；(b) HXMT 是否有月球规避约束——未知；(c) 进/出掩方向每次不同，只能得到沿若干方位角的一维轮廓；(d) 脉冲星与星云要用相位分解分离（反而是优点：脉冲星是点源，可当作刀口响应的在轨标定）。
   - 成本极低：把上表日期与观测日志交叉即可在一天内判定有无事件。
7. **评分**：N5 / F2 / I3 —— 新颖、便宜，但成功与否取决于运气；银心方向的窗口（2023–2026）对分解 1°×5.7° 视场内的源也许有用。

### Gaps
- 未能取回 1970 年代 Crab 月掩硬 X 射线尺寸测量的原始论文（ADS 页面返回 405），也未能核实 RXTE、NuSTAR 是否做过 Crab 月掩。
- HXMT 的月球规避角、Crab 定标观测的具体日期均未知。
- 行星/小行星掩 X 射线源的概率极低，未做估算。

---

## 3. 叠加分析：对单个探测不到的外部样本做群体级搜索

### Takeaway
HXMT 团队已对 GW（O2+O3a，星表文中计 48 个）、高能中微子（41 个）、FRB（39 个）逐事件做过 targeted 搜索并给上限，Guidorzi et al. 还看过"cumulative time profiles"；但没有找到对 O3b/O4 全样本、CHIME/FRB 大样本或 EP 暂现源做**群体级似然叠加**的论文。更"非常规"且物理指向明确的是把叠加对象换成 Crab 射电巨脉冲。

### Cited Findings
- [confirmed in literature] HXMT/HE 对 39 个 FRB 的搜索：从数秒到亚毫秒时标，检查了 "both individual and cumulative time profiles"，无信号；FRB 180916.J0158+65 的 1 s 上限 L_γ,iso<5.5×10⁴⁷ erg/s，其余多为 10⁴⁹–10⁵¹ erg/s，毫秒时标上限高约 100 倍 — [Guidorzi et al. 2020, A&A 637, A69, arXiv:2003.10889](https://arxiv.org/abs/2003.10889)
- [confirmed in literature] 对 SGR 1935+2154 的 FAST 射电暴，HXMT 给出 1 s 时标 3σ 流量上限约 2.7×10⁻⁹（LE）至 4.5×10⁻⁹ erg cm⁻²（HE）— [GCN 27679](https://gcn.nasa.gov/circulars/27679)
- [confirmed in literature] FRB 200428 的 X 射线暴由 HXMT 探测到双峰结构，HE 在该暴期间饱和 — [Ge et al. 2023, arXiv:2302.00176](https://ar5iv.labs.arxiv.org/html/2302.00176)
- [confirmed in literature] GW 搜索覆盖 O2 与 O3a，无显著对应体；O4 期间仍以 GCN 形式逐事件报告（例如 S250223dk 的 targeted 搜索给出 ~4σ 的 1 s 候选体，"very preliminary"）— [Cai et al. 2021, arXiv:2109.12270](https://arxiv.org/abs/2109.12270)；[GCN 39508](https://gcn.nasa.gov/circulars/39508?page=25)
- [confirmed in literature] IceCube 警报的逐事件上限（如 IceCube-190712A、IceCube-201209A：T0±100 s 内无 >3σ 超出）— [GCN 25066](https://gcn.nasa.gov/circulars/25066)；[GCN 29049](https://gcn.nasa.gov/circulars/29049)
- [confirmed in literature] EP 暂现源 EPW20240219aa：无 blind 触发，targeted 亚阈值搜索在 T0+40 s 附近找到 ~50 s 的硬谱暂现信号（4 s 时标最显著），与 Fermi-GBM、Swift/BAT 一致 — [GCN 35785](https://gcn.nasa.gov/circulars/35785)
- [confirmed in literature] 可供叠加的外部样本：CHIME/FRB 第一星表 536 个 FRB（2018-07-25 至 2019-07-01）— [CHIME/FRB Collaboration 2021, arXiv:2106.04352](https://arxiv.org/abs/2106.04352)；GWTC-4.0（O4 前段）— [LVK 2025, arXiv:2508.18082](https://arxiv.org/abs/2508.18082)；O3 的 GBM+BAT 联合分析 — [Fletcher et al., arXiv:2308.13666](https://arxiv.org/abs/2308.13666)
- [confirmed in literature] GW150914-GBM：GW 之后 0.4 s、>50 keV、持续 ~1 s 的弱暂现信号，误报概率 0.0022，1 keV–10 MeV 光度 1.8×10⁴⁹ erg/s — [Connaughton et al. 2016, ApJL 826, L6, arXiv:1602.03920](https://arxiv.org/abs/1602.03920)
- [confirmed in literature] Crab 巨脉冲的 X 射线增强：NICER 在 0.2–12 keV 测得 3.8±0.7%（5.4σ），用了 370 万个自转周期、约 2.6 万个巨脉冲 — [Enoto et al. 2021, Science 372, 187, arXiv:2104.03492](https://arxiv.org/abs/2104.03492)
- [confirmed in literature] Suzaku/HXD 在 15–75 keV 看到主脉冲巨脉冲相位 ±1.5° 内 21.5%（2.70σ）的"统计上边缘"的增强，样本 ~1.3×10⁴ 个巨脉冲 — [Mikami et al. 2013, arXiv:1309.1650](https://arxiv.org/abs/1309.1650)

### Inferences
#### 想法 3：外部星表触发的 CsI 群体似然叠加（BBH 并合、FRB、EP 快暂现源、中微子）
1. **一句话**：把 targeted 相干搜索的对数似然比（或 S/N 加权计数）在外部事件样本上叠加，检验"平均每个事件有多少 0.2–3 MeV 辐射"。
2. **原理与来源**：Blackburn et al. 2015 式相干搜索（Cai et al. 2021 已实现）；GW150914-GBM 候选体引出的"BBH 是否有弱硬谱闪"问题 [confirmed in literature]。
3. **适配度**：HXMT/CsI 在 >200 keV 面积大、对硬而短的信号（GW150914-GBM 正是 >50 keV 的硬信号）比 GBM NaI 更敏感 [inference]；但视场被地球挡掉 ~1/3、SAA 关机、<200 keV 不灵敏，对软的 EP 暂现源或 FRB 的磁星暴型对应体（谱峰 ~几十 keV）并不占优——这点应直说：**对 FRB/EP 的软谱对应体，GBM、GECAM、BAT 更合适** [inference]。
4. **数据**：全任务 CsI 事例（已归档）；外部星表全部公开。
5. **查新**：逐事件上限已做（见上）；群体叠加未检索到 [confirmed in literature：检索未见]。EP×HXMT 的系统性论文是否已发表——未查。
6. **可行性**（本笔记计算）[inference]：以 targeted 灵敏度 1.5×10⁻⁸ erg cm⁻²（1 s，200 keV–3 MeV）为单事件水平，叠加 N≈150 个有覆盖的 BBH → 平均流量上限 ~1.2×10⁻⁹ erg cm⁻²，在 d_L=1.5 Gpc 处对应 E_iso≲3×10⁴⁷ erg，约为 GW150914-GBM（1.8×10⁴⁹ erg/s × ~1 s）的 2%。对 FRB：10 ms 窗内本底 ~100 计数，单事件 5σ≈50 计数；叠加 250 个 → 每个 FRB 平均 ≲3 计数，约 2.5×10⁻⁹ erg cm⁻²（硬谱假设）。系统误差：各事件入射方向不同导致响应差 ~30%；粒子 spike 需要沿用团队的滤除算法。
7. **评分**：N2 / F5 / I3 —— 最不"非常规"，但管线现成、一两个月可出结果；BBH 群体上限有明确物理靶子。

#### 想法 4：射电巨脉冲（GRP）× HE 20–250 keV 叠加 —— 判定增强幅度是否随能量上升
1. **一句话**：把与射电望远镜同时的 HXMT Crab 数据按"有/无巨脉冲的自转周期"分组叠加，测量 20–250 keV 主峰处的增强幅度。
2. **原理与来源**：Enoto et al. 2021（NICER，软 X，3.8%）与 Mikami et al. 2013（Suzaku/HXD，硬 X，21.5%，2.7σ）[confirmed in literature]。两者若都为真，说明增强幅度随能量急剧上升，直接约束巨脉冲时磁层等离子体密度/对产生的变化。
3. **适配度**：HE 的面积远大于 Suzaku/HXD，且无堆积、2 μs 计时；在 >20 keV 做逐周期叠加，**HXMT/HE 是现有数据里最合适的之一**（AstroSat/LAXPC 在 20–80 keV 可比）[inference]。
4. **数据**：HXMT 全部 Crab 指向（公开）；射电单脉冲数据需要合作获取（每日过中天监测 Crab 的阵列、国内 25–40 m 级望远镜的 Crab 监测、FAST 专项观测等）——**这是瓶颈** [speculation：具体哪些台站有与 HXMT 重叠的单脉冲记录未核实]。
5. **查新**：检索 "Insight-HXMT Crab pulsar giant pulses X-ray enhancement FAST radio" 未返回任何 HXMT 或 FAST 的结果 [confirmed in literature：检索未见]。
6. **可行性**（本笔记计算）[inference]：假设 HE 脉冲计数率 200 cts/s、非脉冲（星云+本底）1300 cts/s、取主峰 2% 相位窗、峰值/平均=4（均为假设）→ 每周期窗内脉冲 0.54 计数、非脉冲 0.88 计数。3σ 探测 4% 增强需 ~2.7×10⁴ 个巨脉冲；**若增强真有 ~20%（Mikami 值），只需 ~1.1×10³ 个巨脉冲即可 3σ、~3×10³ 个即可 5σ**。系统误差：射电–X 射线绝对相位对齐（Jodrell 历表系统误差 ~40 μs，约 0.1% 相位，可接受）、巨脉冲选择偏差（闪烁/RFI）。
7. **评分**：N4 / F3 / I4 —— 物理问题明确、所需样本小，成败取决于能否拿到同时的射电单脉冲数据。

### Gaps
- CHIME/FRB 第二星表、IceCube 警报总表、EP 暂现源总表的规模未在本次取回的文献中核实。
- 未能查证 GBM/BAT 是否已对 BBH 做过群体级叠加（Fletcher et al. 的摘要未能取回全文）。
- 磁星 glitch、SN 激波暴、TDE、LHAASO 暂现源：CsI 只对 ≳200 keV 敏感，预期信号谱峰远低于此，未做估算，倾向否决（见 §12）。

---

## 4. 亚阈值/未触发暂现源搜索：透镜回波、未分类暴、磁星巨耀发

### Takeaway
团队的 blind（0.05–1 s 时标）+ targeted 管线已覆盖常规亚阈值搜索；留下的空白是 (a) 用自相关找毫秒–百秒的"引力透镜回波"（检索未见 HXMT 工作），(b) 官方星表里现成的 56 个未分类暴，以及 (c) 超出 0.05–1 s 触发时标的超短/超长暂现源。

### Cited Findings
- [confirmed in literature] blind 搜索把 18 个 CsI 分 6 组，≥3 组在 0.05/0.1/0.2/0.5/1 s 时标上超出本底 3σ 即触发；targeted 搜索在前两年找回 40 个 blind 漏掉的 GRB；改进算法可滤除带电粒子 spike 造成的假触发 — [Cai et al. 2021, arXiv:2109.12270](https://arxiv.org/abs/2109.12270)
- [confirmed in literature] 660 个暴中 56 个未分类、33 个带电粒子事件、245 个 TGF — [Song et al. 2022, arXiv:2203.00878](https://arxiv.org/abs/2203.00878)
- [confirmed in literature] BATSE GRB 950830 的透镜回波证据（Bayesian 分析，透镜质量 (1+z_l)M_l≈5.5×10⁴ M☉，误报概率 ~0.6%）— [Paynter, Webster & Thrane 2021, Nature Astronomy, arXiv:2103.15414](https://arxiv.org/abs/2103.15414)
- [confirmed in literature] GRB 200716C 的透镜解释（延迟 ~1.92 s，放大比 ~1.5，透镜 ~2.4×10⁵ M☉，误报概率 ~0.07%），分析基于 Fermi/GBM — [Yang et al. 2021, ApJL, arXiv:2107.11050](https://arxiv.org/abs/2107.11050)；检索摘要称该暴也触发了 HXMT，但作者因 HXMT 数据未公开而未使用（此点来自检索摘要，arXiv 摘要页未提及，需核实）
- [confirmed in literature] GRB 210812A：两段辐射相隔 33.3 s、流量比 ~4.5，点质量透镜 ~10⁶ M☉；GBM 已观测 >3000 个 GRB，透镜暴比例估计 ~1/1000 — [Veres et al. 2021, ApJL, arXiv:2110.06065](https://arxiv.org/abs/2110.06065)
- [confirmed in literature] 用 GRB 光变中的透镜回波约束致密暗物质：≳10 M☉ 的透镜给出 ≳1 ms 的延迟；现有 GBM/BAT 数据受噪声限制，未来仪器可在 10–1000 M☉ 把 f_DM 压到 ~1% — [Ji, Kovetz & Kamionkowski 2018, PRD 98, 123523, arXiv:1809.09627](https://arxiv.org/abs/1809.09627)
- [confirmed in literature] 河外磁星巨耀发（MGF）样本：5 Mpc 内 4 个短暴与邻近恒星形成星系成协，上升时间与 E_iso 与普通短暴不同（>99.9%），体积发生率 3.8×10⁵ Gpc⁻³ yr⁻¹（>4×10⁴⁴ erg）— [Burns et al. 2021, ApJL, arXiv:2101.05144](https://arxiv.org/abs/2101.05144)
- [confirmed in literature] 检索 "Insight-HXMT gravitationally lensed gamma-ray burst search" 未返回任何 HXMT 的透镜搜索 — [最接近的条目：Cai et al. 2021](https://arxiv.org/pdf/2109.12270)

### Inferences
#### 想法 5：毫秒–百秒自相关"回波"搜索，兼做 GBM 透镜候选体的独立检验
1. **一句话**：对 HXMT/CsI 全部亮暴做自相关/Bayesian 回波搜索，重点是 0.1–10 ms 延迟（对应 ~10–10³ M☉ 透镜），并用 HXMT 的独立数据复核 GRB 200716C、210812A 等 GBM 候选体的"两像光变与能谱是否一致"。
2. **原理与来源**：Paynter et al. 2021；Ji et al. 2018；Veres et al. 2021；Yang et al. 2021 [confirmed in literature]。点质量透镜的两像时间延迟与 (1+z_l)M_l 成正比，毫秒延迟对应恒星级–中等质量致密天体。
3. **适配度**：CsI 在 >200 keV 有效面积 ~1000 cm²、2 μs 时间戳、单事件死时间 4–8 μs [confirmed in literature]，对亮短暴在 1 ms 分辨下仍有足够计数，**在毫秒延迟区间比 GBM 更合适** [inference]；但 HXMT 的 GRB 样本（四年 322 个，十年约 800 个）只有 GBM 的 ~1/4，百秒级延迟的统计不占优 [inference]。
4. **数据**：GRB 星表附带的 T0−100 s 至 T0+300 s 事例数据 [confirmed in literature：星表按此区间选取事例]；是否公开下载——未知。
5. **查新**：未见 HXMT 透镜搜索 [confirmed in literature：检索未见]。
6. **可行性** [inference]：按 Veres et al. 的 ~1/1000 比例，800 个暴期望 ≲1 个候选体，主要产出是 f_DM 上限与对已有候选体的独立裁决。系统误差：(a) 粒子 spike 会伪造毫秒级"回波"，需用 HVT 标记与 18 探测器间的一致性剔除；(b) 极亮暴的饱和/丢数（'IRON' 样本）；(c) 同组探测器共享处理单元带来的相关死时间会在自相关函数的 μs 级延迟上留下结构，ms 级以上影响小。
7. **评分**：N4 / F4 / I3 —— 数据与方法都现成，独特卖点在毫秒延迟区间。

#### 想法 6：56 个"未分类暴"再分类与河外磁星巨耀发候选
1. **一句话**：把官方星表剩下的未分类暴（外推到十年约 150 个）与 IPN/GBM/GECAM/Konus 的未触发列表及邻近星系位置交叉，筛出 MGF 候选并在 CsI 的 μs 数据里找毫秒级准周期振荡。
2. **原理与来源**：Burns et al. 2021 的"按邻近星系成协 + 上升时间"筛选法 [confirmed in literature]。
3. **适配度**：MGF 的初始尖峰硬而短（MeV、毫秒上升），正落在 CsI 最强的能段和时标 [inference]；但 HXMT 没有自主定位能力，必须借助三角定位或其他卫星的定位，**单独使用不占优**。
4. **数据**：需要团队的未分类暴列表（星表论文未列出明细——未核实）。
5. **查新**：未检索到针对 HXMT 未分类暴或 MGF 的专文；HXMT 是否记录到 2020 年以来几例著名的河外 MGF 候选体——未查。
6. **可行性** [inference]：工作量小（列表交叉 + 光变形态学）；主要不确定性是这 56 个事件中有多少其实是 TGF/粒子事件（可用轨道位置、地磁纬度与 HVT 计数率判别）。
7. **评分**：N3 / F4 / I3。

### Gaps
- 超长暴、TDE 喷流、X 射线闪等软/慢信号在 CsI（≳200 keV）中预期很弱；用准直望远镜空白天数据搜这些源的事件率未能估算（缺 LE 在典型 GTI 下的本底涨落数据）。
- HXMT 与 GECAM/GBM 的联合亚阈值搜索是否已有系统论文——未查（配额耗尽）。

