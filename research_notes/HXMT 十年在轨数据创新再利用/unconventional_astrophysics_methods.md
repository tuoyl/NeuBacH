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
- [confirmed in literature] 轨道高度 550 km、倾角 43°（来自检索摘要对大气掩食论文的概括，倾角数值本人未在原文逐字核对）— [Yu et al. 2022, AMT 15, 3141](https://amt.copernicus.org/articles/15/3141/2022/)；导航论文正文写明轨道高度 550 km、半长轴 6922.88 km — [Zheng et al. 2019, arXiv:1908.01922](https://arxiv.org/pdf/1908.01922)
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
3. **利用的 HXMT 特性与适配度**：40 mm 厚 CsI、~1000 cm² 平均有效面积、全事例下传、无星上触发 [confirmed in literature]。在 >300 keV，GBM 的 NaI 很薄、BGO 只有两块，所以 HXMT/CsI 的有效面积应明显更大 [inference]；HXMT 指向观测时姿态惯性固定，源在载荷坐标系中的方向在一次观测内不变，比 Fermi 的摇摆扫天更利于台阶建模 [inference]。**不占优之处**：响应要穿过卫星结构，质量模型弥散达 ~30%，背向入射更差 [confirmed in literature]；<200 keV 基本不可用，所以它不是 GBM/BAT 的替代，只是 >200 keV 的补充 [inference]。另外，沿准直器轴向（θ<6°）入射时 CsI 有效面积远大于离轴 [confirmed in literature，Luo et al. 2020]，所以**指向目标自身**的升/落台阶信噪比最高，可把 NaI 的 250 keV 上限延伸到 ~800 keV；但这些台阶恰好与 NG↔LG 模式切换同步，是最难处理的一类 [inference]。
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
- [confirmed in literature] GW 搜索覆盖 O2 与 O3a，无显著对应体；O4 期间仍以 GCN 形式逐事件报告（例如 S250223dk 的 targeted 搜索给出 ~4σ 的 1 s 候选体，"very preliminary"）— [Cai et al. 2021, arXiv:2109.12270](https://arxiv.org/abs/2109.12270)；[GCN 39508](https://gcn.nasa.gov/circulars/39508)
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

---

## 5. 十年尺度时域：Crab 的"闪烁"、脉冲星长期监测、CsI 全天脉冲折叠

### Takeaway
HXMT 已发表 Crab 脉冲轮廓长期演化与 glitch 前后轮廓/流量不变性的工作，但把 Crab 当作恒定标准烛光来定标，本次未检索到用 HXMT 独立检验星云流量百分级变化的论文。两个"把定标量变成信号"的做法值得做：用脉冲星脉冲流量作内禀参考测星云流量；把只用于质量模型定标的 CsI Crab 脉冲信号扩展成 MeV 脉冲星/磁星硬尾的十年监测。常规的自转/轨道演化研究不在此列。

### Cited Findings
- [confirmed in literature] Crab 星云硬 X 射线流量 2008-08 至 2010-08 下降约 7%（70 mCrab，15–50 keV，GBM 掩食法），Swift/BAT、RXTE/PCA、INTEGRAL/IBIS 独立证实；2005–2008 存在 ~3 年时标的相关变化（2007-04 极小）；100–500 keV 的下降"可能更大"；RXTE/PCA 的脉冲流量与自转减慢一致，因此变化归因于星云 — [Wilson-Hodge et al. 2011, ApJ 727, L40, arXiv:1010.2679](https://arxiv.org/abs/1010.2679)
- [confirmed in literature] HXMT 把 Crab 的幂律模型（Γ=2.11，N=8.76）作为定标基准，并用 Crab 残差导出经验修正函数乘到有效面积上 — [Li X.-B. et al. 2020, arXiv:2003.06998](https://arxiv.org/pdf/2003.06998)
- [confirmed in literature] 扫描巡天用 Crab 长期光变估计流量系统误差（LE 1.8%、ME 1.6%、HE 2.7%），并注明"due to the possible fluctuations of Crab"这是保守估计 — [Sai et al. 2020, arXiv:2003.07741](https://arxiv.org/pdf/2003.07741)
- [confirmed in literature] Crab 两次 glitch（2017、2019）前后 X 射线轮廓不变：HXMT 27–200 keV 在 rms 0.47% 内、NICER 0.5–10 keV 在 0.28% 内 — [Zhang et al. 2022, ApJ 932, 11（ADS，检索摘要）](https://ui.adsabs.harvard.edu/abs/2022ApJ...932...11Z)
- [confirmed in literature] Crab X 射线双峰间距长期漂移约 0.8°/世纪（多个中国卫星数据）— ["Long-term variations of X-ray pulse profiles for the Crab pulsar: data analysis and modeling", SCPMA 63, 109511 (2020)（ADS，检索摘要）](https://ui.adsabs.harvard.edu/abs/2020SCPMA..63j9511S/abstract)
- [confirmed in literature] LE 观测 Crab：脉冲计数率 ~130 cnts/s，"本底"（含星云）~1200 cnts/s — [Zheng et al. 2019, arXiv:1908.01922](https://arxiv.org/pdf/1908.01922)
- [confirmed in literature] "the stable pulse emission of Crab at hard X-rays can be detected with HXMT/CsI in various incident directions (θ, φ)"，被用来检验质量模型；论文把 "MeV-pulsars" 列为 HE 可监测对象；CsI 有效面积在 80–300 keV 随能量快速上升，且"the effective area at 6°<θ<90° is usually much smaller than that of θ<6°"（即沿准直器轴向入射时最大）— [Luo et al. 2020, arXiv:2005.01367](https://arxiv.org/pdf/2005.01367)
- [confirmed in literature] COMPTEL 9 年数据给出 Crab 脉冲星 0.75–30 MeV 的能量分辨脉冲轮廓（主峰、次峰相隔 0.4 相位，有强桥辐射）— [Kuiper et al. 2001, A&A, arXiv:astro-ph/0109200](https://arxiv.org/abs/astro-ph/0109200)
- [confirmed in literature] 磁星 4U 0142+61：INTEGRAL 总谱 Γ=0.93±0.06，探测到 229 keV，20–229 keV 流量 (15.01±0.82)×10⁻¹¹ erg cm⁻² s⁻¹；结合 SPI 首次给出谱拐折证据，峰值能量 228(+65/−41) keV；脉冲辐射测到 160 keV，脉冲谱 Γ=0.40±0.15 — [den Hartog et al. 2008, A&A, arXiv:0804.1640](https://arxiv.org/abs/0804.1640)
- [confirmed in literature] Crab 是 HXMT 最合适的计时定标源，毫秒脉冲星对 HXMT 太暗 — [Tuo et al. 2022, arXiv:2109.04709](https://web3.arxiv.org/pdf/2109.04709)

### Inferences
#### 想法 7：以脉冲星脉冲流量为内禀参考的 Crab 星云流量十年监测（1–250 keV 同时）
1. **一句话**：对每一次 Crab 定标观测测量"非脉冲（星云）流量 / 脉冲流量"之比及其能量依赖，独立检验星云百分级"闪烁"，而不依赖有效面积的绝对定标。
2. **原理与来源**：Wilson-Hodge et al. 2011 用"脉冲流量随自转减慢平稳变化"论证变化来自星云 [confirmed in literature]；这里把同一逻辑反过来当测量方法——脉冲星与星云同时经过同一台仪器，响应漂移在比值中一阶抵消 [inference]。
3. **适配度**：HXMT 的 Crab 观测是全任务最密集的定标数据，三台望远镜同时覆盖 1–250 keV，无堆积 [confirmed in literature 关于定标用途]。软 X 段 NICER 能做同样的事且光子更多，**HXMT 的独特性只在 10–250 keV 的同时覆盖**（Wilson-Hodge et al. 指出 >100 keV 的下降可能更大，至今缺少单一仪器的宽带验证）[inference]。
4. **数据**：全部 Crab 指向观测（公开定标数据）、Jodrell/NICER 历表。
5. **查新**：检索到的 HXMT Crab 长期工作是轮廓演化与 glitch 不变性；专门检索"HXMT Crab 星云流量变化"的查询因配额耗尽未执行，**查新不完整**。
6. **可行性**（本笔记计算）[inference]：LE 脉冲 130 cts/s、非脉冲 1200 cts/s → 10 ks 曝光脉冲计数 1.3×10⁶，统计误差 ≈√(1.33×10⁷)≈3.6×10³，即比值单次精度 ~0.3%；而已知变化幅度为 ~3.5%/yr 量级。系统误差：(a) 星云项需要扣本底——HE 本底约为 Crab 计数的 30%–50%（假设），1% 的本底误差传到星云流量是 0.3%–0.5%，ME 更差，LE 用 blind 探测器最干净；(b) 脉冲星与星云谱形不同，增益漂移会通过能段边界产生二阶效应，需用窄能段并逐次重定标；(c) 脉冲流量自身的长期演化与 glitch 后变化（Zhang et al. 2022 给出 0.47% 的不变性上限，可作约束）；(d) 指向偏差对两者影响相同。
7. **评分**：N4 / F5 / I3 —— 数据全是现成的定标数据，方法简单稳健；影响力中等（独立验证 + 首次给出同一仪器的宽带能量依赖）。

#### 想法 8：CsI 全天脉冲折叠 —— MeV 脉冲星与磁星硬尾的十年监测
1. **一句话**：不管卫星指向哪里，只要目标未被地球遮挡，就把 CsI 事例按已知历表折叠，得到 Crab 0.2–3 MeV 的十年脉冲轮廓/脉冲流量序列，并对 PSR B1509−58、磁星硬尾等做脉冲探测或上限。
2. **原理与来源**：Fermi-GBM 对吸积脉冲星的全天脉冲监测思路（文献本次未取回）；COMPTEL 的 Crab MeV 轮廓（Kuiper et al. 2001）；磁星硬尾在 ~200–300 keV 的拐折问题（den Hartog et al. 2008）[confirmed in literature]。脉冲信号天然免疫于非准直探测器的本底建模问题。
3. **适配度**：CsI 的 Crab 脉冲已被探测并用于定标 [confirmed in literature]，但未见科学利用。对"周期信号"而言没有成像能力不是缺点；十年里约一半时间可见目标，累计曝光 ~10⁸ s 量级，远超任何指向型软 γ 仪器对单个源的曝光 [inference]。沿准直器轴向的 CsI 有效面积远大于离轴 [confirmed in literature]，所以**指向观测期间的脉冲星还能额外得到 250–800 keV 的"准指向"脉冲谱** [inference]。局限：<200 keV 离轴不可用；SPI/IBIS 的信噪比（单位曝光）更高，HXMT 靠的是曝光量。
4. **数据**：全任务 HE 事例文件中的 CsI 事例 + 轨道/姿态；射电或 X 射线历表（Jodrell、NICER、Fermi-LAT）。
5. **查新**：Luo et al. 2020 证实 CsI Crab 脉冲仅用于质量模型检验；针对"CsI 脉冲星科学"的专门检索因配额耗尽未执行，**查新不完整**。
6. **可行性**（本笔记计算）[inference]：
   - 输入：A_eff=1000 cm²、B=4000 cts/s（200–600 keV，假设）、有效曝光 T=1.6×10⁸ s（十年的一半）。
   - 5σ 脉冲计数率 = 5√(B/T) ≈0.025 cts/s → 2.5×10⁻⁵ ph cm⁻² s⁻¹ → 约 1.4×10⁻¹¹ erg cm⁻² s⁻¹（平均光子能量 350 keV）。
   - Crab：若 200–600 keV 脉冲占总流量 20%（假设），CsI 中脉冲 ≈3 cts/s，~10⁴ s 即达 5σ，十年累计 S/N≈600 → 可按年甚至按月给出 MeV 段脉冲轮廓与流量，检验轮廓的长期演化和 glitch 响应是否延伸到 MeV。
   - 4U 0142+61：把 INTEGRAL 总谱（Γ=0.93）不加拐折外推到 200–600 keV 是 ~3×10⁻¹⁰ erg cm⁻² s⁻¹；CsI 的十年脉冲灵敏度约为其 5%，因此即使探测不到，也能对拐折能量给出有约束力的上限。该源赤纬 +62°，靠近轨道极区，很少被地球遮挡 [inference]。
   - 障碍：方向相关响应（每次指向不同）、CsI 增益漂移导致能段定义随时间变化、慢脉冲星（P~6–12 s）对本底红噪声更敏感、磁星计时噪声要求同期历表。
7. **评分**：N4 / F4 / I4 —— 用的是"被当作屏蔽体的晶体 + 只用于定标的信号"，科学靶子（MeV 脉冲、磁星硬尾拐折）明确。

**常规时域方向的判断** [inference]：自转/力矩历史、超轨道周期、轨道周期导数、自由进动搜索属于 HXMT 常规科学的延长线，除非与上面两种"非准直/自校准"手段结合，否则不符合"非常规"要求，未展开。

### Gaps
- 未取回 Fermi-GBM 脉冲星监测项目的文献（尝试的 arXiv 编号不对），该先例无 URL。
- PSR B1509−58 等其他软 γ 脉冲星的 MeV 脉冲流量未在本次取回文献中核实，未给估算。
- CsI 沿轴向（θ<6°）的有效面积数值在 Luo et al. 2020 Fig. 12 中，文本提取不到，需人工读图。
- HXMT Crab 定标观测的次数与总曝光未查到。

---

## 6. 脉冲星导航与守时：做到了哪一步，还剩什么

### Takeaway
已发表的在轨验证是 2017-08-31 至 09-05 的 5 天 Crab 观测，用 SEPO 方法单星定轨到 10 km（3σ）；本次未检索到后续的多脉冲星或长期"脉冲星钟"工作。用户设想的几个剩余角度里，X–射电相位差长期变化与"类 PTA"数据集经量级估算都不可行，建议否决；可行的部分已并入想法 4（巨脉冲）与想法 8（CsI 折叠）。

### Cited Findings
- [confirmed in literature] 用 2017-08-31 至 09-05 的 Crab 观测、SEPO（Significance Enhancement of Pulse-profile with Orbit-dynamics）方法，三台望远镜分别定轨，位置 10 km（3σ）、速度 10 m/s（3σ）；HE/ME/LE 曝光约 180/190/120 ks；文中用 PSR B1821−24、B1509−58、J1811−1925 的轮廓做了仿真；同方法先在 POLAR/TG-2 上用 31 天 Crab 数据验证 — [Zheng et al. 2019, ApJS 244, 1, arXiv:1908.01922](https://arxiv.org/pdf/1908.01922)
- [confirmed in literature] RXTE 八年监测：Crab X 射线主峰领先射电 344±40 μs（0.0102±0.0012 相位），主要误差是射电端 40 μs 系统误差（星际散射与设备定标）；无法区分该延迟是相位恒定还是时间恒定 — [Rots, Jahoda & Lyne 2004, ApJ 605, L129, arXiv:astro-ph/0403187](https://arxiv.org/abs/astro-ph/0403187)
- [confirmed in literature] HXMT 计时定标因 Jodrell 历表 ~40 μs 的系统误差而改用准同时 NICER 观测建历表；HE/ME 的仪器延迟（0.156/0.335 μs）可忽略，能量依赖约 0.26 μs keV⁻¹；文中给出了 Crab 射电延迟随 X 射线能量的关系图 — [Tuo et al. 2022, arXiv:2109.04709](https://arxiv.org/pdf/2109.04709)

### Inferences
- [inference] **X–射电相位差的长期变化**：十年内 Crab 周期变化 ΔP/P≈0.4%（按 P≈33.8 ms、Ṗ≈4.2×10⁻¹³，记忆值），若延迟"相位恒定"，300 μs 的延迟只变 ~1 μs；即使把 RXTE（1996）接到 HXMT（2026），也只有 ~4 μs，远小于射电端 40 μs 的系统误差。否决。
- [inference] **用 X 射线 ToA 反推 DM 变化**：X 射线 ToA 不受色散影响，HE 系统误差 12 μs，而 Crab 的 DM 变化在几百 MHz 可造成 100 μs 以上的延迟，原理上可测；但 Jodrell 自己就有双频 DM 修正，X 射线只能做一致性检查，科学增量小。列为低优先级。
- [inference] **类 PTA 数据集**：Crab 计时噪声达毫秒量级，其他年轻脉冲星同样嘈杂，毫秒脉冲星对 HXMT 太暗 [confirmed in literature]。否决。
- [inference] **不用指向的 CsI 脉冲星导航**：CsI 中 Crab 脉冲 ~3 cts/s、本底 ~4000 cts/s，5 天的轮廓显著性只有指向 HE 的 ~1%，定位误差将从 10 km 退化到 ~10³ km 量级。否决。

### Gaps
- 未能检索 2019 年之后 HXMT 是否发表多脉冲星导航、长期脉冲星钟稳定度的论文（配额耗尽）。
- Crab 的 P、Ṗ 数值为记忆值，未在本次取回文献中核对。

---

## 7. 弥散辐射：CXB、地球反照、银河脊、弥散谱线

### Takeaway
HXMT 已发表 1.5–120 keV 的 CXB 谱（LE 用 blind 探测器、HE 用地球遮挡段作本底，只用了 2 个宽视场 HE 探测器），以及 LE 的银道面弥散本底图；ME/HE 的银河脊硬 X 射线剖面未检索到。最有潜力的非常规方向是把 LE 当作"大集光率 + CCD 级能量分辨"的弥散谱仪去找暗物质衰变线；CXB 偶极、511 keV/正电子素连续谱、Fermi/eROSITA 泡的硬 X 对应体经估算不可行。

### Cited Findings
- [confirmed in literature] HXMT 的 DXRB 谱（1.5–120 keV）：cut-off power law，Γ=1.40，归一化 9.57 ph cm⁻² s⁻¹ keV⁻¹ sr⁻¹（1 keV 处），截断能量 55 keV（已修正地球反照与大气辐射）；谱形与前人一致，但归一化除 HEAO-1 外都偏低 — [Huang et al. 2022, MNRAS 513, 4074, arXiv:2306.10114](https://arxiv.org/abs/2306.10114)
- [confirmed in literature] 该工作的细节：HE 只用两个宽视场探测器；HE "blank-sky" 与 "Earth-occulted" 累计曝光 377 ks 与 187 ks（2021 年前的公开数据）；LE 有效曝光 117 ks；LE 用地球遮挡段作本底会有系统效应（疑似滤光片漏光），故改用 blind 探测器；遮挡段曝光短于 475 s 的被剔除以消除 40–45 keV 的本底偏差 — [Huang et al. 2022, arXiv:2306.10114](https://arxiv.org/pdf/2306.10114)
- [confirmed in literature] LE 本底：7 keV 以下由弥散 X 射线本底主导，谱稳定、对天区依赖小；1 ks 谱的平均系统误差 1–7 keV 为 4.2%、1–10 keV 为 3.7% — [Liao et al. 2020, arXiv:2004.01432](https://arxiv.org/pdf/2004.01432)
- [confirmed in literature] LE 银道面弥散本底：用三年银道面扫描数据做 1°×1° 像素图，谱与 RXTE/PCA 一致 — ["The Diffuse X-Ray Background of the Insight-HXMT/LE Telescope in the Galactic Plane", ApJS 260, 42 (2022)](https://www.scilit.net/article/655e9ab6c90e7ca46351779b691c5e3d)
- [confirmed in literature] INTEGRAL 用地球掩过视场的调制测 5–100 keV 的 CXB，归一化比 HEAO-1 高 ~10%，并评估了反照、宇宙线致大气辐射、极光等污染 — [Churazov et al. 2007, A&A, arXiv:astro-ph/0608250](https://arxiv.org/abs/astro-ph/0608250)
- [confirmed in literature] Swift/BAT 用不同地磁纬度的地球掩食数据同时测 15–200 keV 的 CXB 与地球反照谱，CXB 归一化比 HEAO-1 高 8±3% — [Ajello et al. 2008, ApJ, arXiv:0808.3377](https://arxiv.org/abs/0808.3377)
- [confirmed in literature] 宇宙线致大气硬 X 射线辐射的 Monte Carlo：给出 25–300 keV 面亮度随能量、**太阳调制水平、地磁截止刚度**和天顶角的拟合公式；与 INTEGRAL 测量在 10% 内一致 — [Sazonov et al. 2007, MNRAS 377, 1726, arXiv:astro-ph/0608253](https://arxiv.org/abs/astro-ph/0608253)
- [confirmed in literature] NuSTAR 用"侧孔杂散光在探测器上的空间调制"测 3–20 keV 的 CXB（~7 Ms 河外场），强度 2.8×10⁻¹¹ erg s⁻¹ cm⁻² deg⁻²，高于 HEAO-1、与 INTEGRAL 一致；各方向间的弥散与宇宙方差一致 — [Krivonos et al. 2021, MNRAS, arXiv:2011.11469](https://arxiv.org/abs/2011.11469)
- [confirmed in literature] HEAO-1 A2 的 2–10 keV 偶极：与本星系群运动的 Compton-Getting 效应（幅度 0.0042）加剩余大尺度结构（0.0023–0.0085）一致 — [Scharf et al. 2000, ApJ, arXiv:astro-ph/9908187](https://arxiv.org/abs/astro-ph/9908187)
- [confirmed in literature] INTEGRAL/IBIS 的硬 X 射线银河脊：17–60 keV 的经/纬度剖面与近红外恒星分布一致，总光度 (3.7±0.2)×10³⁷（erg/s），70–200 keV 除点源外无弥散辐射；IBIS 本底模型精度 ~1%–2% — [Krivonos et al. 2007, A&A, arXiv:astro-ph/0605420](https://arxiv.org/abs/astro-ph/0605420)；软 X 段的恒星起源论证 — [Revnivtsev et al. 2006, arXiv:astro-ph/0510050](https://arxiv.org/pdf/astro-ph/0510050)
- [confirmed in literature] HXMT 银道面扫描四年灵敏度：2–6 keV 约 8.2×10⁻¹²、7–40 keV 约 4.21×10⁻¹¹、25–100 keV 约 2.78×10⁻¹¹ erg s⁻¹ cm⁻²；监测 >1300 个源，223 个 S/N>5 — [Wang et al. 2023, ApJS 265, 52, arXiv:2303.00360](https://arxiv.org/abs/2303.00360)
- [confirmed in literature] 弥散暗物质衰变线的先例：Fermi-GBM 全天数据搜 10–25 keV 的 sterile neutrino 线（质量 20–50 keV）— [Ng et al. 2015, PRD 92, 043503, arXiv:1504.04027](https://arxiv.org/abs/1504.04027)；NuSTAR 银河核球场 5–20 keV（质量 10–40 keV）— [Roach et al. 2020, PRD 101, 103011, arXiv:1908.09037](https://arxiv.org/abs/1908.09037)；XMM-Newton >30 Ms 空白天数据否定 3.5 keV 线的暗物质解释 — [Dessert, Rodd & Safdi 2020, Science 367, 1465, arXiv:1812.06976](https://arxiv.org/abs/1812.06976)

### Inferences
#### 想法 9："三孔径差分 + 临边扫描"重测 CXB 归一化，并测地球反照/大气辐射的太阳周调制
1. **一句话**：利用同一台 HE 里立体角为 0（blind）、6.3 deg²（窄）、32.5 deg²（宽）的三类探测器，把随立体角线性增长的天空信号与不随立体角变化的粒子本底分开；同时把十年里上万次目标掩食过程按 ELV 叠加，得到地球反照/大气辐射的临边增亮轮廓及其随地磁截止刚度和太阳活动周（2019 极小至 2024–25 极大）的变化。
2. **原理与来源**：NuSTAR 的"空间调制"法（Krivonos et al. 2021）、INTEGRAL/BAT 的地球掩食法（Churazov et al. 2007；Ajello et al. 2008）、Sazonov et al. 2007 对太阳调制依赖的预言 [confirmed in literature]。
3. **适配度**：HXMT 的窄准直视场扫过临边时能给出角分辨的反照轮廓，而 INTEGRAL/BAT 是宽视场积分；三类视场共存是 HXMT 的结构性特点 [inference]。但 HE 的 CXB 信号只占本底的百分之几，绝对归一化未必能胜过 BAT/INTEGRAL；**真正新的可测量是反照/大气分量的太阳周调制**，这需要跨越一个太阳活动周的同一仪器数据，HXMT 恰好具备 [inference]。
4. **数据**：全部高银纬指向（不限于专门的空白天区提案）的 HE 事例与 EHK 文件（ELV、地磁截止刚度），均为常规公开产品。
5. **查新**：Huang et al. 2022 只用 2 个宽视场探测器和 2021 年前的数据；未见十年更新或太阳周分析 [confirmed in literature：已有工作范围]。
6. **可行性**（本笔记计算）[inference]：用 Huang et al. 的最佳拟合谱，20–100 keV 的 CXB 强度 ≈1.6 ph cm⁻² s⁻¹ sr⁻¹；假设探测效率 0.7，每个窄视场探测器 ≈0.6 cts/s、每个宽视场 ≈3.1 cts/s；15 个窄探测器合计的 CXB 计数约为 2 个宽探测器的 1.5 倍，却未被使用。十年数据使统计误差降到 1% 以下，极限在系统误差：(a) blind 探测器与其他探测器的能量分辨/本底谱不同（Huang et al. 已指出）——需要用深遮挡段逐探测器标定本底比；(b) 遮挡段与非遮挡段的地磁环境不同；(c) LG/NG 模式切换使遮挡段的 NaI 数据处于不同增益。太阳周调制：银河宇宙线流量在太阳极小/极大间变化 ~15%–20%（一般知识，未取文献），应能在叠加的大气辐射分量中看到。
7. **评分**：N3 / F4 / I3。

#### 想法 10：把 LE 当作大集光率弥散谱仪 —— 2–10 keV 暗物质衰变线（含 3.5 keV）搜寻
1. **一句话**：叠加十年 LE 空白天、高银纬指向与银道面扫描中无亮源的时段，以 blind 探测器扣粒子本底，在 2–10 keV 搜寻银晕暗物质衰变的窄线，并利用不同指向的暗物质柱密度差异做"随 D 因子变化"的检验。
2. **原理与来源**：Dessert et al. 2020（XMM 空白天）、Roach et al. 2020（NuSTAR 杂散光）、Ng et al. 2015（GBM 全天）[confirmed in literature]。弥散线灵敏度正比于 √(A·Ω·T)，而不是聚焦望远镜看重的角分辨。
3. **适配度** [inference]：LE 窄视场 1.6°×6° 约 9.6 deg²，约 60 个开放 SCD（每个 4 cm²）；宽视场约 18 个。估算集光率 AΩ≈3×10³ cm² deg²（≈1 cm² sr），比聚焦望远镜在 3.5 keV 的集光率高一个量级以上（XMM、NuSTAR 杂散光的对比值为笔者估计，未核对文献）；LE 在 7 keV 以下本底由 CXB 主导 [confirmed in literature]，且能量分辨为 CCD 级。**如果仪器谱线与响应的系统误差可控，LE 在这个指标上确实优于现有聚焦望远镜**；劣势是 LE 的 GTI 短而碎、亮地球漏光、有效面积含经验修正。
4. **数据**：LE 事例（含 blind 模块）、全部非银道面指向与扫描；公开性同常规数据。
5. **查新**：针对 "Insight-HXMT sterile neutrino / dark matter decay line" 的检索因配额耗尽**未能执行**，这是本笔记中最需要补做的查新。
6. **可行性**（本笔记计算）[inference]：
   - 以 3.5 keV 线的典型参数（sin²2θ=5×10⁻¹¹，m_s=7.1 keV）和内银河中高纬方向的柱密度 5×10²² GeV cm⁻²（假设）为例：衰变率 1.2×10⁻²⁸ s⁻¹，线强度 ≈0.07 ph cm⁻² s⁻¹ sr⁻¹，在 LE 中 ≈0.07 cts/s；同一 150 eV 分辨元内 CXB 连续谱 ≈0.25 cts/s。
   - 取本底为 CXB 的 2 倍、T=1 Ms：5σ 线强度极限 ≈3.5×10⁻³ ph cm⁻² s⁻¹ sr⁻¹，对应 sin²2θ≈2.5×10⁻¹²；若连续谱有 1% 的系统误差，极限约 3.6×10⁻¹²。
   - 障碍：(a) LE 有效面积用 Crab 残差做了经验修正，任何窄的响应残差都会伪装成线或抹掉线，需用 Crab 与 blank-sky 的比值谱交叉检查；(b) 仪器荧光线（Li et al. 2020 列出 S、Ni/Cu/Zn 等）与大气 Ar 荧光的位置要避开；(c) LE 增益随温度变化；(d) 点源与银河脊热线（Fe、S、Ar、Ca）在低银纬的污染——应以 |b|>10° 为主样本。
7. **评分**：N4（待查新）/ F3 / I4 —— 这是"空白天 + 盲探测器"由本底变信号的典型；能否成立取决于响应系统误差能否压到 1% 量级。

#### 想法 11：银道面扫描叠加的 20–100 keV 银河脊（GRXE）纬度剖面与 P(D) 涨落分析
1. **一句话**：把十年数千次银道面扫描中扣除已知点源 PSF 后的 ME/HE 残差按银纬叠加，得到硬 X 射线银河脊的纬度/经度剖面，并用残差涨落分布约束探测阈值以下的源计数。
2. **原理与来源**：INTEGRAL/IBIS 把探测器总计数率当"准直器"来测 GRXE（Krivonos et al. 2007）；RXTE/PCA 扫描 [confirmed in literature]。
3. **适配度**：HXMT 是真正的准直扫描仪器，方法自然；但 IBIS 已给出 17–60 keV 剖面和 70–200 keV 上限，**HXMT 不比 INTEGRAL 更合适，价值在于系统误差完全不同的独立确认**，以及与自家 LE 图的同仪器衔接 [inference]。
4. **数据**：GPSS 扫描事例与团队的点源拟合结果（残差光变是否公开——未知）。
5. **查新**：LE 已做（ApJS 260, 42）；ME/HE 未检索到 [confirmed in literature：检索未见]。
6. **可行性** [inference]：若 GRXE 在 HE 窄视场内相当于 2–3 mCrab（笔者由软 X 面亮度外推的假设值），即 ~2–3 cts/s，对比 ~500 cts/s 的本底只有 0.5%，单次过脊（~1 min）S/N~1；十年 ~10⁴–10⁵ 次过脊在统计上足够，关键是点源扣除残差与地磁调制是否与银纬相关。弱源数（S/N<3 的监测源在 LE/ME/HE 各约 10³ 个，Sai et al. 2020）说明残差里确有大量阈下源可供 P(D) 分析。
7. **评分**：N3 / F3 / I2。

### Gaps
- CXB 偶极与各向异性：Compton-Getting 幅度 0.42%，而 LE 单次 1 ks 的系统误差就有 ~4%，且高银纬指向的天区覆盖稀疏——判断不可行，但没有做正式的模拟。
- 511 keV 线/正电子素连续谱/²⁶Al 相关连续谱：未取回 SPI 的流量文献；仅凭一般知识判断 CsI 的能量分辨无法把 511 keV 线与银心点源连续谱分开（见 §12）。
- Fermi/eROSITA 泡的硬 X 射线对应体：没有找到预期面亮度的文献，无法定量。
- XMM、NuSTAR、eROSITA、HaloSat 等在 3.5 keV 处的集光率对比值未核对文献；LE 在 3–4 keV 是否存在仪器特征未查。

---

## 8. 扫描巡天副产品：暂现源目录与"机动巡天"

### Takeaway
"slew survey"并非完全空白：团队的扫描管线已把扫描前后的机动段纳入，用来监测银道面外的源；论文还明说发现了若干"很短时标"的新 X 射线源候选体但未报道。指向之间的大机动是否被系统利用未能查证。对河外快暂现源，HXMT 的集光率不如 MAXI/EP，这个方向的定位应是"银道面内短时标硬暴的回收"。

### Cited Findings
- [confirmed in literature] "the sources out of the Galactic plane are also monitored as Insight-HXMT is in slew phase (attitude adjustment) before/after the scanning observation" — [Sai et al. 2020, arXiv:2003.07741](https://arxiv.org/pdf/2003.07741)
- [confirmed in literature] "Several new X-ray source candidates are also discovered, some of which can be the X-ray transients with very short time scale, which need to be further confirmed and thus not reported here"；两年内 >1000 次扫描；单次扫描 5σ 灵敏度约 3/20/18 mCrab（LE/ME/HE）；光变中的异常峰用三种视场方向的峰是否同时/有无延迟来区分源与粒子事件 — [Sai et al. 2020, arXiv:2003.07741](https://arxiv.org/pdf/2003.07741)
- [confirmed in literature] 四年星表（2017-06 至 2021-08）与 MAXI 联合给出长期光变，摘要未提暂现源搜索、机动数据或残差图 — [Wang et al. 2023, arXiv:2303.00360](https://arxiv.org/abs/2303.00360)

### Inferences
#### 想法 12：用"三视场判别"在扫描/机动/指向数据中回收短时标暂现源
1. **一句话**：利用窄视场三个取向（相差 60°）+ 宽视场 + blind 探测器的同时响应，对 1–100 s 的短暴做真伪判别与粗定位，建立十年的银道面短暴目录（未触发的磁星短暴、I 型暴、SFXT 耀发），并复核团队未报道的候选体。
2. **原理与来源**：RXTE/PCA 银心扫描、XMM slew survey、MAXI 的思路（先例文献本次未取回）；判别逻辑：天体点源过视场给出宽度可预言的三角形响应，短于穿越时间的内禀暴在三组取向中同时出现且幅度比由准直器透过率决定，粒子事件则同时出现在 blind 探测器中 [inference]。
3. **适配度**：HE 对 20–250 keV 短暴的面积与时间分辨很好，但瞬时视场只有几十 deg²。**对河外快 X 射线暂现源，LE 的集光率比宽视场监测器低两个量级以上，不占优**；优势只在银道面内高发生率的硬短暴和 ms 级计时 [inference]。
4. **数据**：GPSS 与小天区扫描事例、指向间机动段事例（是否归档——未知）。
5. **查新**：机动段已被部分使用（见上）；专门的 "HXMT slew survey" 检索因配额耗尽未执行，**查新不完整**。
6. **可行性** [inference]：算法简单；事件率是主要未知数——没有找到 LE/HE 在机动段的本底涨落统计，无法给出可靠的期望数。建议先在已知活跃磁星所在扫描区做回收率测试。
7. **评分**：N3 / F3 / I2。

### Gaps
- 指向之间机动的角速度、机动段数据是否下传并公开，未查到。
- 团队"未报道的短时标候选体"的数量与性质未知。

---

## 9. 大面积、无堆积带来的计时技巧：高阶统计、亚毫秒变化、TNO 掩食

### Takeaway
这一组里没有找到值得推荐的新想法：TNO 掩 Sco X-1 的"干净重做"在 HXMT 上因计数率比 RXTE/PCA 低一个量级以上而不可行；强度干涉与量子引力"模糊"检验在物理量级上不成立；高阶统计量属于常规计时分析的延伸，且查新未完成。

### Cited Findings
- [confirmed in literature] RXTE/PCA 在 Sco X-1（典型计数率 ~10⁵ cts/s）光变中的毫秒级下陷最初（Chang et al. 2006）被解释为小 TNO 掩食；原作者的后续分析见 "Millisecond dips in the RXTE/PCA light curve of Sco X-1 and TNO occultation" — [Chang et al. 2007, MNRAS, arXiv:astro-ph/0701850](https://arxiv.org/pdf/astro-ph/0701850)
- [confirmed in literature] 2007-06 至 2011-10 共 334 ks 的 RXTE/PCA Sco X-1 数据中没有找到确定的 TNO 掩食事件，给出 30–300 m 天体尺寸分布的上限 — [Chang, Liu & Chen 2013, MNRAS, "Search for serendipitous TNO occultation in X-rays", arXiv:1211.4650](https://arxiv.org/abs/1211.4650)
- [confirmed in literature] 下陷由高能带电粒子事件在 PCA 中引起的电子学死时间产生，至多 10% 可能是 TNO 掩食，"it is a mistake to conclude that any TNOs have been detected" — [Jones et al. 2008, ApJ, arXiv:0710.0837](https://arxiv.org/abs/0710.0837)；更早的同结论短文 — [Jones et al. 2006, arXiv:astro-ph/0612129](https://ar5iv.arxiv.org/html/astro-ph/0612129)
- [confirmed in literature] 2007-06 至 2009-10 的 RXTE/PCA 专项观测只找到一个可能的掩食事件，并给出百米级 TNO 与十米级主带小行星尺寸分布的上限 — [Chang, Liu & Chen 2011, MNRAS, arXiv:1009.2547](https://arxiv.org/abs/1009.2547)
- [confirmed in literature] HE 单事件死时间通常 4–8 μs，高能带电粒子信号的死时间更长；每个 HED 有死时间计数器（步长 1/8 μs），每秒总死时间写入工程数据 — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)

### Inferences
- [inference] **TNO 掩食重做（否决）**：按 open area 比例（LE 300 cm² 对 PCA ~6500 cm²，后者为记忆值）外推，LE 对 Sco X-1 约 4×10³ cts/s，加上 ME、HE 合计约 5 cts/ms，比 PCA 的 ~100 cts/ms 低 20 倍。2 ms 全遮挡的"零计数"概率为 e⁻¹⁰≈4.5×10⁻⁵，100 ks 内有 5×10⁷ 个 2 ms 窗，会出现 ~2×10³ 个纯统计假下陷；只有 ≳4 ms（≳100 m 天体）的下陷才可辨。再加上 LE 的 SCD 读出把到达时间抹开 ~1 ms。X 射线在 40 AU 处的 Fresnel 尺度约 30 m（4 keV），本来是 X 射线法的优势，但 HXMT 的光子数用不上它。HVT 标记与死时间计数器确实能把粒子事件造成的下陷识别出来（这是 RXTE 所缺的），但这只是"更干净"，不是"更灵敏"。
- [inference] **强度干涉/HBT（否决）**：X 射线的相干时间约为 h/ΔE，对 keV 带宽是 10⁻¹⁸ s 量级，而 HE 时间分辨 2 μs，简并参数低十几个量级。
- [inference] **量子引力/Lorentz 破缺（否决）**：keV–MeV 光子的能量杠杆远小于 TeV 观测，没有竞争力。
- [speculation] **高阶统计量（bispectrum/bicoherence）与硬 X 射线 Fourier 域混响**：HE 无堆积、死时间小，技术上合适，但这属于常规计时科学的延伸，且未能查新。

### Gaps
- HXMT 对 Sco X-1 的实际计数率与累计曝光未查到，上面的数字是按面积比例外推的。
- 未能检索 HXMT 数据上已有的 bicoherence、亚毫秒变化工作（配额耗尽）。

---

## 10. 相邻 HE 模块/ME 像素间 Compton 散射偏振测量

### Takeaway
事件级信息足以在离线做"跨组"探测器间的时间符合（2 μs 时间戳），但同组 6 个探测器的同时信号在星上就被丢弃，加上 18 个大晶体的几何对方位角采样很粗，量级估算给出的最小可探测偏振度对最亮的非饱和 GRB 也在 ~70% 左右。结论：不推荐作为主攻方向，可做一次一天量级的"跨组双事件率"实测来最终定论。

### Cited Findings
- [confirmed in literature] 每个 HE 事件带 5 bit 通道号与 19 bit、2 μs 刻度的亚秒时间；同组 HED 在事件处理期间的同时信号被丢弃；CsI 与 NaI 同时沉积的事件可由脉宽识别 — [Liu et al. 2020, arXiv:1910.04955](https://arxiv.org/pdf/1910.04955)
- [confirmed in literature] INTEGRAL/IBIS 用 Compton 模式在 >200 keV 测到 Crab 的强线偏振，偏振角平行于脉冲星自转轴 — [Forot et al. 2008, ApJL, arXiv:0809.1292](https://arxiv.org/abs/0809.1292)
- [confirmed in literature] AstroSat/CZTI 利用其支撑结构在 >100 keV 透明的特点做广角 GRB 监测，并在 100–300 keV 对首年最亮的 11 个 GRB 做了偏振测量（5 个 ≳3σ）— [Chattopadhyay et al. 2019, ApJ 884, 123, arXiv:1707.06595](https://arxiv.org/abs/1707.06595)
- [confirmed in literature] 检索 "Insight-HXMT polarimetry Compton scattering HE detectors" 只返回发射前的 HXMT 偏振计概念论文（标题为 "X-ray polarimetry on-board HXMT" 与 "An X-ray Polarimeter for HXMT Mission"，内容未读），未见利用在轨 HE 数据做偏振的工作 — [arXiv:0810.2708](https://arxiv.org/pdf/0810.2708)；[arXiv:0709.3992](https://arxiv.org/pdf/0709.3992)

### Inferences
- [inference] **符合信息是否保留**：跨组（A–B、B–C、A–C）探测器对的双事件可以靠 2 μs 时间戳离线找回；同组对的第二次击中在星上丢失。三组在安装面上的空间分布（是否相邻扇区）未从文献文本中读出，决定了可用的相邻对数量。
- [inference] **量级估计**：单个 CsI 本底 ~550 cts/s，取 4 μs 符合窗，每对探测器的偶然符合 ~1.2 s⁻¹。假设真 Compton 双事件占 GRB 计数的 ~1%（由"4 cm 厚晶体侧向散射到相邻晶体"的立体角粗估，是最不确定的输入）、调制因子 μ~0.2、可用相邻跨组对 ~20 对：对一个持续 10 s、CsI 总超出 2×10⁴ cts/s 的亮暴，S≈2000、B≈2200，MDP₉₉≈4.29√(S+B)/(μS)≈0.7。更亮的暴会饱和。对 Crab 这样的稳定源，双事件淹没在本底双事件中，只有借助脉冲相位才能分离，估算 MDP 仍在 ~50% 量级。
- [inference] **判断**：INTEGRAL/IBIS、SPI 和 AstroSat/CZTI 有像素化探测面，明显更合适；HXMT 不占优。

### Gaps
- 没有 Geant4 级的双事件效率与调制因子，上述 1% 与 μ=0.2 均为假设；需要质量模型模拟或直接在存档数据中数跨组符合事件。
- ME 像素间的 Compton 散射（5–30 keV，Si-PIN）截面极小，未做估算。

---

## 11. 多信使与跨卫星协同：三角定位、交叉定标、存档符合

### Takeaway
HXMT/CsI 已是 GRB 联合能谱分析与 GW/中微子/FRB 逐事件跟踪的常规成员；在这个方向上没有发现需要新方法的空白，剩下的是若干"查表型"存档核对（如 KM3-230213A、LHAASO 暂现源）以及 §3 的群体叠加。

### Cited Findings
- [confirmed in literature] GOLDEN 样本与 Fermi/GBM、Swift/BAT、GECAM 做联合能谱分析，HXMT 在高能端提供更好的约束 — [Song et al. 2022, arXiv:2203.00878](https://arxiv.org/abs/2203.00878)
- [confirmed in literature] 交叉定标：10 个亮暴联合拟合，HXMT/CsI 相对其他仪器高 7.0±8.8% — [Luo et al. 2020, arXiv:2005.01367](https://arxiv.org/abs/2005.01367)
- [confirmed in literature] HXMT 对 GW170817 有专文观测报告 — ["Insight-HXMT observations of the first binary neutron star merger GW170817", arXiv:1710.06065](https://arxiv.org/pdf/1710.06065)
- [confirmed in literature] HXMT 与 GECAM-C 对 GRB 221009A 的联合观测已发表 — ["Insight-HXMT and GECAM-C observations of the brightest-of-all-time GRB 221009A", arXiv:2303.01203](https://arxiv.org/pdf/2303.01203)

### Inferences
- [inference] **时间三角定位**：CsI 的 2 μs 时间戳和 ~10 μs 级绝对计时不是瓶颈；瓶颈是基线。近地轨道卫星之间最长基线 ~1.3×10⁴ km（光行时 ~43 ms），对互相关精度 ~1 ms 的亮短暴，环带宽度约 1°–2°；与日地 L1 的探测器组合时可到 ~0.1° 量级。这是已有的行星际网络做法，HXMT 只是多一个节点。
- [speculation] **存档符合查表**：对 KM3-230213A（2023-02-13）、LHAASO 报告的暂现事件、IceCube 高能警报总表，在 CsI 数据中各查 T0±数小时是低成本的例行工作；是否已有人做过——未查。UHECR 多重态没有可用的时间窗，否决。

### Gaps
- HXMT 是否正式加入 IPN、是否已有 Li-CCF 类联合定位论文——未能检索（配额耗尽）。
- KM3-230213A 的原始论文未取回，事件时间为记忆值。

---

## 12. 考虑过但否决（或降级）的想法

### Takeaway
下列想法或者在量级上不成立，或者别的卫星明显更合适，或者已被团队做完；列出理由以免重复投入。

### Cited Findings
- [confirmed in literature] Sco X-1 毫秒下陷的仪器起源争议与 RXTE 的后续结果（334 ks 无确定事件）— [Jones et al. 2008, arXiv:0710.0837](https://arxiv.org/abs/0710.0837)；[Chang, Liu & Chen 2011, arXiv:1009.2547](https://arxiv.org/abs/1009.2547)；[Chang, Liu & Chen 2013, arXiv:1211.4650](https://arxiv.org/abs/1211.4650)
- [confirmed in literature] CXB Compton-Getting 偶极幅度 0.0042 — [Scharf et al. 2000, arXiv:astro-ph/9908187](https://arxiv.org/abs/astro-ph/9908187)；LE 1 ks 谱系统误差 ~4% — [Liao et al. 2020, arXiv:2004.01432](https://arxiv.org/pdf/2004.01432)
- [confirmed in literature] 70–200 keV 的银河脊除点源外无弥散辐射（IBIS）— [Krivonos et al. 2007, arXiv:astro-ph/0605420](https://arxiv.org/abs/astro-ph/0605420)
- [confirmed in literature] Crab 掩食的大气密度反演已由团队完成并持续更新 — [Xue et al. 2023, arXiv:2211.00293](https://web3.arxiv.org/pdf/2211.00293)；[Zhang H.-H. et al. 2025, arXiv:2502.19678](https://arxiv.org/abs/2502.19678)
- [confirmed in literature] TGF、太阳耀斑已在团队的暴分类中（四年 245 个 TGF、4 个太阳耀斑）— [Song et al. 2022, arXiv:2203.00878](https://arxiv.org/abs/2203.00878)

### Inferences
| 否决/降级的想法 | 理由 | 标签 |
|---|---|---|
| TNO/小天体掩 Sco X-1 的"干净重做" | HXMT 总计数率 ~5 cts/ms，比 RXTE/PCA 低 ~20 倍，2 ms 下陷被泊松涨落淹没；LE 读出抹开 ~1 ms；预期事件数 ≪1 | [inference] |
| 强度干涉（HBT）式测量 | X 射线相干时间 ~10⁻¹⁸ s，对 2 μs 分辨毫无希望 | [inference] |
| 量子引力"模糊"/Lorentz 破缺 | keV–MeV 能量杠杆不如 TeV 光子 | [inference] |
| CXB 偶极/各向异性 | 信号 0.4%，LE 系统误差 ~4%，高银纬指向天区覆盖稀疏 | [inference] |
| 用非准直 CsI 测 CXB | 地球始终挡住固定比例的天空，没有调制量；与反照简并 | [inference] |
| 511 keV 线/正电子素连续谱/²⁶Al 连续谱 | CsI 能量分辨不足以分出 511 keV 线；掩食法在银心方向被硬态黑洞点源连续谱混淆；SPI 已远超 | [inference]（SPI 流量未取文献） |
| Fermi/eROSITA 泡的硬 X 射线对应体 | 预期面亮度未知且应远低于 CXB；LE 空白天区覆盖不成图 | [speculation] |
| 准直望远镜临边掩食做成像/拥挤场分解 | 大气边缘模糊 ~0.1°–0.2°，不如扫描 PSF 拟合 | [inference] |
| 类 PTA 的 Crab 计时阵 | Crab 计时噪声毫秒级，MSP 对 HXMT 太暗 | [inference] |
| X–射电相位差的长期变化 | 十年变化 ~1 μs，射电系统误差 40 μs | [inference] |
| CsI 无指向脉冲星导航 | 轮廓显著性约为指向 HE 的 1%，定位退化到 ~10³ km | [inference] |
| 引力透镜视差（picolensing，HXMT–GBM 流量比） | 近地基线 ~10⁴ km 远小于 GRB 源投影尺度所要求的基线；两星交叉定标误差 ~10% 已大于信号 | [inference]（先例文献未取回） |
| 相邻模块 Compton 偏振 | 同组符合丢失、几何粗、MDP~70%（见 §10）；降级为一天量级的可行性实测 | [inference] |
| SN 激波暴、TDE、磁星 glitch 的 CsI 叠加 | 预期辐射谱峰远低于 200 keV，CsI 不敏感 | [inference] |
| 大气密度十年趋势、TGF 与雷电成协、SAA 演化 | 已由团队开展或属于地球物理/空间环境，不是天体物理新方法 | [inference] |

### Gaps
- 上表多数否决理由基于量级估算而非模拟；其中 picolensing、511 keV、泡状结构三项没有取回支撑文献。
- 若后续补做查新发现 §7 想法 10（LE 弥散谱线）已有人发表，应将其移入本表"已做"类。
