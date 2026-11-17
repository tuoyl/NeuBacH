# 类比迁移：其他空间任务的"非设计用途（off-label）"科学及其向 Insight-HXMT（慧眼）的可移植性评估

> 调研日期：2026-11-17。标注约定：**[confirmed in literature]** = 本次检索/抓取到原始文献或其摘要；**[inference]** = 我基于已引文献做的推断或量级估算；**[speculation]** = 尚无文献支撑的设想。
> 检索局限（重要）：本轮会话的 WebSearch 配额在约 70 次检索后被用尽（配额与其他并行研究员共享），之后只能对已知 arXiv/DOI/机构页面做定点抓取核实。因此（1）部分任务（HEAO-1、Konus-Wind、eROSITA、Kepler/TESS、WMAP、CSES、ASIM、XPNAV-1、LAXPC、Fermi-LAT ACD、AGILE-ACS 等）未能取得可核实的参考文献，已在 Gaps 中逐条列出，没有编造；（2）"Insight-HXMT + 关键词"的查新只覆盖了下文 KQ2 列出的方向，"未检索到"不等于"不存在"。
> 关于 HXMT 参数的更正（来自文献而非记忆）：HE 为 18 个 NaI(Tl)/CsI(Na) phoswich，总面积约 5100 cm²，15 个 1.1°×5.7°、2 个 5.7°×5.7°、1 个遮挡（blind）[HE 仪器论文](https://arxiv.org/abs/1910.04955)；ME 54 个模块约 950 cm²，48 个 1°×4°（其中 3 个 blind）、6 个 4°×4°；LE 约 380 cm²，63 个 1.6°×6°（3 个 blind）、21 个 6°×6°（3 个 blind）、6 个 60°×2.5° [Huang et al. 2022](https://arxiv.org/abs/2306.10114)。粒子监测器（PM）阈值：电子 1 MeV（另一文写 1.5 MeV）、质子 20 MeV [Lu et al. 2020](https://arxiv.org/abs/1911.01594)；[Liao et al. 2023](https://arxiv.org/abs/2302.14459)。

---

## KQ1：其他任务有哪些"off-label"科学案例？逐条能否移植到 HXMT？（按任务编目，共 67 例）

### Takeaway
跨 30 多个任务共整理出 67 个有文献支撑的案例，"废数据变信号"的套路高度集中在六类：反符合屏蔽当全天监视器、粒子/本底数据当空间天气产品、地球当遮挡体或靶、空白天区/杂散光做基础物理、仪器伪迹（毛刺、活化线）当物理量、本底模型当发现工具。其中约三分之一 HXMT 已经做过（GRB/TGF/太阳耀斑/SAA/大气密度/脉冲星导航/CXB/PBH），真正尚属空白且 HXMT 具备硬件条件的集中在"粒子沉降与地磁暴响应""地球作为 X 射线靶""CsI 地球掩食监测""活化线核数据""本底模型残差搜索"几类。

### Cited Findings
说明：每条"科学内容/数据类型"均为 **[confirmed in literature]**（括号内注明"检索摘要级"的表示只看到检索摘要而未打开原文）；"HXMT 判定"均为 **[inference]**，判定值为 Yes / Partly / No。

**A. RXTE / BeppoSAX / CGRO-BATSE（老一代准直或全天探测器）**

- **C01 RXTE（HEXTE 粒子监测器）→ SAA 强度与漂移**。用 1996–2007 年粒子监测器计数率（纯工程/保护数据）测得 SAA 西漂约 0.3°/yr，并有不规则跳变。— [Fürst et al. 2009, EPSL, arXiv:0902.2873](https://arxiv.org/abs/0902.2873)。**HXMT：Yes，已做一部分**（见 KQ2-H6，仅 2017–2019 两年）；43° 倾角覆盖整个 SAA，优于 RXTE 的 23°，可延长到近十年。
- **C02 RXTE/PCA → Sco X-1 毫秒级 dip 被解释为百米级 TNO 掩食，后被证明主要是高能带电粒子事件造成的死时间伪迹**。— [Chang et al. 2007, arXiv:astro-ph/0701850](https://arxiv.org/abs/astro-ph/0701850)；反驳：[Jones et al., arXiv:astro-ph/0612129](https://arxiv.org/abs/astro-ph/0612129)；后续只剩 1 个可能事件：[Chang et al. 2010, arXiv:1009.2547](https://arxiv.org/abs/1009.2547)。**HXMT：No**（LE 约 380 cm²、ME 约 950 cm²，远小于 PCA，Sco X-1 计数率低一个量级以上）；但"粒子事件伪装成天体信号"的教训可直接移植（见 C21）。
- **C03 RXTE/PCA 与 ARGOS/USA → X 射线掩星测中高层大气密度**。利用源进出地球掩食时的衰减段（通常被 GTI 剔除）。— [Determan et al. 2007, JGR 112, A06323（题录页）](https://earthref.org/ERR/112002)。**HXMT：Yes，已做**（KQ2-H8）。
- **C04 RXTE/PCA 转动（slew）数据 → 3–20 keV 全天巡天 XSS**，294 个源。— [Revnivtsev et al. 2004, A&A 418, 927](https://arxiv.org/abs/astro-ph/0402414)。**HXMT：Partly，价值低**——HXMT 银道面扫描本身是设计目标；机动段数据若有，灵敏度（本底系统误差 1–3%）远逊于 Swift/BAT、INTEGRAL、eROSITA 现有巡天。
- **C05 BeppoSAX/PDS 粒子监测器 → 低倾角 LEO 辐射环境、SAA 西漂约 0.4°/yr**。— [Campana et al. 2014, Exp. Astron. 37, 599](https://arxiv.org/abs/1405.0360)。**HXMT：Yes**（同 C01）。
- **C06 BeppoSAX GRBM：PDS 的反符合侧屏蔽被当作 GRB 监视器**。— [Feroci et al. 1997, arXiv:astro-ph/9708168](https://arxiv.org/abs/astro-ph/9708168)。**HXMT：Yes，已做**（CsI 即同类，KQ2-H1）。
- **C07 BATSE → 发现大气起源的 TGF**（为宇宙 GRB 设计的探测器看到了雷暴）。— Fishman et al. 1994, Science 264, 1313（题录见 [NASA NTRS 综述](https://ntrs.nasa.gov/api/citations/20120014488/downloads/20120014488.pdf)，检索摘要级）。**HXMT：Yes，已做**（KQ2-H4）。
- **C08 BATSE 地球掩食技术（EOT）**：无成像的全天探测器靠源进出地平线的台阶做 20 keV–1 MeV 全天监测。— [Harmon et al. 2002, ApJS 138, 149](https://arxiv.org/abs/astro-ph/0109069)。**HXMT：Partly→值得做**（见 C11、KQ3-S4）。
- **C09 BATSE 未触发暴目录**：离线重扫连续数据，新增 873 个未触发 GRB。— [Kommers et al. 2001（HEASARC 表）](https://heasarc.gsfc.nasa.gov/W3Browse/all/kommersgrb.html)。**HXMT：Yes，已做**（盲搜+定向搜索，KQ2-H2）。
- **C10 BATSE 存档光变曲线 → 引力透镜回声（GRB 950830）→ 中等质量黑洞证据**。— [Paynter, Webster & Thrane 2021, Nature Astron.](https://arxiv.org/abs/2103.15414)。**HXMT：Partly**——CsI 在 >200 keV 统计量大，可做自相关回声搜索，但样本（4 年 322 个）远小于 GBM，未检索到 HXMT 已做；增量价值有限。

**B. Fermi GBM / LAT**

- **C11 GBM 地球掩食监测**：3 年监测 209 个源、探测到 99 个；100 keV 以上两年数据探测到 8 个源，Crab 与 Cyg X-1 在 300 keV 以上可见。— [Wilson-Hodge et al. 2012, ApJS 201, 33](https://arxiv.org/abs/1201.3585)；[Case et al. 2011, ApJ 729, 105](https://arxiv.org/abs/1009.4953)。**HXMT：Partly→入选**（KQ3-S4）。
- **C12 GBM → 雷暴产生的正负电子束（TEB），511 keV 湮灭线**。— [Briggs et al. 2011, GRL 38, L02808（AGU 新闻稿）](https://news.agu.org/press-release/spacecraft-catches-thunderstorms-hurling-antimatter-into-space/)。**HXMT：Partly**——TGF 已做；CsI 能量分辨率差但面积大，TEB 是否已系统搜索未检索到。
- **C13 GBM 全天本底谱 → 20–50 keV 惰性中微子暗物质衰变线上限**。— [Ng et al. 2015, PRD 92, 043503](https://arxiv.org/abs/1504.04027)。**HXMT：Partly**（见 KQ3-S9，坦率说不占优）。
- **C14 GBM 暴谱 → femtolensing 约束 PBH；后被证明因源尺寸和波动光学效应而不成立**。— [Barnacka et al. 2012](https://arxiv.org/abs/1204.2056)；[Katz et al. 2018, JCAP 12, 005](https://arxiv.org/abs/1807.11495)。**HXMT：No**（方法本身已被推翻）。
- **C15 GBM 本底模型变成发现工具**：神经网络预测本底 + 变点检测，找到触发目录之外的长而弱事件（DeepGRB）；物理本底模型用于超长暴。— [Crupi et al. 2023, Exp. Astron. 56, 421](https://arxiv.org/abs/2303.15936)；[Biltzinger et al. 2020, A&A 640, A8](https://arxiv.org/abs/2005.11219)。**HXMT：Yes→入选首位**（KQ3-S1；用户自己的 NeuBacH 就是这个起点）。
- **C16 GBM 把"粒子沉降（particle precipitation）"作为一类触发事件分类**。— [Paciesas et al. 2012, GBM GRB Catalog（前两年）](https://arxiv.org/abs/1201.3099)。**HXMT：Yes→入选**（KQ3-S2）。
- **C17 Fermi-LAT → 月球 γ 射线反推近地宇宙线质子/氦谱及太阳调制**。— [Ackermann et al. 2016, PRD 93, 082001](https://arxiv.org/abs/1604.03349)。**HXMT：No**（能段不符，月球很少进入准直视场）。
- **C18 Fermi-LAT → 宁静太阳的盘面与逆康普顿两成分 γ 辐射**。— [Abdo et al. 2011, ApJ 734, 116](https://arxiv.org/abs/1104.2093)。**HXMT：No**（太阳规避）。
- **C19 Fermi-LAT → 地球临边 γ 射线反推 90 GeV–6 TeV 宇宙线质子谱**（地球临边原是要剔除的本底）。— [Ackermann et al. 2014, PRL](https://arxiv.org/abs/1403.5372)。**HXMT：Partly**——能段不同，但"地球大气辐射当宇宙线示踪"可移植为硬 X 射线大气辐射研究（KQ3-S3）。

**C. INTEGRAL**

- **C20 SPI-ACS：512 kg BGO 反符合屏蔽当准全向 GRB 探测器**（约 0.8 个候选/天）。— [von Kienlin et al. 2003, A&A 411, L299](https://arxiv.org/abs/astro-ph/0308346)。**HXMT：Yes，已做**。
- **C21 SPI-ACS 的"短尖峰"**：约 30 个/天、通常只占一个 50 ms bin，归因于高能宇宙线击中 BGO；作者也指出其中一部分可能是漏检的真 GRB。— [Savchenko, Neronov & Courvoisier 2012, A&A](https://arxiv.org/abs/1203.1344)。**HXMT：Yes→入选**（KQ3-S7）；HXMT 定向搜索论文已把"光变曲线中的 spike"当作假触发剔除 [Cai et al. 2021](https://arxiv.org/abs/2109.12270)，但未见把它当研究对象。
- **C22 INTEGRAL 让地球穿过视场，用地球遮挡测 5–100 keV CXB**。— [Churazov et al. 2007, A&A 467, 529](https://arxiv.org/abs/astro-ph/0608250)；[Türler et al. 2010](https://arxiv.org/abs/1001.2110)。**HXMT：Yes，已做**（KQ2-H11），但留有明确的改进空间。
- **C23 INTEGRAL 的 CXB 地球观测意外拍到极光硬 X 射线；2012 年的地球观测还被近地点穿越质子带后的活化污染**。— [ESA: INTEGRAL X-rays Earth's aurora](https://sci.esa.int/web/integral/-/57257-integral-x-rays-earths-aurora)；[Türler et al. 2013, arXiv:1302.6493](https://arxiv.org/abs/1302.6493)（检索摘要级）。**HXMT：Partly→入选**（KQ3-S3）。
- **C24 INTEGRAL/IREM 与 XMM 辐射监测器的 16–19 年工程数据 → 数据驱动的辐射带电子模型 5DRBM-e**。— [Métrailler et al. 2019, Adv. Space Res. 64, 1701](https://arxiv.org/abs/1907.11029)。**HXMT：Partly**——LEO 只能采样 SAA 与外带内缘（L≲3），做不了全局模型，但可做 550 km 高度的长期经验图。
- **C25 SPI 空天区/银河弥散谱 → 暗物质衰变线与 PBH 蒸发约束**。— [Boyarsky et al. 2008, MNRAS 387, 1345](https://arxiv.org/abs/0710.4922)；[Laha, Muñoz & Slatyer 2020, PRD 101, 123514](https://arxiv.org/abs/2004.00627)。**HXMT：Partly，PBH 已做**（KQ2-H12）。
- **C26 SPI 的 383 条仪器本底线（13.5 年）**：按探测器分布归类来源，本底强度与太阳活动反相关，并记录了 Ge 探测器的辐照退化。— [Diehl et al. 2018, A&A 611, A12](https://arxiv.org/abs/1710.10139)。**HXMT：Yes→入选**（KQ3-S5）。

**D. Swift / Suzaku / Hitomi / NuSTAR**

- **C27 Swift/BAT 在三次安全模式期间地球穿过视场 → 15–200 keV CXB 与地球反照谱**。— [Ajello et al. 2008, ApJ 689, 666](https://arxiv.org/abs/0808.3377)。**HXMT：Yes/Partly**（已测 CXB；地球反照本身未作为研究对象，见 S3）。
- **C28 Swift/BAT GUANO：把本来不下传的事件数据按需抢救下来做亚阈值搜索**。— [Tohuvavohu et al. 2020, ApJ](https://arxiv.org/abs/2005.01751)。**HXMT：Yes（天然具备）**——HXMT 全程下传事件数据，这是它相对 BAT 的固有优势。
- **C29 Suzaku/WAM：HXD 的 BGO 反符合屏蔽当 50 keV–5 MeV 全天监视器**（每面约 800 cm²，约 140 GRB/年）。— [Yamaoka et al. 2009, PASJ 61, S35（DARTS 说明页）](https://data.darts.isas.jaxa.jp/pub/suzaku/wam-1.0/wam-grb/intro/wam_intro.html)（检索摘要级）。**HXMT：Yes，已做**。
- **C30 Suzaku/XIS：检验弥散 X 射线本底是否随视线方向地磁场强度变化，以约束 ALP→光子转换**；99% 上限相当于 2–6 keV CXB 面亮度的 6–15%。— [Yamamoto et al. 2020, JCAP, arXiv:1906.04429](https://arxiv.org/abs/1906.04429)。**HXMT：Partly→入选**（KQ3-S8）。
- **C31 Suzaku/XIS 与 Hitomi/HXI 的 Crab 掩食 → 70–200 km 大气密度；多任务合并得到 71–116 km 约 −5%/十年的长期趋势**。— [Katsuda et al. 2021, JGR 126, e28886](https://arxiv.org/abs/2103.07602)；[Katsuda et al. 2023, JGR 128, e2022JA030797](https://repository.kulib.kyoto-u.ac.jp/dspace/bitstream/2433/279487/1/2022JA030797.pdf)。**HXMT：Yes，已做**（H8）。
- **C32 Suzaku → 地球磁鞘太阳风电荷交换（SWCX）C VI 线**。— [Fujimoto et al. 2007](https://arxiv.org/abs/astro-ph/0609308)。**HXMT：No**（SWCX 主线 <1 keV，LE 下限约 1 keV）。
- **C33 Hitomi/HXI：在轨本底由 SAA 质子活化线主导，用作 Geant4 活化模拟的在轨验证**。— [Odaka et al. 2018](https://arxiv.org/abs/1804.00827)；[Hagino et al. 2018, HXI 在轨性能](https://arxiv.org/abs/1805.08344)。**HXMT：Yes→入选**（KQ3-S5）。
- **C34 NuSTAR 的 0-bounce 杂散光（原为污染）→ 6–40 keV 惰性中微子最强上限**；11 年数据的再分析提到 40 keV 附近的边缘超出是已知仪器效应。— [Roach et al. 2023, PRD 107, 023009](https://arxiv.org/abs/2207.04572)；[Krivonos et al. 2024, arXiv:2405.17861](https://arxiv.org/abs/2405.17861)。**HXMT：Partly**（S9）。
- **C35 NuSTAR StrayCats：视场外 1–4° 亮源的杂散光被编成目录用于科学**（78 个源、436 个视场）。— [Grefenstette et al. 2021, ApJ 909, 30](https://arxiv.org/abs/2102.01236)。**HXMT：Partly**——对应物是大视场探测器与 LE 60°×2.5° 探测器里的"场外源污染"，可做亮源附带监测，但无成像，辨源困难。
- **C36 NuSTAR 用侧孔径 CXB 在焦面上的空间调制测 3–20 keV CXB**（抓取摘要称比 HEAO-1 高"8 倍"，原文应为约 8%，需复核）。— [Krivonos et al. 2021, MNRAS 502, 3966](https://arxiv.org/abs/2011.11469)。**HXMT：Yes，已做**；HXMT 的等价手段是小/大/遮挡视场的差分。
- **C37 NuSTAR 当太阳轴子望远镜**（g_aγ ≲ 7.3×10⁻¹² GeV⁻¹）及首批聚焦硬 X 射线太阳像。— [Ruz et al. 2025, PRL 135, 141001](https://arxiv.org/abs/2407.03828)；[Grefenstette et al. 2016, ApJ 826, 20](https://arxiv.org/abs/1605.09738)。**HXMT：No**（不能指向太阳）。

**E. AstroSat / POLAR / GECAM / 立方星**

- **C38 AstroSat/CPM：为保护仪器而设的带电粒子监测器给出 SAA 信号与粒子通量**。— [Rao et al. 2017, JApA 38, 33](https://arxiv.org/abs/1608.06038)。**HXMT：Yes**（同 PM）。
- **C39 AstroSat/CZTI：编码孔径成像仪在 100 keV 以上当离轴 GRB 监视器与偏振计**（GRB 151006A）。— [Rao et al. 2016, ApJ 833, 86](https://arxiv.org/abs/1608.07388)。**HXMT：Partly**——GRB 监测已做；偏振见 KQ4。
- **C40 POLAR（天宫二号 GRB 偏振计）→ 用 31 天 Crab 脉冲星数据定轨，偏差 20 km（1σ）**。— 见 [Zheng et al. 2019 正文（ar5iv）](https://ar5iv.labs.arxiv.org/html/1908.01922)（检索摘要级）。**HXMT：Yes，已做**（H9）。
- **C41 GECAM → 9 个月 147 个 TGF、2 个 TEB**。— [Zhao et al. 2023, GRL](https://arxiv.org/abs/2306.10255)。**HXMT：Yes，已做**。
- **C42 GECAM → 2024-03-21 磁暴期间连续三轨同一区域的带电粒子沉降事件，其中两个有显著周期性，持续 >3.5 小时，与闪电无关**。— [Wang et al. 2025, arXiv:2505.06192](https://arxiv.org/abs/2505.06192)。**HXMT：Yes→入选**（S2）；未检索到 HXMT 同类工作。
- **C63 GRBAlpha / VZLUSAT-2 立方星 → 测 LEO 本底水平（含外辐射带与 SAA），约 67% 轨道时间可用于暴探测，并看到太阳耀斑**。— [Řípa et al. 2022, Proc. SPIE](https://arxiv.org/abs/2207.03272)；[Pál et al. 2023](https://arxiv.org/abs/2302.10048)。**HXMT：Yes**（本底地理分布已在本底模型论文中给出，可升级为公开产品）。

**F. 宇宙线/粒子实验（AMS-02、DAMPE、PAMELA、CALET、MAXI）**

- **C43 DAMPE → 2017 年 9 月宇宙线电子/正电子的 Forbush 下降及其能量依赖**。— [Alemanno et al. 2021, ApJL 920, L43](https://arxiv.org/abs/2110.00123)。**HXMT：Partly→入选**（S6）；2017 年 9 月 HXMT 已在轨。
- **C44 DAMPE → 相对论性分数电荷粒子搜索**。— [DAMPE Collab. 2022, PRD 106, 063026](https://arxiv.org/abs/2209.04260)。**HXMT：No**（无电荷测量）。
- **C45 AMS-02 → 逐日质子通量的周期性（日球物理副产品）**。— PRL 127, 271102 (2021)（[DOI 可解析，但页面 403 未能打开，题目凭记忆](https://link.aps.org/doi/10.1103/PhysRevLett.127.271102)）。AMS-02 strangelet 搜索：**reference not found**（只见二手综述 [arXiv:2404.12094](https://arxiv.org/abs/2404.12094) 提及）。**HXMT：No**。
- **C46 PAMELA → 在 SAA 发现地磁俘获反质子带**；并做过 strangelet 搜索（仅二手综述级）。— [Adriani et al. 2011, ApJL 737, L29](https://arxiv.org/abs/1107.4882)。**HXMT：No**。
- **C47 CALET 电荷探测器（CHD）计数率 → ISS 上的相对论电子沉降（REP）事件；2024 年 5 月超级磁暴后新生的多 MeV 电子带内移到 L≈2.2 并持续数月**；MAXI/RBM（保护 GSC 的 1 s 计数器）也被用于 REP 研究。— [Kataoka et al. 2016, GRL 43, 4119](https://galileo.dsfta.unisi.it/~maestro/articoli/CALET/SpaceWeather/Geophysical%20Research%20Letters%20-%202016%20-%20Kataoka%20-%20Relativistic%20electron%20precipitation%20at%20International%20Space%20Station%20%20Space.pdf)；[Ficklin et al. 2026, GRL 接收, arXiv:2602.03990](https://arxiv.org/abs/2602.03990)；[MAXI/RBM 数据说明（DARTS）](https://data.darts.isas.jaxa.jp/pub/metadata/analysis-playbooks/maxi-rbm-timeseries/playbook.md)；ISS 多仪器 REP 剂量论文 [Space Weather, doi:10.1029/2019SW002280](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019SW002280)（页面 403，第一作者未核实）。**HXMT：Partly→入选**（S2）。

**G. XMM-Newton / Chandra / HST / Gaia / LISA Pathfinder / Planck**

- **C48 XMM 转动巡天 XMMSL1**（218 次 slew，约 14% 天区）。— [Saxton et al. 2008, A&A 480, 611](https://arxiv.org/abs/0801.3732)。**HXMT：Partly，价值低**（同 C04）。
- **C49 XMM → SWCX 识别**。— [Snowden et al. 2004](https://arxiv.org/abs/astro-ph/0404354)；[Carter & Sembay 2008](https://arxiv.org/abs/0807.3624)。**HXMT：No**。
- **C50 XMM X 射线本底的季节性变化 → 被解释为太阳轴子在地磁场中的转换（有争议）**。— [Fraser et al. 2014, MNRAS 445, 2146](https://arxiv.org/abs/1403.2436)。**HXMT：Partly**（并入 S8）。
- **C51 XMM MOS2 未曝光角落约 255 Ms（15 年）→ 非聚焦本底与 Chandra 本底代理量、EPIC 辐射监测器一致，来源是银河宇宙线**。— [Gastaldello et al. 2022, ApJ 928, 168](https://openaccess.inaf.it/bitstreams/9de1a801-f231-427e-a42a-1a94c8e30f7a/download)。**HXMT：Yes→入选**（S6；遮挡探测器就是"未曝光角落"）。
- **C52 XMM 空白天区 → 否定 3.5 keV 线的暗物质解释；547 Ms 深度搜索 5–16 keV 衰变线**。— [Dessert et al. 2020, Science 367, 1465](https://arxiv.org/abs/1812.06976)；[Foster et al. 2021, PRL 127, 051101](https://arxiv.org/abs/2102.02207)。**HXMT：Partly**（S9）。
- **C53 HST 25 年约 7.6 万幅暗场定标帧 → 12 亿条宇宙线径迹，看到 SAA、太阳周期调制与地磁场结构**。— [Miles et al. 2021, ApJ 918, 86](https://arxiv.org/abs/2006.00909)。**HXMT：Partly**——LE 的 SCD 不成像，无径迹；等价物是遮挡探测器与 ACD 计数率（S6）。
- **C54 Gaia 焦面 106 块 CCD 当 L2 辐射监测器**，记录了 2017-09-10 X8.2 耀斑后的最大粒子通量。— [ESA Gaia: Cosmic Ray detections on Gaia CCDs](https://www.cosmos.esa.int/web/gaia/cosmic-ray-detections-on-gaia-ccds)；[Kirsch 2018 硕士论文](https://www.sternwarte.uni-erlangen.de/docs/theses/2018-10_Kirsch.pdf)。**HXMT：Partly**（同 C53）。
- **C55 LISA Pathfinder 诊断用粒子探测器 → 银河宇宙线的重现性下降与 2016-08-02 Forbush 下降**。— [Armano et al. 2018, ApJ 854, 113](https://arxiv.org/abs/1802.09374)。**HXMT：Partly→入选**（S6）。
- **C56 LISA Pathfinder → 54 次微流星体撞击的动量测量**。— [Thorpe et al. 2019, ApJ 883, 53](https://arxiv.org/abs/1905.02765)。**HXMT：No**（姿态测量精度差很多个量级）。
- **C57 Planck/HFI 的宇宙线毛刺**：三类毛刺、来源定位到硅片衬底，毛刺率跟随宇宙线通量。— [Planck 2013 results X, A&A 571, A10](https://arxiv.org/abs/1303.5071)。**HXMT：Partly**（并入 S7）。

**H. GPS / RHESSI / AGILE / NICER / 其他**

- **C58 GPS 星载原子钟 16 年存档 → 拓扑缺陷（畴壁）暗物质搜索**。— [Roberts et al. 2017, Nature Commun. 8, 1195](https://arxiv.org/abs/1704.06844)。**HXMT：No**（星载晶振需靠 Crab 校准，[Tuo et al. 2022](https://arxiv.org/abs/2109.04709)）。
- **C59 GPS 星座的 CXD 高能粒子数据（16 年以上）于 2017 年公开，成为辐射带研究数据集**。— [Morley et al. 2017, Space Weather](https://www.osti.gov/pages/biblio/1345950)。**HXMT：Partly**——启示是把 PM/ACD/遮挡探测器数据整理成公开的空间天气产品。
- **C60 RHESSI 的太阳指向传感器（工程用）→ 迄今最精确的太阳扁率测量**。— [NASA 2008 新闻稿（Fivian et al., Science）](https://webarchive.library.unt.edu/eot2008/20081003205033/http://www.nasa.gov/topics/solarsystem/features/oblate_sun.html)；[Fivian & Hudson 2014, AGU](https://ui.adsabs.harvard.edu/abs/2014AGUFMSH41C4156F)。**HXMT：No**（无对应硬件）。
- **C61 AGILE/MCAL → TGF 能谱延伸到 40 MeV**。— [Marisaldi et al. 2010, JGR 115, A00E13](https://ui.adsabs.harvard.edu/abs/2010JGRA..115.0E13M)。**HXMT：Yes，已做**（能量上限 3 MeV，不如 MCAL）。
- **C62 NICER/SEXTANT → X 射线脉冲星自主导航在轨演示**。— [Mitchell et al. 2018, NASA NTRS](https://ntrs.nasa.gov/citations/20180001252)。**HXMT：Yes，已做**（H9）。
- **C64 公民科学**：Gravity Spy（LIGO 毛刺分类 + 机器学习）；Burst Chaser（Swift/BAT 与 Fermi/GBM 的 GRB 脉冲形状分类）。— [Zevin et al. 2017, CQG 34, 064003](https://arxiv.org/abs/1611.04596)；[Burst Chaser（Zooniverse）](https://www.zooniverse.org/projects/amylien/burst-chaser/about/research)。**HXMT：Partly**——技术上可行（如给 CsI 尖峰/粒子事件分类打标签），科学驱动弱。
- **C65 用多任务 GRB 时间特征给 Lorentz 破缺设稳健下限**。— [Ellis et al. 2006, Astropart. Phys. 25, 402](https://arxiv.org/abs/astro-ph/0510172)。**HXMT：Partly，价值低**——keV–MeV 能量杠杆远不如 GeV–TeV。
- **C66 概念方案：夜侧低轨 X 射线探测器透过地球看太阳方向，探测太阳轴子在地磁场中转换出的约 4 keV 光子**。— [Davoudiasl & Huber 2006, PRL 97, 141302](https://arxiv.org/abs/hep-ph/0509293)。**HXMT：No**（见 KQ4）。
- **C67（跨领域锚点）NaI(Tl) 晶体的宇宙线活化核素产额**：ANAIS 用两块 12.5 kg 晶体首次直接估计海平面产额（Te、I 同位素与 ²²Na）。— [Amaré et al. 2015, JCAP 02, 046](https://arxiv.org/abs/1411.0106)。**HXMT：Partly→入选**（S5）。

### Inferences
- [inference] 67 例里按"废料类型"粗分：反符合/屏蔽数据 6 例，粒子监测与工程数据 12 例，地球遮挡/地球辐射 9 例，空白天区或杂散光做基础物理 10 例，仪器伪迹（毛刺、活化线、死时间、未曝光像元）9 例，转动/未触发/亚阈值副产品 7 例，其余为导航、公民科学、概念方案等。
- [inference] HXMT 与原任务相比真正占优的硬件特征只有四条：（1）约 575 kg 的 CsI 分成 18 个独立读出模块并带 ACD 标记和微秒计时（比 SPI-ACS 的单路 50 ms 计数率信息多得多；质量按 5100 cm²×约 25 cm×4.51 g/cm³ 估算，CsI 厚度凭记忆，未核实）；（2）三个望远镜都有遮挡探测器，同时记录"纯粒子本底"；（3）43° 倾角介于 GECAM（29°）与 ISS（51.6°）之间，可达 L≈3；（4）全程事件数据下传、近十年同一仪器。其余方面（能量分辨率、成像、视场、能量上限）HXMT 多数不如原任务。
- [inference] 最密集的"共同测试窗口"有三个：2017 年 9 月（X9.3 耀斑、DAMPE 的 Forbush 下降、Gaia 记录的质子事件，HXMT 当时正在观测 Crab）、2024-03-21 磁暴（GECAM 振荡沉降事件）、2024 年 5 月超级磁暴（CALET 看到 L≈2.2 的新电子带）。任何 HXMT 空间环境类想法都应先用这三个窗口验证。

### Gaps
- 以下任务/类别因检索配额用尽**未取得可核实文献，未编入目录**：HEAO-1、HEAO-3 活化线、Konus-Wind、eROSITA（SWCX/滤光轮关闭数据）、Kepler/TESS（宇宙线与散射光）、WMAP、CSES、ASIM、XPNAV-1、AstroSat/LAXPC、AstroSat/CZTI 地球掩食（Singhal et al. 2021，凭记忆存在但未核实）、Fermi-LAT ACD、AGILE 反符合/ratemeter、RHESSI TGF（Smith et al. 2005）与 RHESSI GRB 偏振、Subaru-HSC/DECam/CREDO 宇宙线径迹、单粒子翻转当粒子监测器、轨道衰减反演热层密度、气辉、声化（sonification）。
- RXTE/HEXTE 与 ASM、INTEGRAL/IBIS 没有找到独立于上述条目的 off-label 案例。
- C07、C23（Türler 2013）、C29、C40、C45、C47（Space Weather 论文）只到检索摘要级或 DOI 级，正文未打开。

---

## KQ2：HXMT 自己已经做过哪些 off-label 科学？（查新基线）

### Takeaway
HXMT 团队已把"CsI 当全天 γ 监视器""掩食测大气""粒子监测器看 SAA""脉冲星导航""空白天区测 CXB"都做成了论文，所以这些方向只剩"延长基线、换问题"的增量空间；而本次检索没有找到 HXMT 在 Forbush 下降/磁暴/电子沉降、惰性中微子或轴子、地球掩食源监测目录、透镜回声或 LIV 方面的工作。

### Cited Findings
- **H1 GRB**：CsI 在 0.2–3 MeV 当全天监视器，前四年离线搜索到 322 个 GRB。— [Song et al. 2022, ApJS, arXiv:2203.00878](https://arxiv.org/abs/2203.00878) [confirmed in literature]
- **H2 亚阈值/定向搜索**：盲搜 + 相干定向搜索，两年数据多找回 40 个 GRB，灵敏度约 1.5×10⁻⁸ erg cm⁻²（200 keV–3 MeV）；改进方法以剔除"光变曲线中的 spike"引起的假触发；O2/O3a 引力波事件无显著对应体。— [Cai et al. 2021, MNRAS, arXiv:2109.12270](https://arxiv.org/abs/2109.12270) [confirmed in literature]
- **H3 CsI 响应**：响应主要靠 Geant4 质量模型，用不同入射方向的 Crab 脉冲辐射检验；正常增益与低增益两种模式分别定标；10 个亮暴的流量比其他仪器高约 7%。— [Luo et al. 2020, JHEAp, arXiv:2005.01367](https://arxiv.org/abs/2005.01367) [confirmed in literature]
- **H4 TGF**：Zenodo 数据集包含 282 个 HXMT TGF 的事件数据（含探测器号、道址、脉宽、ACD 标记、增益模式），对应论文题为"First Results of Terrestrial Gamma-ray Flash observations by Insight-HXMT"（期刊信息该页未给出）。— [Zhao Yi 2024, Zenodo 10976454](https://zenodo.org/records/10976454) [confirmed in literature]
- **H5 太阳耀斑**：CsI 观测到 2017-09-06 X9.3 耀斑，100–800 keV 能谱演化与约 22 s 的准周期脉动。— [Zhang et al. 2021, ApJ 918, 42](https://ar5iv.arxiv.org/html/2106.09506)；多波段联合：[Li et al. 2021, arXiv:2108.03781](https://arxiv.org/abs/2108.03781) [confirmed in literature]
- **H6 SAA**：用粒子监测器测得 2017-06 至 2019-06 SAA 西漂约 0.43°/yr、北漂约 0.17°/yr（Zhao et al., JHEAp）。— [检索结果条目（抓取时该链接返回 404，题录与数值来自检索摘要，需复核）](https://bohrium.dp.tech/paper/arxiv/56d81cf1dabfae2eeea10cac) [confirmed in literature，检索摘要级]
- **H7 粒子监测器**：3 台塑料闪烁体 + PMT，用于触发主探测器开关机并可绘制 SAA 图。— [Lu et al. 2020, arXiv:1911.01594](https://arxiv.org/abs/1911.01594) [confirmed in literature]
- **H8 大气密度**：LE 单次 Crab 掩食反演 85–200 km 密度（为 NRLMSISE-00 的 81–89%）；115 组掩食的统计研究（ApJS 264，检索摘要级）；2022–2024 年 83 组掩食、三台望远镜、55–130 km，发现 NRLMSIS 在 90–95 km 与 100–120 km 高估约 20%。— [Yu et al. 2022, AMT 15, 3141](https://arxiv.org/abs/2204.09674)；[ApJS 264 (2022)（DOAJ 条目）](https://doaj.org/article/cafb0de60cb54e51948813f066fb5302)；[Zhang et al. 2025, arXiv:2502.19678](https://arxiv.org/abs/2502.19678) [confirmed in literature]
- **H9 脉冲星导航**：2017-08-31 至 09-05 的 Crab 观测，SEPO 方法定轨，位置 10 km、速度 10 m/s（3σ）。— [Zheng et al. 2019, ApJS 244, 1](https://arxiv.org/abs/1908.01922) [confirmed in literature]
- **H10 时间系统**：晶振温漂与长期漂移修正；与 NICER 的 Crab 到达时间比较，系统误差 HE 12.1 μs、ME 8.6 μs、LE 15.8 μs。— [Tuo et al. 2022, ApJS, arXiv:2109.04709](https://arxiv.org/abs/2109.04709) [confirmed in literature]
- **H11 弥散 X 射线本底**：1.5–120 keV 的 CXB 谱；LE 用遮挡探测器测本底（只用了 2019-07 以前的 117 ks，"因为 LE 后期性能变差"），HE 因各模块能量分辨率不同、遮挡盖板影响谱形而无法用遮挡探测器，只能用地球遮挡数据（空白天区 377 ks、被遮挡 187 ks），并需修正地球反照与大气辐射；拟合得反照因子 A_albedo=1.70±0.14、大气辐射归一化 C=(2.4±0.4)×10⁻³；ME 因"性能不稳定"未用；作者认为再加数据也无改善（系统误差主导），但提出"或可仅用遮挡探测器改进 HE 本底模型"。— [Huang et al. 2022, MNRAS 513, 4074, arXiv:2306.10114](https://arxiv.org/abs/2306.10114) [confirmed in literature]
- **H11b**：同文指出 LE 的地球遮挡数据"似乎存在漏光问题，可能与极光辐射有关"，因此没用它评估本底。— [Huang et al. 2022](https://arxiv.org/abs/2306.10114) [confirmed in literature]
- **H11c 银道面弥散本底**（LE）。— [Jin et al. 2022, ApJS 260, 42（题录页）](https://dissem.in/p/133274586/the-diffuse-x-ray-background-of-the-insight-hxmtle-telescope-in-the-galactic-plane) [confirmed in literature，检索摘要级]
- **H12 原初黑洞**：用 HXMT 数据约束 2×10¹⁶–5×10¹⁷ g 的 PBH，三种本底假设下只有幂律加指数截断模型时优于已有上限，排除到 4×10¹⁷ g。— [Yang, Pan & Zhang, Commun. Theor. Phys. 78 (2026) 045402, arXiv:2412.09297](https://arxiv.org/abs/2412.09297) [confirmed in literature]
- **H13 本底的物理图像**：LE、ME 本底以瞬发成分为主，HE 以晶体被带电粒子活化后的延迟成分为主；HE 本底在每次穿越 SAA 后快速衰减，并随在轨天数长期累积（头几百天上升快，之后变缓）；ME 本底与地磁截止刚度明显反相关；PM 计数率的地理分布五年内几乎不变，磁赤道附近稳定在 1.7–1.8 cts/s；HE 只有一个遮挡探测器，被视为本底估计的潜在隐患，文中建议备用方案是用 ACD 和 PM 当瞬发粒子监测器。— [Liao et al. 2023, RDTM, arXiv:2302.14459](https://arxiv.org/abs/2302.14459) [confirmed in literature]
- **H13b**：HE 模拟与在轨比较显示总谱和约 67 keV 线的相对强度随时间上升；LE 观测到模拟中没有的约 7.5 与 8.0 keV 线。— [Zhang J. et al. 2020, Ap&SS, arXiv:2009.14489](https://arxiv.org/abs/2009.14489) [confirmed in literature]
- **H13c 本底模型**：HE、ME 模型按 5°×5° 地理网格存谱形，再用遮挡探测器的同时计数率定标；HE 系统误差约 1.5%（26–100 keV），ME 约 1.3%（8.9–44 keV）。— [Liao et al. 2020（HE）](https://arxiv.org/abs/2005.01661)；[Guo et al. 2020（ME）](https://arxiv.org/abs/2003.06260) [confirmed in literature]
- **H14 HE 五年性能**：NaI 很稳定；CsI 增益持续上升，需定期定标。— [Li et al. 2023, RDTM, arXiv:2302.03859](https://arxiv.org/abs/2302.03859) [confirmed in literature]
- **H15 CsI 的 Crab 掩食**：2025 年一项工作用 Crab 掩食数据评估指向观测时 CsI 的能量响应。— [检索结果条目："Improved performance of Insight-HXMT/HE CsI detectors"](https://www.bohrium.com/paper-details/improved-performance-of-insight-hxmt-he-csi-detectors/1136687896430903320-491) [confirmed in literature，检索摘要级]
- **未检索到的方向**（各做过一次定向检索，结果为空）：HXMT + Forbush/geomagnetic storm/electron precipitation（只返回本底模型论文和 GECAM 的沉降事件）；HXMT + sterile neutrino/axion（只返回 H12）；HXMT + Earth occultation 源监测目录；HXMT + 透镜 GRB / Lorentz invariance violation。— 例：[GECAM 沉降事件是该检索中唯一相关结果](https://arxiv.org/abs/2505.06192)

### Inferences
- [inference] H11 的作者自己留下了两个"未完成事项"：仅用遮挡探测器重建 HE 本底以摆脱地球反照修正，以及 LE 地球遮挡数据里的"漏光/极光"疑点。两者都可以直接由 NeuBacH 这类模型接手。
- [inference] H13 说明 PM 在磁赤道的计数率太低且对太阳调制不敏感（高截止刚度），做宇宙线调制研究必须用高纬（低截止刚度）轨道段，而这正是标准筛选（如 H11 中 COR>8）丢弃的数据。
- [inference] 约 67 keV 线的长期增长与 ¹²⁵I（半衰期约 59 天，EC 衰变在 NaI 内的和峰约 67 keV）相符；论文摘要没有做这个认证，这是我的核物理推断。

### Gaps
- 未能核实 H4 对应论文的期刊与年份、H6 的完整题录、H15 的作者与期刊。
- 未检索：HXMT 空间环境监测器（SEM）是否有科学论文；HXMT 是否已有 TEB、太阳耀斑目录（除 X9.3 之外）、2024 年 5 月磁暴期间本底的研究。
- 公开数据中是否包含 ACD 计数率、PM、SEM 及全部 HK 文件：未知，需向 HXMT 科学数据中心确认。

---

## KQ3：最值得尝试的 12 项移植（排序后的候选想法）

### Takeaway
排在最前面的是三件"现成数据 + 现成工具"的事：把 NeuBacH 的预测残差当作全档案异常探测器、用被筛掉的高本底段做粒子沉降与磁暴响应目录、把地球遮挡段当作"对地观测"数据研究地球硬 X 射线辐射；基础物理类（ALP、暗物质线）新颖但受系统误差限制，HXMT 并不比原任务更合适，应降低预期。

### Cited Findings
（各想法所依赖的关键事实；想法本身见 Inferences）
- 神经网络本底 + 变点检测能在 GBM 存档里找到触发目录之外的事件，包括类似太阳耀斑的特征。— [Crupi et al. 2023](https://arxiv.org/abs/2303.15936)
- NeuBacH 用 BERT 架构，输入高压、粒子监测器计数率、SAA、仰角、位置、遮挡探测器谱等，HE 模型已完成。— 项目自述 README（本仓库），模型与测试集见 [Zenodo 17476734](https://zenodo.org/records/17476734)（未在线核实）
- GECAM 在磁暴期间看到持续数小时、带周期性的粒子沉降；CALET 看到 2024 年 5 月后内移到 L≈2.2 的新电子带。— [Wang et al. 2025](https://arxiv.org/abs/2505.06192)；[Ficklin et al. 2026](https://arxiv.org/abs/2602.03990)
- HXMT HE 小视场下，地球反照是临边增亮、大气辐射是中心增亮；大气辐射还依赖太阳调制水平与截止刚度。— [Huang et al. 2022](https://arxiv.org/abs/2306.10114)
- INTEGRAL 的地球观测曾被极光硬 X 射线"淹没"。— [ESA](https://sci.esa.int/web/integral/-/57257-integral-x-rays-earths-aurora)
- GBM 用 NaI 在 100 keV 以上两年探测到 8 个源。— [Case et al. 2011](https://arxiv.org/abs/1009.4953)
- HE 本底由活化主导并有长期累积；SPI 与 Hitomi 都把活化线做成了独立的仪器物理/核数据论文。— [Liao et al. 2023](https://arxiv.org/abs/2302.14459)；[Diehl et al. 2018](https://arxiv.org/abs/1710.10139)；[Odaka et al. 2018](https://arxiv.org/abs/1804.00827)
- XMM 未曝光角落的计数率是银河宇宙线的良好代理量。— [Gastaldello et al. 2022](https://openaccess.inaf.it/bitstreams/9de1a801-f231-427e-a42a-1a94c8e30f7a/download)
- Suzaku 的 ALP 上限为 2–6 keV CXB 的 6–15%。— [Yamamoto et al. 2020](https://arxiv.org/abs/1906.04429)
- XMM 的衰变暗物质搜索用了 547 Ms；NuSTAR 杂散光用了每探测器约 7–20 Ms。— [Foster et al. 2021](https://arxiv.org/abs/2102.02207)；[Roach et al. 2023](https://arxiv.org/abs/2207.04572)

### Inferences
以下 12 张"想法卡"整体属于 **[inference]**；其中的量级估算都写明了假设，个别设想标为 [speculation]。评分为 1–5 的初步分（新颖性 N / 可行性 F / 潜在影响 I）。

**S1（排名 1）NeuBacH 残差当全档案异常探测器** — N3 / F5 / I4
1. 一句话：用"实测 − NeuBacH 预测"的残差时间序列扫描近十年全部数据，找触发算法漏掉的慢变、长时标或非天体事件。
2. 原理与灵感：C15（Crupi 的 DeepGRB、Biltzinger 的物理本底模型）。
3. HXMT 特征与是否更合适：遮挡探测器 + PM + 丰富 HK 让本底可预测性高于 GBM（GBM 没有遮挡探测器）；**在本底可预测性上 HXMT 确实更合适**，但视场小、对天区覆盖远不如 GBM，所以找到的更多会是粒子事件、太阳事件和视场内弱暂现源，而不是大量新 GRB。
4. 数据：Level-1 事件数据 + EHK/HK；公开性据我所知为公开存档，但未在本次检索中核实。
5. 查新：未检索到 HXMT 做过；GBM 已有（C15）。
6. 可行性：障碍是模型在"训练分布之外"（SAA 边缘、低 COR）是否可靠；残差中活化成分的慢漂会产生假阳性；需要用 H2 的 GRB 样本、H4 的 TGF、H5 的耀斑做召回率检验。
7. 这是其余多项（S2、S3、S6）的共用基础设施，应最先做。

**S2（排名 2）被筛掉的高本底段 → 粒子沉降/REP 事件目录与磁暴响应** — N4 / F4 / I3
1. 一句话：用 PM、ACD、CsI、三台望远镜遮挡探测器在低截止刚度段和 SAA 边缘的数据，建立 2017–2026 年带电粒子沉降事件目录。
2. 原理与灵感：C16、C42、C47。
3. HXMT 特征：43° 倾角、550 km 高度可达 L≈1.09/cos²(53°)≈3（假设地磁纬度最大约 53°），正好覆盖槽区和 2024 年 5 月后内移到 L≈2.2 的新电子带；多探测器、多能段可区分电子直接击中与轫致辐射。**坦率说 HXMT 不如 CALET/ISS（L 可达 4–6）适合研究外带 REP**，优势只在槽区/内带边缘、时间基线比 GECAM 长 3 年多、以及有遮挡探测器。
4. 数据：PM 文件、事件文件中 GTI 之外的部分；SAA 内主探测器高压关闭，只有 PM 可用。公开产品是否含 PM/ACD 计数率：未知。
5. 查新：定向检索未发现 HXMT 同类论文。
6. 可行性：先验证三个窗口（2024-03-21、2024-05 及其后数月、2017-09）；系统误差主要是活化余辉与真实沉降的区分（用遮挡探测器与 ACD 的时间结构）。可检验的预言：2024 年 5 月之后数月，HXMT 在 L≈2.2–3 的轨道段 PM（>1 MeV 电子）计数率应出现并缓慢衰减的增强。
7. 产出可与 GECAM、CALET、MAXI/RBM 做多点联合。

**S3（排名 3）把地球当靶：地球遮挡段的硬 X 射线反照、大气辐射与极光/沉降轫致辐射** — N4 / F3 / I3
1. 一句话：利用每个指向观测都附带的地球遮挡段，测量地球大气 1–250 keV 辐射随视角、截止刚度、太阳周期和昼夜的变化。
2. 原理与灵感：C22、C23、C27、C19；H11 与 H11b。
3. HXMT 特征：窄准直视场可以分辨临边增亮（反照）与中心增亮（大气辐射），INTEGRAL、BAT 只能看整个地球盘；十年数据跨越太阳活动极小到极大。**这一点上 HXMT 确有独特性**。从 43° 纬度向极区方向看临边时，切点可再远约 23°（arccos[6371/6921]），极光卵在几何上可见 [inference]。
4. 数据：ELV<0 的事件数据（标准流程丢弃）；H11 只用了 187 ks 的 HE 遮挡数据，实际存量应大两个量级以上（未核实）。
5. 查新：H11 把 A_albedo 与 C 当作干扰参数拟合过；未检索到以地球辐射为主题的 HXMT 论文。
6. 可行性：需要可靠的"纯粒子本底"预测（正是 NeuBacH 的用途）；HE 遮挡探测器与其他模块分辨率不同是已知障碍；LE 的"漏光"可能只是可见光负载而非 X 射线，需要先甄别。[speculation] 日照面遮挡段的 LE/ME 可能看到大气对太阳耀斑 X 射线的散射与 Ar 荧光（2.96 keV），从而成为间接的耀斑监测，本次未找到文献支撑。
7. 反过来可回馈 CXB 测量：摆脱 H11 中对反照模型的依赖。

**S4（排名 4）CsI 地球掩食监测：200 keV 以上亮源的长期光变** — N3 / F3 / I3–4
1. 一句话：把 BATSE/GBM 的掩食台阶法用到 HXMT/CsI，监测 Crab、Cyg X-1 及亮暂现源在 200 keV–3 MeV 的流量。
2. 原理与灵感：C08、C11。
3. HXMT 特征：CsI 总面积约 5100 cm²，比 GBM 的 BGO 大一个量级。量级估算 [inference]：Crab 在 200–800 keV 的光子流量约 0.02 ph cm⁻² s⁻¹（取 9.7E⁻²·¹），若有效面积 1000–4000 cm²，则台阶 20–80 cts/s；若 CsI 本底约 5000 cts/s（**未知，需实测**），单次台阶（两侧各 100 s）约 2–8σ，每天约 30 次台阶。**是否真的优于 GBM 取决于离轴响应与本底稳定性，目前不能断言**。
4. 数据：CsI 事件数据（正常增益 40–800 keV、低增益 0.2–3 MeV）；增益模式的时间分配未知。
5. 查新：H15 表明 Crab 掩食台阶在 CsI 中已被用于响应评估；未检索到源监测目录。
6. 可行性障碍：CsI 增益持续漂移（H14）；离轴响应依赖卫星质量模型（H3）；HXMT 频繁改变指向，响应随时间变；活化本底在台阶时间尺度上不平坦。
7. 首个目标应是指向 Crab 时的在轴掩食，验证后再推广到离轴源。

**S5（排名 5）活化线当核数据实验：NaI/CsI 在轨同位素产额与衰变曲线** — N4 / F3 / I3
1. 一句话：用十年间数千次 SAA 穿越后的衰减曲线和长期累积，提取 NaI(Tl)/CsI(Na) 中各活化核素的产额，检验 Geant4 活化物理。
2. 原理与灵感：C26、C33、C67。
3. HXMT 特征：约 6.5 kg NaI 与数百 kg CsI（按面积×厚度估算，厚度凭记忆未核实）长期暴露于已知质子环境，18 个模块独立；**在"NaI/CsI 闪烁体 + 长基线"这一点上没有其他任务可比**，但能量分辨率远不如 SPI 的 Ge，线认证会有简并。
4. 数据：遮挡探测器谱、地球遮挡段谱、空白天区谱；H13b 已显示约 67 keV 线长期增长。
5. 查新：H13、H13b 描述了现象并用经验折线拟合，未见逐核素分析（本次未能专门检索）。
6. 可行性：需要 PM 给出每次穿越的质子注量；长寿命核素与增益漂移简并。[speculation] 结果对地面 NaI 暗物质实验（ANAIS、COSINE 等）的宇宙线活化模型有交叉检验价值，但空间质子谱与海平面中子谱不同，换算并不直接。
7. 也直接服务于后续 NaI/CsI 类任务的本底预估。

**S6（排名 6）遮挡探测器 + PM + ACD 当十年宇宙线/辐射环境监测器** — N3 / F4 / I2–3
1. 一句话：把遮挡探测器、PM、ACD 计数率整理成 2017–2026 年的均一化序列，研究 Forbush 下降、太阳调制、SAA 演化，并给出"空间天气 → X 射线本底"的经验传递函数。
2. 原理与灵感：C01、C05、C43、C51、C53–C55、C59。
3. HXMT 特征：三个能段同时有"纯粒子"通道。**作为宇宙线监测器 HXMT 不如中子堆、AMS-02、DAMPE**：H13 显示磁赤道 PM 计数率五年几乎不变；只有高纬段（截止刚度约 3–5 GV）才有可测调制，Forbush 下降幅度估计为极区中子堆的一半左右，即 2–8% [inference]。真正的独特产出是本底传递函数，对 eXTP 等后续任务有工程价值。
4. 数据：同 S2。
5. 查新：SAA 只做了头两年（H6）；Forbush/太阳调制未检索到。
6. 可行性：统计上容易（HE 计数率数百 cts/s）；难点是轨道调制，需按相同地理网格逐日比较或用 NeuBacH 残差。首个测试：2017 年 9 月 8 日前后的 Forbush 下降（C43），当时 HXMT 在连续观测 Crab（H9）。
7. 建议把结果按 GPS/CXD 的方式（C59）做成公开数据产品。

**S7（排名 7）大气密度：从"能测"推进到十年趋势与磁暴响应** — N2 / F5 / I3
1. 一句话：把已有的 Crab 掩食反演扩展为 2017–2026 年跨太阳周期的均一序列，并检验 2024 年 5 月磁暴等事件期间的中高层大气响应。
2. 灵感：C03、C31。
3. HXMT 特征：单一仪器、1–250 keV、55–130 km，每个 Crab 观测日约 15 次掩食；**确实比拼接六个任务的 Katsuda 2023 更均一**。
4. 数据：Crab（及其他亮源）观测的掩食段，公开。
5. 查新：H8 已有三篇，最新覆盖 2022–2024；十年趋势、磁暴响应、非 Crab 源未检索到。
6. 可行性高；主要系统误差是 LE 性能退化（H11）与源本身的变化。
7. 新颖性有限，但几乎必然出结果。

**S8（排名 8）TGF/TEB 的中纬度与闪电触发叠加搜索** — N3 / F4 / I3
1. 一句话：利用 43° 倾角覆盖 26°–43° 纬度带（GBM、GECAM、AGILE 到不了），做闪电定位触发的亚阈值叠加和 TEB 搜索。
2. 灵感：C07、C12、C41、C61。
3. HXMT 特征：CsI 面积可能是飞过的最大 TGF 探测器之一 [inference]，但能量上限 3 MeV、高计数率下的死时间与堆积是短板；ASIM 同样覆盖中纬度且带光学。
4. 数据：H4 的 282 个 TGF 事件数据已公开于 Zenodo；闪电定位数据需另行获取。
5. 查新：首批结果已有（H4）；闪电触发叠加与 TEB 是否已做未知。
6. 可行性：依赖绝对时间精度（H10 显示为 10 μs 量级，足够）。

**S9（排名 9）CsI 太阳耀斑 γ 射线目录（第 25 太阳周）** — N3 / F4 / I3
1. 一句话：系统整理 2017–2026 年经卫星侧面进入 CsI 的耀斑，利用大面积研究 >200 keV 的亚秒结构与 QPP。
2. 灵感：C21（SPI-ACS 把太阳耀斑列为三类事件之一）、C63（立方星也看到耀斑）、H5。
3. HXMT 特征：统计量大；**但无成像、响应依赖质量模型，能谱学不如 Fermi/GBM**（专用太阳仪器的比较未检索）。
4. 数据：CsI 事件数据。
5. 查新：只检索到 X9.3 一个事件的两篇论文。
6. 可行性：S1 的残差搜索可自动给出候选；低增益模式下 2.2 MeV 中子俘获线是否可见属 [speculation]。

**S10（排名 10）CsI"尖峰"与多模块符合事件目录** — N3 / F4 / I2–3
1. 一句话：把被当作假触发剔除的 spike 做成目录，用 18 模块 + ACD + 脉宽信息区分宇宙线簇射与真正的亚毫秒 γ 暂现。
2. 灵感：C21、C57、C02 的教训。
3. HXMT 特征：SPI-ACS 只有单路 50 ms 计数率，无法判别尖峰本质；HXMT 的逐事件信息**确实更适合回答"尖峰里有没有真信号"**。但作为宇宙线物理仪器，HXMT 无电荷与径迹测量，远不如专用实验。
4. 数据：事件数据中的 ACD 标记、脉宽、探测器号（H4 的数据格式说明了这些字段存在）。
5. 查新：H2 仅把它当噪声处理。
6. 可行性：产出更可能是对极短暴率的上限和一套假触发分类器；[speculation] 沿模块依次触发的慢速重粒子搜索几何因子太小，不抱期望。

**S11（排名 11）LE 空白天区计数率随地磁场的变化 → ALP 转换约束** — N4 / F3 / I2–3
1. 一句话：检验扣除粒子本底后的 2–6 keV 弥散本底是否随视线上的 (B⊥L)² 变化。
2. 灵感：C30、C50。
3. HXMT 特征：LE 的集光本领约 1.6 cm² sr（60 个小视场 + 18 个大视场模块，几何估算），比 Suzaku 单台 XIS（约 0.01 cm² sr 量级）大两个量级，CXB 计数率估计为每秒数个 [inference]；有同时的遮挡探测器。统计上有望把 6–15% 的上限推进到百分之几。**但粒子本底本身随地磁纬度变化，与信号简并，最终受系统误差限制；LE 只有早期约 117 ks 被团队认为质量足够（H11）**。
4. 数据：空白天区观测（P0101293、P0202041、P0301293 等）。
5. 查新：未检索到 HXMT 的 ALP/轴子工作。
6. 可行性：需要 IGRF 磁场沿视线积分；信号模型（宇宙 ALP 背景）本身较冷门。

**S12（排名 12）ME/LE/HE 的暗物质衰变线搜索** — N3–4 / F2 / I2
1. 一句话：把准直望远镜当"大集光本领、无聚焦"的弥散线探测器，类比 NuSTAR 杂散光。
2. 灵感：C13、C25、C34、C52。
3. 是否更合适：**不**。量级估算 [inference]：ME 集光本领约 1.5 cm² sr，是 NuSTAR 0-bounce（约 0.02–0.04 cm² sr）的数十倍，但能量分辨率差约 7 倍、粒子本底占比高得多，净统计灵敏度至多与 NuSTAR 相当；LE 要追平 XMM 的 547 Ms 需要约 10 Ms 高质量空白天区曝光，而实际可用的只有亚 Ms 量级；HE 能段有 SPI 的 Ge 分辨率压制。
4. 数据：空白天区 + 高银纬弱源指向。
5. 查新：HXMT 只有 PBH 连续谱约束（H12）。
6. 可行性：ME 的 Ag 荧光线区、1–2% 的本底系统误差是硬限制；只有当遮挡探测器差分能把系统误差压到千分之几才有意义。保留在清单末尾，主要因为方法新、可作为 S1/S3 的副产品顺带得到。

### Gaps
- 所有量级估算缺少三个关键输入：CsI 各模式下的实际本底计数率、CsI 离轴有效面积、ACD/PM 计数率及其在公开数据中的可得性。
- S2、S3、S6 的新颖性判断只基于各一次定向检索；建议后续补查"Insight-HXMT + radiation belt / Forbush / aurora / albedo / activation"。
- 未能核实 HXMT 在机动（slew）期间是否保持科学数据采集，以及低增益模式的时间占比。
- 评分为主观初判，未经领域专家校验。

---

## KQ4：哪些移植不可行或不值得？

### Takeaway
被否决的移植大多败在 HXMT 的硬性约束上：不能看太阳、无成像、LE 下限约 1 keV、有效面积或能量上限不够，或者原方法本身已被后续工作推翻。

### Cited Findings
- **TNO 掩食毫秒 dip（C02）**：原始信号已被归因于粒子事件造成的死时间。— [Jones et al.](https://arxiv.org/abs/astro-ph/0612129)
- **Femtolensing 约束 PBH（C14）**：考虑源尺寸与波动光学后，大多数 GRB 不适用，原有上限被撤销。— [Katz et al. 2018](https://arxiv.org/abs/1807.11495)
- **太阳轴子（C37、C66）**：NuSTAR 的方法需要直接对日观测；Davoudiasl–Huber 方案需要夜侧探测器朝向被地球挡住的太阳方向。— [Ruz et al. 2025](https://arxiv.org/abs/2407.03828)；[Davoudiasl & Huber 2006](https://arxiv.org/abs/hep-ph/0509293)
- **月球/宁静太阳 γ 射线作宇宙线探针（C17、C18）**：信号在 30 MeV 以上。— [Ackermann et al. 2016](https://arxiv.org/abs/1604.03349)
- **SWCX（C32、C49）**：特征线为 C VI 等 1 keV 以下软线。— [Fujimoto et al. 2007](https://arxiv.org/abs/astro-ph/0609308)
- **原子钟暗物质（C58）**：依赖 GPS 原子钟网络；HXMT 的晶振需靠 Crab 脉冲星修正漂移。— [Roberts et al. 2017](https://arxiv.org/abs/1704.06844)；[Tuo et al. 2022](https://arxiv.org/abs/2109.04709)
- **微流星体撞击（C56）**：依赖 LPF 的 μN·s 级动量测量。— [Thorpe et al. 2019](https://arxiv.org/abs/1905.02765)
- **奇异粒子搜索（C44、C46）**：需要电荷/刚度测量。— [DAMPE 2022](https://arxiv.org/abs/2209.04260)
- **转动巡天（C04、C48）**：RXTE XSS 的价值在于当时没有 3–20 keV 全天巡天。— [Revnivtsev et al. 2004](https://arxiv.org/abs/astro-ph/0402414)
- **全局辐射带模型（C24）**：5DRBM 依赖 INTEGRAL/XMM 的高椭圆轨道穿越整个辐射带。— [Métrailler et al. 2019](https://arxiv.org/abs/1907.11029)
- **LIV（C65）**：稳健下限来自大样本、已知红移与尽可能大的能量跨度。— [Ellis et al. 2006](https://arxiv.org/abs/astro-ph/0510172)

### Inferences
- [inference] **TNO dip：No**。HXMT 对 Sco X-1 的计数率比 PCA 低一个量级以上，毫秒 bin 内光子数不足；且该现象本身多半是伪迹。
- [inference] **太阳轴子两种方案：No**。HXMT 有太阳规避角，准直视场（包括 LE 的 60°×2.5° 探测器）永远不含太阳方向，无论太阳是否被地球遮挡。
- [inference] **SWCX：No**。LE 下限约 1 keV，只能碰到很弱的 Mg XI、Si XIII。
- [inference] **CCD 宇宙线径迹（C53、C54）：No（按原样）**。SCD 不成像；替代物是计数率（S6）。
- [inference] **月球掩食或月面辐射：No**。月球只占 LE 小视场约 2%，且很少进入视场。
- [inference] **转动巡天：不值得**。HE 本底系统误差 1–3% 对应约 10 mCrab 量级，Swift/BAT 与 INTEGRAL 的巡天深一个量级以上。
- [inference] **透镜回声、LIV：低优先级**。能做，但样本量与能量杠杆都不如 GBM/LAT/LHAASO，难有新结论。
- [speculation] **CsI 模块间康普顿散射做 GRB 偏振：大概率 No**。18 个大圆柱模块的方位角采样太粗，偶然符合在亮暴时占主导；未找到可类比的已核实文献（RHESSI 的类似尝试凭记忆存在且有争议，本次未检索）。
- [inference] **公民科学、声化：可行但不属于"科学发现"类**，可作为 S10 分类任务的附带形式。
- [inference] **坦率的总体判断**：除 S1（本底可预测性）、S3（窄视场分辨地球盘）、S5（NaI/CsI 长基线）、S7（均一的掩食序列）、S10（逐事件判别尖峰）之外，HXMT 在其余入选方向上都不比原任务更合适，价值来自"独立轨道、独立仪器、十年基线"而非性能优势。

### Gaps
- "不可行"判断基于几何与量级估算，没有逐项用真实数据验证；例如 HXMT 对 Sco X-1 的实际计数率、太阳规避角的确切数值均未检索核实。
- 未检索 HXMT 星敏感器、陀螺等平台工程数据的可得性，因此没有评估任何依赖姿态数据的移植。
