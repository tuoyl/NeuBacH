# Insight-HXMT（慧眼）仪器参数、归档数据与已发表"非常规"用途 —— 事实底座（检索截止 2026-11-17）

> 标注约定：**[confirmed]** = confirmed in literature（来自本次实际取回的论文/官方文档/官网页面，行尾给出 URL）；**[inference]** = 由已证实事实推出、但原文未直接这样说；**[speculation]** = 无直接文献支持的猜测。
> 检索局限：本轮联网搜索配额在调研中途耗尽（全体 agent 共享的上限），此后只能直接抓取已知 URL（arXiv、官网 hxmten.ihep.ac.cn、官方用户手册、AO-09 White Book、官方论文列表）。NASA ADS 页面被代理拒绝（HTTP 405），故若干 2024–2026 年论文只核实到"官方论文列表中的题目+期刊卷页+ADS 链接"，未读到摘要，文中逐条注明。
> 最关键的四份"一手"文件（建议写报告时优先引用）：AO-09 White Book（2026-05-09，[PDF](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)）；HXMT Data Reduction Guide v2.06（[PDF](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)）；官方论文列表（2026-05-19 更新，411 篇，[网页](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)）；宋黎明 2025-08-22 昆明报告幻灯片（[PDF](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)）。

---

## 1. 仪器参数核实与补全（用户凭记忆给出的参数哪些对、哪些要改）

### Takeaway
用户记忆的主干参数（LE/SCD、ME/Si-PIN、HE/NaI-CsI phoswich、550 km/43°、2017-06-15 发射、CsI 兼作全天伽马监视器、大小视场+全遮挡盲探测器、粒子监测器+SEM）全部正确，但需要四处修正：(a) 三台望远镜**当前实际可用能段**比设计值窄（LE 1–12 keV 且 2020-06 后建议 ≥2 keV；ME 8–35 keV、可靠 ≥10 keV；HE 20–250 keV）；(b) CsI 正常增益模式能段文献有 40–600 keV 与 40–800 keV 两种写法，低增益 0.2–3 MeV；(c) HE 事例**不在星上被反符合剔除**，而是每个事例带 18 位 HVT 标签和 8 位脉宽下传，被 veto 的带电粒子事例（空白天区约占 30%）与 CsI 事例都留在 Level-1 事例文件里；(d) HE 准直器不是纯钽，而是铝结构+0.14–0.15 mm 钽片，另有 2 mm 钽环/钽盖，~57 keV 本底线被模拟论文明确归因于钽准直器。

### Cited Findings

**1.1 卫星与轨道**
- [confirmed] 2017-06-15 北京时间 11 时于酒泉由长征四号乙发射，约 600 s 后入轨；轨道高度 550 km、倾角 43°；整星约 2500 kg（White Book 写约 2700 kg）、载荷约 1000 kg、设计寿命 4 年；指向精度 ±0.1°(3σ)、姿态测量 ±0.01°(3σ)、稳定度 ±0.005°/s；数传 120 Mbps；三个地面站（海南、北京、喀什）；一轨约 96 min — [Zhang S.-N. et al. 2020, SCPMA 63, 249502](https://arxiv.org/abs/1910.09613)；[AO-09 White Book §3.1](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)；[官网 Post-Launch activities](http://hxmten.ihep.ac.cn/status/261.jhtml)
- [confirmed] 2017-06-16 HE 加电，06-19 LE/ME 加电，06-26 至 11-30 在轨标定，2017-12-01 起科学运行（AO-01）；2018-01-30 通过在轨测试验收、01-31 交付 — [官网 status/261](http://hxmten.ihep.ac.cn/status/261.jhtml)；[宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)
- [confirmed] 三种姿态模式：全天巡天（绕太阳方向慢旋）、定点指向（三轴稳定，单次一轨到最长 20 天）、小天区扫描（扫描速度 0.01/0.03/0.06 °/s）；银道面被划为 22 个 20°×20° 天区，单块扫描 2 小时到约 5 天 — [Zhang S.-N. et al. 2020](https://arxiv.org/abs/1910.09613)
- [confirmed] 观测约束：太阳规避角 70°–180°；地球遮挡时以地心为中心半径 83° 内的源不可见；月球遮挡：距月心 15.5′ 内不可见，计划工具取月球规避角 6°（理由写的是 "CXB and soft X-ray and visible light pollution"）；SAA 内不做科学观测 — [AO-09 White Book §5.2, §10](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)

**1.2 三台望远镜主参数（用户记忆值 vs 核实值）**

| 项目 | 用户记忆 | 核实值（[confirmed]，除非另注） | 来源 |
|---|---|---|---|
| LE 探测器 | SCD，~1–15 keV | SCD（e2v CCD236，第二代 swept charge device）；96 片 = 3 机箱×32 片（24 个模块×4 片）；每片 ~4 cm²、4 象限、每象限 100×100 像素；总几何面积 384 cm²。能段：设计 1–15 keV；仪器论文写 0.7–13 keV；标定论文可用 1–10 keV；2026 年 AO-9 官方口径 **1–12 keV**；2020-06-25 之后因阈值上调，官方建议谱分析只用 **2–10 keV**；FAQ 写软件能段限 1.5–10 keV | [Chen Y. et al. 2020, SCPMA 63, 249505](https://arxiv.org/abs/1910.08319)；[Zhang S.-N. 2020](https://arxiv.org/abs/1910.09613)；[Li X.B. et al. 2023, RDTM 7, 25](https://arxiv.org/abs/2302.10714)；[AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml)；[FAQ](http://hxmten.ihep.ac.cn/FAQ.jhtml) |
| LE 能量/时间分辨 | — | 地面 140 eV@5.9 keV（−50 °C；辐照后预期退化到 ~220 eV@5.9 keV）；连续读出一帧 0.98 ms ≈ 1 ms（光子到达时刻有最长 ~1 ms 的未知延迟）；工作温度设计 −80~−30 °C，在轨 −75~−40 °C（5 年实测 −55~−38 °C，平均 −47.9 °C，每轨变化约 5 °C，被动辐射制冷）；噪声阈值每 64 s 自动调整 | [Chen Y. 2020](https://arxiv.org/abs/1910.08319)；[White Book §3.4](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)；[Li X.B. 2023](https://arxiv.org/abs/2302.10714) |
| ME 探测器 | Si-PIN，~5–30 keV | 1728 个 Si-PIN 像素 = 3 机箱×576；54 片 ASIC（每片读 32 像素；Cao 2020 写 VA32TA6，Overview 写 VA32TA2）；像素 12.5 mm×4.5 mm×1 mm 厚（56.25 mm²）；总几何面积 952 cm²；50 μm Be 窗。能段：设计 5–30 keV；本底论文写 5–40 keV；因电子学噪声"当前可靠能段从 10 keV 起"；AO-9 口径 **8–35 keV**；FAQ 10–35 keV | [Cao X.L. et al. 2020, SCPMA 63, 249504](https://arxiv.org/abs/1910.04451)；[Tan Y. et al. 2023, RDTM 7, 15](https://arxiv.org/abs/2303.00339)；[Liao J.Y. et al. 2023, RDTM 7, 34](https://arxiv.org/abs/2302.14459)；[Zhang J. et al. 2020, Ap&SS 365, 158](https://arxiv.org/abs/2009.14489)；[AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml) |
| ME 能量/时间分辨 | — | ~3 keV FWHM @17.8 keV（约 15%@20 keV）；时间分辨/单事例死时间 255 μs（Overview 表写 280 μs，White Book 表写 240 μs）；最大计数率 2.2×10⁴ cts/s；工作温度设计 −50~−5 °C，在轨 −40~−10 °C；每机箱 2 个 ²⁴¹Am 源各照 4 个像素（共 24 个标定像素，4 个因噪声关闭，剩 20 个） | [Cao X.L. 2020](https://arxiv.org/abs/1910.04451)；[Tan Y. 2023](https://arxiv.org/abs/2303.00339)；[Data Reduction Guide v2.06 §2.2](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf) |
| HE 探测器 | NaI/CsI phoswich，~20–250 keV，~5000 cm² | 18 个 NaI(Tl)/CsI(Na) phoswich，单个直径 190 mm、几何面积 283.5 cm²，合计 5096 cm²（常写 ~5000/5100）；NaI 厚 3.5 mm、CsI 厚 40 mm；1.5 mm Be 入射窗；共用一只 5 英寸 PMT（负高压约 −1000 V），靠脉冲形状甄别（PSD）区分 NaI/CsI 事例；准直观测能段 20–250 keV（标定论文可用 28–250 keV）；最大计数率 >30 000 cts/s | [Liu C.Z. et al. 2020, SCPMA 63, 249503](https://arxiv.org/abs/1910.04955)；[Zhang S.-N. 2020](https://arxiv.org/abs/1910.09613)；[Li X.B. et al. 2020, JHEAp 27, 64](https://arxiv.org/abs/2003.06998) |
| HE 能量/时间分辨 | — | 能量分辨 ~15%@60 keV（Liu 2020）/18%@60 keV（Li 2020）/≤17%（White Book）/指标 19%；时间标签由 500 kHz 时钟给出，±2 μs；timing accuracy <10 μs；指标 25 μs。NaI 温度主动控制在 18±2 °C | [Liu C.Z. 2020](https://arxiv.org/abs/1910.04955)；[Li X.B. 2020](https://arxiv.org/abs/2003.06998)；[White Book 表 3-5](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf) |
| 在轨绝对时间精度 | — | 以 Crab 脉冲轮廓标定：HE 22 μs、ME 51 μs、LE 21 μs；另有随能量 0.26±0.02 μs/keV 的到达时间依赖（Tuo et al. 2022） | [Zhang S.-N. 2020](https://arxiv.org/abs/1910.09613)；[White Book §7.4](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf) |
| "Open area"（准直器后净面积） | — | 标定论文 Table 1：HE 4270 cm²、ME 850 cm²、LE 300 cm²（按 PDF 版式解析的列对应关系，建议使用前再核对原表） | [Li X.B. 2020](https://arxiv.org/abs/2003.06998) |

- [confirmed] 时间分辨的数字在不同文献里彼此不一致：标定论文 Table 1 给 "Time resolution (us)" = HE 2 / ME 6.4 / LE 10，并写 LE "readout cycle is 100 us"；仪器论文给 ME 255 μs、LE 0.98 ms；Overview 表 4 给 HE 25 μs / ME 280 μs / LE 1 ms；LE 五年论文引言甚至写 "LE … timing resolution: 1 µs" — [Li X.B. 2020](https://arxiv.org/abs/2003.06998)；[Cao 2020](https://arxiv.org/abs/1910.04451)；[Chen 2020](https://arxiv.org/abs/1910.08319)；[Zhang S.-N. 2020](https://arxiv.org/abs/1910.09613)；[Li X.B. 2023](https://arxiv.org/abs/2302.10714)

**1.3 视场配置与盲探测器（逐探测器编号见 Data Reduction Guide 表 2.1–2.3）**
- [confirmed] HE：15 个窄视场 1.1°×5.7°、2 个宽视场 5.7°×5.7°（DetID 2、9）、1 个盲探测器（**DetID 16**，1.1°×5.7° 准直器上加盖 2 mm 钽盖）；18 个准直器分三组，长轴方位相差 60°（0°、+60°、−60°）— [Data Reduction Guide v2.06 表 2.1](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)；[Liu C.Z. 2020](https://arxiv.org/abs/1910.04955)
- [confirmed] ME：每机箱 15 片 ASIC 窄视场 1°×4°、2 片宽视场 4°×4°、1 片盲视场（ASIC 10/28/46，像素 320–351、896–927、1472–1503）；全机 45/6/3 片 ASIC — [Data Reduction Guide v2.06 表 2.2](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] LE：每机箱 20 片窄视场 1.6°×6°、6 片宽视场 4°×6°、2 片盲视场（DetID 13/45/77 与 21/53/85，其中一片带 ⁵⁵Fe 源，0.1–1 μCi）、4 片"巡天"超大视场约 50–60°×2–6°（梯形，DetID 16–19/48–51/80–83）；全机 60/18/6/12 片。盲视场用 1 mm 铝盖遮挡 — [Data Reduction Guide v2.06 表 2.3](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)；[Chen Y. 2020](https://arxiv.org/abs/1910.08319)
- [confirmed] Overview 正文把 LE 视场写成 "1°×6°, 6°×6° and 2.5°×60°"，但同文表 2 及后续文献是 1.6°×6°、4°×6°、60°×2.5°（应以后者为准）— [Zhang S.-N. 2020](https://arxiv.org/abs/1910.09613)
- [confirmed] 盲探测器是为了定点观测而对原设计做的改动：HXMT 原本只为扫描巡天设计，无法做 on/off，于是把部分探测器准直器完全封死来估计仪器本底 — [Zhang S.-N. 2020 §7](https://arxiv.org/abs/1910.09613)
- [confirmed] ME 盲视场的遮挡材料文献说法不一：模拟论文写 "covered FOV is sheltered by 0.6 mm tantalum"，ME 本底论文写低能带电粒子 "can hardly penetrate the aluminum plate above the blind FOV detector" — [Zhang J. 2020](https://arxiv.org/abs/2009.14489)；[Guo C.C. et al. 2020, JHEAp 27, 44](https://arxiv.org/abs/2003.06260)

**1.4 CsI 全天伽马监视（"GRB 模式"）**
- [confirmed] 两种增益模式：NOrmal Mode（NOM/NG，自动增益控制把 ²⁴¹Am 59.5 keV 峰锁在 NaI 第 50 道）与 Low Gain Mode（LGM/LG，关闭 AGC、降低 PMT 高压）。仪器论文/Overview/White Book 表：NOM 下 NaI 20–250 keV、CsI 40–600 keV；LGM 下 NaI 100–1250 keV、CsI 200–3000 keV（均为沉积能量）— [Liu C.Z. 2020](https://arxiv.org/abs/1910.04955)；[Zhang S.-N. 2020 表 3](https://arxiv.org/abs/1910.09613)；[White Book 表 3-3](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] 较新的文献把 NOM 下 CsI 写成 **40–800 keV**（与用户记忆一致）；GRB 星表实际谱分析用的可靠沉积能段是 NG 约 150–800 keV、LG 约 200–3000 keV — [Li X.F. et al. 2023, RDTM 7, 3](https://arxiv.org/abs/2302.03859)；[Song X.Y. et al. 2022, ApJS 259, 46](https://arxiv.org/abs/2203.00878)
- [confirmed] LGM "通常安排在卫星进入地球遮挡、目标不可见、正常观测被打断时"；模式切换靠遥控指令，且会同时改变 NaI 与 CsI 的能段 — [Li X.F. 2023 §2](https://arxiv.org/abs/2302.03859)；[White Book §3.2.2](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] 能量 >~200 keV 的伽马光子可穿透星体从任意方向在 CsI 中沉积能量，视场为未被地球遮挡的全天；MeV 能段有效面积 ~1000 cm² 量级（"up to thousands cm²"）；18 个 CsI 总本底计数率 5000–12000 cts/s，随地理位置、太阳活动和地磁活动变化；LGM 本底低于 NOM；**无星上触发、无实时下传**，所有 GRB 由地面从 CsI 事例数据中搜出；GRB 瞬时辐射探测灵敏度约 8×10⁻⁸ erg/cm²（10–1000 keV）— [White Book §3.2.3.2](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)；[Cai C. et al. 2021, MNRAS 508, 3910](https://arxiv.org/abs/2109.12270)
- [confirmed] CsI 响应主要靠 Geant4+整星质量模型，用 Crab 脉冲星在不同入射方向的脉冲辐射检验；与 Fermi/GBM、Swift/BAT、Konus-Wind 联合拟合 10 个亮暴，HXMT/CsI 流量系统偏高 7.0±8.8% — [Luo Q. et al. 2020, JHEAp 27, 1](https://arxiv.org/abs/2005.01367)
- [confirmed] 在轨数据分析的脉宽选择：NOM 下 PSA 道 54–70 为 NaI 事例、80–120 为 CsI 事例；LGM 下 78–120 为 CsI 事例；HXMTDAS `hescreen` 默认 minpulsewidth=54、maxpulsewidth=70（即默认只留 NaI）— [Li X.F. 2023 §3.2](https://arxiv.org/abs/2302.03859)；[Data Reduction Guide v2.06 §4](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)

**1.5 HE 事例格式、反符合屏蔽（HVT/ACD）、自动增益、死时间**
- [confirmed] 每个 HE "good event" 为 8 字节数据包：能量 8 bit（256 道）、脉宽 8 bit、事例发生时 **18 个 HVT 的标签 18 bit**、通道 5 bit、事例类型 2 bit、亚秒时间 19 bit（500 kHz 时钟）、CRC 4 bit；秒信息来自 GPS PPS — [Liu C.Z. 2020 §2.6](https://arxiv.org/abs/1910.04955)
- [confirmed] 原始设计是脉宽超过 PSD 阈值的 "CsI-involved event" 被丢弃；但通过遥控把 PSD 阈值调大，NaI 与 CsI 事例全部记录并完整下传（GRB 模式在发射前不久才加入）— [Liu C.Z. 2020 §2.7](https://arxiv.org/abs/1910.04955)；[Zhang S.-N. 2020](https://arxiv.org/abs/1910.09613)
- [confirmed] HVT（即 ACD）共 18 块塑料闪烁体（BC-408，厚 6.4 mm）：6 块顶部（每块 ~2250 cm²，覆盖 3 个探测器及其准直器，准直器正上方减薄到 3.4 mm 并去掉铝壳）+ 12 块侧面（每块 1000 cm²，外包 1 mm 钽壳）；带电粒子探测效率要求 >95%；HVT 信号超阈时产生 veto 信号，与之符合的 good event **只被打标签，不在星上剔除**，"在地面数据分析时按一定规则剔除"；18 路 HVT 计数率每秒记录一次 — [Liu C.Z. 2020 §2.4, §2.6](https://arxiv.org/abs/1910.04955)；[White Book §3.2.1](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] 空白天区观测中，1 μs 内与 ACD 符合而被标记的事例约占 30% — [Liao J.Y. et al. 2020b, JHEAp 27, 14](https://arxiv.org/abs/2005.01661)
- [confirmed] AGC：每个准直器角上装一个 ²⁴¹Am 源（~200 Bq，半衰期 433 年）嵌在 BC-448M 塑料闪烁体里，5.5 MeV α 与 59.5 keV X 射线在 2 μs 窗内符合即标记为标定事例（单独存为 Radioactive Events）— [Liu C.Z. 2020 §2.3](https://arxiv.org/abs/1910.04955)
- [confirmed] 死时间：读出电子学约 6 μs；单个 NaI 事例最大 <8 μs，五年无变化；每 6 个探测器共用一路 PDAU/ADC，一个事例会同时给同组另 5 个探测器带来死时间；高能带电粒子信号死时间更长；在轨本底条件下累计死时间占比约 2%–6%；每秒记录一次死时间并下传（Level-1 的 HE-DTime 文件）— [Zhang S.-N. 2020 图 19](https://arxiv.org/abs/1910.09613)；[Li X.F. 2023 §3.4](https://arxiv.org/abs/2302.03859)；[Song X.Y. 2022 §4.4](https://arxiv.org/abs/2203.00878)
- [confirmed] ME 死时间：单事例 255 μs（受 ASIC 读出速度限制）；每 192 个像素共用一套数据采集，一个噪声像素会抬高全部 192 像素的死时间比例；未过数据选择阈值的触发记为 ADC=0 的 pseudo-event 并计入死时间 — [Tan Y. 2023](https://arxiv.org/abs/2303.00339)

**1.6 粒子监测器（HPM/PM）与空间环境监测器（SEM）**
- [confirmed] HE 配 3 个相同的 HPM：BC-440M 塑料闪烁体圆柱（Ø10 mm×10 mm，探测面积 ~1 cm²）+ R647 PMT，1 mm 铝帽；设计阈值电子 1 MeV / 质子 20 MeV，模拟灵敏下限电子 >1.5 MeV、质子 >20 MeV；SAA 外预期 ~1 cps，SAA 内平均 <4000 cps；每秒收集一次三路计数率；计数率 3 s 内超过 10 cps（可遥控改）即判定进入 SAA，关 HED 与 HVT 高压 — [Lu X.F. et al. 2020, JHEAp 26, 77](https://arxiv.org/abs/1911.01594)；[Liu C.Z. 2020 §2.5](https://arxiv.org/abs/1910.04955)
- [confirmed] SAA 模式由两路并行判定：(1) 3 个 PM 的总粒子通量以及 SEM 的质子、电子通量任一超阈即触发，全部低于阈值至少 10 s 后自动退出；(2) 按发射后实测平均通量预报的轨道时间。两种方式都可遥控启用/禁用 — [Zhang S.-N. 2020 §2.1](https://arxiv.org/abs/1910.09613)
- [confirmed] SEM 是独立于三台望远镜的载荷（国家空间科学中心研制）：18 个探测单元，其中 1 个电子谱仪（0.4–1.5 MeV 6 道 + >1.5 MeV 1 道，视场 30°）、1 个质子谱仪（3–150 MeV 6 道 + >150 MeV 1 道，视场 40°）、16 个方向探测器（视场 180°×15°；电子 ≥200 keV、质子 1.5–200 MeV）；时间分辨 1 s — [White Book 表 3-4](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)；[Zhang S.-N. 2020 §2.2.4](https://arxiv.org/abs/1910.09613)
- [confirmed] SEM 自己的论文摘要给的质子谱仪上限是约 300 MeV（"质子约 3–300 MeV，7 个能道"），与 Overview/White Book 的 150 MeV 不一致 — [王春琴等 2019,《空间科学学报》39(6), 770](https://cjss.ac.cn/cn/article/doi/10.11728/cjss2019.06.770)

**1.7 准直器/屏蔽材料及其对荧光线、活化线的含义**
- [confirmed] HE 准直器：外框和筒由整块铝合金圆柱切出（框厚 1 mm），内插 0.14–0.15 mm 钽片并贴钽条；每个 NaI/CsI 探测器外围再装 2 mm 厚钽环；盲准直器盖 2 mm 钽盖；侧面 HVT 外包 1 mm 钽壳 — [Zhang S.-N. 2020 §2.2.1](https://arxiv.org/abs/1910.09613)；[Liu C.Z. 2020 §2.2, §2.4](https://arxiv.org/abs/1910.04955)
- [confirmed] ME 准直器：铝合金框内插钽箔，可屏蔽到 30 keV — [Cao X.L. 2020 §2.5](https://arxiv.org/abs/1910.04451)
- [confirmed] LE 准直器：铝合金 Al7075（壁厚约 120 μm，可挡住 <10 keV），外贴 0.2 mm 钽片；遮光膜（OBF）为铝/聚酰亚胺/铝（100/400/100 nm）夹层，由镍框加固 — [Chen Y. 2020 §2.2](https://arxiv.org/abs/1910.08319)；[Zhang J. 2020 §2](https://arxiv.org/abs/2009.14489)
- [confirmed] 模拟论文原话：HE 本底谱中 "around 57 keV, which originates from HE tantalum collimators" 的线结构清晰可见；而 SAA 诱发成分在 ~31、56、67、191 keV 的峰 "result from the radioactive decay of iodine isotopes" — [Zhang J. et al. 2020](https://arxiv.org/abs/2009.14489)

### Inferences
- [inference] 对"把卫星当辐射探测器阵列"最重要的硬件事实是：HE 的反符合是**标签式**而非剔除式——Level-1 HE 事例里保留了被 18 块 HVT 中任意几块 veto 的事例及其 18 位命中图样，也保留了 CsI 事例（脉宽 >70）。也就是说标准流程丢掉的"带电粒子事例"和"背面/侧面入射事例"在数据里原封不动，且带有粗略的方向信息（哪几块 HVT 着火）。
- [inference] HE 的 ~56–57 keV 线很可能是钽 Kα 荧光（57.5 keV）与碘活化线的混合：标定论文把 31/56/67/191 keV 统称为碘活化线，模拟论文则单独把 ~57 keV 归于钽准直器。两种说法并不矛盾，但意味着这条线的强度同时反映"瞬时粒子通量"（荧光）与"累积活化"（衰变）两个时间尺度。
- [inference] 时间分辨数字的矛盾可以这样理解：2 μs（HE）/6.4 μs（ME）/10 μs（LE）是时间标签的量化步长，而 ME 255 μs、LE ~1 ms 是受读出机制限制的有效分辨率/单事例死时间；做毫秒以下的 LE 时间分析时应按 ~1 ms 处理。
- [inference] 由于 LGM 常安排在地球遮挡期，地球遮挡期的 HE 数据有相当一部分是低增益模式（NaI 100–1250 keV、CsI 0.2–3 MeV），这对想用遮挡期 NaI 20–250 keV 数据的方案是限制，对想用 MeV 段 CsI 数据的方案是利好。两种模式各占多少时间未见公开统计。

### Gaps
- 未找到三台望远镜有效面积随能量的数值表（只有 Overview 图 10 / White Book 图 3-32 的曲线）；CsI 全天有效面积随入射角的数值需看 Luo et al. 2020 正文图表。
- 未找到 NOM 与 LGM 的在轨时间占比统计；也未找到 HVT 高压/阈值历史调整的公开记录。
- LE 是否存在常规意义的死时间修正、其数值：未找到。
- SEM 质子谱仪上限（150 vs 300 MeV）以哪个为准：未能确认。
- "HE 能量分辨"四个数字（15%/17%/18%/19%@60 keV）分别对应地面、指标还是在轨，未逐一核对原表。

---

## 2. 2025–2026 年在轨现状

### Takeaway
截至 2026 年卫星仍在正常运行并继续征集提案：2026-05-11 发布 AO-9（观测期 2026-09-01 至 2027-08-31），公告原话 "currently in service smoothly"；Data Release 16（2026-09-14）里已有 2026-06-01 的观测。已公开的退化主要是：LE 约 8% 的 CCD 关闭、2020-06 后低能阈值上调；ME 约 17% 像素关闭、10 keV 以下受噪声影响；HE 18 个探测器五年评估时全部健康，仅 DetID 2 增益演化异常、CsI 增益持续漂移。2025-02 做过一次轨道控制，但当前轨道高度的数值未能确认。

### Cited Findings
- [confirmed] AO-9 公告（2026-05-11，PI 张双南署名）："It was launched on 15th June 2017 and is currently in service smoothly"；三台望远镜口径写作 HE 20–250 keV/5100 cm²、ME 8–35 keV/952 cm²、LE 1–12 keV/384 cm²；AO-9 观测期 2026-09-01 至 2027-08-31（公告里误写成 "AO-08 Observations"）— [AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml)；截止日延至 2026-06-22 — [AO-9 延期通知](http://hxmten.ihep.ac.cn/notice/997.jhtml)
- [confirmed] Data Release 16 于 2026-09-14 发布，含 186 个观测，列表中最晚的观测到 2026-06-01（XTE J1859+083，2026-05-03 至 06-01），并含 2024-12 至 2025-08 的空白天区（HXMT-Blanksky-10）、Cas A、Crab 核心标定观测 — [DR16 通知](http://hxmten.ihep.ac.cn/notice/1019.jhtml)
- [confirmed] 2025-08-22 报告："目前已在轨稳定运行 8 年 2 个月"；"在轨温度满足设计指标；时间系统工作正常；中能载荷关闭像素（17%）；低能载荷关闭 CCD（8%）；满足科学观测需求"；"轨道控制 2025 年 2 月 25–28 日；25 年轨控耗费燃料 ~20 kg，现剩余 ~129.8 kg"（幻灯片上"高能载荷 N 个工作正常"、轨道高度、倾角等数字是图片，文本层取不到）— [宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)
- [confirmed] 同一报告给出的本底估计系统误差：LE（2–10 keV）4.6%、ME（10–40 keV）1.6%、HE（25–250 keV）2.1%；到 2024-03 底共探测到 346 个 GRB（其中 34 个为慧眼单独探测）；截至 2024-12 ADS 统计使用慧眼数据的论文 320+ 篇、引用约 7600 次 — [宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)
- [confirmed] LE 坏 CCD 表（截至五年论文）：detID 29、87（2017-06-24，小视场）、1（2017-07-28，大视场）、15（2017-09-15，大视场）、69（2019-04-14，大视场）、54（2019-06-18，小视场）、76（2020-06-30，小视场），共 7 片；论文称坏 CCD 数量 "kept changing over the last 5 years" — [Li X.B. et al. 2023, RDTM 7, 25](https://arxiv.org/abs/2302.10714)
- [confirmed] LE 辐照损伤表现：forced trigger 噪声峰位和宽度随时间和温度增大（峰位漂移 200 天前 <50 道，1500 天后约 250 道）；Cas A 各线峰位逐年下降（归因于电荷转移效率下降，用二次多项式描述）；增益不确定度 <20 eV（2–9 keV）、分辨率不确定度 <15 eV；为压噪声，多数 CCD 阈值调到约 450 道，导致 2020-06-25 之后 2 keV 以下有效面积下降且随温度剧烈变化，官方建议此后只用 2–10 keV — [Li X.B. 2023](https://arxiv.org/abs/2302.10714)
- [confirmed] ME 五年状态：298 个像素因噪声过大被关闭、20 个仅用于标定、91 个无法给出可靠能谱，1319 个像素（742 cm²，原 952 cm²）正常；增益缓慢升高 1.43%；探测效率无可测变化；每个像素独立读出，单像素失效不影响其他像素 — [Tan Y. et al. 2023, RDTM 7, 15](https://arxiv.org/abs/2303.00339)
- [confirmed] 2026 年 White Book 做谱模拟时 ME 好像素数取 800，并称 "In most observations, this value can be reached or exceeded"；HE 本底取 2023-07 至 2024-06 的平均谱 — [AO-09 White Book §10.4](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] HE 五年状态：发射约 90 天后性能稳定；NaI 增益变化 <1%（AGC 的功劳），能量分辨略有变好；CsI 增益持续升高，需按月标定，LGM 高压在 2017-07、2020-06、2021-10 三次遥控调整；多数探测器高压五年内下降（作者解释为闪烁光产额/收集效率缓慢改善的补偿），DetID 0、3、5、15 稳定，**DetID 2 行为异常且未被理解**（"data of DetID 2 can still be used normally"）；PSD 品质因数 2–2.7 且每年改善约 1.2%–2.8%；结论 "HE is still in good health" — [Li X.F. et al. 2023, RDTM 7, 3](https://arxiv.org/abs/2302.03859)
- [confirmed] 状态综述类文献（题目与卷页来自官方论文列表）：Lu Fangjun 2023, "In-orbit performance evolution of Insight-HXMT in the first five years", RDTM 7, 1（[Springer](https://link.springer.com/article/10.1007/s41605-023-00387-z)）；Li X.B. et al. 2025, "The In-orbit Operation and Calibration of the Insight-HXMT Satellite", 天文学报 AcASn 66, 36（[ADS](https://ui.adsabs.harvard.edu/abs/2025AcASn..66...36L/abstract)）；Liao J.Y. 2025, "The First 6-year In-orbit Background of Insight-HXMT", AcASn 66, 37（[ADS](https://ui.adsabs.harvard.edu/abs/2025AcASn..66...37L/abstract)）；Li Wei et al. 2025, "Performance study of swept charge device array in orbit", ExA 60, 14（[ADS](https://ui.adsabs.harvard.edu/abs/2025ExA....60...14L/abstract)）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] 软件现状：hxmtsoft v2.06 + CALDB v2.07（2023-12）；hxmtsoft **v2.07** + CALDB **v2.08**（2026-04-15/16 发布），更新内容包括修正 megticorr/legticorr 错误、修正三个 bkgmap 的数据类型错误、**取消 hegtigen 的温度筛选条件**、HE/ME/LE 本底数据库更新并迁入 CALDB；用户手册仍沿用 v2.06 — [官网 Software 页](http://hxmten.ihep.ac.cn/software.jhtml)；[hxmtsoft v2.07 更新说明](http://hxmten.ihep.ac.cn/documents/978.jhtml)

### Inferences
- [inference] 17%×1728 ≈ 294 个 ME 像素、8%×96 ≈ 8 片 LE CCD：与五年论文的 298 个关闭像素、7 片坏 CCD 基本一致，说明 2022–2025 年间"被关闭"的硬件数量几乎没有再增加；但"能用于科学分析的 ME 像素"（White Book 取 800）明显少于 2022 年的 1319，差额可能来自噪声逐步升高后被软件剔除的像素，或 800 只统计小视场像素（原文未说明口径）。
- [inference] 2025-02 的轨道控制与"剩余燃料约 130 kg、单次约 20 kg"说明轨道高度在被主动维持，寿命短期内不受燃料限制；实际高度史可直接从 Level-1 的 Orbit/EHK 文件读出，无需依赖文献。
- [inference] 2017-06 至 2026 年覆盖了第 24 太阳周下降段、2019–2020 极小期与第 25 太阳周上升段和峰值期，这是"十年辐射环境记录"的天然时间基线；但五年本底论文说 PM 计数率的地理分布 2017–2022 "varied little"，说明 43° 倾角轨道上 >20 MeV 质子/>1.5 MeV 电子对太阳周的响应不强，想做太阳周调制需要更灵敏的示踪量（如 HE 本底、CsI 计数率、LE 盲探测器高能段）。

### Gaps
- 当前轨道高度/衰减史：未找到公开数字（幻灯片上是图片）。
- 2025–2026 年 HE 正常工作探测器的确切个数：幻灯片数字取不到；AO-9 只说运行平稳。
- 2022-09 之后的 LE 坏 CCD、ME 坏像素逐个清单：应在 CALDB v2.08 的 bad detector 文件里，未下载核对。
- 2025 年天文学报专刊三篇综述（运行与标定、六年本底、六年银道面星表）只核实到题目与卷页，未读到正文；NASA ADS 被代理拒绝。
- 任务延寿的正式批复文件、预计退役时间：未找到。

---

## 3. 归档与数据可得性

### Takeaway
公开的是 Level-1（官方文件名后缀 L1P）FITS 产品，经 [archive.hxmt.cn/proposal](http://archive.hxmt.cn/proposal) 下载（NSSDC 有镜像，退役后由空间中心永久归档）；专有期常规观测 1 年、预批准 ToO 3 个月、临时 ToO 约 3 个工作日即公开。Level-1 里除事例文件外，还公开了 HE 的高压、温度、计数率、死时间、**粒子监测器**文件，LE 的 **forced trigger 事例**与**超大视场事例**，三台望远镜各自的标定源事例与**盲探测器事例**，以及姿态、轨道、EHK。地球遮挡期数据确定包含在公开事例文件中（多篇论文直接使用）；SAA 内 HE 高压关闭不记事例、LE 只传 forced trigger；SEM 数据不在 Level-1 文件清单里。空白天区和 Crab/Cas A 标定观测在专有期后同样公开。

### Cited Findings

**3.1 数据级别与政策**
- [confirmed] 数据政策（2019-11-14 版）：Level 0 = 初步解码解帧的下传数据，**不对外发布**；Level 1 = 由 Level 0 去重、匹配、排序、解包后的原始数据（对外发布的就是它）；Level 2 = 由 Level 1 提取的光变、能谱、图像、星表。数据分三类：核心科学数据、客座科学数据、非提案数据（后者由 PI 决定用途）；专有期通常为收到 Level-1 产品后 1 年；标定数据随科学数据一起发布；姿态、轨道、时间、载荷状态等辅助工程数据作为辅助数据发布；在轨期间由高能所分发，退役后由国家空间科学中心负责分发与永久归档 — [Data Policy of Insight-HXMT](http://hxmten.ihep.ac.cn/policy/174.jhtml)
- [confirmed] 现行口径：常规提案专有期 1 年、预批准 ToO 3 个月；数据按批次发布，实际公开日可能比专有期结束晚几个月；临时（ad hoc）ToO 数据通常 3 个工作日后公开（可申请 3 个月保护）；核心组导师可为博士课题申请保护最多 20% 的常规观测数据；非注册用户也可下载公开数据（填姓名、单位、邮箱后邮件收到链接）— [FAQ 第 9–13 条](http://hxmten.ihep.ac.cn/FAQ.jhtml)；[AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml)
- [confirmed] 银道面扫描、本底观测、标定观测的提案 "ONLY open to the core science groups"（指申请权限，不是数据永久不公开）— [AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml)

**3.2 归档规模（注意两个口径不同的数字）**
- [confirmed] 截至 2025-08："累计产生有效一级定点观测数据产品 4338 条；分 13 个批次公开释放一级数据产品共计约 142.4 TB；数据公开率超过 92%"；下载网址 http://archive.hxmt.cn/proposal 与 http://hxmt.nssdc.ac.cn — [宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)
- [confirmed] 截至 2026-05-01："10573 data product and 728 GRB data have been publicly released" — [AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml)
- [confirmed] 近期批次：DR14（专有期在 2025-06-30 前结束的观测，220 个）、DR15（2025-11-30 前，267 个，2025-12-11 发布）、DR16（2026-08-31 前，186 个，2026-09-14 发布）— [DR14](http://hxmten.ihep.ac.cn/notice/935.jhtml)；[DR15](http://hxmten.ihep.ac.cn/notice/944.jhtml)；[DR16](http://hxmten.ihep.ac.cn/notice/1019.jhtml)
- [confirmed] 提案轮次统计（提案数/观测申请数/ToO 数）：AO01（2017.11–2019.06）90/517/39；AO02 35/349/26；AO03 34/329/16；AO04 33/333/30；AO05 43/339/30；AO06 50/337/10；AO07 49/330/5；AO08 48/592/— — [宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)
- [confirmed] 年度时间预算：AO-9 名义可分配曝光约 47 Ms（非 ToO 12 Ms + ToO 35 Ms，后者按约 10% 触发概率超额分配 + 2 Ms 临时观测），其中核心提案 18 Ms（含银道面巡天 2.5 Ms）、客座 27 Ms、标定 2 Ms；"for 50% observation efficiency, the actual observation time is two times longer"。AO-08 实际：非 ToO 批准约 9.0 Ms（银道面扫描约 1.9 Ms），预批准 ToO 触发 5 个共 0.9 Ms，临时 ToO 28 个共 3.8 Ms — [AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml)；[AO-09 White Book §13](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)

**3.3 Level-1 产品的目录与文件清单**
- [confirmed] 层级为 Proposal–Observation–Exposure：Proposal ID 形如 Paattmmm；Observation ID = Proposal ID + 三位序号；一个 observation 被人为切成多个约 3 小时的 exposure 以控制文件大小（单个 observation 数据量常超过数 GB）；根目录有 FileList.FITS、ExpoList.XML，以及覆盖整个观测的 ACS（姿态、轨道）和 AUX（地面生成的辅助数据如 EHK）目录；每个 exposure 目录下有 ACS、AUX、HE、ME、LE 子目录 — [Data Reduction Guide v2.06 §3.1–3.2](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] 文件类型（表 3.1，文件名 `HXMT_<expoID>_<type>_FFFFFF_Vn_L1P.FITS`，P = Pointing）：
  - HE：HE-Evt（事例）、HE-RadEvt（标定源事例）、HE-ShltEvt（盲探测器事例）、HE-InsStat（仪器状态）、HE-TH（温度）、HE-Cnts（计数）、HE-DTime（死时间）、HE-HV（高压）、**HE-PM（粒子监测器）**
  - ME：ME-Evt、ME-RadEvt、ME-ShltEvt、ME-CirPara（电路参数）、ME-InsStat、ME-TH、ME-Cnts
  - LE：LE-Evt、LE-RadEvt、LE-ShltEvt、**LE-WFOVEvt（超大视场事例）**、**LE-ForcedEvt（forced trigger 事例）**、LE-CirPara、LE-InsStat、LE-TH、LE-Cnts
  - 公共：Att（姿态）、Orbit（轨道）、EHK（扩展 housekeeping）
  — [Data Reduction Guide v2.06 表 3.1](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] EHK 中可用于筛选的量：ELV、DYE_ELV（指向与亮地球边缘夹角）、SAA_FLAG、T_SAA、TN_SAA、COR（截止刚度，GV）、MOON_ANGLE、SUN_ANGLE、ANG_DIST；HE 还可用 PM 文件做筛选，示例 `pmexpr="Cnt_PM_0<100&&Cnt_PM_1<100&&Cnt_PM_2<100"`（说明 PM 文件里有 Cnt_PM_0/1/2 三列）— [Data Reduction Guide v2.06 §5.3.1](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] LE forced trigger：每 32 ms 记录每片 CCD 的噪声幅度/基线，按普通事例格式保存但类型不同；HXMTDAS 用其 1 s 平均值做基线扣除 — [Data Reduction Guide v2.06 §2.3](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)；[Li X.B. 2023](https://arxiv.org/abs/2302.10714)
- [confirmed] PI 道与能量的换算（FAQ）：LE PI=(E−0.1)/13×1536，ME PI=(E−3)/60×1024，HE PI=256×(E−15)/370（E 以 keV 计）— [FAQ 第 15 条](http://hxmten.ihep.ac.cn/FAQ.jhtml)
- [confirmed] 一份公开的 HE 事例样例（TGF 数据集）说明了列含义：Time 为自 2012-01-01T00:00:00 起算的 MET 秒；Det_ID 0–17；Channel 0–255；Pulse_Width（数值 N 对应 N/48 ms，原文如此）；ACD 为 18 个布尔量；Event_Type 0=正常增益、1=低增益 — [Zenodo 10.5281/zenodo.10976454](https://zenodo.org/records/10976454)

**3.4 "通常被丢弃"的数据是否在公开文件里**
- [confirmed] **地球遮挡期**：在。GTI 由用户用 EHK 的 ELV>10 自行生成，说明 Level-1 事例文件含 ELV<10 的数据；HE 五年论文用 "the data of earth occultation" 标定 LGM 下的 CsI，并每两个月取一天地球遮挡期本底数据监测 PSD 与死时间；五年本底论文把 "time interval of earth occlusion"（此时不同视场探测器光变重合、没有 CXB 光子）列为 LE 光变曲线的三类时段之一；DXRB 论文用地球遮挡数据构建 HE 本底（遮挡曝光 187 ks）；大气密度系列论文直接用定点观测中的掩食段 — [Data Reduction Guide v2.06](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)；[Li X.F. 2023](https://arxiv.org/abs/2302.03859)；[Liao J.Y. 2023](https://arxiv.org/abs/2302.14459)；[Huang R. et al. 2022](https://arxiv.org/abs/2306.10114)
- [confirmed] **SAA 内**：HE 进入 SAA 模式后 HED 高压关闭，"The instrument will not record any scientific event"；LE 在 SAA 模式下 "only the FT events are transferred"（即只有 forced trigger 噪声事例）；PM 在 SAA 内继续计数（官方用 PM 绘制 SAA 图）— [Li X.F. 2023 §2](https://arxiv.org/abs/2302.03859)；[Chen Y. 2020 §3](https://arxiv.org/abs/1910.08319)；[Zhang S.-N. 2020 图 22](https://arxiv.org/abs/1910.09613)
- [confirmed] **SAA 前后各 300 s、低截止刚度区、偏离指向时段**：标准 GTI 用 T_SAA、TN_SAA、COR、ANG_DIST 剔除，这些都是用户侧筛选条件 — [Data Reduction Guide v2.06](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] **空白天区与标定观测**：公开。DR14–DR16 的目标列表中有 BLANK_SKY2/3/6/8/10/11/14、HXMT-Blanksky-1/-10、Cas_A、Crab 等核心观测 — [DR16](http://hxmten.ihep.ac.cn/notice/1019.jhtml)；[DR15](http://hxmten.ihep.ac.cn/notice/944.jhtml)；[DR14](http://hxmten.ihep.ac.cn/notice/935.jhtml)
- [confirmed] **CsI/GRB 数据**：官方统计里单列 "GRB data"（至 2026-05-01 公开 728 个）；昆明报告的数据释放图例中 "GRB" 与 AO01–AO07 并列；AO-9 公告称 HXMT（GRB）数据可用 GECAMTools 分析（https://github.com/zhangpeng-sci/GECAMTools-Public）；第三方论文称从 Insight-HXMT 数据档案取得约 248 个 GRB 并从 HE-Evt 文件提取光变（后者仅见于搜索摘要，未读原文）— [AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml)；[宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)
- [confirmed] **被 veto 事例、分裂事例、高 grade 事例**：都在 Level-1 里，由 `hescreen`（anticoincidence 选项）、`megrade`/`mescreen`（grade）、`lerecon`/`lescreen` 在用户侧剔除 — [Data Reduction Guide v2.06](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)

**3.5 空白天区与银道面扫描的规模**
- [confirmed] 截至 2019-12-12，空白天区观测 200 多次，有效曝光约 3.5 Ms（HE 本底模型：约 3 Ms 建模、0.5 Ms 检验）；ME 本底模型用 2017-09-20 至 2019-04-29 的 195 次以上空白天区（约 1 Ms）；LE 本底模型用 150 次以上高银纬空白天区、21 个空白天区检验 — [Liao 2020b](https://arxiv.org/abs/2005.01661)；[Guo 2020](https://arxiv.org/abs/2003.06260)；[Liao J.Y. et al. 2020a, JHEAp 27, 24](https://arxiv.org/abs/2004.01432)
- [confirmed] 五年本底论文表 2 的空白天区 ObsID 范围：P0101293（001–191，2017-11-02 至 2019-06-26）、P0202041（001–161，至 2020-07-22）、P0301293（001–115，至 2021-08-30）、P0401293（001–115，至 2022-08-29），合计 582 个观测；另把 PSR B0540-69、GW170817 天区、Cas A 的观测当作 HE 的"空白天区" — [Liao J.Y. 2023 表 2](https://arxiv.org/abs/2302.14459)
- [confirmed] 例行空白天区观测频度约每月 10 次、每次 3 ks — [AO-09 White Book §6.4](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] 银道面扫描巡天：头两年 1000 多次扫描（每次 2–3 小时），完整覆盖 0°<l<360°、|b|<10°；单次扫描 5σ 灵敏度 LE ~3 mCrab（1–6 keV）、ME ~20 mCrab（7–40 keV）、HE ~18 mCrab（25–100 keV）；4 年星表（2017-06 至 2021-08）监测 1300 多个源、223 个 S/N>5；已有 6 年星表（2025，天文学报）— [Sai N. et al. 2020, JHEAp 26, 1](https://arxiv.org/abs/2003.07741)；[Wang C. et al. 2023, ApJS 265, 52](https://arxiv.org/abs/2303.00360)；[Liao J.Y. et al. 2025, AcASn 66, 38](https://ui.adsabs.harvard.edu/abs/2025AcASn..66...38L/abstract)

### Inferences
- [inference] "4338 条定点观测产品"（2025-08）与"10573 个数据产品"（2026-05）显然不是同一计数单位（前者限定"定点"，后者可能把扫描观测或 exposure 级产品也算在内）；总公开数据量约 140 TB 量级。
- [inference] 累计有效曝光可粗估：约 50% 观测效率 × 31.6 Ms/年 ≈ 16 Ms/年，2017-12 至 2026 年底约 9 年 → **上限约 140 Ms**（未扣除故障、姿态机动）；这与 AO-08 实际执行的非 ToO 9 Ms + ToO 约 4.7 Ms ≈ 14 Ms/年自洽。
- [inference] 空白天区累计曝光：前两年 3.5 Ms（2017–2019 年明显多于例行频度），此后按每月 10 次×3 ks ≈ 0.36 Ms/年，至 2026 年合计约 **6 Ms 量级**；加上被当作 HE 空白天区的 Cas A、PSR B0540-69 观测会更多。
- [inference] 文件后缀 L1P 中 P 表示 Pointing，且 HXMTDAS 明确 "specified for analyzing the Insight-HXMT data from pointed observations"，因此银道面扫描模式的 Level-1 数据即使可下载，文件命名与处理流程也不同于定点产品。
- [inference] 存在 ANG_DIST<0.04 这一筛选条件，说明事例文件里包含指向尚未稳定（机动/偏离）的时段，但机动段覆盖是否完整未见说明。

### Gaps
- **SEM 数据是否公开**：Level-1 文件清单（表 3.1）里没有 SEM 文件；未找到 SEM 数据的公开下载入口或申请流程。SEM 由国家空间科学中心研制，数据很可能在其体系内。——未确认。
- **银道面扫描 Level-1 数据是否对公众开放下载**：未确认（archive.hxmt.cn 是 JavaScript 单页应用，无法抓取检索界面）。
- **GRB data 的具体内容**（暴前后多长时间窗、是否只有 CsI 事例）：未确认；TGF 论文的 Zenodo 数据集是 T0±1 s。
- 归档总观测数、总曝光、空白天区总曝光的官方单一数字：未找到，以上为拼合与推断。
- ME 在 SAA 内是否记录事例：未找到说明。
- HE-Cnts 文件的列定义（是否包含 18 路 HVT 每秒计数率，仪器论文说这些计数率每秒记录并送 OBDH）：未核对 FITS 头，建议数据分析者直接打开文件确认。
- hxmt.nssdc.ac.cn 页面出现"提交申请/使用目的"对话框，说明该镜像可能需要申请；具体流程未确认。

---

## 4. 标准筛选（GTI）与被丢弃的时间比例

### Takeaway
官方推荐的 GTI 表达式三台望远镜几乎相同：`ELV>10 && COR>8 && SAA_FLAG==0 && ANG_DIST<0.04 && T_SAA>300 && TN_SAA>300`，LE 另加 `DYE_ELV>30`；之后 ME、LE 还要用大小视场光变之差剔除"flare"时段。官方给出的总体数字是"观测效率约 50%"（实际占用时间是有效曝光的两倍）；在已通过常规 GTI 的 ME 数据里，flare 还要再去掉约 7%。更细的逐项占比没有公开统计。

### Cited Findings
- [confirmed] 推荐表达式（defaultexpr=NONE 时）：
  HE：`ELV>10&&COR>8&&SAA_FLAG==0&&ANG_DIST<0.04&&T_SAA>300&&TN_SAA>300`
  ME：同 HE
  LE：`ELV>10&&DYE_ELV>30&&COR>8&&SAA_FLAG==0&&ANG_DIST<0.04&&T_SAA>300&&TN_SAA>300`
  — [Data Reduction Guide v2.06 §4](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] 各参数含义与默认门限：ELV 指向高出地球边缘 >10°；DYE_ELV 高出亮地球 >10°（默认值）；T_SAA/TN_SAA 距上次/下次 SAA >300 s；COR >8 GV；MOON_ANGLE >1°；SUN_ANGLE >70°；ANG_DIST 为偏离指向方向的角度 — [Data Reduction Guide v2.06 §5.3.1](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] 清洗步骤还包括：剔除坏时段、剔除坏探测器（baddetfile）、剔除高 grade 事例；HE 可选 anticoincidence=yes 去掉粒子本底，并按脉宽 54–70 选 NaI；常规 GTI 之后 "there are still some bad time intervals"，需用 megticorr、legticorr 再剔一次，LE 推荐使用 legticorr 更新后的 GTI — [Data Reduction Guide v2.06 §4, §7.3](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] hxmtsoft v2.07（2026-04）取消了 hegtigen 中的温度筛选条件 — [v2.07 更新说明](http://hxmten.ihep.ac.cn/documents/978.jhtml)
- [confirmed] HE 本底论文的选择理由：ELV>10 保证看到目标；高纬度 COR 小、本底很高故取 COR>8；SAA 前后 300 s 剔除，且 "Even a thousand seconds after the satellite passes through SAA" 本底仍偏高；活化最强的地理区域（论文图 7 紫色方块）也从 GTI 中剔除（hegtigen 的 ascendfile/descendfile 两张 "high bkg" 地图即为此用）— [Liao 2020b](https://arxiv.org/abs/2005.01661)；[Data Reduction Guide v2.06 §5.3.1](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)
- [confirmed] LE 本底论文的做法与官方默认不同：因为低 COR 区 LE 本底"高但稳定"，建模时**不用 COR**；DYE_ELV 一般取 >20°，但即使 >60° 在"赤道附近和高纬度的沉降粒子区"仍有异常峰，于是也不用 DYE_ELV，改用"大、小视场光变之差保持恒定超过 100 s"作为先验 GTI — [Liao 2020a §2](https://arxiv.org/abs/2004.01432)
- [confirmed] ME flare：流量正比于视场、持续几十到几百秒、可出现在高 COR 的特定区域、与 ELV 无关；被筛出的 flare 平均持续约 200 s，**占常规 GTI 曝光的约 7%**；另有像素噪声突增（0–100 道即 3.0–8.9 keV 内峰值可达正常值的数千倍并指数衰减），每次观测通常剔除 1–2 个像素 — [Guo 2020 §2](https://arxiv.org/abs/2003.06260)
- [confirmed] 总体效率："for 50% observation efficiency, the actual observation time is two times longer" — [AO-9 公告](http://hxmten.ihep.ac.cn/notice/984.jhtml)
- [confirmed] 同一观测 HE、ME、LE 的有效曝光并不相同，"because the data processing has some procedures related to the intrinsic instrument performance itself" — [Zhang J. 2020 表 2 注](https://arxiv.org/abs/2009.14489)

### Inferences
- [inference] "一半时间不算曝光"主要由地球遮挡（地球约占卫星天空的 1/3，White Book §6.2 原话）、SAA 及其前后 300 s、低截止刚度（COR<8，43° 倾角轨道的高纬段）三项贡献；被丢弃的这一半正是"辐射探测器"视角下信息量最大的部分（高纬高粒子通量、SAA 边缘活化衰减、地球大气反照）。
- [inference] 被丢弃数据可分为三种不同性质：(i) 仪器根本没记录（SAA 内 HE 事例）；(ii) 记录了但在 GTI 之外（地球遮挡、SAA 前后、低 COR、flare、机动）；(iii) 在 GTI 之内但被事例级筛选去掉（HVT veto 约 30%、CsI 脉宽事例、ME 高 grade、LE 分裂事例、标定源事例、盲探测器事例）。后两类全部在公开的 Level-1 数据里。

### Gaps
- 未找到官方公布的"各筛选条件分别丢掉多少时间"的分解，也未找到三台望远镜各自 GTI 占比的十年统计。
- megticorr/legticorr 的具体算法阈值、hegtigen 的 prefr/postfr 参数含义未细读。
- 无星上触发的情况下，事例是否存在因存储/下传限制而丢失（数传饱和）的系统统计：GRB 星表只对个别极亮暴标注了饱和/数据丢失，整体统计未找到。

---

## 5. 已发表的本底知识（成分、谱线、长期演化、辐射损伤）

### Takeaway
三台望远镜的本底模型都建在"空白天区数据库 + 盲探测器实时归一"之上，并已有 2 年（2020）、5 年（2023）、6 年（2025）三轮评估。成分上：HE 以 SAA 与宇宙线质子诱发的延迟（活化）成分加大气反照为主，ME 全能段由宇宙线质子瞬时成分主导，LE 在 ~7 keV 以下由 CXB 主导、以上由粒子主导。已识别的谱线有 HE 的 31/56/67/191 keV（碘活化，~57 keV 另归因于钽）、CsI 的 67–2767 keV 八条线、ME 的银 K 线、LE 的 Ni/Cu/Zn 荧光线；但**文献没有给出逐条的同位素证认**。长期演化方面，HE 本底随活化累积上升并趋于饱和，ME 缓慢上升，LE 的"上升"主要是辐照损伤后阈值上调与分裂事例重建造成的假象。

### Cited Findings

**5.1 本底成分（模拟与在轨对比）**
- [confirmed] 成分分类：两类延迟成分（穿越 SAA 引起的 SAA 成分；高能宇宙线质子引起的 CRP_D）+ 四类瞬时成分（CRP_P、地球大气反照 γ（Albedo）、非孔径入射 CXB（CXB_N）、孔径入射 CXB（CXB_A））；HE 主要贡献者是 SAA、CRP_D 和 Albedo；ME 全能段由 CRP 主导；LE 高能段由 CRP、低能段由 CXB 主导；地球对卫星的立体角约为全天的 1/3 — [AO-09 White Book §6.2](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] 模拟（Geant4 + 整星质量模型）与在轨两年对比：HE 17 个大小视场探测器 30–300 keV 平均本底率 观测 538.6±0.1 cts/s、模拟 563.2±3.6 cts/s（差约 5%）；ME 小视场 10–40 keV 观测 37.8–40.2 cts/s、模拟 35.1±0.2 cts/s（低约 15%，相当于 1.6–1.7×10⁻³ cts s⁻¹ keV⁻¹ cm⁻²）；LE 小视场 1–12 keV 观测 12.1–12.7 cts/s、模拟 12.8±0.2 cts/s（4.1–4.3×10⁻³ cts s⁻¹ keV⁻¹ cm⁻²）— [Zhang J. et al. 2020, Ap&SS 365, 158](https://arxiv.org/abs/2009.14489)
- [confirmed] HE：CXB 占比 <10%；本底与 COR 显著反相关且谱形随 COR 变化；同一地理位置升轨（由南向北，刚出 SAA）与降轨本底差别很大；PM 计数率赤道附近 <3 cts/s、高纬 >10 cts/s、SAA 内高得多 — [Liao 2020b](https://arxiv.org/abs/2005.01661)
- [confirmed] ME：本底随 COR 可变化 3 倍——赤道附近（最高 COR）约 30 cts/s，高纬（最低 COR）约 150 cts/s，每轨（约 95 min）各经过赤道和高纬两次；每次穿越 SAA 后本底水平没有显著升高，说明 SAA 诱发成分较弱；同一 COR 区内本底率仍可变化达 50%（后一句来自搜索摘要）；谱形随地理位置变化不可忽略 — [Guo 2020](https://arxiv.org/abs/2003.06260)
- [confirmed] LE：粒子本底谱形在不同 COR、不同地理位置下稳定，盲探测器 10–12.5 keV 计数率可作强度标尺；由于"大量低能电子和高能质子散射"，盲探测器低能段本底不为零且同样受地磁调制；21 个高银纬空白天区之间 3–7 keV CXB 计数率的内禀弥散为 4.0±0.7%（按 7%/平方度涨落和 1.6°×6° 视场的理论预期约 2.3%，作者认为差值来自标定系统误差和未分辨源）— [Liao 2020a](https://arxiv.org/abs/2004.01432)；[AO-09 White Book §6.3](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] White Book 承认 "The in-orbit background of ME and LE have become more complicated due to the light pollution and other problems"；LE 本底论文说大量可见光或低能带电粒子进入准直器会使探测器饱和 — [AO-09 White Book §6.3](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)；[Liao 2020a](https://arxiv.org/abs/2004.01432)
- [confirmed] ME 与 LE 的 "flare"：流量正比于视场、盲探测器看不到、持续几十到几百秒、LE 在 7–13 keV 比 1–7 keV 更显著，作者推测是"低地球轨道局部区域的低能带电粒子"经准直器进入，Geant4 模拟支持这一推测 — [Guo 2020](https://arxiv.org/abs/2003.06260)；[Liao 2020a](https://arxiv.org/abs/2004.01432)

**5.2 已识别的谱线**
- [confirmed] HE/NaI：31、56、67、191 keV 四条本底线，"due to activation of iodine by cosmic and SAA protons"，在空白天区或地球遮挡谱中都出现，用于修正地面 E–C 关系与分辨率模型；31 keV 峰实际是 25.25–40.25 keV 内多条线的混合；~57 keV 另被归因于钽准直器；50–70 keV 活化线强度第一年快速上升后约一年趋于饱和，~30 keV 与 ~190 keV 的延迟成分初期快速上升 — [Li X.B. 2020](https://arxiv.org/abs/2003.06998)；[Zhang J. 2020](https://arxiv.org/abs/2009.14489)；[AO-09 White Book §6.2, §7.1](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] HE/CsI：用约 67、191、478、692、1182、1372、1729、2767 keV 八条本底线按月标定线性 E–C 关系，文献称它们来自"卫星结构中被高能粒子诱发的放射性同位素"；NOM 下监测 ~692 keV 峰位、LGM 下监测 ~2767 keV 峰位 — [Li X.F. 2023 §3.2.2](https://arxiv.org/abs/2302.03859)
- [confirmed] ME：约 22.5 keV 的银 K 荧光线（Si-PIN 用银胶粘在陶瓷片上；入射光子能量超过 Ag K 边 25.5 keV 时可穿透硅与银作用），模拟取银胶厚 14 μm 但实际厚度未知且各像素不同；~25 keV 处的"峰"不是真峰，是电子学噪声与阈值截断造成；²⁴¹Am 标定线 13.94、17.58（17.8）、21.30（21.6）、26.34 keV — [Zhang J. 2020](https://arxiv.org/abs/2009.14489)；[Data Reduction Guide v2.06 §2.2](http://www.hxmt.cn/u/cms/www/202312/HXMT_manual_V2_06_English.pdf)；[Tan Y. 2023](https://arxiv.org/abs/2303.00339)
- [confirmed] LE：7.472（Ni）、8.041（Cu）、8.631（Zn）keV 荧光线，文中称锌是 LE 铝合金（Al7075）准直器的主要合金成分；7.5 与 8.0 keV 两条线在观测谱中有而模拟中没有，作者归因于 LE 结构建模的不确定；另有 Si K 边 1.839 keV 与 1.74 keV 的 Si Kα 荧光峰；在轨增益用 Cas A 的 Mg/Al、Si、S、Ar、Ca、Fe 线与空白天区的 Ni、Zn 线标定 — [Zhang J. 2020 §3.4](https://arxiv.org/abs/2009.14489)；[Li X.B. 2020](https://arxiv.org/abs/2003.06998)；[Li X.B. 2023](https://arxiv.org/abs/2302.10714)

**5.3 本底模型及其精度**
- [confirmed] HE 模型（Liao et al. 2020b, JHEAp 27, 14）：全球划分 72×18=1296 个 5°×5° 地理网格，区分升/降轨，对每个能道用时间的经验函数（2020 年用二次多项式；五年论文改用两段折线）描述长期演化，再用盲探测器同期计数率做修正；8 ks 曝光下 26–100 keV 能谱平均系统误差 1.5%（1 ks 为 3.6%，32 ks 为 0.9%），30–150 keV 光变误差约 2–3% — [Liao 2020b](https://arxiv.org/abs/2005.01661)；[Liao 2023 §4.2](https://arxiv.org/abs/2302.14459)
- [confirmed] ME 模型（Guo et al. 2020, JHEAp 27, 44）：同样的 5°×5° 地理网格数据库 + 盲视场探测器归一；平均曝光约 5.5 ks 时 8.9–44.0 keV 系统误差约 1.3%，>80 ks 时 <0.5%，<1 ks 时可达 3%；作者指出模型严重依赖盲探测器，正在并行开发不依赖盲探测器的参数化模型 — [Guo 2020](https://arxiv.org/abs/2003.06260)
- [confirmed] LE 模型（Liao et al. 2020a, JHEAp 27, 24）：粒子本底谱形取自盲探测器数据库、强度由盲探测器 10–12.5 keV 计数率定，弥散 X 射线本底另行估计；1 ks 能谱平均系统误差 1–7 keV 约 4.2%、1–10 keV 约 3.7%，1–10 keV 光变 <8% — [Liao 2020a](https://arxiv.org/abs/2004.01432)
- [confirmed] 2 ks 曝光下的官方综合数字：HE（26–100 keV）~2.6%、ME（8.9–44 keV）~1.9%、LE（1–10 keV）~3.2% — [AO-09 White Book §6.4](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] 参数化（不依赖数据库网格的）模型：You Yuan, Liao Jin-Yuan, Zhang Shuang-Nan et al. 2021, "A Parametric Model to Reproduce the Background of the Insight-HXMT/HE Blind Detector", ApJS 256, 47（只核实到题目与卷页）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)；[ADS](https://ui.adsabs.harvard.edu/abs/2021ApJS..256...47Y/abstract)
- [confirmed] DXRB 论文没有用 HE 盲探测器估计粒子本底（搜索摘要转述的理由是盲探测器能量分辨不同且多了遮盖物会使谱产生偏差），改用地球遮挡数据建 HE 本底；并称因 ME 性能不稳定只用了 HE 和 LE — [Huang R. et al. 2022, MNRAS 513, 4074](https://arxiv.org/abs/2306.10114)

**5.4 五年长期演化（Liao et al. 2023, RDTM 7, 34）**
- [confirmed] 空间环境本身：PM 计数率的地理分布五年变化很小（含低纬、高纬和 SAA），磁赤道附近低计数率区稳定在 1.7–1.8 cts/s，"the space environment has changed very little during the last five years"；SAA 强度五年无显著变化 — [Liao 2023 §1, §4](https://arxiv.org/abs/2302.14459)
- [confirmed] HE：本底由延迟成分主导，随晶体持续活化而明显上升，五年内上升趋势逐渐变缓；谱形五年稳定；每次穿越 SAA 后快速衰减 + 随在轨天数长期累积，累积在发射后初期很快、数百天后变慢；刚出 SAA 的位置（如 lon 345°, lat 15° 升轨）长期演化最不明显，因为短时标成分占主导；逐年系统误差均 <3% — [Liao 2023 §4](https://arxiv.org/abs/2302.14459)
- [confirmed] ME：第五年本底略高于第一年，是微弱延迟成分的累积效应，低能段比高能段更明显；SAA 附近区域（330°<lon<360°, 0°<lat<30°）明显高于同纬度其他区域，证明 ME 也有延迟成分；部分像素辐照损伤使低能噪声分布变宽，第五年 11 keV 以下谱显著抬高；小视场探测器的银线中心随时间漂移而盲视场探测器不明显，作者据此认为盲视场探测器受辐照损伤更轻；10–40 keV 系统误差约 1.6%（10–15 keV 约 2%）— [Liao 2023 §3](https://arxiv.org/abs/2302.14459)
- [confirmed] LE：本底地理分布形状不变但强度在 2020-06-30 之后明显升高；作者解释为辐照损伤使噪声分布变宽 → 阈值上调 → 更多双分裂事例无法重建而被当作低能单事例，谱逐年"左移"，即"上升是辐照损伤与分裂事例重建共同造成的"；发射线持续展宽反映能量分辨下降；低能段计数率稳定、高能段受地磁场显著调制 — [Liao 2023 §2](https://arxiv.org/abs/2302.14459)
- [confirmed] 模拟论文关于长期趋势的结论：HE 总本底谱与 ~67 keV 线的相对丰度在模拟和观测中都长期上升；ME、LE 两年内未见明显长期上升 — [Zhang J. 2020](https://arxiv.org/abs/2009.14489)

**5.5 增益、分辨率与辐射损伤（详见第 2 节）**
- [confirmed] HE：59.5 keV 峰位日变化 <0.01 道；191 keV 峰位五年变化 <1%；两年内 HE、ME 增益与分辨稳定并按月监测 — [Li X.B. 2020](https://arxiv.org/abs/2003.06998)；[Li X.F. 2023](https://arxiv.org/abs/2302.03859)
- [confirmed] ME：²⁴¹Am 13.94 keV 线 FWHM 700 天内仅增加 0.5%；峰位缓慢上移，五年增益 +1.43%；Si-PIN 漏电流在典型工作温度下低至 10 pA，污染沉积与辐照损伤会通过增大漏电流使分辨变差、噪声计数率升高 — [Li X.B. 2020](https://arxiv.org/abs/2003.06998)；[Tan Y. 2023](https://arxiv.org/abs/2303.00339)
- [confirmed] LE 探测器专门的本底/性能论文（只核实到题目与卷页）：Li Wei, Lu Jingbin, Yang Yanji et al. 2024, "Background experiment of the low energy x-ray telescope detectors on insight-HXMT", AIP Advances 14, 025252（[ADS](https://ui.adsabs.harvard.edu/abs/2024AIPA...14b5252L/abstract)）；Zhu Y.X. et al. 2021, "Calibration of the energy response matrix for X-ray detector CCD236", JINST 16, P05016（[ADS](https://ui.adsabs.harvard.edu/abs/2021JInst..16P5016Z/abstract)）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] 死时间专题：Xiao S. et al. 2020, "Deadtime calculation method of the High Energy X-ray telescope (HE) onboard the Insight-HXMT satellite", JHEAp 26, 58；Tuo Y.L. et al. 2024, "Revisiting the dead time effects of Insight-HXMT/ME on timing analysis", MNRAS 532, 4317；时间系统：Tuo Y.L. et al. 2022, "In-orbit timing calibration of the Insight-Hard X-ray Modulation Telescope", ApJS 259, 14；PSF：Nang Y. et al. 2020, JHEAp 25, 39（均只核实到题目与卷页）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)

### Inferences
- [inference] 文献对谱线只做到"碘活化""结构材料活化"这一层，没有给出同位素。按 NaI 探测器活化的一般规律，191 keV 与 67 keV 常分别对应 ¹²³I（159 keV + K 壳层结合能）与 ¹²⁵I（35.5 keV + K 壳层结合能）的电子俘获衰变，~31 keV 对应 Te/I 的 K X 射线——这是通用知识，**不是 HXMT 论文的结论**，本次未检索到出处。
- [speculation] CsI 的 1372、1729、2767 keV 三条线在能量上与 ²⁴Na（铝结构经 ²⁷Al(n,α) 活化）的 1369 keV、2754 keV 及后者的双逃逸峰 1732 keV 吻合；478 keV 与 ⁷Be（477.6 keV）或 ¹⁰B(n,α)⁷Li* 吻合；1182 keV 与 67 keV 可能与 ¹⁸²Ta（1189/1221 keV、67.7 keV，半衰期 114 天）有关；692 keV 可能是碘/铯同位素电子俘获的 γ + K X 射线和峰。以上全部是凭能量做的猜测，无文献支持；若成立，这些线的强度时间序列就是"中子通量/质子通量的星上活化剂量计"。
- [inference] 用盲探测器时要注意它们并非"与开放探测器完全相同只是看不到天空"：HE 盲探测器多了 2 mm 钽盖（额外的钽荧光与活化源，DXRB 论文因此弃用），ME 盲视场辐照损伤更轻（遮盖物挡住了低能粒子），LE 盲视场中有一片带 ⁵⁵Fe 源（半衰期 2.7 年，十年后已衰减到初始的约 1/13）。
- [inference] ME/LE 的 "flare" 被本底组当作需要剔除的污染，文献只给了现象描述和"低能带电粒子"的推测，没有后续的地理分布、粒子种类、能谱或与地磁活动关系的系统研究（官方论文列表题目中无对应条目）。

### Gaps
- 逐条谱线的同位素证认、半衰期拟合：未在所取回文献中找到（也许在 Li et al. 2009、Xie et al. 2015 的发射前模拟论文中，未检索）。
- 2025 年"六年本底"综述（天文学报）与 2024 年 LE 本底实验论文的具体结论：未读到。
- HE 本底上升速率的数值（每年百分之几）：五年论文只给图不给数。
- 本底与太阳活动、地磁暴的定量关系：White Book 只定性说 CsI 本底依赖太阳活动与地磁活动；未找到定量研究。
- "light pollution"（杂散光）问题的专门论文：未找到。

---

## 6. 已发表的 off-label / 非常规用途清单（避免把旧闻当新点子）

### Takeaway
已经成体系的非常规方向有四个：地球伽马闪（TGF，2 篇专题+1 篇联合，但只发布了前 4 年样本）、X 射线掩星测中高层大气密度（5 篇，含汤加火山事件）、GRB/引力波/中微子/FRB 对应体搜寻（成熟流水线与多份星表）、磁星暴/FRB 200428（挖得最深）。做过但只是单篇或少量论文的有：SAA 漂移（2020、2024 两篇）、SEM 初步结果（1 篇中文）、脉冲星导航在轨验证（1 篇+若干算法文章）、弥散 X 射线本底（高银纬 1 篇、银道面 LE 1 篇）、太阳耀斑（约 5 篇个例）、CsI 地球掩星测 Crab（1 篇）、原初黑洞等暗物质限制（借用 DXRB 能谱）。**在两轮搜索和官方 411 篇论文题目扫描中均未发现**：Lorentz 不变性检验、惰性中微子/轴子谱线搜寻、月掩、偏振测量、机器学习本底模型、地球电子束（TEB）、CsI 全天掩星监测星表、针对带电粒子事例或 ME/LE flare 的专门研究、基于 PM/HVT 的地磁暴或粒子沉降研究。

### Cited Findings

**6.1 地球伽马闪（TGF）**
- [confirmed] Yi Qibin, Zhao Yi, Xiong Shaolin et al. 2025, "First results of terrestrial gamma-ray flash observed by Insight-HXMT", SCPMA 68, 251013（[ADS 2025SCPMA..6851013Y](https://ui.adsabs.harvard.edu/abs/2025SCPMA..6851013Y/abstract)）。数据：HE（NaI+CsI）事例；前 4 年识别出 282 个亮 TGF；中纬度（30°–43°）TGF 较罕见、性质与低纬（<30°）无显著差异；HE 测得的硬度比与持续时间无关（与此前研究不同）；配套数据集公开（282 个 TGF，T0±1 s 事例，CC BY 4.0）— [宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)；[Zenodo 数据集](https://zenodo.org/records/10976454)；[官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] Yi Qibin, Guo Haoxuan, Xiong Shaolin et al. 2025, "Unveil the TGF-lightning relation with a large sample of TGFs detected by Insight-HXMT", SCPMA 68, 281011（[ADS](https://ui.adsabs.harvard.edu/abs/2025SCPMA..6881011Y/abstract)；只核实到题目与卷页）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] Zhang HongBo et al. 2021, "On the Terrestrial Gamma-Ray Flashes Preceding Narrow Bipolar Events", GRL 48, e92160（[ADS](https://ui.adsabs.harvard.edu/abs/2021GeoRL..4892160Z/abstract)；列入官方论文列表，HXMT 数据的具体用法未核实）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] 背景数字：2017-06-26 至 2021-06-30 的暴搜索共得 660 个 burst，分类为 322 个 GRB、245 个 TGF、4 个太阳耀斑、**33 个 "charged particles"**、56 个未分类；White Book 称约每年 100 个 TGF，并给出有/无闪电关联的星下点分布图 — [Song X.Y. et al. 2022, ApJS 259, 46](https://arxiv.org/abs/2203.00878)；[AO-09 White Book §3.2.3.2](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- 挖掘深度：IHEP 暴发组的系统性工作，但公开样本只到 2021 年中。

**6.2 太阳耀斑**
- [confirmed] Zhang P. et al. 2021, "Non-thermal Electron Energization During the Impulsive Phase of an X9.3 Flare Revealed by Insight-HXMT", ApJ 918, 42 — 用 CsI 观测 2017-09-06 X9.3 耀斑，小波分析得约 22 s 准周期脉动，光子谱指数由 ~1.8 → ~2.0 → ~1.8 演化 — [arXiv:2106.09506](https://arxiv.org/abs/2106.09506)
- [confirmed] 只核实到题目与卷页的四篇：Zhang Ping et al. 2022, "X-Ray Fine Structure of a Limb Solar Flare Revealed by Insight-HXMT, RHESSI and Fermi", RAA 22, 055006（[ADS](https://ui.adsabs.harvard.edu/abs/2022RAA....22e5006Z/abstract)）；Zhao Hai-Sheng, Li Dong, Xiong Shao-Lin et al. 2023, "Paired quasi-periodic pulsations of hard X-ray emission in a solar flare", SCPMA 66, 259611（[ADS](https://ui.adsabs.harvard.edu/abs/2023SCPMA..6659611Z/abstract)）；Li Dong, Warmuth A., Wang Jincheng, Zhao Haisheng et al. 2023, "Global Energetics of Solar Powerful Events on 2017 September 6", RAA 23, 095017（[ADS](https://ui.adsabs.harvard.edu/abs/2023RAA....23i5017L/abstract)）；Zhang Haiyun et al. 2026, "Patterns of X-Ray and γ-Ray Flares: From Blazar to Magnetar and Sun", ApJ 999, 246（[ADS](https://ui.adsabs.harvard.edu/abs/2026ApJ...999..246Z/abstract)）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] Yu Xiao Xia, Zhang Shuang Nan, Lu Hong et al. 2024, "A Comparative Study of Two Contrasting Cosmic-Ray Events Caused by Solar Eruptions from NOAA AR 12673 in 2017 September", ApJ 960, 85（[ADS](https://ui.adsabs.harvard.edu/abs/2024ApJ...960...85Y/abstract)；列入官方论文列表，HXMT 数据在其中的角色未核实）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- 挖掘深度：都是个例研究；前四年暴搜索只归类出 4 个太阳耀斑（当时处于太阳活动极小期）。

**6.3 SAA、粒子监测器与空间环境**
- [confirmed] Zhao H.S., Liu C.Z., Li X.Q. et al. 2020, "The observation of the South Atlantic Anomaly with the particle monitors onboard Insight-HXMT", JHEAp 26, 95（[ADS](https://ui.adsabs.harvard.edu/abs/2020JHEAp..26...95Z/abstract)）：用 3 个 PM 跟踪 2017-06-19 至 2019-06-19 的 SAA；搜索摘要给出峰值位置西漂 0.43±0.01 °/yr、北漂 0.17±0.01 °/yr（另一条搜索摘要写约 0.3 °/yr 且纬度无明显变化，两者不一致，需读原文）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] Zhao Hai-Sheng, Zheng Shi-Jie, Li Liu-Yuan et al. 2024, "Analysis of the drift of the South Atlantic Anomaly using particle monitors onboard Insight-HXMT", JHEAp 44, 187（[ADS](https://ui.adsabs.harvard.edu/abs/2024JHEAp..44..187Z/abstract)；只核实到题目与卷页）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] Lu X.F., Liu C.Z., Li X.B. et al. 2020, "Design and calibration of the high energy particle monitor onboard the Insight-HXMT", JHEAp 26, 77：HPM 数据除给 SAA 告警外也用于绘制 SAA 图 — [arXiv:1911.01594](https://arxiv.org/abs/1911.01594)
- [confirmed] 王春琴、沈国红、张珅毅等 2019,《HXMT 卫星空间环境监测器及初步观测结果》,《空间科学学报》39(6), 770–777：SAA 带电粒子辐射集中在约 80°W–20°E、0°–40°S，高能电子方向各向异性明显强于质子；2017-09 太阳质子事件未对该轨道产生可观测影响；地磁扰动使北美上空（约 40°–43°N）和澳大利亚西南（约 40°–43°S）出现粒子辐射增强且方向性很强 — [CJSS 页面](https://cjss.ac.cn/cn/article/doi/10.11728/cjss2019.06.770)
- 挖掘深度：SAA 漂移做过两轮；SEM 只见一篇 2019 年的初步结果；PM 数据的其他用途未见。

**6.4 脉冲星导航与计时应用**
- [confirmed] Zheng S.J., Zhang S.N., Lu F.J. et al. 2019, "In-orbit Demonstration of X-ray Pulsar Navigation with the Insight-HXMT Satellite", ApJS 244, 1：用 2017-08-31 至 09-05 共 5 天的 Crab 观测和 SEPO 方法，HE/ME/LE 各自独立定轨，合并后位置精度 10 km（3σ）、速度 10 m/s（3σ）— [arXiv:1908.01922](https://arxiv.org/abs/1908.01922)
- [confirmed] 只核实到题目与卷页：Wang Yidi, Zhang Shuangnan, Ge Mingyu et al. 2023, "Fast On-Orbit Pulse Phase Estimation of X-Ray Crab Pulsar for XNAV Flight Experiments", IEEE TAES 59, 3395（[ADS](https://ui.adsabs.harvard.edu/abs/2023ITAES..59.3395W/abstract)）；Wang Yidi, Zheng Wei, Zhang Shuangnan et al. 2023, "Review of X-ray pulsar spacecraft autonomous navigation", Chinese Journal of Aeronautics 36(10), 44（[ADS](https://ui.adsabs.harvard.edu/abs/2023ChJAn..36j..44Y/abstract)）；Yan L.L. et al. 2022, "X-ray timing observation of navigation pulsars",《航空学报》（[期刊页](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2022.26588)）；Bachetti M. et al. 2026, "A Simple, Flexible Method for Timing Cross-calibration of Space Missions", PASP 138, 014501（[ADS](https://ui.adsabs.harvard.edu/abs/2026PASP..138a4501B/abstract)）；Zhao Haisheng et al. 2023, "Long-term evolution of the X-ray flux of the Crab pulsar", RDTM 7, 48（[Springer](https://link.springer.com/article/10.1007/s41605-023-00392-2)）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] "脉冲星导航"是核心科学团队第 5 组、官网成果栏目之一，也是 2025 年报告列出的四项科学目标之一 — [Zhang S.-N. 2020 §4.3](https://arxiv.org/abs/1910.09613)；[宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)
- 挖掘深度：在轨验证 1 篇 + 算法/综述若干；属于设计目标之内，不算新。

**6.5 X 射线地球掩星测中高层大气密度**
- [confirmed] Yu Dao-Chun et al. 2022, "Measurement of the vertical atmospheric density profile from the X-ray Earth occultation of the Crab Nebula with Insight-HXMT", AMT 15, 3141：2018-09-30 的 Crab 掩食个例 — [AMT](https://amt.copernicus.org/articles/15/3141/2022/)
- [confirmed] Yu Daochun, Li Haitao, Li Baoquan, Ge Mingyu, Tuo Youli et al. 2022, "New method for Earth neutral atmospheric density retrieval based on energy spectrum fitting during occultation with LE/Insight-HXMT", AdSpR 69, 3426：用 LE 2017-11-16 的 Crab 掩食做能谱拟合反演，90–200 km 密度显著低于 NRLMSISE-00（110–120 km 低 34.4%）— [arXiv:2203.05419](https://arxiv.org/abs/2203.05419)
- [confirmed] Xue Wang-Chen, Li Xiao-Bo, Xiong Shao-Lin et al. 2023, "Neutral Atmospheric Density Measurement Using Insight-HXMT Data by Earth Occultation Technique", ApJS 264, 5：115 组 Crab 掩食、6–100 keV、三台望远镜联合、贝叶斯反演；55–80/80–90/90–100 km 分别比 NRLMSISE-00 低约 10%/20%/25%，与 NRLMSIS 2.0 大体一致 — [arXiv:2211.00293](https://arxiv.org/abs/2211.00293)
- [confirmed] Katsuda S., Shinagawa H., Fujiwara H. et al. 2024, "X-Raying Neutral Density Disturbances in the Mesosphere and Lower Thermosphere induced by the 2022 Hunga-Tonga Volcano Eruption-Explosion", GRL 51(20), DOI 10.1029/2024GL112025：利用恰好在 2022-01-14 至 01-16 进行的 Cas A 观测中的大气掩食，测 ~90–150 km 中性密度；喷发后震中约 1000 km 内密度严重亏损，较弱亏损延伸到约 7000 km 并持续 8 小时以上，剖面有垂直约 20 km、水平约 1000 km 的波状结构 — [arXiv:2410.09410](https://arxiv.org/abs/2410.09410)
- [confirmed] Zhang Hao-Hui, Xue Wang-Chen, Li Xiao-Bo et al. 2025, "Measurement of Neutral Atmosphere Density During the Years of Increasing Solar Activity Using Insight-HXMT Data with the Earth Occultation Technique", ApJ 988, 233：2022–2024 年 83 组 Crab 掩食、三台望远镜、最大似然反演，覆盖 55–130 km；NRLMSIS 在 90–95 与 100–120 km 高估约 20%；55–65 km 可能有选择偏差 — [arXiv:2502.19678](https://arxiv.org/abs/2502.19678)
- 挖掘深度：已成系列（2 个团队、5 篇），但样本集中在 Crab（外加 1 次 Cas A 事件）。

**6.6 弥散 X 射线本底（CXB）与银河系弥散辐射**
- [confirmed] Huang Rui, Cui Wei, Liao Jin-Yuan et al. 2022, "Insight-HXMT measurements of the diffuse X-ray background", MNRAS 513, 4074：LE+HE 空白天区测 1.5–120 keV DXRB 谱；LE 本底由盲探测器直接测、HE 本底由地球遮挡数据导出；截断幂律光子指数 1.40、截断能 55 keV、1 keV 处归一 9.57 ph cm⁻² s⁻¹ keV⁻¹ sr⁻¹（已修正地球反照与大气辐射，HE 能段显著）；归一化低于除 HEAO-1 外的多数已发表结果；HE 用了空白天区 377 ks、地球遮挡 187 ks（前两年 166 段遮挡过渡数据）— [arXiv:2306.10114](https://arxiv.org/abs/2306.10114)
- [confirmed] "The Diffuse X-Ray Background of the Insight-HXMT/LE Telescope in the Galactic Plane", ApJS 260(2), 42 (2022)：用前三年 LE 银道面扫描数据，把银道面划成 360×20 个 1°×1° 像元逐像元给出 DXB 谱，受银河 H I 吸收与银河脊 X 射线辐射影响，与 RXTE/PCA 结果一致（作者名单未核实；信息来自搜索摘要）— [Scilit 记录](https://www.scilit.net/article/655e9ab6c90e7ca46351779b691c5e3d)；官网另有"Diffuse X-ray Background (DXB) of Insight-HXMT/LE"本底动态页 — [官网 Background 页](http://hxmten.ihep.ac.cn/background.jhtml)
- [confirmed] 成像方法：Guan J., Lu F.-J., Zhang S. et al. 2020, "A modified direct demodulation method applied to Insight-HXMT Galactic plane scanning survey", JHEAp 26, 11（[ADS](https://ui.adsabs.harvard.edu/abs/2020JHEAp..26...11G/abstract)）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- 挖掘深度：高银纬 CXB 只用了前两年数据、未用 ME；银道面只做了 LE。

**6.7 CsI 地球掩星技术（EOT）与 MeV 段**
- [confirmed] Zheng Chao, Xiong Shao-Lin, Li Cheng-Kui et al. 2025, "Improved performance of Insight-HXMT/HE CsI detectors", SCPMA 68, 271011（[ADS](https://ui.adsabs.harvard.edu/abs/2025SCPMA..6871011Z/abstract)）：搜索摘要称该文首次用地球掩星技术测 Crab 来评估 CsI 在定点观测下的能量响应，并把定点观测能段从 1–250 keV 扩展到 1–750 keV（未读到原文）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] Takata J. et al. 2024, "Efficiency of Nonthermal Pulsed Emission from Eight MeV Pulsars", ApJ 965, 126（[ADS](https://ui.adsabs.harvard.edu/abs/2024ApJ...965..126T/abstract)；列入官方论文列表，HXMT 数据用法未核实）；White Book 称 CsI 全天监测 "can also monitor the MeV pulsars" — [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)；[AO-09 White Book §3.2.3.2](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- 挖掘深度：刚起步（1 篇、1 个源）。

**6.8 FRB 200428 / 磁星暴（已被充分挖掘，仅列代表）**
- [confirmed] Li C.K., Lin L., Xiong S.L. et al. 2021, Nature Astronomy（arXiv:2005.11071）：FRB 200428 与 SGR J1935+2154 X 射线暴的证认 — [宋黎明 2025 昆明报告](https://ep.bao.ac.cn/ep/cms/files/%E6%85%A7%E7%9C%BC-HXMT%E5%8D%AB%E6%98%9F%E7%A7%91%E5%AD%A6%E8%BF%90%E8%A1%8C%E5%92%8C%E6%88%90%E6%9E%9C-%E6%98%86%E6%98%8E-20250822.pdf)
- [confirmed] Cai Ce et al. 2022, "An Insight-HXMT Dedicated 33 day Observation of SGR J1935+2154. I. Burst Catalog" 与 "II. Burst Spectral Catalog", ApJS 260, 24 与 25；Li Xiaobo et al. 2022, ApJ 931, 56（FRB 200428 相关 X 射线暴的 QPO）；Xiao Shuo et al. 2024, "The Self-organized Criticality Behaviors of Pulses in Magnetar Bursts", ApJS 274, 14；Xue Wang-Chen et al. 2026, ApJ 998, L44（2022-10 活跃期与 KM40m 联合）等十余篇 — [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)

**6.9 GRB、引力波、中微子、FRB 对应体搜寻**
- [confirmed] Li T.P., Xiong S.L., Zhang S.N. et al. 2018, "Insight-HXMT observations of the first binary neutron star merger GW170817", SCPMA 61, 031011 — [AO-09 White Book 参考文献](http://hxmten.ihep.ac.cn/u/cms/enwww/202605/111005108h32.pdf)
- [confirmed] Cai C., Xiong S.L., Li C.K. et al. 2021, MNRAS 508, 3910：盲搜（SNR 法）+ 定向相干搜索流水线；前两年数据中定向搜索多找回 40 个 Swift/BAT 或 Fermi/GBM 已知但盲搜漏掉的 GRB；灵敏度约 1.5×10⁻⁸ erg/cm²（200 keV–3 MeV）；对 LIGO-Virgo O2、O3a 事件未发现显著成协暴；解决了光变曲线中 spike 引起的假触发 — [arXiv:2109.12270](https://arxiv.org/abs/2109.12270)
- [confirmed] Song X.Y., Xiong S.L., Zhang S.N. et al. 2022, "The First Insight-HXMT Gamma-Ray Burst Catalog: The First Four Years", ApJS 259, 46：322 个 GRB；对 41 个高能中微子、48 个引力波、39 个 FRB 外部触发做了定向搜索 — [arXiv:2203.00878](https://arxiv.org/abs/2203.00878)
- [confirmed] Guidorzi C., Marongiu M., Martone R. et al. 2020, "A search for prompt γ-ray counterparts to fast radio bursts in the Insight-HXMT data", A&A 637, A69：39 个 FRB，无信号，1 s 尺度光度上限多为 10⁴⁹–10⁵¹ erg/s — [arXiv:2003.10889](https://arxiv.org/abs/2003.10889)；Guidorzi C. et al. 2020, A&A 642, A160（FRB 180916.J0158+65 跟踪观测）— [arXiv:2008.11965](https://arxiv.org/abs/2008.11965)
- [confirmed] 定位类：Xiao Shuo et al. 2025, "The First Insight-HXMT Localization Catalog of Short-duration Gamma-Ray Bursts: The First Six Years", ApJS 278, 60；Xiao S. et al. 2021, "Enhanced Localization of Transients Based on a Novel Cross-correlation Method", ApJ 920, 43；Luo Qi et al. 2023, "Localization method for the pointing observation with the Collimated Telescopes of Insight-HXMT", RAA 23（官方列表写 "23, 5004"）；Svinkin D.S. et al. 2022, IPN 短暴定位星表，ApJS 259, 34 — [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] GRB 221009A：An Zheng-Hua et al. 2023（arXiv:2303.01203，HXMT+GECAM-C，E_iso ~1.5×10⁵⁵ erg）；Zhang Wen-Long et al. 2025, ApJ 986, 170；Zheng Chao et al. 2024, ApJ 962, L2；Zhang Yan-Qiu et al. 2024, ApJ 972, L25；Kann D.A. et al. 2023, ApJ 948, L12 — [arXiv:2303.01203](https://arxiv.org/abs/2303.01203)；[官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)

**6.10 暗物质 / 新物理（都是"借用 HXMT 已发表能谱结果"）**
- [confirmed] Yang Chen, Pan Jun-Da, Zhang Xin 2026, "Constraining primordial black hole abundance with Insight-HXMT", Commun. Theor. Phys. 78, 045402：限制 2×10¹⁶–5×10¹⁷ g 的原初黑洞；直接用观测数据或加天体物理本底模型时 HXMT 无优势，采用截断幂律本底模型时把"PBH 构成全部暗物质"的排除上限推到 4×10¹⁷ g — [arXiv:2412.09297](https://arxiv.org/abs/2412.09297)
- [confirmed] Ge Shao-Feng, He Xiao-Gang, Ma Xiao-Dong, Sheng Jie 2022, "Revisiting the fermionic dark matter absorption on electron target", JHEP 05, 191：列入官方论文列表，但其 arXiv 摘要未提 Insight-HXMT，只说用了天体物理 X/γ 射线观测限制 — [arXiv:2201.11497](https://arxiv.org/abs/2201.11497)
- [confirmed] Caputo Andrea et al. 2026, "Heavy dark photon handbook: Cosmological and astrophysical bounds", PRD 113, 075001（[ADS](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113g5001C/abstract)；列入官方论文列表，HXMT 数据用法未核实）— [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)

**6.11 检索后"未发现"的方向（证据强度见各条）**
- [confirmed] 官方论文列表（411 篇，2026-05-19 更新）的题目中没有出现 Lorentz、invariance violation（只有 "Scale-invariant"）、sterile、axion、neutrino、lunar、Moon、machine learning、deep learning、neural、polarimetry（只有与 IXPE 联合的 Sco X-1 偏振论文 La Monaca et al. 2024, ApJL 960, L11）、electron beam、geomagnetic、storm、precipitation 等词 — [官方论文列表](http://hxmten.ihep.ac.cn/LatestPapers/992.jhtml)
- [confirmed] 针对 "Insight-HXMT + Lorentz invariance / GRB 221009A time lag" 的两次联网搜索只返回基于 LHAASO、Fermi/GBM、Swift 的 LIV 工作；针对 "sterile neutrino / dark matter decay line + Insight-HXMT blank sky"、"polarization / lunar occultation"、"machine learning background" 的搜索各一次，均无 HXMT 专门结果（XMM-Newton 空白天区 3.5 keV 线搜索是他星工作）— 检索入口示例：[LIV 检索返回的代表性论文](https://arxiv.org/abs/2308.03031)；[DM 线检索返回的代表性论文](https://arxiv.org/abs/1812.06976)

### Inferences
各方向"还有多少余量"的判断（全部为 [inference]，供选题时排除旧闻）：

| 方向 | 已做到哪一步 | 余量判断 |
|---|---|---|
| TGF | 前 4 年 282 个亮 TGF 样本、TGF–闪电关系 | 2021-07 之后约 5 年数据未见发表；地球电子束（TEB）、弱 TGF、与 GECAM/ASIM 联合、用 18 位 HVT 标签区分电子与光子，均未见 |
| 太阳耀斑 | 约 5 篇个例（2017-09 事件为主） | 第 25 太阳周（2022–2026）的系统耀斑样本、CsI 的 MeV 段核线/中子信号未见系统工作 |
| SAA / PM | 2 年与（题目显示的）更长基线漂移分析 | 只用了 PM 的峰位；SAA 形状/面积、太阳周调制、PM+HVT+CsI+LE forced trigger 联合、十年基线未见 |
| SEM | 1 篇 2019 年初步结果 | 方向性/能谱数据几乎未被天文界使用；但数据可得性未确认 |
| 脉冲星导航 | 5 天 Crab 定轨验证 + 算法论文 | 属设计目标；十年 Crab 数据用于"时间保持/星上时钟长期稳定度"未见 |
| 大气密度掩星 | 约 200 组 Crab 掩食 + 1 次 Cas A 事件 | 其他亮源掩食、全任务期气候学、太阳周/地磁暴响应、成分（O₂/N₂/O/Ar）分离、重力波谱统计仍有空间 |
| CXB | 前两年 LE+HE、截断能 55 keV、归一化偏低 | 十年数据重做、纳入 ME、用地球掩星调制直接测 CXB（而非仅用遮挡数据当本底）、大气反照谱本身当科学目标，未见 |
| 银河弥散/脊 | LE 三年扫描 1°×1° 图 | ME/HE 段的银河脊硬 X 射线成分、十年叠加未见 |
| CsI 地球掩星 | Crab 一个源的响应验证 | 类 BATSE/GBM 的全天亮源掩星监测、0.2–3 MeV 弥散辐射/511 keV 线未见 |
| GW/中微子/FRB | O2、O3a、39 个 FRB、41 个中微子 | O4 时期结果只见 GCN 通告（搜索摘要），未见论文；亚阈值叠加分析有空间 |
| LIV | 未发现 | HXMT 的 μs 级时间标签 + keV–MeV 大面积对谱延迟型 LIV 检验是空白；但 GRB 221009A 主暴段有饱和问题（GRB 星表对饱和样本单独标注） |
| 暗物质 | 3 篇，均借用已发表 DXRB 谱 | 直接在十年空白天区/地球遮挡"差谱"上做线搜索未见；难点是能量分辨（HE ~17%@60 keV）与活化线系统误差 |
| 月掩 / 偏振 | 未发现 | 月球只作为规避对象出现；准直+非成像探测器做偏振无文献先例 |
| 带电粒子事例 | 暴搜索中归类出 33 个 "charged particles" 事件；ME/LE flare 仅被当作污染剔除 | 没有专门论文——"粒子沉降区地图""flare 的地磁活动依赖"是明确空白 |
| 机器学习本底 | 未发现；已有物理参数化模型（You et al. 2021） | 空白，但需证明优于"数据库+盲探测器"的 1–3% 系统误差才有价值 |

- [inference] 暴搜索流水线把 5% 的触发（33/660）归为带电粒子事件、8.5%（56/660）未分类，这两类样本从未被单独发表，是现成的"废料"入口。
- [inference] 官方团队自己已经把 HE 定位为"多用途仪器"（昆明报告转述 TGF 论文结论："尽管 HE 是为天文观测而专门设计的，但它实际上是一个多用途仪器，适用于研究来自地球的高能辐射现象"），说明向地球与空间物理方向的 off-label 选题在团队内部是被认可的叙事。

### Gaps
- 约 15 篇 2023–2026 年论文只核实到官方列表中的题目、作者、期刊卷页与 ADS 链接，未读到摘要（NASA ADS 返回 HTTP 405，搜索配额耗尽）：两篇 2025 年 TGF 论文的定量结论（除昆明报告转述的部分）、2024 年 SAA 漂移论文的新漂移率、2024 年"两次宇宙线事件"论文用了 HXMT 的什么数据、2025 年 CsI 性能论文的掩星细节、三篇 2025 年天文学报综述、2026 年 dark photon handbook 与 2022 年 JHEP 论文对 HXMT 数据的具体用法。写最终报告前若要引用这些论文的结论，需补读原文。
- "未发现"只能说明在官方 411 篇列表（截至 2026-05-19）的**题目**和有限次搜索中没有出现；第三方用公开数据做的工作若未被官方收录、或 2026-05 之后发表的论文，可能漏掉。LIV、暗物质线搜索两项建议在 ADS 全文检索中再确认一次。
- Zhao et al. 2020 SAA 论文的漂移率存在两条互相矛盾的搜索摘要（0.43 °/yr 西漂+0.17 °/yr 北漂 vs 约 0.3 °/yr 且纬度无明显变化），未能读原文裁定。
- GECAM 已发表 TGF 与地球电子束（TEB）结果（arXiv:2306.10255，搜索中出现），HXMT 是否有 TEB 样本未见报道。
- 官方 ADS 公共文库（https://ui.adsabs.harvard.edu/public-libraries/R_dGKc_6RSyswz9mVTsUuA）与高能所机构知识库（http://ir.ihep.ac.cn/handle/311005/264769）是做完整查新的权威入口，本次均未能访问。
