#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detector_response_calc.py
=========================
Insight-HXMT 各子探测器对慢速（beta~1e-3）、大质量、强相互作用暗物质的响应估算。
配套笔记: detector_response.md （同目录）。

本脚本只做"量级 + 不确定带"的估算，全部物理输入在下面集中列出并注明来源：
  [M]  = measured / 文献实测或仪器论文给出的数（来源见笔记中的 URL）
  [I]  = inference / 由 [M] 推出的假设或外推（笔记中逐条说明）
两种散射模型:
  模型 A (默认, ff=False): 纯接触/几何截面, 无靶核形状因子, 反冲谱在 [0, E_max] 平坦;
  模型 B (ff=True)       : 再乘靶核 Helm 形状因子 F^2 (DEAP-3600 Model I 的保守做法).
未建模: 风的方向性与地球遮挡、卫星物质对粒子的减速(仅表 G1 给解析上限)、核反冲脉冲形状偏移、
        晶体倒角/封装等几何细节、级联重叠区(sigma >~ 1e-18 cm^2)的额外淬灭.
运行:  python3 detector_response_calc.py            (约 5 分钟, 仅依赖 numpy; 随机种子固定)
"""
import numpy as np

rng = np.random.default_rng(20261118)
C_KMS = 299792.458
AMU = 0.93149410  # GeV
NA = 6.02214076e23
G2GEV = 5.609588e23            # 1 g = 5.61e23 GeV
CM2G_TO_CM2GEV = 1.0 / G2GEV   # 1 cm^2/g = 1.783e-24 cm^2/GeV

# --------------------------------------------------------------------------
# 1. 靶核与材料  (标准物性常数; BC-408 取自 Saint-Gobain 数据表 [M])
# --------------------------------------------------------------------------
M = {  # 核质量 GeV
    'H': 0.938272, 'C': 12.000 * AMU, 'Na': 22.98977 * AMU, 'Si': 27.97693 * AMU,
    'I': 126.90447 * AMU, 'Cs': 132.90545 * AMU, 'Ta': 180.94788 * AMU, 'Al': 26.98154 * AMU}
A = {'H': 1, 'C': 12, 'Na': 23, 'Si': 28, 'I': 127, 'Cs': 133}

MATERIALS = {  # 数密度 cm^-3
    'NaI':   {'rho': 3.67,  'n': {'Na': 3.67 / 149.894 * NA, 'I': 3.67 / 149.894 * NA}},
    'CsI':   {'rho': 4.51,  'n': {'Cs': 4.51 / 259.810 * NA, 'I': 4.51 / 259.810 * NA}},
    'BC408': {'rho': 1.032, 'n': {'H': 5.23e22, 'C': 4.74e22}},          # [M] 数据表
    'Si':    {'rho': 2.329, 'n': {'Si': 2.329 / 28.0855 * NA}},
}

# --------------------------------------------------------------------------
# 2. 标准晕模型 (Baxter et al. 2021 推荐值 [M]): v0=238, v_esc=544, |v_sun+v0|=250.2 km/s
# --------------------------------------------------------------------------
V0, VESC, VE = 238.0, 544.0, 250.2


def sample_speeds(n, flux_weighted=True):
    """返回地球系速率样本 (km/s)。flux_weighted=True 时按 v f(v) 抽样(即"穿过探测器的粒子")."""
    out = []
    need = n * 3
    while sum(len(o) for o in out) < need:
        vg = rng.normal(0.0, V0 / np.sqrt(2.0), size=(need, 3))
        vg = vg[np.sum(vg**2, axis=1) < VESC**2]
        vg[:, 1] += VE          # 地球系: v_lab = v_gal - v_E; 取模无关符号
        out.append(np.sqrt(np.sum(vg**2, axis=1)))
    v = np.concatenate(out)
    if flux_weighted:
        p = v / v.sum()
        v = rng.choice(v, size=n, p=p, replace=True)
    else:
        v = v[:n]
    return v


def emax_keV(target, v_kms, m_chi=np.inf):
    """最大反冲能 E_max = 2 mu^2 v^2 / m_A  (keV)."""
    mA = M[target]
    mu = mA if np.isinf(m_chi) else m_chi * mA / (m_chi + mA)
    return 2.0 * mu**2 / mA * (np.asarray(v_kms) / C_KMS)**2 * 1e6


# --------------------------------------------------------------------------
# 3. Helm 形状因子 (Lewin & Smith 1996 参数化; 标准做法 [I])
# --------------------------------------------------------------------------
def helm_F2(target, E_keV):
    Aa = A[target]
    q = np.sqrt(2.0 * M[target] * np.asarray(E_keV, float) * 1e-6)  # GeV
    qf = q / 0.1973269804                                             # fm^-1
    s, a = 0.9, 0.52
    c = 1.23 * Aa**(1.0 / 3.0) - 0.60
    rn = np.sqrt(max(c * c + 7.0 / 3.0 * np.pi**2 * a * a - 5.0 * s * s, 0.01))
    x = np.maximum(qf * rn, 1e-9)
    j1 = (np.sin(x) - x * np.cos(x)) / x**2
    F = 3.0 * j1 / x * np.exp(-0.5 * (qf * s)**2)
    if Aa == 1:
        return np.ones_like(x)
    return F**2


# --------------------------------------------------------------------------
# 4. 淬灭因子 / 电离产额模型: 每种给出 central / low / high 三套
# --------------------------------------------------------------------------
def qf_Na(E, case='c'):
    """NaI(Tl) 中 Na 反冲. [M] Lee 2024 (3.8-146 keV: 11.2%->23.7%), Joo 2019, Bignell 2021
    (36-401 keV: 16%->30.5%).  central = 幂律拟合 0.112*(E/3.8)^0.215  [I]
    low = central*0.8 (Collar/Xu 低能偏低) ; high = 0.30 常数 (DAMA, 6.5-97 keV)"""
    E = np.asarray(E, float)
    cen = 0.112 * (np.maximum(E, 1e-3) / 3.8)**0.215 * (1.0 - np.exp(-E / 1.5))
    cen = np.minimum(cen, 0.33)
    if case == 'c':
        return cen
    if case == 'l':
        return 0.8 * cen
    return np.full_like(E, 0.30) * (1.0 - np.exp(-E / 1.5))


def qf_I(E, case='c'):
    """NaI(Tl) 中 I 反冲. [M] Lee 2024/Joo 2019: 6-75 keV 为 4.3%-6.2%; Cintas 2024: 6.0+-2.2% @14 keV;
    DAMA: 0.09+-0.01 (22-330 keV); Fushimi 1993: 0.05+-0.02 (40-300 keV).
    central = 0.05*(E/10)^0.10 [I, >75 keV 为外推]; low = 0.04; high = 0.09."""
    E = np.asarray(E, float)
    cut = (1.0 - np.exp(-E / 3.0))
    if case == 'c':
        return 0.05 * (np.maximum(E, 1e-3) / 10.0)**0.10 * cut
    if case == 'l':
        return 0.04 * cut
    return 0.09 * cut


def qf_CsI(E, case='c'):
    """CsI(Na) 中 Cs/I 反冲. [M] COHERENT 2022 四次多项式(2-70 keV):
       E_vis = 0.05546 E + 4.307 E^2 - 111.7 E^3 + 840.4 E^4  (MeV)  => QF 约 7%-10.7%.
    >60 keV 无现代数据: central = QF(60)*(E/60)^-0.15 [I]; high = 常数 QF(60);
    low = Park2002/Guo2016 一类偏低数据 0.065*(E/60)^-0.30 [I]."""
    E = np.asarray(E, float)
    x = np.minimum(E, 60.0) * 1e-3
    poly = 0.05546 + 4.307 * x - 111.7 * x**2 + 840.4 * x**3
    q60 = 0.05546 + 4.307 * 0.06 - 111.7 * 0.06**2 + 840.4 * 0.06**3
    hi = np.where(E <= 60.0, poly, q60)
    if case == 'h':
        return hi
    if case == 'c':
        return np.where(E <= 60.0, poly, q60 * (np.maximum(E, 60.0) / 60.0)**-0.15)
    lowv = 0.065 * (np.maximum(E, 5.0) / 60.0)**-0.30
    return np.minimum(lowv, poly)


def qf_H(E, case='c'):
    """塑闪中质子反冲. [M] Awe 2018 (EJ-301): 12.3%@17 keV, 9.3%@28, 8.9%@53; 约 100 keV 处极小 ~5%;
    0.1-1 keV >20% (NE-110, Ahlen 1985 / Ficenec 1987, 仅一组实验).  Cappiello 2021 取 20%."""
    E = np.asarray(E, float)
    xs = np.log(np.array([0.05, 4.0, 17.2, 28.1, 52.6, 100.0, 1000.0]))
    ys = np.array([0.20, 0.20, 0.123, 0.093, 0.089, 0.05, 0.05])
    cen = np.interp(np.log(np.maximum(E, 0.05)), xs, ys)
    if case == 'c':
        return cen
    if case == 'l':
        return 0.5 * cen
    return np.minimum(1.5 * cen, 0.30)


def qf_C(E, case='c'):
    """塑闪中碳反冲: ~1% (Awe 2018 引 Hong 2002, Verbinski 1968) [M]; 0.5%-2% [I]."""
    E = np.asarray(E, float)
    return np.full_like(E, {'c': 0.01, 'l': 0.005, 'h': 0.02}[case])


_SI_E = np.array([0.0, 0.68, 0.86, 1.05, 1.22, 1.79, 2.26, 2.97, 3.75, 4.87, 6.07, 7.67, 8.93,
                  10.77, 12.52, 14.48, 16.23, 18.19, 20.67])
_SI_Y = np.array([0.0, 0.088, 0.105, 0.114, 0.123, 0.168, 0.193, 0.215, 0.243, 0.258, 0.274,
                  0.280, 0.302, 0.308, 0.320, 0.329, 0.344, 0.357, 0.360])


def lindhard_si(E):
    eps = 11.5 * np.asarray(E, float) * 14.0**(-7.0 / 3.0)
    k = 0.133 * 14.0**(2.0 / 3.0) * 28.0855**-0.5
    g = 3.0 * eps**0.15 + 0.7 * eps**0.6 + eps
    return k * g / (1.0 + k * g)


def qf_Si(E, case='c'):
    """硅电离产额. [M] Chavarria 2016 (0.7-2.3 keV), Izraelevitch 2017 (1.8-20.7 keV, 表1);
    >20 keV 用 Lindhard(k=0.146) 归一到 20.67 keV 处实测值 [I]. low/high = +-10%."""
    E = np.asarray(E, float)
    scale = 0.360 / lindhard_si(20.67)
    y = np.where(E <= 20.67, np.interp(E, _SI_E, _SI_Y), scale * lindhard_si(np.maximum(E, 20.67)))
    return y * {'c': 1.0, 'l': 0.9, 'h': 1.1}[case]


QF = {('NaI', 'Na'): qf_Na, ('NaI', 'I'): qf_I, ('CsI', 'Cs'): qf_CsI, ('CsI', 'I'): qf_CsI,
      ('BC408', 'H'): qf_H, ('BC408', 'C'): qf_C, ('Si', 'Si'): qf_Si}


def mean_qE(mat, sp, Emax, case='c', n=4000):
    """平坦反冲谱 [0,Emax] 上 <QF(E) E> (keVee)."""
    E = (np.arange(n) + 0.5) / n * Emax
    return float(np.mean(QF[(mat, sp)](E, case) * E))


# --------------------------------------------------------------------------
# 5. 阈值与量程 (keVee).  [M]/[I] 见笔记 "电子学" 一节
# --------------------------------------------------------------------------
G_CSI = 0.42          # CsI 每 keVee 的脉冲幅度 / NaI 每 keVee 的脉冲幅度 [I]: 由 NaI 20-250(350) 与 CsI 40-600(800) 量程比
HE_NOM = (20.0, 350.0)     # NaI 等效 keV: 下阈 ~20 (PHA 道20附近), 上阈 ~350 (Liu 2020: 16-350 keV) [M]
HE_LGM = (100.0, 1500.0)   # 低增益: 增益 1/5 [M]; 上阈取 CsI ~3.6 MeV (Luo 2020: 3-4 MeV 处有效面积陡降)
HVT_THR = (100.0, 50.0, 200.0)   # 反符合阈: 发射前模拟取 100 keV (Xie 2015) [M-模拟假设]; 在轨值未公开
LE_WIN = (1.0, 13.0)       # LE: 0.7-13 keV 设计; 2020-06 后下阈约 2 keV [M]
ME_WIN = (10.0, 35.0)      # ME: 可靠 >=10 keV, 上限 ~35(-44) keV [M]

# --------------------------------------------------------------------------
# 6. 复合泊松: 一条径迹上多次散射的 Sum(QF*E)
# --------------------------------------------------------------------------
def deposit(mat, L_cm, sigma, v_kms, case='c', m_chi=np.inf, return_n=False, ff=False):
    """对每条径迹(路径长 L_cm, 速率 v)抽样: 返回电子等效沉积(keVee) 及 物理反冲能之和(keV).
    ff=False: 纯接触/几何截面, 质心系各向同性 -> 反冲谱在 [0,Emax] 平坦 (Cappiello 2021, Bhoonah 2021 的做法).
    ff=True : dsigma/dE_R 再乘靶核 Helm 形状因子 F^2(E_R) (DEAP-3600 2022 "Model I" 的保守做法);
              sigma 仍指零动量转移截面, 每次"名义散射"以概率 F^2 实际发生."""
    L_cm = np.asarray(L_cm, float)
    ee = np.zeros_like(L_cm)
    er = np.zeros_like(L_cm)
    ntot = np.zeros_like(L_cm)
    for sp, n in MATERIALS[mat]['n'].items():
        lam = n * sigma * L_cm
        Emax = emax_keV(sp, v_kms, m_chi)
        qf = QF[(mat, sp)]
        big = lam > 40.0
        N = np.zeros(L_cm.shape, dtype=np.int64)
        N[~big] = rng.poisson(lam[~big])
        ntot += np.where(big, lam, N)
        # 少次散射: 逐次相加
        kmax = int(N.max()) if N.size else 0
        for k in range(1, kmax + 1):
            idx = np.nonzero(N >= k)[0]
            if idx.size == 0:
                break
            E = rng.random(idx.size) * Emax[idx]
            if ff:
                E = np.where(rng.random(idx.size) < helm_F2(sp, E), E, 0.0)
            ee[idx] += qf(E, case) * E
            er[idx] += E
        # 多次散射: 高斯近似 (复合泊松: mean = lam<x>, var = lam<x^2>)
        if big.any():
            ib = np.nonzero(big)[0]
            # 用 64 个分位点近似 <x>, <x^2>
            u = (np.arange(64) + 0.5) / 64.0
            Eg = Emax[ib][:, None] * u[None, :]
            w = helm_F2(sp, Eg) if ff else 1.0
            x = qf(Eg, case) * Eg
            m1, m2 = (w * x).mean(axis=1), (w * x**2).mean(axis=1)
            ee[ib] += np.maximum(rng.normal(lam[ib] * m1, np.sqrt(lam[ib] * m2)), 0.0)
            e1, e2 = (w * Eg).mean(axis=1), (w * Eg**2).mean(axis=1)
            er[ib] += np.maximum(rng.normal(lam[ib] * e1, np.sqrt(lam[ib] * e2)), 0.0)
    if return_n:
        return ee, er, ntot
    return ee, er


# --------------------------------------------------------------------------
# 7. 各向同性弦长抽样
# --------------------------------------------------------------------------
def iso_lines(n, rb):
    """各向同性均匀通量: 方向 u 均匀分布于球面, 垂直平面内半径 rb 圆盘上均匀取点."""
    mu = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    u = np.stack([st * np.cos(ph), st * np.sin(ph), mu], axis=1)
    # 垂直于 u 的两个基矢
    a = np.where(np.abs(u[:, [2]]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(u, e1)
    r = rb * np.sqrt(rng.random(n))
    al = rng.uniform(0, 2 * np.pi, n)
    p = (r * np.cos(al))[:, None] * e1 + (r * np.sin(al))[:, None] * e2
    return p, u


def chord_cyl(p, u, R, z0, z1):
    """直线 p+t u 与有限圆柱 (x^2+y^2<R^2, z0<z<z1) 的弦长."""
    a = u[:, 0]**2 + u[:, 1]**2
    b = 2 * (p[:, 0] * u[:, 0] + p[:, 1] * u[:, 1])
    c = p[:, 0]**2 + p[:, 1]**2 - R * R
    with np.errstate(divide='ignore', invalid='ignore'):
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.maximum(disc, 0))
        t1 = np.where(a > 1e-14, (-b - sq) / (2 * a), -np.inf)
        t2 = np.where(a > 1e-14, (-b + sq) / (2 * a), np.inf)
        ok = np.where(a > 1e-14, disc > 0, c < 0)
        tz1 = (z0 - p[:, 2]) / u[:, 2]
        tz2 = (z1 - p[:, 2]) / u[:, 2]
    ta, tb = np.minimum(tz1, tz2), np.maximum(tz1, tz2)
    lo, hi = np.maximum(t1, ta), np.minimum(t2, tb)
    return np.where(ok & (hi > lo), hi - lo, 0.0)


def chord_box(p, u, half):
    lo = np.full(len(p), -np.inf)
    hi = np.full(len(p), np.inf)
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = (-half[i] - p[:, i]) / u[:, i]
            tb = (half[i] - p[:, i]) / u[:, i]
        lo = np.maximum(lo, np.minimum(ta, tb))
        hi = np.minimum(hi, np.maximum(ta, tb))
    return np.where(hi > lo, hi - lo, 0.0)


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def hdr(s):
    print('\n' + '=' * 100 + '\n' + s + '\n' + '=' * 100)


def table_kinematics():
    hdr('表 K1  最大反冲能 E_max = 2 mu^2 v^2 / m_A  (keV)   [运动学, 精确]')
    vs = [100, 250.2, 300, 544, 794]
    for mchi in [10.0, 100.0, 1e3, 1e4, np.inf]:
        print(f'\n m_chi = {mchi:g} GeV' + ('   (m_chi >> m_A 极限: E_max = 2 m_A v^2)' if np.isinf(mchi) else ''))
        print('   target   m_A[GeV]  ' + '  '.join(f'v={v:>6g}' for v in vs) + '   q_max(300)[MeV]  F2(Emax/2,300) F2(Emax,300)')
        for t in ['H', 'C', 'Na', 'Si', 'I', 'Cs']:
            mu = M[t] if np.isinf(mchi) else mchi * M[t] / (mchi + M[t])
            em = [float(emax_keV(t, v, mchi)) for v in vs]
            q = 2 * mu * 300 / C_KMS * 1e3
            e300 = float(emax_keV(t, 300, mchi))
            print(f'   {t:>4}   {M[t]:9.3f}  ' + '  '.join(f'{e:8.2f}' for e in em) +
                  f'   {q:10.2f}       {float(helm_F2(t, e300/2)):9.2e}   {float(helm_F2(t, e300)):9.2e}')


def table_halo():
    hdr('表 K2  地球系速率分布 (SHM: v0=238, v_esc=544, v_E=250.2 km/s)')
    v = sample_speeds(400000, flux_weighted=False)
    vf = sample_speeds(400000, flux_weighted=True)
    print(f'  数密度加权: <v>={v.mean():.1f}  rms={np.sqrt((v**2).mean()):.1f}  中位={np.median(v):.1f} km/s')
    print(f'  通量加权  : <v>={vf.mean():.1f}  rms={np.sqrt((vf**2).mean()):.1f}  中位={np.median(vf):.1f} km/s')
    print('  通量加权分位数 (km/s): ' + ', '.join(f'{int(q*100)}%={np.quantile(vf, q):.0f}' for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
    for vc in [140, 300, 400, 500, 600, 700]:
        print(f'  P(v>{vc}) 数密度加权 = {np.mean(v > vc):.4f}   通量加权 = {np.mean(vf > vc):.4f}')
    print(f'  暗物质通量 Phi = rho <v>/m = 0.3 GeV/cm3 * {v.mean():.0f} km/s / m = {0.3*v.mean()*1e5:.3g}/m_GeV  cm^-2 s^-1 (4pi 积分)')
    return v.mean()


def table_mean_qe():
    hdr('表 Q1  单次散射的电子等效能量 (平坦反冲谱, m_chi>>m_A): <QF*E> 与 QF(Emax)*Emax, keVee   central [low, high]')
    for v in [250.2, 300.0, 544.0, 794.0]:
        print(f'\n  v = {v:g} km/s')
        for (mat, sp) in QF:
            Em = float(emax_keV(sp, v))
            ms = [mean_qE(mat, sp, Em, c) for c in 'clh']
            mx = [float(QF[(mat, sp)](np.array([Em]), c)[0] * Em) for c in 'clh']
            print(f'    {mat:>5}:{sp:<2}  Emax={Em:8.2f} keV   <QF E>={ms[0]:7.3f} [{ms[1]:.3f},{ms[2]:.3f}]   max={mx[0]:7.2f} [{mx[1]:.2f},{mx[2]:.2f}]')


def table_mean_qe_ff():
    hdr('表 Q2  变体: 乘靶核 Helm 形状因子后, 每次"名义散射"的平均电子等效能量 <F^2 QF E> (keVee) 与平均反冲能 <F^2 E> (keV)\n'
        '       (central QF; m_chi>>m_A).  与表 Q1 之比即形状因子对连续能损 dE/dx 的压低倍数')
    for v in [250.2, 300.0, 405.0, 544.0]:
        print(f'\n  v = {v:g} km/s')
        for (mat, sp) in QF:
            Em = float(emax_keV(sp, v))
            E = (np.arange(4000) + 0.5) / 4000 * Em
            F2 = helm_F2(sp, E)
            x = QF[(mat, sp)](E, 'c') * E
            print(f'    {mat:>5}:{sp:<2}  <F2>={F2.mean():6.4f}   <F2 E>={np.mean(F2*E):8.3f} keV (无FF {Em/2:8.2f})   '
                  f'<F2 QF E>={np.mean(F2*x):7.3f} keVee (无FF {x.mean():7.3f})   压低倍数={x.mean()/max(np.mean(F2*x),1e-12):6.1f}')


def classify_he(Hlo, Hhi, H):
    return np.where(H < Hlo, 0, np.where(H <= Hhi, 1, 2))


def table_deterministic():
    hdr('表 R1  接触截面 sigma (每核, 与 A 无关) 下的"典型值"响应: v=300 km/s, 垂直穿过, m_chi>>m_A\n'
        '       N=散射次数期望; SumER=物理反冲能之和; Eee=电子等效 central [low,high]; 类别见笔记')
    v = 300.0
    elems = [('NaI 层 3.5 mm', 'NaI', 0.35), ('CsI 层 40 mm', 'CsI', 4.0), ('HVT 塑闪 6.4 mm', 'BC408', 0.64),
             ('SCD 耗尽层 50 um', 'Si', 0.005), ('Si-PIN 1 mm', 'Si', 0.1)]
    for name, mat, L in elems:
        print(f'\n  {name}   (rho*sigma*v^2 = {MATERIALS[mat]["rho"]*(v*1e5)**2*6.2415e11:.3e} eV/cm per cm^2)')
        print('    sigma[cm2]    N_tot      SumER[keV]     Eee_c[keVee]   [low, high]          类别')
        for ex in range(-24, -13):
            s = 10.0**ex
            N = sum(n * s * L for n in MATERIALS[mat]['n'].values())
            er = sum(n * s * L * float(emax_keV(sp, v)) / 2 for sp, n in MATERIALS[mat]['n'].items())
            ee = [sum(n * s * L * mean_qE(mat, sp, float(emax_keV(sp, v)), c) for sp, n in MATERIALS[mat]['n'].items()) for c in 'clh']
            cls = classify_det(name, N, ee)
            print(f'    1e{ex:<4d}   {N:10.3g}  {er:12.4g}   {ee[0]:12.4g}   [{ee[1]:.3g}, {ee[2]:.3g}]   {cls}')


def classify_det(name, N, ee):
    e = ee[0]
    if name.startswith('NaI'):
        lo, hi = HE_NOM
        if N < 0.3:
            return f'单次散射区: P(散射)={N:.2g}, 绝大多数低于阈(20 keVee); 仅高速尾可过阈'
        if e < lo:
            return '多次散射但总和仍 < 20 keVee: 多数不触发'
        if e <= hi:
            return 'NaI 量程内(NOM 20-350 keVee) [但贯穿径迹同时在 CsI 沉积更多 -> 实为 CsI/混合类]'
        if e <= HE_LGM[1]:
            return '超出 NOM 上阈(不记录); LGM 量程内'
        return '超量程: 不记录 + 死时间(>=20us) ; 远超时预期 spike'
    if name.startswith('CsI'):
        lo, hi = HE_NOM[0] / G_CSI, HE_NOM[1] / G_CSI
        if N < 0.3:
            return f'单次散射区: P(散射)={N:.2g}, 单次 <~20 keVee, 低于 CsI 阈(~48 keVee)'
        if e < lo:
            return '多次散射但总和 < ~48 keVee: 多数不触发'
        if e <= hi:
            return f'CsI 事例, NOM 量程内(~{lo:.0f}-{hi:.0f} keVee)'
        if e <= HE_LGM[1] / G_CSI:
            return '超出 NOM 上阈(不记录); LGM 量程内(~240-4200 keVee)'
        return '超量程: 不记录 + 死时间 ; NOM 下预期 spike(巨脉冲后假触发串)'
    if name.startswith('HVT'):
        if e < HVT_THR[1]:
            return '不触发反符合'
        if e < HVT_THR[2]:
            return '阈值附近(阈值本身不确定 50-200 keVee)'
        return '反符合触发 -> 计入 HVT 每秒计数; 1 us 内的 HE 事例被打标'
    if name.startswith('SCD'):
        if N < 0.3:
            return f'单次散射区: P(散射)={N:.2g}; 单次反冲 0-20 keVee -> 普通单事例(约一半落在 1-13 keV)'
        if e <= LE_WIN[1]:
            return '量程内(1-13 keV), 可能跨像素成分裂事例'
        return '超过 LE 上阈(记为超上阈计数, 不进能谱)'
    if name.startswith('Si-PIN'):
        if N < 0.3:
            return f'单次散射区: P(散射)={N:.2g}; 仅 E_R>~27 keV 的反冲 (>10 keVee) 可记录'
        if e < ME_WIN[0]:
            return '多数低于 ME 可靠下限(10 keV)'
        if e <= ME_WIN[1]:
            return 'ME 量程内(10-35 keV) 普通事例'
        return '超过 ME 量程(处理方式文献未述)'
    return ''


def mc_phoswich(nline=600000):
    hdr('表 R2  蒙特卡罗: 各向同性弦长 + SHM 速率(通量加权) + 复合泊松, HE 单个 phoswich (NaI 3.5mm + CsI 40mm, r=9.5cm)\n'
        '       每个"穿过 phoswich 的暗物质粒子"落入各类的概率; A_eff = (S/4) x 概率 (cm^2/模块)\n'
        '       脉冲幅度 H = E_NaI + g*E_CsI (NaI 等效 keV, g=0.42); NOM 记录窗 20-350, LGM 100-1750\n'
        '       PSD 类: f=g*E_CsI/H <0.10 记 NaI 类(脉宽54-70); >0.75 记 CsI 类(>=80); 其余混合')
    R, tN, tC = 9.5, 0.35, 4.0
    rb = np.sqrt(R**2 + ((tN + tC) / 2)**2)
    zc = (tN - tC) / 2.0
    p, u = iso_lines(nline, rb)
    p[:, 2] += zc
    LN = chord_cyl(p, u, R, 0.0, tN)
    LC = chord_cyl(p, u, R, -tC, 0.0)
    hit = (LN + LC) > 0
    S4 = np.pi * rb**2 * hit.mean()
    S4_exact = (2 * np.pi * R**2 + 2 * np.pi * R * (tN + tC)) / 4
    print(f'  平均投影面积 S/4: MC={S4:.1f} cm2, 解析={S4_exact:.1f} cm2 ; 18 模块合计 {18*S4_exact:.0f} cm2')
    LN, LC = LN[hit], LC[hit]
    n = len(LN)
    print(f'  穿过 phoswich 的径迹中: 只过 NaI {np.mean((LN>0)&(LC==0)):.3f}, 只过 CsI {np.mean((LN==0)&(LC>0)):.3f}, 两者都过 {np.mean((LN>0)&(LC>0)):.3f}')
    print(f'  平均弦长: NaI(过 NaI 的径迹) {LN[LN>0].mean():.3f} cm ; CsI(过 CsI 的径迹) {LC[LC>0].mean():.3f} cm')
    v = sample_speeds(n)
    res = {}
    for case, ff in [('c', False), ('l', False), ('h', False), ('c', True)]:
        print(f'\n  QF 取值: {dict(c="central", l="low", h="high")[case]}' +
              ('   *** 变体: 乘靶核 Helm 形状因子 F^2 (I/Cs 的高能反冲被压低) ***' if ff else '   (纯接触截面, 无核形状因子)'))
        print('    sigma      |-------------------- NOM (正常增益) --------------------|   |------ LGM (低增益) ------|')
        print('    [cm2]      P(<阈)   P(NaI类)  P(混合)  P(CsI类)  P(超量程)  Aeff_rec[cm2]    P(记录)  P(超量程)  <N_NaI> <N_CsI>')
        for ex2 in range(-48, -27):
            s = 10.0**(ex2 / 2.0)
            eN, _, nN = deposit('NaI', LN, s, v, case, return_n=True, ff=ff)
            eC, _, nC = deposit('CsI', LC, s, v, case, return_n=True, ff=ff)
            H = eN + G_CSI * eC
            cl = classify_he(*HE_NOM, H)
            f = np.where(H > 0, G_CSI * eC / np.maximum(H, 1e-12), 0)
            rec = cl == 1
            pN = np.mean(rec & (f < 0.10))
            pM = np.mean(rec & (f >= 0.10) & (f <= 0.75))
            pC = np.mean(rec & (f > 0.75))
            cl2 = classify_he(*HE_LGM, H)
            res[(case, ff, ex2)] = (pN, pM, pC, np.mean(cl == 2), np.mean(cl2 == 1), np.mean(cl2 == 2))
            print(f'    {s:9.2e}  {np.mean(cl==0):8.5f} {pN:9.2e} {pM:9.2e} {pC:9.2e} {np.mean(cl==2):9.2e}   {S4_exact*(pN+pM+pC):10.3e}   '
                  f'{np.mean(cl2==1):9.2e} {np.mean(cl2==2):9.2e}  {nN.mean():7.3g} {nC.mean():7.3g}')
    return res, S4_exact


def mc_slabs(nline=400000):
    hdr('表 R3  蒙特卡罗: HVT 反符合板 / LE-SCD / ME-Si-PIN (各向同性通量, SHM 速率)')
    v = sample_speeds(nline)
    # 大平板: 通量加权 cos 分布 p(mu)=2mu
    mu = np.sqrt(rng.random(nline))
    print('\n  (a) HVT: BC-408 6.4 mm 平板 (弦长 t/cos, 上限截断 30 cm); P(E_ee >= 阈) , 阈 = 100 [50, 200] keVee')
    print('      sigma[cm2]   QF central: P(>=100)  P(>=50)  P(>=200)     QF low: P(>=100)    QF high: P(>=100)   (3.4 mm 减薄区 central P(>=100))')
    L = np.minimum(0.64 / mu, 30.0)
    L2 = np.minimum(0.34 / mu, 30.0)
    for ex2 in range(-44, -33):
        s = 10.0**(ex2 / 2.0)
        ec, _ = deposit('BC408', L, s, v, 'c')
        el, _ = deposit('BC408', L, s, v, 'l')
        eh, _ = deposit('BC408', L, s, v, 'h')
        e2, _ = deposit('BC408', L2, s, v, 'c')
        print(f'      {s:9.2e}     {np.mean(ec>=100):8.4f} {np.mean(ec>=50):8.4f} {np.mean(ec>=200):8.4f}        {np.mean(el>=100):8.4f}          {np.mean(eh>=100):8.4f}          {np.mean(e2>=100):8.4f}')
    print('\n  (b) LE SCD: 硅耗尽层 50 um 平板; 每个穿过 SCD 的粒子: P(1-13 keVee) [2020-06 后下阈 2 keV: P(2-13)], P(>13), <N>')
    print('      sigma[cm2]    P(1-13)c   P(2-13)c   P(>13)c    P(1-13)low  P(1-13)high   <N>    P(N>=2 且横向位移>100um)')
    L = np.minimum(0.005 / mu, 1.0)
    disp = 0.005 * np.sqrt(1 - mu**2) / mu
    for ex2 in range(-48, -35):
        s = 10.0**(ex2 / 2.0)
        ec, _, nn = deposit('Si', L, s, v, 'c', return_n=True)
        el, _ = deposit('Si', L, s, v, 'l')
        eh, _ = deposit('Si', L, s, v, 'h')
        print(f'      {s:9.2e}    {np.mean((ec>=1)&(ec<=13)):9.2e}  {np.mean((ec>=2)&(ec<=13)):9.2e}  {np.mean(ec>13):9.2e}  '
              f'{np.mean((el>=1)&(el<=13)):9.2e}  {np.mean((eh>=1)&(eh<=13)):9.2e}  {nn.mean():7.3g}   {np.mean((nn>=2)&(disp>0.01)):9.2e}')
    print('\n  (c) ME Si-PIN: 12.5 x 4.5 x 1 mm 硅块 (各向同性弦); 每个穿过像素的粒子: P(10-35 keVee), P(8-44), P(>35), <N>')
    half = np.array([0.625, 0.225, 0.05])
    rb = np.linalg.norm(half)
    p, u = iso_lines(nline * 3, rb)
    Lb = chord_box(p, u, half)
    hit = Lb > 0
    S4 = np.pi * rb**2 * hit.mean()
    Lb = Lb[hit][:nline]
    vb = sample_speeds(len(Lb))
    print(f'      像素平均投影面积 S/4 = {S4:.4f} cm2 (解析 {2*(1.25*0.45+1.25*0.1+0.45*0.1)/4:.4f}); 平均弦长 {Lb.mean()*10:.3f} mm')
    print('      sigma[cm2]    P(10-35)c  P(8-44)c   P(>35)c    P(10-35)low P(10-35)high  <N>')
    for ex2 in range(-48, -37):
        s = 10.0**(ex2 / 2.0)
        ec, _, nn = deposit('Si', Lb, s, vb, 'c', return_n=True)
        el, _ = deposit('Si', Lb, s, vb, 'l')
        eh, _ = deposit('Si', Lb, s, vb, 'h')
        print(f'      {s:9.2e}    {np.mean((ec>=10)&(ec<=35)):9.2e}  {np.mean((ec>=8)&(ec<=44)):9.2e}  {np.mean(ec>35):9.2e}  '
              f'{np.mean((el>=10)&(el<=35)):9.2e}  {np.mean((eh>=10)&(eh<=35)):9.2e}  {nn.mean():7.3g}')


def table_si_fraction(nsamp=2000000):
    hdr('表 S1  单次散射过阈比例 (每个发生的散射 / 每个"名义截面"散射), 含 Helm 形状因子\n'
        '       eps0 = <F^2 * 1[窗内]>  (相对零动量转移截面 sigma_A0 的效率);  fFF = <F^2> (形状因子总压低);\n'
        '       eps0/fFF = 已发生散射中落入窗内的比例.  速率按通量加权 SHM.  "contact" 行 = 不乘核形状因子')
    v = sample_speeds(nsamp)
    uu = rng.random(nsamp)
    wins = [('NaI:Na', 'NaI', 'Na', (20.0, 350.0)), ('NaI:I ', 'NaI', 'I', (20.0, 350.0)),
            ('CsI:Cs', 'CsI', 'Cs', (20.0 / G_CSI, 350.0 / G_CSI)), ('CsI:I ', 'CsI', 'I', (20.0 / G_CSI, 350.0 / G_CSI)),
            ('LE Si 1-13 ', 'Si', 'Si', (1.0, 13.0)), ('LE Si 2-13 ', 'Si', 'Si', (2.0, 13.0)),
            ('ME Si 10-35', 'Si', 'Si', (10.0, 35.0)), ('ME Si 8-44 ', 'Si', 'Si', (8.0, 44.0)),
            ('HVT:H >=1keVee*', 'BC408', 'H', (1.0, 1e9))]
    for mchi in [10.0, 100.0, 1e3, 1e4, np.inf]:
        store = {}
        print(f'\n  m_chi = {mchi:g} GeV')
        print('    通道             Emax(300km/s)   fFF      eps0(c)    [low, high]         eps_contact(c) [low, high]   (SI 权重 A^2 mu^2 相对值)')
        for name, mat, sp, (lo, hi) in wins:
            Em = emax_keV(sp, v, mchi)
            E = uu * Em
            F2 = helm_F2(sp, E)
            out = []
            for case in 'clh':
                ee = QF[(mat, sp)](E, case) * E
                inw = (ee >= lo) & (ee <= hi)
                out.append((np.mean(F2 * inw), np.mean(inw)))
            mu = M[sp] if np.isinf(mchi) else mchi * M[sp] / (mchi + M[sp])
            wgt = A[sp]**2 * mu**2
            store[name.strip()] = (wgt, float(np.mean(F2)), [o[0] for o in out], [o[1] for o in out])
            print(f'    {name:<15} {float(emax_keV(sp,300,mchi)):9.2f} keV  {np.mean(F2):7.4f}  {out[0][0]:9.3e}  [{out[1][0]:.2e}, {out[2][0]:.2e}]   '
                  f'{out[0][1]:9.3e}  [{out[1][1]:.2e}, {out[2][1]:.2e}]   {wgt:10.3e}')
        muN = 0.938272 if np.isinf(mchi) else mchi * 0.938272 / (mchi + 0.938272)
        for lab, k1, k2 in [('NaI (Na+I)', 'NaI:Na', 'NaI:I'), ('CsI (Cs+I)', 'CsI:Cs', 'CsI:I')]:
            (w1, f1, e1, c1), (w2, f2, e2, c2) = store[k1], store[k2]
            nom = [(w1 * e1[i] + w2 * e2[i]) / (w1 + w2) for i in range(3)]
            act = [(w1 * e1[i] + w2 * e2[i]) / (w1 * f1 + w2 * f2) for i in range(3)]
            con = [(c1[i] + c2[i]) / 2 for i in range(3)]
            print(f'      => {lab}: SI(A^2 mu^2 加权) 过阈比例: 相对名义截面 {nom[0]:.2e} [{nom[1]:.1e},{nom[2]:.1e}]; '
                  f'相对实际发生的散射 {act[0]:.2e} [{act[1]:.1e},{act[2]:.1e}]; 接触截面(两种核等权) {con[0]:.3f} [{con[1]:.3f},{con[2]:.3f}]')
        print(f'      sigma_A0/sigma_chiN = A^2 mu_A^2/mu_N^2 :  Na {store["NaI:Na"][0]/muN**2:.3e}  Si {store["LE Si 1-13"][0]/muN**2:.3e}  '
              f'I {store["NaI:I"][0]/muN**2:.3e}  Cs {store["CsI:Cs"][0]/muN**2:.3e}')
    print('  * HVT 行仅示意单次质子反冲的可见能量量级; 反符合阈(~100 keVee)远高于单次散射.')


def table_shielding():
    hdr('表 G1  自遮蔽: E(x)=E0 exp(-2 Sigma sigma/m_chi) (m_chi>>m_A, 接触截面; 只依赖质量柱密度 Sigma)\n'
        '       给出动能降到 1/e、10%、1% 时的 sigma/m')
    print('    Sigma[g/cm2]   说明                                   sigma/m(1/e)        sigma/m(10%)       sigma/m(1%)   [cm2/GeV]')
    cases = [(0.3, '顶部 HVT 之上 (~1 mm Al 壳)'), (1.0, 'HE 准直器视场内 (HVT+Be 窗+封装)'),
             (2.0, '侧面 HVT 之上 (1 mm Ta 壳)'), (7.0, '外圈 HE 模块侧向 (HVT+Ta 环+壳体)'),
             (15.0, 'HE 正面离轴 (穿过 30 cm 准直器的 Ta 片)'), (25.0, 'NaI 背面 (CsI 18 g/cm2 + 石英 + PMT...)'),
             (60.0, '穿过相邻模块/平台')]
    for S, d in cases:
        vals = [np.log(f) / (2 * S) * CM2G_TO_CM2GEV for f in (np.e, 10.0, 100.0)]
        print(f'    {S:8.1f}      {d:<40s} {vals[0]:12.2e}      {vals[1]:12.2e}      {vals[2]:12.2e}')
    print('  对比: 大气 ~1030 g/cm2 -> sigma/m(1/e) = %.2e cm2/GeV' % (1 / (2 * 1030.0) * CM2G_TO_CM2GEV))


def table_areas():
    hdr('表 G2  几何有效面积 (cm^2): 各向同性通量的平均投影面积 S/4, 及单向"风"与指向轴夹角 theta 的投影面积')
    r, tN, tC = 9.5, 0.35, 4.0
    items = {
        'HE NaI 层 (x18)': lambda th: 18 * (np.pi * r**2 * abs(np.cos(th)) + 2 * r * tN * np.sin(th)),
        'HE CsI 层 (x18)': lambda th: 18 * (np.pi * r**2 * abs(np.cos(th)) + 2 * r * tC * np.sin(th)),
        'HVT 顶部 6 块 (13500 cm2)': lambda th: 13500 * abs(np.cos(th)),
        'HVT 侧面 12 块 (12000 cm2, 方位平均)': lambda th: 12000 * (2 / np.pi) * np.sin(th),
        'LE SCD 384 cm2': lambda th: 384 * abs(np.cos(th)),
        'ME Si-PIN 952 cm2': lambda th: 952 * abs(np.cos(th)),
    }
    iso = {'HE NaI 层 (x18)': 18 * (2 * np.pi * r**2 + 2 * np.pi * r * tN) / 4,
           'HE CsI 层 (x18)': 18 * (2 * np.pi * r**2 + 2 * np.pi * r * tC) / 4,
           'HVT 顶部 6 块 (13500 cm2)': 13500 / 2, 'HVT 侧面 12 块 (12000 cm2, 方位平均)': 12000 / 2,
           'LE SCD 384 cm2': 384 / 2, 'ME Si-PIN 952 cm2': 952 / 2}
    ths = [0, 30, 60, 80, 90]
    print('    子探测器                                 各向同性 S/4    ' + '  '.join(f'th={t:>2d}' for t in ths))
    for k, f in items.items():
        print(f'    {k:<38s} {iso[k]:10.0f}      ' + '  '.join(f'{f(np.radians(t)):6.0f}' for t in ths))
    print('  靶质量: NaI %.2f kg, CsI %.1f kg (18 模块); LE 耗尽层硅 %.2f g; ME 硅 %.0f g; HVT 塑闪 %.1f kg' % (
        18 * np.pi * r**2 * tN * 3.67 / 1e3, 18 * np.pi * r**2 * tC * 4.51 / 1e3, 384 * 0.005 * 2.329, 952 * 0.1 * 2.329,
        (13500 + 12000) * 0.64 * 1.032 / 1e3))
    print('  靶核数: NaI %.3e (Na=I 各); CsI %.3e (Cs=I 各); LE Si %.3e; ME Si %.3e; HVT H %.3e, C %.3e' % (
        18 * np.pi * r**2 * tN * MATERIALS['NaI']['n']['I'], 18 * np.pi * r**2 * tC * MATERIALS['CsI']['n']['I'],
        384 * 0.005 * MATERIALS['Si']['n']['Si'], 952 * 0.1 * MATERIALS['Si']['n']['Si'],
        25500 * 0.64 * 5.23e22, 25500 * 0.64 * 4.74e22))


def table_nuclearite():
    hdr('表 N1  nuclearite 黑体发光公式 (De Rujula & Glashow 1984, 取自 MACRO 2000 式(3)) 的发光效率 eta = dL/dE\n'
        '       dL/dx = sigma/(6 pi^2 sqrt2) * w_max^(5/2) * (m/n)^(3/2) * v^2 ,  dE/dx = sigma rho v^2  (自然单位)')
    eV4_per_gcm3 = 5.609588e32 / (5.067731e4)**3      # 1 g/cm3 -> eV^4
    mats = [('水', 18.015, 3, 1.0, 3.14), ('NaI', 149.894, 2, 3.67, 3.5), ('CsI', 259.81, 2, 4.51, 3.5),
            ('塑闪(CH)', 13.02, 2, 1.032, 3.5), ('硅', 28.0855, 1, 2.329, 1.1)]
    for name, mw, nn, rho, wmax in mats:
        m_over_n = mw * AMU * 1e9 / nn
        eta = wmax**2.5 * m_over_n**1.5 / (6 * np.pi**2 * np.sqrt(2)) / (rho * eV4_per_gcm3)
        dedx = rho * (250.2e5)**2 * 6.2415e11 * 1e-16   # eV/cm for sigma=1e-16
        print(f'    {name:<8s} w_max={wmax:.2f} eV  eta={eta:.2e}   sigma=1e-16 cm2, v=250 km/s: dE/dx={dedx/1e9:.1f} GeV/cm, 可见光 dL/dx={eta*dedx/1e6:.2f} MeV/cm')


def table_charged():
    hdr('表 C1  单位电荷慢重粒子(CHAMP 类)的电子阻止本领估计: NIST PSTAR 1 keV 质子电子阻止本领按 v 线性外推到 v=300 km/s\n'
        '       (等效质子动能 0.470 keV; 外推到 PSTAR 表下限以下, 且未计绝缘体带隙阈效应 -> 上限性质的估计 [I])')
    pstar = {'BC408': 190.1, 'NaI': 41.10, 'CsI': 34.67, 'Si': 105.0}   # MeV cm2/g @ 1 keV [M]
    Ep = 0.5 * 938.272e3 * (300.0 / C_KMS)**2
    for mat, s1 in pstar.items():
        se = s1 * np.sqrt(Ep / 1.0) * MATERIALS[mat]['rho']
        print(f'    {mat:<6s} S_e(1 keV)={s1:6.1f} MeV cm2/g -> v=300 km/s: {se:6.1f} MeV/cm   等效接触截面 sigma_eff = dE/dx/(rho v^2) = {se*1e6/(MATERIALS[mat]["rho"]*(300e5)**2*6.2415e11):.1e} cm2')


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    table_kinematics()
    table_halo()
    table_mean_qe()
    table_mean_qe_ff()
    table_deterministic()
    mc_phoswich()
    mc_slabs()
    table_si_fraction()
    table_shielding()
    table_areas()
    table_nuclearite()
    table_charged()
