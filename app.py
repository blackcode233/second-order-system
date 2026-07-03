import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
import streamlit as st
import numpy as np
import control as ct

# 设置网页配置
st.set_page_config(page_title="二阶控制系统参数分析专家系统", layout="wide")
st.title("🎓 二阶控制系统参数分析交互式教学软件")
st.markdown("---")

# ================= 侧边栏：参数输入与调节 =================
st.sidebar.header("🎛️ 系统参数配置")

# 1. 输入模式选择
input_mode = st.sidebar.selectbox(
    "选择系统输入模式",
    ["标准形式 (ωn, ζ)", "一般多项式形式 (num, den)", "主导极点形式"]
)

# 初始化系统变量
num, den = [1], [1, 1, 1]
wn, zeta = 1.0, 0.707

if input_mode == "标准形式 (ωn, ζ)":
    wn = st.sidebar.slider("自然震荡频率 (ωn)", 0.1, 10.0, 2.0, 0.1)
    zeta = st.sidebar.slider("阻尼比 (ζ)", 0.0, 2.0, 0.5, 0.05)
    num = [wn**2]
    den = [1, 2 * zeta * wn, wn**2]

elif input_mode == "一般多项式形式 (num, den)":
    a0 = st.sidebar.number_input("分子常数项 (b0)", value=1.0)
    c1 = st.sidebar.number_input("分母s项系数 (2ζωn)", value=1.0)
    c0 = st.sidebar.number_input("分母常数项 (ωn^2)", value=1.0)
    num = [a0]
    den = [1, c1, c0]
    # 反算 wn 和 zeta
    wn = np.sqrt(c0) if c0 > 0 else 0.1
    zeta = c1 / (2 * wn)

elif input_mode == "主导极点形式":
    real_part = st.sidebar.slider("极点实部 (影响衰减快慢)", -5.0, -0.1, -1.0, 0.1)
    imag_part = st.sidebar.slider("极点虚部 (影响震荡频率)", 0.0, 5.0, 2.0, 0.1)
    # (s - p1)(s - p2) = s^2 - 2*real*s + (real^2 + imag^2)
    num = [real_part**2 + imag_part**2]
    den = [1, -2 * real_part, real_part**2 + imag_part**2]
    wn = np.sqrt(real_part**2 + imag_part**2)
    zeta = -real_part / wn

# 创建系统传递函数
sys = ct.tf(num, den)

# ================= 主界面布局 =================
col1, col2 = st.columns([1, 2])

# ------ 左侧：性能指标计算 ------
with col1:
    st.subheader("📊 系统特征与性能指标")
    
    # 特征参数
    st.metric("自然固有频率 $ω_n$", f"{wn:.3f} rad/s")
    st.metric("阻尼比 $ζ$", f"{zeta:.3f}")
    
    # 判定阻尼状态
    if zeta == 0: status = "无阻尼 (Undamped)"
    elif 0 < zeta < 1: status = "欠阻尼 (Underdamped)"
    elif zeta == 1: status = "临界阻尼 (Critically Damped)"
    else: status = "过阻尼 (Overdamped)"
    st.info(f"**当前状态**：{status}")
    
    st.markdown("### ⏱️ 时域响应指标 (阶跃输入)")
    # 计算时域指标 (针对标准的欠阻尼/过阻尼近似或精确计算)
    if 0 < zeta < 1:
        wd = wn * np.sqrt(1 - zeta**2)
        tr = (np.pi - np.arccos(zeta)) / wd
        tp = np.pi / wd
        mp = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100
        ts = 3.5 / (zeta * wn) # 5% 准则
        
        st.write(f"**上升时间 ($t_r$):** {tr:.3f} s")
        st.write(f"**峰值时间 ($t_p$):** {tp:.3f} s")
        st.write(f"**超调量 ($M_p$):** {mp:.1f} %")
        st.write(f"**调节时间 ($t_s$, 5%):** {ts:.3f} s")
    else:
        st.write("**上升时间 ($t_r$):** 见图（无震荡系统）")
        st.write("**峰值时间 ($t_p$):** 无峰值")
        st.write("**超调量 ($M_p$):** 0 %")
        # 粗略估算过阻尼调节时间
        poles = ct.poles(sys)
        ts = 3.0 / abs(max(poles.real))
        st.write(f"**调节时间 ($t_s$):** ~{ts:.3f} s")

# ------ 右侧：图形显示（响应曲线与零极点图） ------
with col2:
    st.subheader("📈 动态仿真曲线")
    
    response_type = st.selectbox("选择响应类型", ["单位阶跃响应", "冲激响应", "斜坡响应"])
    
    t = np.linspace(0, 10, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # 1. 绘制时域响应
    if response_type == "单位阶跃响应":
        t, y = ct.step_response(sys, t)
        ax1.plot(t, y, 'b', lw=2, label='Output')
        ax1.axhline(1.0, color='r', linestyle='--', alpha=0.6, label='Target')
    elif response_type == "冲激响应":
        t, y = ct.impulse_response(sys, t)
        ax1.plot(t, y, 'g', lw=2)
    elif response_type == "斜坡响应":
        t, y = ct.step_response(sys, t) # 积分得到斜坡
        y_ramp = np.cumsum(y) * (t[1]-t[0])
        ax1.plot(t, y_ramp, 'm', lw=2, label='Output')
        ax1.plot(t, t, 'r--', alpha=0.6, label='Input')
        
    ax1.set_title(f"{response_type} 曲线")
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("响应幅值")
    ax1.grid(True, alpha=0.4)
    if response_type in ["单位阶跃响应", "斜坡响应"]:
        ax1.legend()
        
    # 2. 绘制零极点图
    poles = ct.poles(sys)
    zeros = ct.zeros(sys)
    
    ax2.axhline(0, color='black', lw=1, alpha=0.5)
    ax2.axvline(0, color='black', lw=1, alpha=0.5)
    ax2.scatter(poles.real, poles.imag, marker='x', color='red', s=100, lw=3, label='Poles')
    if len(zeros) > 0:
        ax2.scatter(zeros.real, zeros.imag, marker='o', color='blue', s=100, label='Zeros')
        
    ax2.set_title("S 平面零极点分布图")
    ax2.set_xlabel("实部 (Real)")
    ax2.set_ylabel("虚部 (Imag)")
    ax2.grid(True, alpha=0.4)
    ax2.set_xlim([-6, 1])
    ax2.set_ylim([-5, 5])
    ax2.legend()
    
    st.pyplot(fig)