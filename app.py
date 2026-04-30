import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import os
import time

st.set_page_config(page_title="HxAim 终极动画模拟器", layout="wide")

# ==========================================
# 0. 直接加载本地中文字体 (无需下载)
# ==========================================
@st.cache_resource
def load_font():
    # 自动尝试加载你可能上传的文件名
    font_candidates = ["msyh.ttf", "msyh.ttc", "simhei.ttf", "SimHei.ttf"]
    font_loaded = False
    
    for font_path in font_candidates:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams['axes.unicode_minus'] = False # 修复负号显示
            font_loaded = True
            break
            
    if not font_loaded:
        st.warning("⚠️ 未检测到字体文件，请确认你上传的字体文件名是 msyh.ttf 或 simhei.ttf。")

load_font()

# ==========================================
# 1. 核心算法复刻 (对齐 C++ 逻辑)
# ==========================================
def smooth_noise(x):
    xi = int(np.floor(x))
    f = x - xi
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    def hash_func(n):
        n = (n << 13) ^ n
        nn = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff
        return 1.0 - (nn / 1073741824.0)
    return hash_func(xi) * (1.0 - u) + hash_func(xi + 1) * u

class KalmanFilter1D:
    def __init__(self):
        self.p = 0.0; self.v = 0.0
        self.P00 = 1.0; self.P01 = 0.0; self.P10 = 0.0; self.P11 = 1.0
        self.q_pos = 0.001; self.q_vel = 8.0; self.r_obs = 2.0
    def set_noise(self, p, v, r):
        self.q_pos = p; self.q_vel = v; self.r_obs = r
    def reset(self, init_pos):
        self.p = init_pos; self.v = 0.0
        self.P00 = 1.0; self.P01 = 0.0; self.P10 = 0.0; self.P11 = 1.0
    def update_and_predict(self, measured_pos, dt, predict_time):
        if dt <= 0.0: return measured_pos
        p_pred = self.p + self.v * dt
        v_pred = self.v
        P00_pred = self.P00 + dt * (self.P10 + self.P01) + (dt * dt) * self.P11 + self.q_pos
        P01_pred = self.P01 + dt * self.P11
        P10_pred = self.P10 + dt * self.P11
        P11_pred = self.P11 + self.q_vel
        S = P00_pred + self.r_obs
        K0 = P00_pred / S; K1 = P10_pred / S
        y = measured_pos - p_pred
        self.p = p_pred + K0 * y
        self.v = v_pred + K1 * y
        self.P00 = (1.0 - K0) * P00_pred; self.P01 = (1.0 - K0) * P01_pred
        self.P10 = -K1 * P00_pred + P10_pred; self.P11 = -K1 * P01_pred + P11_pred
        return self.p + self.v * predict_time

class IncrementalPID:
    def __init__(self):
        self.kp = 0.0; self.ki = 0.0; self.kd = 0.0; self.max_out = 0.0
        self.error_prev = 0.0; self.error_prev2 = 0.0
    def set_params(self, p, i, d, max_out):
        self.kp = p; self.ki = i; self.kd = d; self.max_out = max_out
    def compute(self, current, target):
        error = target - current
        p_term = self.kp * (error - self.error_prev)
        i_term = self.ki * error
        d_term = self.kd * (error - 2.0 * self.error_prev + self.error_prev2)
        delta = p_term + i_term + d_term
        self.error_prev2 = self.error_prev
        self.error_prev = error
        if self.max_out > 0: delta = max(-self.max_out, min(self.max_out, delta))
        return delta

def calc_dynamic_param(distance, p_min, p_max, p_factor, max_dist, reverse):
    if max_dist <= 0: return p_max
    norm_dist = min(1.0, distance / max_dist)
    base_val = norm_dist if reverse else (1.0 - norm_dist)
    factor_val = base_val ** p_factor if p_factor not in [1.0, 2.0] else (base_val*base_val if p_factor==2.0 else base_val)
    val = p_min + (p_max - p_min) * factor_val
    return max(p_min, min(p_max, val))

# ==========================================
# 2. UI 面板 (严格对齐 C++ Config)
# ==========================================
st.sidebar.title("🎛️ HxAim 参数台")

# --- 布尔类型 (Boolean) ---
with st.sidebar.expander("🟢 布尔类型开关 (Boolean)", expanded=False):
    c1, c2 = st.columns(2)
    AIM_ENABLED = c1.checkbox("自瞄开关", True)
    TRIGGER_ENABLED = c2.checkbox("扳机开关", True)
    DYNAMIC_DELAY_ENABLED = c1.checkbox("智能延迟", False)
    TIME_DYNAMIC_ENABLED = c2.checkbox("时间动态", False)
    DYNAMIC_KP_REVERSE = c1.checkbox("动态反转", False)
    AUTO_MELEE_ENABLED = c2.checkbox("自动按键", False)
    MOUSE_MASK_ENABLED = c1.checkbox("鼠标屏蔽", False)
    PREDICTION_ENABLED = c2.checkbox("预测开关", True)
    BOW_MODE = c1.checkbox("弓箭模式", False)
    PLAYBACK_PATTERN_ENABLED = c2.checkbox("回放弹道", False)

# --- 小数类型 (Double) ---
with st.sidebar.expander("🔵 小数类型参数 (Double)", expanded=True):
    st.markdown("**🎯 PID 参数矩阵**")
    pc1, pc2 = st.columns(2)
    # PID X
    KPX_MIN = pc1.number_input("KPX_MIN (X阻尼-小)", 0.0, 1.0, 0.05, 0.01)
    KPX_MAX = pc1.number_input("KPX_MAX (X阻尼-大)", 0.0, 1.0, 0.15, 0.01)
    KIX_MIN = pc1.number_input("KIX_MIN (X动力-小)", 0.0, 0.1, 0.005, 0.001)
    KIX_MAX = pc1.number_input("KIX_MAX (X动力-大)", 0.0, 0.1, 0.020, 0.001)
    KDX_MIN = pc1.number_input("KDX_MIN", 0.0, 0.5, 0.0, 0.01)
    KDX_MAX = pc1.number_input("KDX_MAX", 0.0, 0.5, 0.0, 0.01)
    # PID Y
    KPY_MIN = pc2.number_input("KPY_MIN (Y阻尼-小)", 0.0, 1.0, 0.05, 0.01)
    KPY_MAX = pc2.number_input("KPY_MAX (Y阻尼-大)", 0.0, 1.0, 0.15, 0.01)
    KIY_MIN = pc2.number_input("KIY_MIN (Y动力-小)", 0.0, 0.1, 0.005, 0.001)
    KIY_MAX = pc2.number_input("KIY_MAX (Y动力-大)", 0.0, 0.1, 0.020, 0.001)
    KDY_MIN = pc2.number_input("KDY_MIN", 0.0, 0.5, 0.0, 0.01)
    KDY_MAX = pc2.number_input("KDY_MAX", 0.0, 0.5, 0.0, 0.01)
    
    st.markdown("---")
    IXMAX = st.slider("IXMAX (最大输出X)", 0.0, 100.0, 50.0)
    IYMAX = st.slider("IYMAX (最大输出Y)", 0.0, 100.0, 50.0)
    c1, c2 = st.columns(2)
    P_FACTOR = c1.number_input("P_FACTOR_X", 0.1, 5.0, 1.0)
    P_FACTOR_Y = c2.number_input("P_FACTOR_Y", 0.1, 5.0, 1.0)
    DEADBAND = c1.number_input("死区X (像素)", 0.0, 20.0, 2.0)
    DEADBAND_Y = c2.number_input("死区Y (像素)", 0.0, 20.0, 2.0)
    PREDICT_TIME_MS = c1.number_input("预测时间X (ms)", 0.0, 100.0, 20.0)
    PREDICT_TIME_MS_Y = c2.number_input("预测时间Y (ms)", 0.0, 100.0, 10.0)
    
    st.markdown("**🧬 拟人化与噪声**")
    FINAL_RANGE = st.slider("最终直线范围", 0.0, 100.0, 25.0)
    MAX_CURVE_PIXELS = st.slider("最大曲线偏移", 0.0, 20.0, 6.0)
    CURVE_FREQUENCY = st.number_input("曲线抖动频率", 0.0001, 0.05, 0.003, format="%.4f")
    KF_Q_POS_X = st.number_input("KF_Q_POS_X", value=0.001, format="%.4f")
    KF_Q_VEL_X = st.number_input("KF_Q_VEL_X", value=8.0)
    CAMERA_SENS = st.slider("镜头补偿 Sens", 0.1, 5.0, 1.0)

# --- 整数类型 (Integer) ---
with st.sidebar.expander("🟠 整数类型参数 (Integer)", expanded=False):
    TARGET_RANGE = st.slider("瞄准范围", 10, 500, 200)
    X_OFFSET = st.number_input("X偏移", -100, 100, 0)
    TRIGGER_PERCENT = st.slider("触发比例 (%)", 1, 100, 30)
    PREDICT_ENABLE_DISTANCE_X = st.slider("预判启动距X", 0, 500, 150)
    PREDICT_ENABLE_DISTANCE_Y = st.slider("预判启动距Y", 0, 500, 100)
    PREDICT_MAX_OFFSET_X = st.slider("最大预判偏移X", 0, 100, 40)
    PREDICT_MAX_OFFSET_Y = st.slider("最大预判偏移Y", 0, 100, 20)
    AIM_DELAY = st.number_input("瞄准延迟 (ms)", 0, 500, 0)
    TRIGGER_PRESS_DELAY = st.number_input("扳机延迟 (ms)", 0, 500, 0)

# --- 环境模拟 (辅助) ---
st.sidebar.markdown("---")
st.sidebar.markdown("🏃 **移动靶环境模拟**")
target_vel_x = st.sidebar.slider("目标移速 X", -20.0, 20.0, -5.0)
target_vel_y = st.sidebar.slider("目标移速 Y", -20.0, 20.0, -1.0)
detect_w = 400.0 # 假定识别框宽

# ==========================================
# 3. 轨迹数据生成引擎
# ==========================================
def generate_simulation_data():
    dt_ms = 16.0; dt_sec = dt_ms / 1000.0
    total_frames = 120
    
    px, py = IncrementalPID(), IncrementalPID()
    kx, ky = KalmanFilter1D(), KalmanFilter1D()
    kx.set_noise(KF_Q_POS_X, KF_Q_VEL_X, 2.0)
    
    cx, cy = 0.0, 0.0
    tx_start, ty_start = 150.0, 80.0
    kx.reset(tx_start); ky.reset(ty_start)
    
    data = {"xhair_x":[], "xhair_y":[], "target_x":[], "target_y":[], "pred_x":[], "pred_y":[], "fire_x":[], "fire_y":[]}
    
    for i in range(total_frames):
        rtx = tx_start + target_vel_x * i
        rty = ty_start + target_vel_y * i
        etx, ety = rtx + X_OFFSET, rty
        data["target_x"].append(rtx); data["target_y"].append(rty)
        
        if PREDICTION_ENABLED:
            px_pred = kx.update_and_predict(etx, dt_sec, PREDICT_TIME_MS / 1000.0)
            py_pred = ky.update_and_predict(ety, dt_sec, PREDICT_TIME_MS_Y / 1000.0)
            
            if abs(etx - cx) <= PREDICT_ENABLE_DISTANCE_X:
                offset_x = np.clip(px_pred - etx, -PREDICT_MAX_OFFSET_X, PREDICT_MAX_OFFSET_X)
                fx = etx + offset_x
            else: fx = etx; kx.reset(etx)
            
            if abs(ety - cy) <= PREDICT_ENABLE_DISTANCE_Y:
                offset_y = np.clip(py_pred - ety, -PREDICT_MAX_OFFSET_Y, PREDICT_MAX_OFFSET_Y)
                fy = ety + offset_y
            else: fy = ety; ky.reset(ety)
        else:
            fx, fy = etx, ety
            
        data["pred_x"].append(fx); data["pred_y"].append(fy)
        
        err_x, err_y = fx - cx, fy - cy
        dist = np.sqrt(err_x**2 + err_y**2)
        
        if abs(err_x) <= DEADBAND and abs(err_y) <= DEADBAND_Y:
            px.compute(cx, cx); py.compute(cy, cy)
            dx, dy = 0.0, 0.0
        else:
            c_kpx = calc_dynamic_param(dist, KPX_MIN, KPX_MAX, P_FACTOR, detect_w, DYNAMIC_KP_REVERSE)
            c_kix = calc_dynamic_param(dist, KIX_MIN, KIX_MAX, P_FACTOR, detect_w, DYNAMIC_KP_REVERSE)
            c_kpy = calc_dynamic_param(dist, KPY_MIN, KPY_MAX, P_FACTOR_Y, detect_w, DYNAMIC_KP_REVERSE)
            c_kiy = calc_dynamic_param(dist, KIY_MIN, KIY_MAX, P_FACTOR_Y, detect_w, DYNAMIC_KP_REVERSE)
            
            px.set_params(c_kpx, c_kix, KDX_MAX, IXMAX)
            py.set_params(c_kpy, c_kiy, KDY_MAX, IYMAX)
            
            tx_calc = cx if abs(err_x) <= DEADBAND else fx
            ty_calc = cy if abs(err_y) <= DEADBAND_Y else fy
            dx = px.compute(cx, tx_calc); dy = py.compute(cy, ty_calc)
            
            len_d = np.sqrt(dx**2 + dy**2)
            if len_d > 0.1 and dist > FINAL_RANGE:
                prx, pry = -dy/len_d, dx/len_d
                nv = 0.0; amp = 1.0; freq = CURVE_FREQUENCY
                for _ in range(3):
                    nv += smooth_noise(i * dt_ms * freq) * amp
                    amp *= 0.5; freq *= 2.0
                fd = min(1.0, (dist - FINAL_RANGE)/60.0)
                off = nv * MAX_CURVE_PIXELS * min(1.0, len_d/12.0) * fd
                dx += prx * off; dy += pry * off
                
        cx += dx * CAMERA_SENS; cy += dy * CAMERA_SENS
        data["xhair_x"].append(cx); data["xhair_y"].append(cy)
        
        if TRIGGER_ENABLED:
            tr = TRIGGER_PERCENT * 0.01
            if (rtx - 30*tr <= cx <= rtx + 30*tr) and (rty - 50*tr <= cy <= rty + 50*tr):
                data["fire_x"].append(cx); data["fire_y"].append(cy)
    return data

sim_data = generate_simulation_data()

# ==========================================
# 4. 图表动画渲染引擎
# ==========================================
st.title("🎬 动态弹道模拟中心")
c_play, c_info = st.columns([1, 4])
play_btn = c_play.button("▶️ 播放动画", use_container_width=True)

plot_placeholder = st.empty()

def draw_frame(frame_idx):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=100)
    fig.patch.set_facecolor('#1e1e2e'); ax.set_facecolor('#282a36')
    ax.tick_params(colors='#f8f8f2')
    for spine in ax.spines.values(): spine.set_edgecolor('#6272a4')
    
    tx = sim_data["target_x"][:frame_idx]; ty = sim_data["target_y"][:frame_idx]
    cx = sim_data["xhair_x"][:frame_idx]; cy = sim_data["xhair_y"][:frame_idx]
    
    ax.plot(tx, ty, '--', color='#6272a4', label="目标真实轨迹")
    ax.plot(cx, cy, color='#bd93f9', linewidth=2, label="准星拉枪轨迹")
    ax.scatter(cx, cy, color='#ff79c6', s=5, zorder=3)
    
    if len(tx) > 0:
        ltx, lty = tx[-1], ty[-1]
        lcx, lcy = cx[-1], cy[-1]
        
        ax.add_patch(patches.Rectangle((ltx-30, lty-50), 60, 100, ec='white', fc='none', alpha=0.3))
        tw, th = 60 * (TRIGGER_PERCENT/100), 100 * (TRIGGER_PERCENT/100)
        ax.add_patch(patches.Rectangle((ltx-tw/2, lty-th/2), tw, th, lw=1.5, ec='#ff5555', fc='none', label="扳机区"))
        
        if PREDICTION_ENABLED:
            lpx, lpy = sim_data["pred_x"][frame_idx-1], sim_data["pred_y"][frame_idx-1]
            ax.scatter(lpx, lpy, color='#f1fa8c', marker='X', s=80, label="预测瞄准点", zorder=4)
        
        ax.scatter(lcx, lcy, color='#50fa7b', marker='o', s=50, label="当前准星")
        ax.add_patch(patches.Rectangle((lcx-DEADBAND, lcy-DEADBAND_Y), DEADBAND*2, DEADBAND_Y*2, 
                                     ec='#50fa7b', fc='rgba(80,250,123,0.2)', label="当前死区"))
                                     
    fx = [x for x in sim_data["fire_x"] if x in cx]
    fy = [y for y in sim_data["fire_y"] if y in cy]
    if fx: ax.scatter(fx, fy, color='#ff5555', marker='*', s=60, zorder=5, label="开火 Fired")

    ax.set_title(f"逐帧计算演示 (当前帧: {frame_idx}/120) | 参数解算中...", color='#f8f8f2', fontsize=14)
    ax.grid(color='#6272a4', ls=':', alpha=0.3)
    ax.legend(facecolor='#282a36', edgecolor='#6272a4', labelcolor='#f8f8f2', loc='upper left', bbox_to_anchor=(1.02, 1))
    
    ax.set_xlim(-50, 250); ax.set_ylim(-20, 120)
    plt.tight_layout()
    return fig

# --- 动画播放控制 ---
if play_btn:
    for i in range(1, 121, 3): 
        fig = draw_frame(i)
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.01)
    fig = draw_frame(120)
    plot_placeholder.pyplot(fig)
    plt.close(fig)
else:
    fig = draw_frame(120)
    plot_placeholder.pyplot(fig)
    plt.close(fig)
