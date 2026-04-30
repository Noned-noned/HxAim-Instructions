import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import os
import time
import json

st.set_page_config(page_title="HxAim 终极动画模拟器", layout="wide")

# ==========================================
# 0. 直接加载本地中文字体
# ==========================================
@st.cache_resource
def load_font():
    font_candidates = ["msyh.ttf", "msyh.ttc", "simhei.ttf", "SimHei.ttf"]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams['axes.unicode_minus'] = False 
            break

load_font()

# ==========================================
# 1. 核心算法复刻
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
    return val

# ==========================================
# 2. UI 面板与数据持久化逻辑 (全组件绑定 Key)
# ==========================================
st.sidebar.title("🎛️ HxAim 控制台")

# --- 导入配置模块 (必须放在最前面) ---
uploaded_file = st.sidebar.file_uploader("📂 导入 JSON 配置", type=['json'], help="上传你之前保存的配置文件以恢复所有参数")
if uploaded_file is not None:
    if "loaded_file" not in st.session_state or st.session_state.loaded_file != uploaded_file.name:
        try:
            config_data = json.load(uploaded_file)
            for k, v in config_data.items():
                st.session_state[k] = v
            st.session_state.loaded_file = uploaded_file.name
            st.sidebar.success("✅ 配置导入成功！")
            time.sleep(0.5)
            st.rerun() # 重新渲染页面应用新参数
        except Exception as e:
            st.sidebar.error(f"读取配置失败: {e}")

st.sidebar.markdown("---")

with st.sidebar.expander("🖥️ 硬件与系统环境", expanded=True):
    target_fps = st.select_slider("游戏帧率 (FPS)", options=[30, 60, 90, 120, 144, 240, 360, 500, 1000], value=60, key="target_fps")
    sens_multiplier = st.number_input("DPI/游戏内灵敏度倍率", value=1.0, step=0.1, key="sens_multiplier")

with st.sidebar.expander("🟢 布尔类型开关 (Boolean)", expanded=False):
    c1, c2 = st.columns(2)
    AIM_ENABLED = c1.checkbox("自瞄开关", True, key="AIM_ENABLED")
    TRIGGER_ENABLED = c2.checkbox("扳机开关", True, key="TRIGGER_ENABLED")
    DYNAMIC_DELAY_ENABLED = c1.checkbox("智能延迟", False, key="DYNAMIC_DELAY_ENABLED")
    TIME_DYNAMIC_ENABLED = c2.checkbox("时间动态", False, key="TIME_DYNAMIC_ENABLED")
    DYNAMIC_KP_REVERSE = c1.checkbox("动态反转", False, key="DYNAMIC_KP_REVERSE")
    AUTO_MELEE_ENABLED = c2.checkbox("自动按键", False, key="AUTO_MELEE_ENABLED")
    MOUSE_MASK_ENABLED = c1.checkbox("鼠标屏蔽", False, key="MOUSE_MASK_ENABLED")
    PREDICTION_ENABLED = c2.checkbox("预测开关", True, key="PREDICTION_ENABLED")
    BOW_MODE = c1.checkbox("弓箭模式", False, key="BOW_MODE")
    PLAYBACK_PATTERN_ENABLED = c2.checkbox("回放弹道", False, key="PLAYBACK_PATTERN_ENABLED")

with st.sidebar.expander("🔵 小数类型参数 (Double)", expanded=False):
    st.markdown("**🎯 PID 参数矩阵**")
    pc1, pc2 = st.columns(2)
    KPX_MIN = pc1.number_input("KPX_MIN (X阻尼-小)", value=0.05, step=0.01, key="KPX_MIN")
    KPX_MAX = pc1.number_input("KPX_MAX (X阻尼-大)", value=0.15, step=0.01, key="KPX_MAX")
    KIX_MIN = pc1.number_input("KIX_MIN (X动力-小)", value=0.005, step=0.001, key="KIX_MIN")
    KIX_MAX = pc1.number_input("KIX_MAX (X动力-大)", value=0.020, step=0.001, key="KIX_MAX")
    KDX_MIN = pc1.number_input("KDX_MIN", value=0.0, step=0.01, key="KDX_MIN")
    KDX_MAX = pc1.number_input("KDX_MAX", value=0.0, step=0.01, key="KDX_MAX")
    
    KPY_MIN = pc2.number_input("KPY_MIN (Y阻尼-小)", value=0.05, step=0.01, key="KPY_MIN")
    KPY_MAX = pc2.number_input("KPY_MAX (Y阻尼-大)", value=0.15, step=0.01, key="KPY_MAX")
    KIY_MIN = pc2.number_input("KIY_MIN (Y动力-小)", value=0.005, step=0.001, key="KIY_MIN")
    KIY_MAX = pc2.number_input("KIY_MAX (Y动力-大)", value=0.020, step=0.001, key="KIY_MAX")
    KDY_MIN = pc2.number_input("KDY_MIN", value=0.0, step=0.01, key="KDY_MIN")
    KDY_MAX = pc2.number_input("KDY_MAX", value=0.0, step=0.01, key="KDY_MAX")
    
    st.markdown("---")
    IXMAX = st.number_input("IXMAX (最大输出X)", value=50.0, key="IXMAX")
    IYMAX = st.number_input("IYMAX (最大输出Y)", value=50.0, key="IYMAX")
    c1, c2 = st.columns(2)
    P_FACTOR = c1.number_input("P_FACTOR_X", value=1.0, key="P_FACTOR")
    P_FACTOR_Y = c2.number_input("P_FACTOR_Y", value=1.0, key="P_FACTOR_Y")
    DEADBAND = c1.number_input("死区X (像素)", value=2.0, key="DEADBAND")
    DEADBAND_Y = c2.number_input("死区Y (像素)", value=2.0, key="DEADBAND_Y")
    PREDICT_TIME_MS = c1.number_input("预测时间X (ms)", value=20.0, key="PREDICT_TIME_MS")
    PREDICT_TIME_MS_Y = c2.number_input("预测时间Y (ms)", value=10.0, key="PREDICT_TIME_MS_Y")
    
    st.markdown("**🔮 卡尔曼预测噪声 (Kalman)**")
    kc1, kc2 = st.columns(2)
    KF_Q_POS_X = kc1.number_input("位置噪声X (Q_POS)", value=0.001, format="%.4f", step=0.001, key="KF_Q_POS_X")
    KF_Q_VEL_X = kc1.number_input("速度噪声X (Q_VEL)", value=8.0, step=0.1, key="KF_Q_VEL_X")
    KF_R_OBS_X = kc1.number_input("观测噪声X (R_OBS)", value=2.0, step=0.1, key="KF_R_OBS_X")
    
    KF_Q_POS_Y = kc2.number_input("位置噪声Y", value=0.001, format="%.4f", step=0.001, key="KF_Q_POS_Y")
    KF_Q_VEL_Y = kc2.number_input("速度噪声Y", value=8.0, step=0.1, key="KF_Q_VEL_Y")
    KF_R_OBS_Y = kc2.number_input("观测噪声Y", value=2.0, step=0.1, key="KF_R_OBS_Y")
    
    st.markdown("**🧬 拟人化与杂项**")
    FINAL_RANGE = st.number_input("最终直线范围", value=25.0, key="FINAL_RANGE")
    MAX_CURVE_PIXELS = st.number_input("最大曲线偏移", value=6.0, key="MAX_CURVE_PIXELS")
    CURVE_FREQUENCY = st.number_input("曲线抖动频率", value=0.003, format="%.4f", step=0.001, key="CURVE_FREQUENCY")
    CAMERA_SENS = st.number_input("镜头补偿 Sens", value=1.0, step=0.1, key="CAMERA_SENS")

with st.sidebar.expander("🟠 整数类型参数 (Integer)", expanded=False):
    TARGET_RANGE = st.number_input("瞄准范围", value=200, key="TARGET_RANGE")
    X_OFFSET = st.number_input("X偏移", value=0, key="X_OFFSET")
    TRIGGER_PERCENT = st.number_input("触发比例 (%)", value=30, key="TRIGGER_PERCENT")
    PREDICT_ENABLE_DISTANCE_X = st.number_input("预判启动距X", value=150, key="PREDICT_ENABLE_DISTANCE_X")
    PREDICT_ENABLE_DISTANCE_Y = st.number_input("预判启动距Y", value=100, key="PREDICT_ENABLE_DISTANCE_Y")
    PREDICT_MAX_OFFSET_X = st.number_input("最大预判偏移X", value=40, key="PREDICT_MAX_OFFSET_X")
    PREDICT_MAX_OFFSET_Y = st.number_input("最大预判偏移Y", value=20, key="PREDICT_MAX_OFFSET_Y")
    AIM_DELAY = st.number_input("瞄准延迟 (ms)", value=0, key="AIM_DELAY")
    TRIGGER_PRESS_DELAY = st.number_input("扳机延迟 (ms)", value=0, key="TRIGGER_PRESS_DELAY")

st.sidebar.markdown("---")
st.sidebar.markdown("🏃 **移动靶测试参数**")
target_vel_x_sec = st.sidebar.number_input("目标移速 X (像素/秒)", value=-60.0, step=10.0, key="target_vel_x_sec")
target_vel_y_sec = st.sidebar.number_input("目标移速 Y (像素/秒)", value=-10.0, step=10.0, key="target_vel_y_sec")
detect_w = 400.0 

# --- 导出配置模块 (收集上方所有被 Key 绑定的变量) ---
PARAM_KEYS = [
    "target_fps", "sens_multiplier",
    "AIM_ENABLED", "TRIGGER_ENABLED", "DYNAMIC_DELAY_ENABLED", "TIME_DYNAMIC_ENABLED", "DYNAMIC_KP_REVERSE", "AUTO_MELEE_ENABLED", "MOUSE_MASK_ENABLED", "PREDICTION_ENABLED", "BOW_MODE", "PLAYBACK_PATTERN_ENABLED",
    "KPX_MIN", "KPX_MAX", "KIX_MIN", "KIX_MAX", "KDX_MIN", "KDX_MAX",
    "KPY_MIN", "KPY_MAX", "KIY_MIN", "KIY_MAX", "KDY_MIN", "KDY_MAX",
    "IXMAX", "IYMAX", "P_FACTOR", "P_FACTOR_Y", "DEADBAND", "DEADBAND_Y", "PREDICT_TIME_MS", "PREDICT_TIME_MS_Y",
    "KF_Q_POS_X", "KF_Q_VEL_X", "KF_R_OBS_X", "KF_Q_POS_Y", "KF_Q_VEL_Y", "KF_R_OBS_Y",
    "FINAL_RANGE", "MAX_CURVE_PIXELS", "CURVE_FREQUENCY", "CAMERA_SENS",
    "TARGET_RANGE", "X_OFFSET", "TRIGGER_PERCENT", "PREDICT_ENABLE_DISTANCE_X", "PREDICT_ENABLE_DISTANCE_Y", "PREDICT_MAX_OFFSET_X", "PREDICT_MAX_OFFSET_Y", "AIM_DELAY", "TRIGGER_PRESS_DELAY",
    "target_vel_x_sec", "target_vel_y_sec"
]

current_config = {k: st.session_state[k] for k in PARAM_KEYS if k in st.session_state}
json_str = json.dumps(current_config, indent=4)

st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 将当前参数导出为 JSON",
    data=json_str,
    file_name="hxaim_config.json",
    mime="application/json",
    use_container_width=True
)

# ==========================================
# 3. 轨迹数据生成引擎 
# ==========================================
def generate_simulation_data():
    dt_ms = 1000.0 / target_fps
    dt_sec = dt_ms / 1000.0
    sim_duration_sec = 2.0
    total_frames = int(sim_duration_sec * target_fps)
    
    px, py = IncrementalPID(), IncrementalPID()
    kx, ky = KalmanFilter1D(), KalmanFilter1D()
    
    kx.set_noise(KF_Q_POS_X, KF_Q_VEL_X, KF_R_OBS_X)
    ky.set_noise(KF_Q_POS_Y, KF_Q_VEL_Y, KF_R_OBS_Y)
    
    cx, cy = 0.0, 0.0
    tx_start, ty_start = 150.0, 80.0
    kx.reset(tx_start); ky.reset(ty_start)
    
    data = {"xhair_x":[], "xhair_y":[], "target_x":[], "target_y":[], "pred_x":[], "pred_y":[], "is_fire":[]}
    
    for i in range(total_frames):
        current_time_sec = i * dt_sec
        rtx = tx_start + target_vel_x_sec * current_time_sec
        rty = ty_start + target_vel_y_sec * current_time_sec
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
                    nv += smooth_noise(current_time_sec * 1000.0 * freq) * amp
                    amp *= 0.5; freq *= 2.0
                fd = min(1.0, (dist - FINAL_RANGE)/60.0) if FINAL_RANGE >= 0 else 1.0
                off = nv * MAX_CURVE_PIXELS * min(1.0, len_d/12.0) * fd
                dx += prx * off; dy += pry * off
        
        cx += dx * CAMERA_SENS * sens_multiplier
        cy += dy * CAMERA_SENS * sens_multiplier
        data["xhair_x"].append(cx); data["xhair_y"].append(cy)
        
        is_firing = False
        if TRIGGER_ENABLED:
            tr = TRIGGER_PERCENT * 0.01
            if (rtx - 30*tr <= cx <= rtx + 30*tr) and (rty - 50*tr <= cy <= rty + 50*tr):
                is_firing = True
        data["is_fire"].append(is_firing)
        
    return data, total_frames, sim_duration_sec

sim_data, t_frames, duration_sec = generate_simulation_data()

# ==========================================
# 4. 图表动画渲染引擎
# ==========================================
st.title(f"🎬 动态弹道模拟中心 (模拟时长 {duration_sec}秒 | {t_frames}帧)")
c_play, c_info = st.columns([1, 4])
play_btn = c_play.button("▶️ 重新播放轨迹动画", use_container_width=True)

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
                                     ec='#50fa7b', fc='#50fa7b33', label="当前死区"))
                                     
    fx = [cx[j] for j in range(frame_idx) if sim_data["is_fire"][j]]
    fy = [cy[j] for j in range(frame_idx) if sim_data["is_fire"][j]]
    if fx: ax.scatter(fx, fy, color='#ff5555', marker='*', s=60, zorder=5, label="开火 Fired")

    ax.set_title(f"逐帧计算演示 (当前帧: {frame_idx}/{t_frames}) | {target_fps} FPS | 灵敏度: {sens_multiplier}x", color='#f8f8f2', fontsize=14)
    ax.grid(color='#6272a4', ls=':', alpha=0.3)
    ax.legend(facecolor='#282a36', edgecolor='#6272a4', labelcolor='#f8f8f2', loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    return fig

# --- 动画自适应播放控制 ---
if play_btn:
    step = max(1, t_frames // 40)
    for i in range(1, t_frames + 1, step): 
        fig = draw_frame(i)
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.01)
    if i != t_frames:
        fig = draw_frame(t_frames)
        plot_placeholder.pyplot(fig)
        plt.close(fig)
else:
    fig = draw_frame(t_frames)
    plot_placeholder.pyplot(fig)
    plt.close(fig)
