import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="HxAim 终极参数模拟中心", layout="wide")

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
    def __init__(self, q_pos, q_vel, r_obs):
        self.p = 0.0; self.v = 0.0
        self.P00 = 1.0; self.P01 = 0.0; self.P10 = 0.0; self.P11 = 1.0
        self.q_pos = q_pos; self.q_vel = q_vel; self.r_obs = r_obs

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
        K0 = P00_pred / S
        K1 = P10_pred / S
        y = measured_pos - p_pred

        self.p = p_pred + K0 * y
        self.v = v_pred + K1 * y
        self.P00 = (1.0 - K0) * P00_pred
        self.P01 = (1.0 - K0) * P01_pred
        self.P10 = -K1 * P00_pred + P10_pred
        self.P11 = -K1 * P01_pred + P11_pred
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
        if self.max_out > 0:
            delta = max(-self.max_out, min(self.max_out, delta))
        return delta

def calc_dynamic_param(distance, p_min, p_max, p_factor, max_dist):
    if max_dist <= 0: return p_max
    norm_dist = min(1.0, distance / max_dist)
    base_val = 1.0 - norm_dist
    factor_val = base_val ** p_factor if p_factor != 1.0 and p_factor != 2.0 else (base_val*base_val if p_factor==2.0 else base_val)
    val = p_min + (p_max - p_min) * factor_val
    return max(p_min, min(p_max, val))

# ==========================================
# 2. UI 面板
# ==========================================
st.sidebar.title("🎛️ 终极控制台")

with st.sidebar.expander("🖥️ 系统与硬件环境 (新)", expanded=True):
    st.info("调整这两项可以模拟不同的电脑配置和游戏设置，直接影响拉枪手感。")
    target_fps = st.select_slider("游戏帧率 (FPS)", options=[30, 60, 120, 144, 240, 360], value=60)
    sens_multiplier = st.slider("DPI/游戏内灵敏度倍率", 0.1, 5.0, 1.0, step=0.1)

with st.sidebar.expander("🎯 目标运动设置"):
    target_start_x = st.slider("目标起始 X", 0, 300, 250)
    target_start_y = st.slider("目标起始 Y", 0, 300, 200)
    # 改为像素/秒，使目标移动不受 FPS 影响
    target_vel_x_sec = st.slider("目标速度 X (像素/秒)", -300.0, 300.0, -60.0, step=10.0)
    target_vel_y_sec = st.slider("目标速度 Y (像素/秒)", -300.0, 300.0, -30.0, step=10.0)
    target_width = st.number_input("目标框宽度", value=60)
    target_height = st.number_input("目标框高度", value=100)

with st.sidebar.expander("🚀 增量 PID 基础参数"):
    col1, col2 = st.columns(2)
    with col1:
        kpx_max = st.number_input("KPX_MAX (刹车)", value=0.15, step=0.01)
        kix_max = st.number_input("KIX_MAX (动力)", value=0.015, step=0.001)
    with col2:
        kpy_max = st.number_input("KPY_MAX", value=0.15, step=0.01)
        kiy_max = st.number_input("KIY_MAX", value=0.015, step=0.001)
    kdx_max = st.number_input("KDX_MAX (阻尼)", value=0.0, step=0.01)
    ixmax = st.number_input("最大输出限制", value=50.0)

with st.sidebar.expander("⚡ 动态距离衰减与死区"):
    p_factor = st.slider("衰减因子 (P_FACTOR)", 0.5, 3.0, 1.0)
    kix_min = st.number_input("KIX_MIN (近战动力)", value=0.005, step=0.001)
    deadband_x = st.slider("死区 X", 0.0, 10.0, 2.0)
    deadband_y = st.slider("死区 Y", 0.0, 10.0, 2.0)
    x_offset = st.slider("X 轴偏差", -50, 50, 0)

with st.sidebar.expander("🔮 卡尔曼预测 (打移动靶)"):
    prediction_enabled = st.checkbox("开启预测", value=True)
    predict_time_x = st.slider("预判时间 X (ms)", 0.0, 100.0, 20.0)
    predict_time_y = st.slider("预判时间 Y (ms)", 0.0, 100.0, 10.0)

with st.sidebar.expander("🧬 拟人化 fBm 曲线"):
    final_range = st.slider("直线范围 (FINAL_RANGE)", 0.0, 100.0, 25.0)
    max_curve = st.slider("最大偏移 (MAX_CURVE)", 0.0, 20.0, 6.0)
    curve_freq = st.slider("抖动频率", 0.001, 0.01, 0.003, step=0.001)

with st.sidebar.expander("🔫 自动扳机"):
    trigger_percent = st.slider("触发比例 (%)", 1, 100, 30)

# ==========================================
# 3. 运行模拟逻辑
# ==========================================
# 根据 FPS 动态计算每帧时间
dt_ms = 1000.0 / target_fps
dt_sec = dt_ms / 1000.0
sim_duration_sec = 2.0 # 固定模拟 2 秒钟的时间
total_frames = int(sim_duration_sec * target_fps)

pid_x = IncrementalPID(); pid_y = IncrementalPID()
kf_x = KalmanFilter1D(0.001, 8.0, 2.0); kf_y = KalmanFilter1D(0.001, 8.0, 2.0)

current_x, current_y = 0.0, 0.0
kf_x.reset(target_start_x); kf_y.reset(target_start_y)

path_xhair_x, path_xhair_y = [current_x], [current_y]
path_target_x, path_target_y = [target_start_x], [target_start_y]
path_pred_x, path_pred_y = [], []
trigger_points_x, trigger_points_y = [], []

max_detect_dist = np.sqrt(target_width**2 + target_height**2)

for i in range(total_frames):
    # 目标基于时间移动，确保不同 FPS 下速度一致
    current_time_sec = i * dt_sec
    real_target_x = target_start_x + target_vel_x_sec * current_time_sec
    real_target_y = target_start_y + target_vel_y_sec * current_time_sec
    path_target_x.append(real_target_x)
    path_target_y.append(real_target_y)

    effective_target_x = real_target_x + x_offset
    effective_target_y = real_target_y

    if prediction_enabled:
        pred_x = kf_x.update_and_predict(effective_target_x, dt_sec, predict_time_x / 1000.0)
        pred_y = kf_y.update_and_predict(effective_target_y, dt_sec, predict_time_y / 1000.0)
        final_target_x, final_target_y = pred_x, pred_y
        path_pred_x.append(pred_x); path_pred_y.append(pred_y)
    else:
        final_target_x, final_target_y = effective_target_x, effective_target_y

    error_x = final_target_x - current_x
    error_y = final_target_y - current_y
    distance = np.sqrt(error_x**2 + error_y**2)

    if abs(error_x) <= deadband_x and abs(error_y) <= deadband_y:
        pid_x.compute(current_x, current_x)
        pid_y.compute(current_y, current_y)
        dx, dy = 0.0, 0.0
    else:
        cur_kpx = calc_dynamic_param(distance, kpx_max * 0.3, kpx_max, p_factor, max_detect_dist)
        cur_kix = calc_dynamic_param(distance, kix_min, kix_max, p_factor, max_detect_dist)
        
        pid_x.set_params(cur_kpx, cur_kix, kdx_max, ixmax)
        pid_y.set_params(cur_kpx, cur_kix, kdx_max, ixmax) # 简化Y轴调用同等逻辑

        target_calc_x = current_x if abs(error_x) <= deadband_x else final_target_x
        target_calc_y = current_y if abs(error_y) <= deadband_y else final_target_y

        dx = pid_x.compute(current_x, target_calc_x)
        dy = pid_y.compute(current_y, target_calc_y)

        # 拟人化叠加，注意时间因子传入真实时间 current_time_sec，防止受FPS干扰
        len_dir = np.sqrt(dx**2 + dy**2)
        if len_dir > 0.1 and distance > final_range:
            perp_x = -dy / len_dir; perp_y = dx / len_dir
            noise_val = 0.0
            amp = 1.0; freq = curve_freq
            for _ in range(3):
                # 将频率乘数扩大，以适配真实的秒级时间戳
                noise_val += smooth_noise(current_time_sec * 1000.0 * freq) * amp
                amp *= 0.5; freq *= 2.0
            
            fade_factor = min(1.0, (distance - final_range) / 60.0)
            offset = noise_val * max_curve * min(1.0, len_dir / 12.0) * fade_factor
            dx += perp_x * offset; dy += perp_y * offset

    # 核心：DPI/灵敏度转换系统
    # 代码输出的 dx 是鼠标的设备移动单元，乘以灵敏度倍率，才是屏幕上真正的准星位移
    actual_crosshair_move_x = dx * sens_multiplier
    actual_crosshair_move_y = dy * sens_multiplier

    current_x += actual_crosshair_move_x
    current_y += actual_crosshair_move_y
    path_xhair_x.append(current_x)
    path_xhair_y.append(current_y)

    t_ratio = trigger_percent * 0.01
    half_w = (target_width * t_ratio) * 0.5
    half_h = (target_height * t_ratio) * 0.5
    if (real_target_x - half_w <= current_x <= real_target_x + half_w) and \
       (real_target_y - half_h <= current_y <= real_target_y + half_h):
        trigger_points_x.append(current_x)
        trigger_points_y.append(current_y)

# ==========================================
# 4. 图表可视化绘制
# ==========================================
st.title(f"📊 弹道模拟（共模拟 {sim_duration_sec} 秒 / {total_frames} 帧）")

fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#282a36')
ax.tick_params(colors='#f8f8f2')
for spine in ax.spines.values(): spine.set_edgecolor('#6272a4')

ax.plot(path_target_x, path_target_y, linestyle='--', color='#6272a4', label="目标真实轨迹")

last_tx, last_ty = path_target_x[-1], path_target_y[-1]
rect_detect = patches.Rectangle((last_tx - target_width/2, last_ty - target_height/2), target_width, target_height, 
                                linewidth=1, edgecolor='white', facecolor='none', alpha=0.3, label="视觉检测框")
trig_w, trig_h = target_width * (trigger_percent/100), target_height * (trigger_percent/100)
rect_trigger = patches.Rectangle((last_tx - trig_w/2, last_ty - trig_h/2), trig_w, trig_h, 
                                 linewidth=1.5, edgecolor='#ff5555', facecolor='none', label="扳机区")
ax.add_patch(rect_detect)
ax.add_patch(rect_trigger)

if prediction_enabled and path_pred_x:
    ax.plot(path_pred_x, path_pred_y, color='#f1fa8c', alpha=0.6)
    ax.scatter(path_pred_x[-1], path_pred_y[-1], color='#f1fa8c', marker='X', s=80, label="卡尔曼预判落点", zorder=4)

# 准星轨迹
ax.plot(path_xhair_x, path_xhair_y, color='#bd93f9', linewidth=2, label="准星拉枪轨迹")
ax.scatter(path_xhair_x, path_xhair_y, color='#ff79c6', s=5, zorder=3)

if trigger_points_x:
    ax.scatter(trigger_points_x, trigger_points_y, color='#ff5555', marker='*', s=40, zorder=5, label="触发开火")

ax.set_title(f"当前配置: {target_fps} FPS | 灵敏度倍率: {sens_multiplier}x | 目标速度解耦", color='#f8f8f2', fontsize=14)
ax.grid(color='#6272a4', linestyle=':', alpha=0.3)
ax.legend(facecolor='#282a36', edgecolor='#6272a4', labelcolor='#f8f8f2')
plt.tight_layout()

st.pyplot(fig)

st.success("""
**环境测试建议：**
* **当你调高灵敏度 (DPI) 时**：你会发现原本平稳的紫线突然变得极不稳定甚至严重超调（绕着目标打转）。这是因为同样的 PID 鼠标输出被放大了。**解决办法**：降低 `KI` 动力，并适当增加 `KP` 阻尼。
* **当你调高 FPS 时**：因为计算频率变高，积分累加得更快。如果在 60 FPS 下表现完美，换到 144 FPS 下可能会略显抽搐。你需要通过这个模拟器找到适用于你高刷屏的最佳参数衰减比例。
""")

