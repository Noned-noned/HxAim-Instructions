import streamlit as st
import streamlit.components.v1 as components

# 设置 Streamlit 网页标题和页面布局为宽屏
st.set_page_config(page_title="HxAim 手册", layout="wide")

# 这里是我们整合了修正版 PID 逻辑的 HTML 模板
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-color: #1e1e2e;
            --panel-bg: #282a36;
            --text-main: #f8f8f2;
            --text-muted: #6272a4;
            --accent: #bd93f9;
            --accent-hover: #ff79c6;
            --success: #50fa7b;
            --warning: #f1fa8c;
            --danger: #ff5555;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            margin: 0;
            padding: 0;
            display: flex;
        }

        /* 侧边栏导航 */
        .sidebar {
            width: 200px;
            background-color: var(--panel-bg);
            height: 100vh;
            position: fixed;
            padding: 20px 0;
            box-shadow: 2px 0 10px rgba(0,0,0,0.3);
            overflow-y: auto;
        }

        .sidebar h2 {
            text-align: center;
            color: var(--accent);
            margin-bottom: 30px;
            font-size: 20px;
        }

        .sidebar a {
            display: block;
            padding: 12px 20px;
            color: var(--text-main);
            text-decoration: none;
            transition: all 0.3s ease;
            border-left: 4px solid transparent;
            font-size: 14px;
        }

        .sidebar a:hover {
            background-color: rgba(189, 147, 249, 0.1);
            border-left: 4px solid var(--accent);
            color: var(--accent-hover);
        }

        /* 主内容区 */
        .content {
            margin-left: 200px;
            padding: 30px;
            max-width: 900px;
            width: 100%;
        }

        .section {
            background-color: var(--panel-bg);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        h1 { color: var(--accent); border-bottom: 2px solid var(--text-muted); padding-bottom: 10px; font-size: 24px;}
        h2 { color: var(--success); margin-top: 0; font-size: 20px;}
        h3 { color: var(--warning); font-size: 16px;}

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 14px;
        }

        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid var(--text-muted);
        }

        th { background-color: rgba(0,0,0,0.2); color: var(--accent); }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
        }
        .badge-bool { background-color: var(--accent); color: #fff; }
        .badge-num { background-color: var(--success); color: #000; }

        .tip-box {
            background-color: rgba(80, 250, 123, 0.1);
            border-left: 4px solid var(--success);
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
            font-size: 14px;
        }
        
        .warn-box {
            background-color: rgba(255, 85, 85, 0.1);
            border-left: 4px solid var(--danger);
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
            font-size: 14px;
        }
        p, li { font-size: 14px; }
    </style>
</head>
<body>

    <div class="sidebar">
        <h2>HxAim 手册</h2>
        <a href="#intro">模块简介与启动</a>
        <a href="#basic">基础功能开关</a>
        <a href="#pid">增量 PID 追踪调参</a>
        <a href="#kalman">卡尔曼滤波与预测</a>
        <a href="#human">拟人化轨迹系统</a>
        <a href="#trigger">扳机与武器模式</a>
    </div>

    <div class="content">
        <div class="section" id="intro">
            <h1>模块简介与启动前置条件</h1>
            <p>HxAim 是一个集成了多种高级算法的辅助模块，采用 <b>Lua + C++ DLL</b> 架构。它主要包含以下核心技术：增量式 PID 鼠标平滑速度控制、1D 卡尔曼滤波器目标预测、分形布朗运动（fBm）拟人化曲线等。</p>
            
            <div class="warn-box">
                <strong>启动前必看：</strong><br>
                1. <strong>网络验证：</strong> 模块启动时会向云端请求用户授权列表，请确保网络通畅且授权未过期。<br>
                2. <strong>弹道回放文件：</strong> 默认从 <code>D:\hxaim_recoil.csv</code> 读取弹道数据。如果开启了“回放弹道”，请确保该路径下存在正确的 CSV 文件（格式：<code>时间ms,X偏移,Y偏移</code>）。
            </div>
        </div>

        <div class="section" id="basic">
            <h2>基础功能开关说明</h2>
            <p>界面上的复选框主要用于控制各大核心逻辑的启动与关闭：</p>
            <table>
                <tr><th>功能名称</th><th>作用说明</th></tr>
                <tr><td><span class="badge badge-bool">自瞄开关</span></td><td>主控开关。关闭后将停止一切追踪与鼠标移动逻辑。</td></tr>
                <tr><td><span class="badge badge-bool">智能延迟/时间动态</span></td><td>控制参数（如 KIX 等）会随 <b>目标距离</b> 或 <b>瞄准时间</b> 动态变化，实现“先快后慢”的拟人效果。</td></tr>
                <tr><td><span class="badge badge-bool">动态反转</span></td><td>反转动态参数的计算逻辑。</td></tr>
                <tr><td><span class="badge badge-bool">鼠标屏蔽</span></td><td>准星极近（死区范围内）时，暂停自身微调或屏蔽真实鼠标输入，防“抢鼠标”。</td></tr>
                <tr><td><span class="badge badge-bool">预测开关</span></td><td>开启基于<b>卡尔曼滤波</b>的运动趋势预测，适用于打移动靶。</td></tr>
            </table>
        </div>

        <div class="section" id="pid">
            <h2>增量式 PID 平滑追踪调参指南</h2>
            <p>本模块的鼠标移动由 <b>增量式 PID 控制器</b> 驱动，X 轴和 Y 轴独立调参。注意：增量式 PID 直接输出鼠标的<b>相对移动速度</b>，因此参数含义与常规位置 PID 有所不同。</p>
            
            <h3>核心参数解析：</h3>
            <ul>
                <li><span class="badge badge-num">KI (基础速度/拉枪力)</span>：<b>这是决定拉枪速度的核心参数。</b> 值越大，准星向目标靠拢的基础速度越快。如果觉得锁人太慢、跟不上，请优先调大 KI。</li>
                <li><span class="badge badge-num">KP (阻尼/刹车)</span>：<b>用于防止准星“飞过头”和抑制画面抖动。</b> 当你把 KI 调得很大导致准星在目标左右来回摇摆（超调）时，适当增加 KP 可以提供阻尼，让准星平稳停在目标上。</li>
                <li><span class="badge badge-num">KD (加速度缓冲)</span>：应对目标极其剧烈的方向突变。通常情况下保持为 0 即可，过大会导致鼠标移动手感变得粘滞。</li>
            </ul>

            <div class="tip-box">
                <strong>快速调参步骤：</strong><br>
                1. 先将 KP 和 KD 设为 0。<br>
                2. 慢慢调大 <code>KIX_MIN</code> 和 <code>KIX_MAX</code>，直到拉枪速度达到你的预期。此时可能会出现准星飞过目标来回晃的情况。<br>
                3. 缓慢增加 <code>KPX_MIN</code> 和 <code>KPX_MAX</code>，你会发现准星的摇晃逐渐被“刹住”，最终平滑地吸附在目标上。<br>
                4. Y 轴（上下）参数同理。
            </div>
            <p><b>死区X / 死区Y：</b> 当准星与目标的像素距离小于该值时，停止 PID 计算，防止在目标中心发生 1 像素级别的来回抽搐。</p>
        </div>

        <div class="section" id="kalman">
            <h2>卡尔曼滤波与预测设置</h2>
            <p>当勾选“预测开关”时生效，系统会分析目标的当前位置和移动速度，预判未来的位置，以解决子弹飞行延迟导致的“跟枪落后”问题。</p>
            <table>
                <tr><th>参数</th><th>说明</th></tr>
                <tr><td>预测时间 X/Y (ms)</td><td>预测目标在未来多少毫秒后的位置。狙击枪或远距离时需调大，步枪近战调小。</td></tr>
                <tr><td>位置/速度噪声</td><td>卡尔曼滤波信任度。越小越信任预测模型，越大越灵敏但也易受假动作欺骗。</td></tr>
                <tr><td>观测噪声</td><td>对图像识别结果的信任度。如果检测框闪烁跳动严重，请调大此值。</td></tr>
            </table>
        </div>

        <div class="section" id="human">
            <h2>拟人化轨迹系统</h2>
            <p>本程序采用 <b>fBm (分形布朗运动)</b> 生成平滑的伪随机噪声，让鼠标移动轨迹呈现轻微的不规则弧线，高度模拟真实玩家的手部微颤。</p>
            <ul>
                <li><b>最终直线范围：</b> 当距离目标大于此数值时，鼠标走曲线；进入此范围后，回归直线精准锁定。</li>
                <li><b>最大曲线偏移：</b> 决定拉枪时画弧线的幅度。数值越大，“甩枪”的弧度越夸张。</li>
                <li><b>曲线抖动频率：</b> 决定轨迹的弯曲频率。数值低是平滑长弧线；数值高是高频小碎步抖动。</li>
            </ul>
        </div>
        
        <div class="section" id="trigger">
            <h2>扳机、武器模式与按键</h2>
            <p>开启扳机开关后，满足条件会自动开火。弓箭模式开启时，程序会自动长按左键进行蓄力，满足配置时间后自动松开。</p>
        </div>
    </div>

</body>
</html>
"""

# 使用 components 渲染 HTML，设置足够的高度以容纳内容并允许滚动
components.html(HTML_CONTENT, height=1000, scrolling=True)

