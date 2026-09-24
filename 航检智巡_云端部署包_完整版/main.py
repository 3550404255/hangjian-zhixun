
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="航检智巡 V3.3", page_icon="📡", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "航检智巡_V3_真实参数.csv"

st.title("📡 航检智巡 V3.3")
st.caption("民航导航设施智能巡检与故障辅助诊断系统")
st.info("V3.3：真实维护数据 + 多参数同步连续仿真 + Isolation Forest 本地异常检测 + 初步故障特征提示。")

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_FILE, encoding="utf-8-sig")

df = load_data()
if df.empty:
    st.error("没有找到：航检智巡_V3_真实参数.csv")
    st.stop()

df["实测数值"] = pd.to_numeric(df["实测值"], errors="coerce")
df["下限"] = pd.to_numeric(df["下限"], errors="coerce")
df["上限"] = pd.to_numeric(df["上限"], errors="coerce")

def judge(v, lo, hi):
    if pd.isna(v):
        return "状态量/待人工判断"
    if pd.isna(lo) or pd.isna(hi):
        return "暂无门限"
    return "正常" if lo <= v <= hi else "超限"

def build_stats(device_df):
    out = {}
    num = device_df[device_df["实测数值"].notna()]
    for p in num["参数"].dropna().unique():
        rows = num[num["参数"] == p]
        vals = rows["实测数值"].dropna().astype(float).values
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if not np.isfinite(std) or std == 0:
            std = max(abs(mean) * 0.005, 0.001)
        lo = rows["下限"].dropna()
        hi = rows["上限"].dropna()
        out[p] = {
            "mean": mean, "std": std,
            "low": float(lo.iloc[0]) if len(lo) else np.nan,
            "high": float(hi.iloc[0]) if len(hi) else np.nan,
            "unit": "" if pd.isna(rows.iloc[0]["单位"]) else str(rows.iloc[0]["单位"]),
            "limit": "未设置" if pd.isna(rows.iloc[0]["门限"]) or str(rows.iloc[0]["门限"]).strip() in ("", "''", "nan") else str(rows.iloc[0]["门限"])
        }
    return out

def find_param(stats, names):
    for name in names:
        if name in stats:
            return name
    return None

def make_iforest_model(stats, selected, samples=300):
    """
    用当前设备真实维护数据的均值/标准差生成“正常工况”训练样本。
    这不是把异常数据拿来训练，而是建立一个本地基线模型。
    """
    if not selected:
        return None

    normal_rows = []
    for _ in range(samples):
        row = []
        for p in selected:
            s = stats[p]
            sigma = max(float(s["std"]), abs(float(s["mean"])) * 0.001, 1e-6)
            row.append(float(s["mean"]) + np.random.normal(0, sigma * 0.35))
        normal_rows.append(row)

    X = np.asarray(normal_rows, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        return None

    model = IsolationForest(
        n_estimators=150,
        contamination=0.06,
        random_state=42
    )
    model.fit(X)
    return model


def ai_result(model, stats, selected, row):
    """返回 Isolation Forest 的异常判断、分数和中文提示。"""
    if model is None or not selected:
        return "未启用", np.nan, "暂无AI检测结果"

    x = []
    for p in selected:
        s = stats[p]
        sigma = max(float(s["std"]), abs(float(s["mean"])) * 0.001, 1e-6)
        # 标准化后再送入模型，避免不同量纲参数互相支配。
        x.append((float(row[p]) - float(s["mean"])) / sigma)

    # 模型训练时也是标准化空间，因此重新训练一个标准化模型更合理。
    # 这里直接使用模型的训练空间；模型由调用处的标准化训练器创建。
    pred = int(model.predict([x])[0])
    score = float(model.decision_function([x])[0])

    if pred == -1:
        if score < -0.12:
            level = "高"
        elif score < 0:
            level = "中"
        else:
            level = "低"
        return "异常", score, f"Isolation Forest判定：疑似异常（{level}）"
    return "正常", score, "Isolation Forest判定：当前点位于正常数据分布附近"


def make_standard_iforest_model(stats, selected, samples=400):
    """在标准化空间训练 Isolation Forest，解决频率/功率/延迟等量纲不同的问题。"""
    if not selected:
        return None

    X = []
    for _ in range(samples):
        row = []
        for p in selected:
            s = stats[p]
            sigma = max(float(s["std"]), abs(float(s["mean"])) * 0.001, 1e-6)
            value = float(s["mean"]) + np.random.normal(0, sigma * 0.35)
            row.append((value - float(s["mean"])) / sigma)
        X.append(row)

    model = IsolationForest(
        n_estimators=150,
        contamination=0.06,
        random_state=42
    )
    model.fit(np.asarray(X, dtype=float))
    return model


def ai_predict(model, stats, selected, row):
    if model is None or not selected:
        return "未启用", np.nan, "暂无AI检测结果"

    x = []
    for p in selected:
        s = stats[p]
        sigma = max(float(s["std"]), abs(float(s["mean"])) * 0.001, 1e-6)
        x.append((float(row[p]) - float(s["mean"])) / sigma)

    x = np.asarray(x, dtype=float).reshape(1, -1)
    pred = int(model.predict(x)[0])
    score = float(model.decision_function(x)[0])

    if pred == -1:
        level = "高" if score < -0.12 else ("中" if score < 0 else "低")
        return "异常", score, f"疑似异常（{level}）"
    return "正常", score, "正常"


device = st.sidebar.selectbox("选择导航设备", ["DVOR", "DME"])
machine = st.sidebar.selectbox("选择设备编号", ["No.1", "No.2"])

device_df = df[(df["设备"] == device) & (df["设备编号"] == machine)].copy()
device_df["状态"] = [
    judge(v, lo, hi) for v, lo, hi in zip(
        device_df["实测数值"], device_df["下限"], device_df["上限"]
    )
]
stats = build_stats(device_df)

if device == "DME":
    names = {
        "delay": find_param(stats, ["运行参数-延迟"]),
        "interval": find_param(stats, ["运行参数-间隔"]),
        "power": find_param(stats, ["运行参数-射频 功率(1kW设备)"]),
        "eff": find_param(stats, ["运行参数-效率"]),
        "tx": find_param(stats, ["运行参数-TX脉冲率"]),
        "decode": find_param(stats, ["运行参数-解码脉冲率"]),
        "width": find_param(stats, ["运行参数-脉冲宽度"]),
        "rise": find_param(stats, ["运行参数-脉冲上升时间"]),
        "fall": find_param(stats, ["运行参数-脉冲下降时间"])
    }
else:
    names = {
        "mod": find_param(stats, ["30Hz调制度"]),
        "carrier": find_param(stats, ["载波功率"]),
        "upper": find_param(stats, ["上边带功率"]),
        "lower": find_param(stats, ["下边带功率"]),
        "azimuth": find_param(stats, ["方位准确度"]),
        "voltage": find_param(stats, ["电池电压"])
    }

mode = st.sidebar.radio("功能", ["设备总览", "多参数连续仿真", "历史维护数据"])

if mode == "设备总览":
    st.subheader(f"🔎 {device} / {machine} 设备总览")
    numeric = device_df[device_df["实测数值"].notna()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("监测参数", len(device_df))
    c2.metric("正常", int((numeric["状态"] == "正常").sum()))
    c3.metric("超限", int((numeric["状态"] == "超限").sum()))
    c4.metric("暂无门限", int((numeric["状态"] == "暂无门限").sum()))
    show = device_df[["参数","单位","门限","实测值","状态"]].copy()
    show["门限"] = show["门限"].replace([np.nan,"","''","nan"], "未设置")
    st.dataframe(show, use_container_width=True, hide_index=True)

elif mode == "多参数连续仿真":
    st.subheader(f"🧪 {device} / {machine} 多参数同步连续仿真")
    if not stats:
        st.warning("当前设备没有可用于仿真的数值参数。")
        st.stop()

    if device == "DME":
        scenario_map = {
            "正常运行": [],
            "应答时序异常": [names["delay"], names["interval"], names["tx"], names["decode"]],
            "发射链路异常": [names["power"], names["eff"], names["tx"]],
            "脉冲波形异常": [names["width"], names["rise"], names["fall"]],
            "综合异常": list(stats.keys())
        }
    else:
        scenario_map = {
            "正常运行": [],
            "调制异常": [names["mod"]],
            "发射功率异常": [names["carrier"], names["upper"], names["lower"]],
            "方位性能异常": [names["azimuth"]],
            "综合异常": list(stats.keys())
        }

    scenario = st.selectbox("选择仿真场景", list(scenario_map.keys()))
    selected = st.multiselect(
        "参与仿真的参数",
        list(stats.keys()),
        default=[p for p in scenario_map[scenario] if p is not None]
    )
    if not selected:
        st.warning("请至少选择一个参数。")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        interval = st.slider("采样间隔（秒）", 0.2, 2.0, 0.5, 0.1)
    with c2:
        duration = st.slider("仿真时长（秒）", 5, 60, 20)
    with c3:
        severity = st.slider("异常强度", 1.0, 6.0, 3.0, 0.5)

    st.caption("仿真基线来自真实维护数据的均值和波动程度；异常为人为注入，用于平台演示和算法开发。")
    start = st.button("▶ 开始多参数连续仿真", type="primary", use_container_width=True)

    if "v32_history" not in st.session_state:
        st.session_state.v32_history = pd.DataFrame()
    if "v33_ai_history" not in st.session_state:
        st.session_state.v33_ai_history = pd.DataFrame()

    if st.button("🗑 清空仿真记录", use_container_width=True):
        st.session_state.v32_history = pd.DataFrame()
        st.session_state.v33_ai_history = pd.DataFrame()
        st.rerun()

    if start:
        hist = []
        ai_hist = []
        n = max(1, int(duration / interval))

        # 本地 Isolation Forest：不调用网络、不需要付费API。
        ai_model = make_standard_iforest_model(stats, selected)
        ai_status_box = st.empty()
        progress = st.progress(0)
        status_box = st.empty()
        chart_box = st.empty()
        table_box = st.empty()

        for i in range(n):
            row = {"时间": datetime.now().strftime("%H:%M:%S")}
            t = i / max(n - 1, 1)

            for p in selected:
                s = stats[p]
                value = s["mean"] + np.random.normal(0, s["std"] * 0.35)
                active = p in scenario_map[scenario]

                if active and scenario != "正常运行":
                    # 渐进漂移；使用门限中点决定向哪边推
                    if not pd.isna(s["low"]) and not pd.isna(s["high"]):
                        mid = (s["low"] + s["high"]) / 2
                        direction = 1 if s["mean"] < mid else -1
                    else:
                        direction = 1
                    value += direction * s["std"] * severity * t

                row[p] = value

            # Isolation Forest：把不同单位参数标准化后进行联合异常检测。
            ai_state, ai_score, ai_text = ai_predict(ai_model, stats, selected, row)
            row["AI异常状态"] = ai_state
            row["AI异常分数"] = ai_score

            hist.append(row)
            ai_hist.append({
                "时间": row["时间"],
                "AI异常状态": ai_state,
                "AI异常分数": ai_score,
                "AI诊断": ai_text
            })

            live = pd.DataFrame(hist)

            bad = 0
            no_limit = 0
            for p in selected:
                s = stats[p]
                stt = judge(row[p], s["low"], s["high"])
                bad += stt == "超限"
                no_limit += stt == "暂无门限"

            if bad:
                status_box.error(f"🔴 当前检测到 {bad} 个参数超限")
            elif no_limit == len(selected):
                status_box.warning("🟡 当前参数均没有可解析门限")
            else:
                status_box.success("🟢 当前门限判断正常")

            if ai_state == "异常":
                ai_status_box.error(
                    f"🤖 AI异常检测：{ai_text} | 分数：{ai_score:.4f}"
                )
            else:
                ai_status_box.success(
                    f"🤖 AI异常检测：{ai_text} | 分数：{ai_score:.4f}"
                )

            # 图表只显示数值参数，避免把AI文字列混入折线图。
            chart_data = live[[p for p in selected if p in live.columns]]
            chart_box.line_chart(chart_data)

            display_live = live.tail(10).copy()
            table_box.dataframe(
                display_live,
                use_container_width=True,
                hide_index=True
            )
            progress.progress((i + 1) / n)
            time.sleep(interval)

        st.session_state.v32_history = pd.DataFrame(hist)
        st.session_state.v33_ai_history = pd.DataFrame(ai_hist)

        # 初步规则提示：只根据真实表里存在的参数做判断
        final = st.session_state.v32_history.iloc[-1]
        hints = []

        def out_of_limit(p):
            if not p or p not in final.index:
                return False
            s = stats[p]
            return not pd.isna(s["low"]) and not pd.isna(s["high"]) and not (s["low"] <= final[p] <= s["high"])

        if device == "DME":
            if out_of_limit(names["delay"]):
                hints.append("应答延迟偏离原始门限，建议检查应答时序相关链路。")
            if out_of_limit(names["power"]) and out_of_limit(names["eff"]):
                hints.append("射频功率与效率同时异常，建议重点检查发射链路。")
            wave = [out_of_limit(names[k]) for k in ["width","rise","fall"]]
            if sum(wave) >= 2:
                hints.append("多个脉冲波形参数同时异常，建议检查脉冲形成/波形相关链路。")
        else:
            if out_of_limit(names["mod"]):
                hints.append("30Hz调制度偏离原始门限，建议检查调制相关链路。")
            power = [out_of_limit(names[k]) for k in ["carrier","upper","lower"]]
            if sum(power) >= 2:
                hints.append("载波/边带功率出现联动异常，建议检查发射功率相关链路。")
            if out_of_limit(names["azimuth"]):
                hints.append("方位准确度偏离原始门限，建议检查方位相关链路。")

        st.divider()
        st.subheader("🧠 初步故障特征提示")
        if hints:
            for h in hints:
                st.warning(h)
        else:
            st.success("当前仿真末端没有形成明确的多参数异常组合。")

        st.success(f"本次仿真完成，共生成 {n} 个同步采样点。")

        st.divider()
        st.subheader("🤖 Isolation Forest 异常检测结果")
        if not st.session_state.v33_ai_history.empty:
            ai_df = st.session_state.v33_ai_history.copy()
            abnormal_count = int((ai_df["AI异常状态"] == "异常").sum())
            c_ai1, c_ai2, c_ai3 = st.columns(3)
            c_ai1.metric("AI检测采样点", len(ai_df))
            c_ai2.metric("AI判定异常", abnormal_count)
            c_ai3.metric(
                "异常比例",
                f"{abnormal_count / len(ai_df) * 100:.1f}%"
            )
            st.dataframe(
                ai_df.tail(20),
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "说明：Isolation Forest 使用当前设备维护数据的均值和波动范围构建本地正常基线，"
                "再对多参数联合状态进行无监督异常检测。当前版本用于仿真验证，不替代真实设备检测规程。"
            )

    if not st.session_state.v32_history.empty:
        st.divider()
        st.subheader("📈 已保存的仿真记录")
        st.dataframe(st.session_state.v32_history.tail(30), use_container_width=True, hide_index=True)

else:
    st.subheader(f"📋 {device} / {machine} 历史维护数据")
    show = device_df[["维护记录","设备","型号","设备编号","参数","单位","门限","实测值","状态"]].copy()
    show["门限"] = show["门限"].replace([np.nan,"","''","nan"], "未设置")
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()
st.caption("航检智巡 V3.3 | 数据基础：用户提供的 VORDME 设备维护参数表 | AI：本地 Isolation Forest")
