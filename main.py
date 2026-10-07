
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="航检智巡 V3.5", page_icon="📡", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "航检智巡_V3_真实参数.csv"

st.title("📡 航检智巡 V3.5")
st.caption("民航导航设施智能巡检与故障辅助诊断系统")
st.info("V3.5：真实维护数据 + 多参数同步连续仿真 + Isolation Forest + 参数单位/工程门限 + 标准化偏离 + 异常等级 + 故障辅助诊断 + AI巡检报告。")

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

# 单位补充表：
# 原始CSV中部分DVOR参数“单位”字段为空，因此这里按参数本身的工程含义补充显示单位。
# 这些补充单位仅用于界面展示，不改变原始数据和门限。
UNIT_FALLBACK = {
    "30Hz调制度": "%",
    "载波功率": "W",
    "上边带功率": "W",
    "下边带功率": "W",
    "方位准确度": "°",
    "电池电压": "V",
}

def get_unit(rows, parameter):
    raw = rows.iloc[0]["单位"]
    if not pd.isna(raw) and str(raw).strip():
        return str(raw).strip()
    return UNIT_FALLBACK.get(parameter, "—")

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
            "unit": get_unit(rows, p),
            "limit": "未设置" if pd.isna(rows.iloc[0]["门限"]) or str(rows.iloc[0]["门限"]).strip() in ("", "''", "nan") else str(rows.iloc[0]["门限"])
        }
    return out

def find_param(stats, names):
    for name in names:
        if name in stats:
            return name
    return None



def parameter_abnormality(stats, selected, row):
    """计算参数相对正常基线的标准化偏离程度，用于解释AI结果。"""
    result = []
    for p in selected:
        s = stats[p]
        sigma = max(float(s["std"]), abs(float(s["mean"])) * 0.001, 1e-6)
        z = abs((float(row[p]) - float(s["mean"])) / sigma)
        result.append({"参数": p, "偏离程度": float(z), "当前值": float(row[p]), "正常均值": float(s["mean"])})
    return sorted(result, key=lambda x: x["偏离程度"], reverse=True)


def abnormal_level(z):
    """将标准化偏离程度转换为便于展示的异常等级。"""
    if z >= 3.0:
        return "高"
    if z >= 2.0:
        return "中"
    if z >= 1.5:
        return "低"
    return "正常"


def build_abnormal_table(stats, abnormal_params, top_n=8):
    """
    生成工程人员更容易理解的异常参数表：
    实际值 + 单位 + 工程门限 + 相对均值的实际偏差 + σ偏离程度 + 异常等级。
    """
    rows = []
    for x in abnormal_params[:top_n]:
        p = x["参数"]
        s = stats[p]
        current = float(x["当前值"])
        mean = float(x["正常均值"])
        delta = current - mean

        if not pd.isna(s["low"]) and not pd.isna(s["high"]):
            limit_text = f'{s["low"]:.4g} ～ {s["high"]:.4g} {s["unit"]}'.strip()
        else:
            limit_text = "未设置"

        rows.append({
            "参数": p,
            "当前值": round(current, 4),
            "单位": s["unit"] if s["unit"] else "—",
            "基线均值": round(mean, 4),
            "实际偏差": round(delta, 4),
            "工程门限": limit_text,
            "偏离程度": round(float(x["偏离程度"]), 2),
            "异常等级": abnormal_level(float(x["偏离程度"]))
        })

    return pd.DataFrame(rows)


def auxiliary_diagnosis(device, abnormal_params):
    """基于观测到的参数偏离程度给出辅助诊断，不直接读取仿真场景。"""
    names = [x["参数"] for x in abnormal_params if x["偏离程度"] >= 2.0]

    if device == "DME":
        if any("延迟" in n for n in names) and any(
            k in n for n in names for k in ["解码", "TX脉冲率", "间隔"]
        ):
            return "疑似应答时序/处理链路异常", "较高", (
                f"检测到 {'、'.join(names)} 出现协同偏离，建议重点检查应答处理、时序及相关接口链路。"
            )
        if any("功率" in n for n in names) and any("效率" in n for n in names):
            return "疑似发射链路异常", "较高", (
                f"检测到 {'、'.join(names)} 同时偏离正常基线，建议重点检查发射链路及功率相关模块。"
            )
        wave = [any(k in n for k in ["脉冲宽度", "脉冲上升时间", "脉冲下降时间"]) for n in names]
        if sum(wave) >= 2:
            return "疑似脉冲波形异常", "较高", (
                f"检测到多个脉冲波形参数同时偏离（{'、'.join(names)}），建议检查脉冲形成及波形相关链路。"
            )
    else:
        if any("调制度" in n for n in names):
            return "疑似调制相关异常", "中", (
                f"检测到 {'、'.join(names)} 明显偏离，建议重点检查调制相关链路。"
            )
        power = [any(k in n for k in ["载波功率", "上边带功率", "下边带功率"]) for n in names]
        if sum(power) >= 2:
            return "疑似发射功率相关异常", "较高", (
                f"检测到载波/边带功率联动偏离（{'、'.join(names)}），建议检查发射功率相关链路。"
            )
        if any("方位准确度" in n for n in names):
            return "疑似方位性能异常", "中", (
                "检测到方位准确度明显偏离，建议检查方位相关链路。"
            )

    if len(names) >= 2:
        return "疑似多参数综合异常", "中", (
            f"多个参数同时偏离正常基线（{'、'.join(names)}），建议结合维护记录进一步排查。"
        )
    if names:
        return "疑似单参数异常", "中", (
            f"主要异常参数为 {names[0]}，建议结合门限、历史趋势及维护记录进一步确认。"
        )
    return "暂未发现明显故障特征", "低", "当前数据与正常运行基线较为接近。"



def build_report(device, ai_state, ai_score, diagnosis, confidence, explanation, abnormal_params):
    top = abnormal_params[:3]
    top_text = "；".join(f"{x['参数']}（偏离{x['偏离程度']:.2f}σ）" for x in top) if top else "无明显异常参数"
    return (f"设备：{device}\n总体状态：{ai_state}\nAI异常分数：{ai_score:.4f}\n"
            f"辅助诊断：{diagnosis}\n诊断置信程度：{confidence}\n"
            f"主要异常参数：{top_text}\n分析说明：{explanation}\n"
            "说明：本报告用于智能巡检辅助分析，不替代专业检测规程和最终故障判定。")

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


def make_standard_iforest_model(stats, selected, samples=600):
    """建立标准化正常基线，并用训练分数分位点校准异常阈值。"""
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

    X = np.asarray(X, dtype=float)
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )
    model.fit(X)

    # 用正常基线自身的得分校准阈值，避免不同参数维度下出现
    # “正常仿真点大面积被判异常”的问题。
    train_scores = model.decision_function(X)
    threshold = float(np.percentile(train_scores, 3))
    model._hjzj_threshold = threshold
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
    score = float(model.decision_function(x)[0])
    threshold = float(getattr(model, "_hjzj_threshold", 0.0))
    max_z = float(np.max(np.abs(x)))

    # Isolation Forest 负责联合分布检测；最大标准化偏离作为第二道保险。
    # 这样可以避免“单个关键参数已经明显越界，但树模型没有及时隔离”的情况。
    is_anomaly = (score < threshold) or (max_z >= 3.0)

    if is_anomaly:
        if max_z >= 5.0 or score < threshold - 0.08:
            level = "高"
        elif max_z >= 3.0 or score < threshold - 0.03:
            level = "中"
        else:
            level = "低"
        return "异常", score, f"疑似异常（{level}，最大参数偏离 {max_z:.2f}σ）"
    return "正常", score, f"正常（最大参数偏离 {max_z:.2f}σ）"



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
        ai_diag_box = st.empty()
        ai_rank_box = st.empty()
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
                    # 渐进异常注入：优先把数值推向门限外，保证仿真场景
                    # 与“正常运行”在观测特征上有明显区别。
                    lo, hi = s["low"], s["high"]
                    if not pd.isna(lo) and not pd.isna(hi) and hi > lo:
                        margin = max((hi - lo) * 0.15, s["std"] * severity)
                        if s["mean"] <= (lo + hi) / 2:
                            target = hi + margin
                        else:
                            target = lo - margin
                        value = value * (1 - t) + target * t
                    else:
                        direction = 1 if np.random.random() >= 0.5 else -1
                        value += direction * s["std"] * severity * 1.5 * t

                row[p] = value

            # Isolation Forest：把不同单位参数标准化后进行联合异常检测。
            ai_state, ai_score, ai_text = ai_predict(ai_model, stats, selected, row)
            abnormal_params = parameter_abnormality(stats, selected, row)
            diagnosis, confidence, explanation = auxiliary_diagnosis(device, abnormal_params)
            row["AI异常状态"] = ai_state
            row["AI异常分数"] = ai_score
            row["辅助诊断"] = diagnosis

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

            ai_diag_box.info(f"🔎 辅助诊断：{diagnosis}｜置信程度：{confidence}\n\n分析说明：{explanation}")
            top3 = abnormal_params[:8]
            rank_text = "；".join(
                f"{i+1}. {x['参数']}：{x['偏离程度']:.2f}σ"
                for i, x in enumerate(top3)
            ) if top3 else "暂无"
            ai_rank_box.caption(f"📊 当前异常参数排名：{rank_text}")

            # 工程化展示：每个参数单独列出实际单位，同时保留σ标准化偏离程度。
            detail_table = build_abnormal_table(stats, abnormal_params, top_n=8)
            ai_rank_box.dataframe(
                detail_table,
                use_container_width=True,
                hide_index=True
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

        # 最终故障特征提示：与AI异常检测采用同一份“异常参数排名”逻辑，
        # 避免出现“AI判定异常很多，但故障提示仍显示正常”的信息冲突。
        final = st.session_state.v32_history.iloc[-1]
        final_abnormal = parameter_abnormality(stats, selected, final)
        final_diag, final_conf, final_explain = auxiliary_diagnosis(device, final_abnormal)

        st.divider()
        st.subheader("🧠 初步故障特征提示")
        top_final = final_abnormal[:8]
        if top_final and top_final[0]["偏离程度"] >= 2.0:
            st.warning(
                f"检测到主要异常参数："
                + "、".join(f"{x['参数']}（{x['偏离程度']:.2f}σ）" for x in top_final[:3])
            )
            st.info(f"辅助诊断：{final_diag}｜置信程度：{final_conf}\n\n{final_explain}")

            st.markdown("#### 📊 异常参数工程化明细")
            st.caption(
                "当前值和实际偏差保留原始工程单位；“偏离程度”采用σ表示，"
                "用于消除不同参数量纲差异，便于跨参数比较。"
            )
            st.dataframe(
                build_abnormal_table(stats, final_abnormal, top_n=8),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("当前仿真末端未发现达到辅助诊断阈值的明显异常参数。")

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
            st.subheader("📝 AI巡检报告")
            last = st.session_state.v32_history.iloc[-1]
            last_ai = st.session_state.v33_ai_history.iloc[-1]
            last_abnormal = parameter_abnormality(stats, selected, last)
            last_diag, last_conf, last_explain = auxiliary_diagnosis(device, last_abnormal)
            report = build_report(device, str(last_ai["AI异常状态"]), float(last_ai["AI异常分数"]), last_diag, last_conf, last_explain, last_abnormal)
            st.text_area("当前巡检摘要", report, height=220)
            st.download_button("⬇️ 导出AI巡检报告（TXT）", data=report.encode("utf-8"), file_name=f"航检智巡_{device}_AI巡检报告.txt", mime="text/plain", use_container_width=True)

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
st.caption("航检智巡 V3.5 | 数据基础：用户提供的 DVOR/DME 设备维护参数表 | AI：本地 Isolation Forest")
