# -*- coding: utf-8 -*-
"""
Steam 游戏数据分析 · 交互式网站
================================
数据来源：和鲸社区（Heywhale）| 原始数据来自 Steam 平台公开数据 + SteamSpy 统计接口
数据集页面：https://www.heywhale.com/mw/dataset/6a166f8230366a83c0b27a69
分析内容：价格倾向、类型倾向、受欢迎程度、机器学习建模（分类 + 聚类）

文件清单（4 个，都在同一层）：
    app.py                      本文件，网站主程序
    requirements.txt            依赖清单
    NotoSansSC-Regular.otf      中文字体
    steam.csv                   数据文件
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, recall_score,
                             precision_score, roc_curve, confusion_matrix)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ============================================================
# 一、路径与中文字体（云端 Linux 没有中文字体，必须自带一个）
# ============================================================
# 以 app.py 所在文件夹为基准，这样本地和云端都能找到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "NotoSansSC-Regular.otf")

if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)                      # 把字体注册进 matplotlib
    cn_font = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.sans-serif"] = [cn_font]
else:
    # 本地没放字体文件时，退回到系统自带中文字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False                 # 让负号正常显示

st.set_page_config(page_title="Steam 游戏数据分析", page_icon="🎮", layout="wide")

PRICE_ORDER = ["免费", "0-5美元", "5-10美元", "10-20美元", "20美元以上"]

# 游戏类型中英对照
GENRE_CN = {
    "Indie": "独立游戏", "Action": "动作", "Adventure": "冒险", "Casual": "休闲",
    "Strategy": "策略", "Simulation": "模拟", "RPG": "角色扮演",
    "Early Access": "抢先体验", "Free to Play": "免费游玩", "Sports": "体育",
}


def cn(g):
    """英文类型名换成中文"""
    return GENRE_CN.get(g, g)


# ============================================================
# 二、读取数据与清洗
# ============================================================
@st.cache_data(show_spinner="正在读取数据…")
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "steam.csv"))
    df = df.drop_duplicates(subset="appid").reset_index(drop=True)

    # 新增好评率：好评数 ÷（好评数 + 差评数）
    df["total_ratings"] = df["positive_ratings"] + df["negative_ratings"]
    df["good_rate"] = df["positive_ratings"] / df["total_ratings"]

    # owners 是 "0-20000" 这样的区间文字，拆出上下界取中间值
    parts = df["owners"].str.split("-", expand=True)
    df["owners_mid"] = (parts[0].astype(int) + parts[1].astype(int)) / 2

    # 取出年份
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # 价格分五档，免费单独一档
    df["price_level"] = pd.cut(df["price"], bins=[-1, 0, 5, 10, 20, 10000], labels=PRICE_ORDER)

    # 游玩时长有 77% 是 0，这些 0 代表「没采集到」
    df["has_playtime"] = (df["median_playtime"] > 0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def get_play_subset():
    """只保留游玩时长有记录的游戏，这些行的时长数字才是可信的"""
    df = load_data()
    return df[df["median_playtime"] > 0].copy()


@st.cache_data(show_spinner=False)
def get_genre_long():
    """把 genres 一列里用分号隔开的多值拆开，一行一个类型"""
    df = load_data()
    return df.assign(genre=df["genres"].str.split(";")).explode("genre")


# ============================================================
# 三、训练模型（第一次运行会训练，之后缓存，不用重复跑）
# ============================================================
@st.cache_resource(show_spinner="正在训练模型，第一次大约需要 1 分钟（之后会直接缓存）…")
def train_models():
    """把分类、聚类、稳健性检验全部跑一遍，结果打包返回"""
    R = {}
    df = load_data()

    # ---------- 3.1 构造建模样本 ----------
    # 只保留评价数 ≥ 10 的游戏：评价太少的游戏好评率非 0% 即 100%，是纯噪声
    d = df[df["total_ratings"] >= 10].copy()

    # 目标：拥有量 ≥ 10 万 且 好评率 ≥ 80%，两个条件同时满足才算「热门」
    d["is_hit"] = ((d["owners_mid"] >= 100000) & (d["good_rate"] >= 0.8)).astype(int)

    # 特征构造
    d["price_log"] = np.log1p(d["price"])                      # 价格取对数，压缩长尾
    d["is_free"] = (d["price"] == 0).astype(int)
    d["has_ach"] = (d["achievements"] > 0).astype(int)
    d["has_multi"] = d["categories"].fillna("").str.contains("Multi-player", case=False).astype(int)
    d["genre_count"] = d["genres"].fillna("").apply(lambda s: len([g for g in s.split(";") if g]))

    # 类型字段用 one-hot：每个类型一列，0/1 表示有没有
    gd = d["genres"].str.get_dummies(sep=";")
    top_genres = gd.sum().sort_values(ascending=False).head(10).index.tolist()

    base = ["price_log", "is_free", "year", "has_multi", "genre_count", "has_ach"]
    X = pd.concat([d[base].reset_index(drop=True),
                   gd[top_genres].reset_index(drop=True)], axis=1)
    y = d["is_hit"].reset_index(drop=True)

    R["features"] = list(X.columns)
    R["top_genres"] = top_genres
    R["n_model_rows"] = int(len(d))
    R["n_hot"] = int(y.sum())
    R["pos_rate"] = float(y.mean())

    # ---------- 3.2 划分训练集 / 测试集 ----------
    # stratify=y 分层抽样：保证两边热门游戏占比一致
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    R["baseline_acc"] = float(1 - yte.mean())                  # 全猜「不热门」的准确率

    models = {
        "逻辑回归": make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced")),
        "随机森林": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42, n_jobs=-1),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics, roc = {}, {}
    for name, m in models.items():
        scores = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc")
        m.fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics[name] = {
            "cv_mean": float(scores.mean()), "cv_std": float(scores.std()),
            "auc": float(roc_auc_score(yte, prob)),
            "acc": float(accuracy_score(yte, pred)),
            "f1": float(f1_score(yte, pred)),
            "recall": float(recall_score(yte, pred)),
            "precision": float(precision_score(yte, pred)),
        }
        fpr, tpr, _ = roc_curve(yte, prob)
        step = max(1, len(fpr) // 60)                          # 抽稀，减小数据量
        roc[name] = {"fpr": [float(v) for v in fpr[::step]],
                     "tpr": [float(v) for v in tpr[::step]]}
    R["metrics"] = metrics
    R["roc"] = roc
    R["cm_rf"] = confusion_matrix(
        yte, (models["随机森林"].predict_proba(Xte)[:, 1] >= 0.5).astype(int)).tolist()

    # ---------- 3.3 特征重要性 ----------
    imp = pd.Series(models["随机森林"].feature_importances_, index=X.columns).sort_values(
        ascending=False).head(12)
    R["importance"] = {k: float(v) for k, v in imp.items()}

    # 逻辑回归的系数（网站的交互式预测要用）
    lr = models["逻辑回归"]
    sc, clf = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
    R["lr"] = {
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "mean": [float(v) for v in sc.mean_],
        "scale": [float(v) for v in sc.scale_],
    }

    # ---------- 3.4 稳健性检验：只保留 2015 年及以后 ----------
    sub = d[d["year"] >= 2015].copy()
    gd15 = sub["genres"].str.get_dummies(sep=";")
    X15 = pd.concat([sub[base].reset_index(drop=True),
                     gd15[top_genres].reset_index(drop=True)], axis=1)
    y15 = sub["is_hit"].reset_index(drop=True)
    Xa, Xb, ya, yb = train_test_split(X15, y15, test_size=0.25, random_state=42, stratify=y15)
    rf15 = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                  class_weight="balanced", random_state=42, n_jobs=-1).fit(Xa, ya)
    p15 = rf15.predict_proba(Xb)[:, 1]
    R["robust"] = {
        "n": int(len(sub)),
        "auc": float(roc_auc_score(yb, p15)),
        "recall": float(recall_score(yb, (p15 >= 0.5).astype(int))),
    }
    # 对照：去掉 year 特征再看
    X15b = X15.drop(columns=["year"])
    Xc, Xd, yc, yd = train_test_split(X15b, y15, test_size=0.25, random_state=42, stratify=y15)
    rf15b = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=-1).fit(Xc, yc)
    R["robust"]["auc_no_year"] = float(roc_auc_score(yd, rf15b.predict_proba(Xd)[:, 1]))
    imp15 = pd.Series(rf15.feature_importances_, index=X15.columns).sort_values(
        ascending=False).head(5)
    R["robust"]["importance_top"] = {k: float(v) for k, v in imp15.items()}

    # ---------- 3.5 「多人模式」对照：单看 vs 控制价格和年份 ----------
    g = d.groupby("has_multi")["is_hit"].agg(["count", "mean"])
    R["multi_single"] = {"n0": int(g.loc[0, "count"]), "r0": float(g.loc[0, "mean"]),
                         "n1": int(g.loc[1, "count"]), "r1": float(g.loc[1, "mean"])}
    d["price_bin"] = pd.cut(d["price"], [-1, 0, 5, 10, 20, 10000],
                            labels=["免费", "0-5", "5-10", "10-20", "20+"])
    d["year_bin"] = pd.cut(d["year"], [1996, 2013, 2015, 2017, 2020],
                           labels=["2013前", "2014-15", "2016-17", "2018后"])
    rows = []
    for pb in ["免费", "0-5", "5-10", "10-20"]:
        for yb in ["2014-15", "2016-17", "2018后"]:
            s = d[(d.price_bin == pb) & (d.year_bin == yb)]
            if len(s) < 200:
                continue
            m1 = s[s.has_multi == 1]["is_hit"].mean()
            m0 = s[s.has_multi == 0]["is_hit"].mean()
            if m0 > 0:
                rows.append([str(pb), str(yb), int(len(s)), float(m1), float(m0)])
    R["multi_ctrl"] = rows

    # ---------- 3.6 聚类 ----------
    cd = d.copy()
    cd["log_ratings"] = np.log1p(cd["positive_ratings"])
    cd["log_owners"] = np.log1p(cd["owners_mid"])
    cfeat = ["good_rate", "log_ratings", "log_owners", "price_log"]
    Z = StandardScaler().fit_transform(cd[cfeat])              # 聚类前必须标准化

    scan = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=5000, random_state=42)
        scan.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    R["k_scan"] = scan

    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
    cd["cluster"] = km.labels_
    prof = cd.groupby("cluster").agg(
        n=("appid", "count"), rate=("good_rate", "mean"), pos=("positive_ratings", "median"),
        own=("owners_mid", "median"), price=("price", "mean")).round(3)
    R["cluster_profile"] = {str(k): {"n": float(v["n"]), "rate": float(v["rate"]),
                                     "pos": float(v["pos"]), "own": float(v["own"]),
                                     "price": float(v["price"])}
                            for k, v in prof.iterrows()}
    R["cluster_top"] = {str(c): cd[cd.cluster == c].nlargest(3, "positive_ratings")["name"].tolist()
                        for c in range(3)}

    pca = PCA(n_components=2, random_state=42)
    P2 = pca.fit_transform(Z)
    step = max(1, len(P2) // 1500)                             # 抽样，画图够用
    R["pca"] = {"x": [float(v) for v in P2[::step, 0]],
                "y": [float(v) for v in P2[::step, 1]],
                "c": [int(v) for v in cd["cluster"].values[::step]]}
    return R


MD = train_models()
df = load_data()


# ============================================================
# 四、侧边栏导航
# ============================================================
st.sidebar.title("🎮 Steam 游戏数据分析")
st.sidebar.caption("数据来源：和鲸社区（Heywhale）\n\n27,075 款游戏 · 1997–2019 年")

PAGES = [
    "📊 数据概览",
    "💰 价格分析",
    "🎯 类型分析",
    "🏆 最受欢迎的游戏",
    "🤖 机器学习建模",
    "ℹ️ 关于本项目",
]
page = st.sidebar.radio("选择页面", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(
    "**三个核心问题**\n\n"
    "1. 玩家喜欢什么类型？\n"
    "2. 玩家能接受什么价格？\n"
    "3. 什么样的游戏最受欢迎？"
)


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# 页面 1：数据概览
# ============================================================
if page == "📊 数据概览":
    st.title("📊 数据概览")
    st.markdown("先看看这份数据长什么样、干不干净，后面的所有分析都建立在它上面。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("游戏总数", "{:,}".format(len(df)))
    c2.metric("覆盖年份", "%d – %d" % (int(df["year"].min()), int(df["year"].max())))
    c3.metric("平均好评率", "{:.1%}".format(df["good_rate"].mean()))
    c4.metric("免费游戏占比", "{:.1%}".format((df["price"] == 0).mean()))

    st.divider()

    st.subheader("1. 数据质量检查")
    st.markdown("检查了三件事：有没有缺失值、有没有重复行、有没有异常的 0。")
    n_missing = int(df[["developer", "publisher"]].isna().sum().sum())
    q1, q2, q3 = st.columns(3)
    q1.metric("缺失值", "{} 个（0.06%）".format(n_missing), help="占比极低，选择保留不填")
    q2.metric("重复行", "0 行", help="以游戏编号 appid 为准，没有重复")
    q3.metric("游玩时长为 0", "{:.1%}".format((df["median_playtime"] == 0).mean()),
              help="这是「假 0」，代表数据没采集到")

    st.markdown("#### 关键发现：游玩时长里的「假 0」")
    st.markdown(
        "游玩时长这一列有 **77.2%** 是 0。常规的缺失值检查查不出任何问题，表面看数据很干净。\n\n"
        "但把这些 0 对应的游戏翻出来看，发现它们很多其实有几百条评价。"
        "**一款已经积累了一百多条评价的游戏，玩家不可能玩 0 分钟就去写评价**——"
        "所以这个 0 只可能是「没采集到」，不是「没玩」。\n\n"
        "这是整个数据处理里最关键的一步：**这种「假 0」只能靠常识判断，代码不会报错。**"
    )

    play = get_play_subset()
    a, b = st.columns(2)
    a.metric("把 0 当真实值算 · 中位游玩时长",
             "{:.0f} 分钟".format(df["median_playtime"].median()), help="错误做法")
    b.metric("只算有记录的 · 中位游玩时长",
             "{:.0f} 分钟（{:.1f} 小时）".format(play["median_playtime"].median(),
                                              play["median_playtime"].median() / 60),
             help="正确做法")
    st.caption("同一个数据集，两种做法差了整整一个量级 —— 而且程序不会报任何错。")

    st.divider()

    st.subheader("2. 主要字段说明")
    st.dataframe(pd.DataFrame([
        ["name", "游戏名称", "展示和识别具体游戏"],
        ["genres", "游戏类型（一格可能有好几种，用分号隔开）", "回答类型倾向"],
        ["price", "售价（美元）", "回答价格倾向"],
        ["positive_ratings / negative_ratings", "好评数 / 差评数", "计算好评率"],
        ["owners", "拥有该游戏的玩家数量区间", "判断游戏热度"],
        ["median_playtime", "玩家游玩时长中位数", "判断玩家黏性"],
        ["release_date", "发行日期", "取出年份，分析时间趋势"],
        ["achievements", "成就数量", "作为建模特征"],
        ["categories", "玩法分类（是否含多人模式）", "作为建模特征"],
    ], columns=["字段名", "含义", "在本次分析中的作用"]), hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("3. 数据预览")
    st.dataframe(df[["name", "genres", "price", "positive_ratings",
                     "negative_ratings", "good_rate", "owners_mid"]].head(20),
                 hide_index=True, use_container_width=True)


# ============================================================
# 页面 2：价格分析
# ============================================================
elif page == "💰 价格分析":
    st.title("💰 价格分析")
    st.markdown("玩家能接受什么价格？哪个价位的游戏口碑最好？")

    count = df["price_level"].value_counts().reindex(PRICE_ORDER)
    avg_rate = df.groupby("price_level", observed=False)["good_rate"].mean().reindex(PRICE_ORDER)

    x = np.arange(len(PRICE_ORDER))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    b1 = ax1.bar(x - 0.2, count.values, width=0.4, color="steelblue", label="游戏数量")
    ax1.set_ylabel("游戏数量", color="steelblue")
    ax1.set_xlabel("价格区间")
    ax1.set_xticks(x)
    ax1.set_xticklabels(PRICE_ORDER)
    for i, v in enumerate(count.values):
        ax1.text(i - 0.2, v, str(int(v)), ha="center", va="bottom", fontsize=9, color="steelblue")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.2, avg_rate.values, width=0.4, color="crimson", label="平均好评率")
    ax2.set_ylabel("平均好评率", color="crimson")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(avg_rate.values):
        ax2.text(i + 0.2, v, "{:.1%}".format(v), ha="center", va="bottom", fontsize=9, color="crimson")
    ax1.set_title("各价格区间：游戏数量 与 平均好评率")
    ax1.legend([b1, b2], ["游戏数量", "平均好评率"], loc="upper right")
    fig.tight_layout()
    show_fig(fig)

    st.markdown("#### 从图里能看出什么")
    st.markdown(
        "**蓝色柱子（游戏数量）** 回答「开发者都在把游戏定在什么价位」："
        "13293 款游戏定价在 5 美元以下，占全部的 **49.1%**，几乎是一半；"
        "10 美元以上越来越少，20 美元以上的只有 923 款。\n\n"
        "**红色柱子（平均好评率）** 回答「玩家对哪个价位评价最高」，结论就反过来了："
        "好评率并不是越便宜越高，而是呈「中间高、两头低」——"
        "**0-5 美元这一档最低，只有 68.4%；10-20 美元最高，达到 75.7%。**\n\n"
        "**为什么便宜的反而口碑差？** 因为 0-5 美元同时是竞争最激烈的一档，"
        "占了一半的游戏，里面混着大量质量参差不齐的作品。"
    )

    st.success("**结论：玩家的真实价格偏好落在 5 到 20 美元这个中价区间，而不是越便宜越好。**")

    st.divider()
    st.subheader("自己看看：各档位的具体数字")
    st.dataframe(pd.DataFrame({
        "价格区间": PRICE_ORDER,
        "游戏数量": count.values,
        "占全部比例": ["{:.1%}".format(v / len(df)) for v in count.values],
        "平均好评率": ["{:.1%}".format(v) for v in avg_rate.values],
        "好评数中位数": [int(df[df.price_level == p]["positive_ratings"].median()) for p in PRICE_ORDER],
        "拥有量中位数": ["{:,.0f}".format(df[df.price_level == p]["owners_mid"].median()) for p in PRICE_ORDER],
    }), hide_index=True, use_container_width=True)


# ============================================================
# 页面 3：类型分析
# ============================================================
elif page == "🎯 类型分析":
    st.title("🎯 类型分析")
    st.markdown("玩家喜欢什么类型的游戏？类型真的能决定口碑吗？")

    topn = st.slider("看数量最多的前几个类型", min_value=5, max_value=15, value=10, step=1)

    gl = get_genre_long()
    count_n = gl["genre"].value_counts().head(topn)
    rate_n = gl.groupby("genre")["good_rate"].mean()[count_n.index]
    labels = [cn(g) for g in count_n.index]

    x = np.arange(len(count_n))
    fig, ax1 = plt.subplots(figsize=(11, 5))
    b1 = ax1.bar(x - 0.2, count_n.values, width=0.4, color="seagreen", label="游戏数量")
    ax1.set_ylabel("游戏数量", color="seagreen")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_xlabel("游戏类型")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.2, rate_n.values, width=0.4, color="darkorange", label="平均好评率")
    ax2.set_ylabel("平均好评率", color="darkorange")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(rate_n.values):
        ax2.text(i + 0.2, v, "{:.1%}".format(v), ha="center", va="bottom", fontsize=8, color="darkorange")
    ax1.set_title("各游戏类型：游戏数量 与 平均好评率")
    ax1.legend([b1, b2], ["游戏数量", "平均好评率"], loc="upper right")
    fig.tight_layout()
    show_fig(fig)

    st.markdown("#### 从图里能看出什么")
    st.markdown(
        "**数量最多的类型**是独立游戏、动作、休闲、冒险这几个，"
        "说明大多数开发者选择的是门槛较低、体量较小的品类。\n\n"
        "**但各类型的平均好评率差别很小**：最高的类型和最低的类型只差 "
        "**{:.1f} 个百分点**，而且都紧贴整体均值。\n\n"
        "**这说明「类型」并不是决定口碑好坏的关键因素。**"
        "玩家不会因为「这是一款独立游戏」就给好评，也不会因为「这是模拟游戏」就给差评。".format(
            (rate_n.max() - rate_n.min()) * 100)
    )
    st.info("这个结论在后面建模时也得到了印证 —— 所有类型特征的重要性加起来还不到 0.09。")


# ============================================================
# 页面 4：最受欢迎的游戏
# ============================================================
elif page == "🏆 最受欢迎的游戏":
    st.title("🏆 最受欢迎的游戏")
    st.markdown("什么样的游戏才算「最受欢迎」？这里的判定标准是可以调的。")

    st.markdown(
        "判定「受欢迎」必须同时看两个维度：**好评数**（玩的人多不多）和 **好评率**（评价好不好）。"
        "只看人数，会把差评一大堆的爆款算进来；只看评价，会把只有三条评价的小游戏算进来。"
    )

    c1, c2 = st.columns(2)
    th_ratings = c1.slider("好评数门槛", min_value=100, max_value=50000,
                           value=10000, step=100,
                           help="越高越严格。1 万大约只占全部游戏的 1.5%")
    th_rate = c2.slider("好评率门槛", min_value=0.50, max_value=0.99,
                        value=0.90, step=0.01, format="%.2f",
                        help="越高越严格。0.90 大约高于四分之三的游戏")

    hot = df[(df["positive_ratings"] >= th_ratings) & (df["good_rate"] >= th_rate)]
    hot = hot.sort_values("positive_ratings", ascending=False)

    m1, m2, m3 = st.columns(3)
    m1.metric("筛出游戏数", "{:,} 款".format(len(hot)))
    m2.metric("占全部比例", "{:.2%}".format(len(hot) / len(df)))
    m3.metric("好评数中位数", "{:,.0f}".format(hot["positive_ratings"].median()) if len(hot) else "—")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["positive_ratings"], df["good_rate"], s=6, alpha=0.25,
               color="lightsteelblue", label="全部游戏")
    if len(hot):
        ax.scatter(hot["positive_ratings"], hot["good_rate"], s=14, alpha=0.9,
                   color="crimson", label="落在右上角的（最受欢迎）")
    ax.axvline(th_ratings, color="red", linestyle="--", linewidth=1)
    ax.axhline(th_rate, color="red", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("好评数（对数刻度）→ 玩的人多不多")
    ax.set_ylabel("好评率 → 评价好不好")
    ax.set_title("好评数 × 好评率：右上角才是「最受欢迎」")
    ax.legend(loc="lower left")
    fig.tight_layout()
    show_fig(fig)

    st.caption("横轴用对数刻度，因为好评数是长尾分布 —— 大部分游戏只有几十条评价，"
               "少数爆款有上百万条。不用对数刻度的话，绝大多数点会挤在最左边看不清。")

    if len(hot):
        st.subheader("最受欢迎的 Top 10")
        show = hot.head(10)[["name", "genres", "price", "owners", "positive_ratings", "good_rate"]].copy()
        show.columns = ["游戏名", "类型", "售价（美元）", "拥有量区间", "好评数", "好评率"]
        show["类型"] = show["类型"].apply(lambda s: "、".join(cn(g) for g in str(s).split(";")))
        show["好评率"] = show["好评率"].apply(lambda v: "{:.1%}".format(v))
        st.dataframe(show, hide_index=True, use_container_width=True)

        # 受欢迎的游戏是不是也更耐玩
        play = get_play_subset()
        play = play.assign(is_hot=((play["positive_ratings"] >= th_ratings) &
                                   (play["good_rate"] >= th_rate)).astype(int))
        n_hot, n_oth = int(play["is_hot"].sum()), int((1 - play["is_hot"]).sum())
        if n_hot > 0 and n_oth > 0:
            md_hot = play[play.is_hot == 1]["median_playtime"].median()
            md_oth = play[play.is_hot == 0]["median_playtime"].median()
            st.divider()
            st.subheader("受欢迎的游戏是不是也更耐玩？")
            st.markdown("这个问题只能在有游玩时长记录的 **{:,} 款**游戏里回答 —— "
                        "其余游戏这项数据没有采集到，不能替它们下结论。".format(len(play)))
            g1, g2 = st.columns(2)
            g1.metric("最受欢迎的游戏 · 中位游玩时长",
                      "{:.0f} 分钟（{:.1f} 小时）".format(md_hot, md_hot / 60))
            g2.metric("其他游戏 · 中位游玩时长",
                      "{:.0f} 分钟（{:.1f} 小时）".format(md_oth, md_oth / 60),
                      delta="{:.1f} 倍".format(md_hot / md_oth) if md_oth else None)
            st.success("**受欢迎的游戏和其他游戏，玩家的游玩时长正好差一倍。**\n\n"
                       "这说明「受欢迎」不只是人多、评价好，玩家还愿意在它身上花更多时间。")
    else:
        st.warning("当前门槛太严，没有游戏能同时满足。试试把滑块往左拖。")

    st.divider()
    with st.expander("看一下默认门槛下的结果（好评数 ≥ 1 万 且 好评率 ≥ 90%）"):
        st.markdown(
            "27075 款游戏里，同时做到「玩的人多」和「评价好」的只有 **189 款，占 0.70%**。"
            "这个比例说明「受欢迎」在 Steam 上是一件相当难的事。\n\n"
            "这 189 款里还有一个明显的现象：**含多人模式的占 38.6%**（73 款），"
            "而全部游戏里含多人模式的只有 21.5% —— 头部明显更集中在多人游戏上。\n\n"
            "不过这只是**现象**，不能倒过来说「做多人模式就能火」。"
            "在建模页面有专门的验证，结论是：**多人模式和热度相关，但不能当作「做了多人就会火」的依据。**"
        )


# ============================================================
# 页面 5：机器学习建模
# ============================================================
elif page == "🤖 机器学习建模":
    st.title("🤖 机器学习建模")
    st.markdown("做了两个任务：**分类**（预测游戏会不会成为热门）和 **聚类**（把游戏自动分组）。")

    tab1, tab2, tab3, tab4 = st.tabs(["分类模型", "模型评估", "稳健性检验", "聚类分析"])

    # ---------- 分类模型 ----------
    with tab1:
        st.subheader("先说清楚要预测什么")
        st.markdown(
            "把「热门游戏」定义为：**拥有量 ≥ 10 万**（玩的人多）**且 好评率 ≥ 80%**（评价好）。"
            "两个条件必须同时满足，否则「只有三条评价全是好评」的小游戏也会被算成热门。\n\n"
            "建模样本剔除掉了评价数少于 10 条的游戏 —— 这些游戏的好评率非 0% 即 100%，"
            "波动极大，属于纯噪声。"
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("建模样本", "{:,} 款".format(MD["n_model_rows"]))
        k2.metric("其中热门", "{:,} 款".format(MD["n_hot"]))
        k3.metric("热门占比", "{:.2%}".format(MD["pos_rate"]))
        k4.metric("特征数量", "%d 个" % len(MD["features"]))

        st.markdown("#### 为什么选这三个模型")
        st.dataframe(pd.DataFrame([
            ["逻辑回归", "把特征加权求和，再压成 0~1 的概率",
             "最简单、最好解释，系数能直接看出每个特征的影响方向。用它当基准线"],
            ["随机森林", "造几百棵决策树，每棵独立判断，最后投票",
             "不要求线性关系，能给出特征重要性，回答「哪个因素最关键」"],
            ["KNN", "看和它最相似的 15 个游戏是不是热门",
             "不做任何假设、只靠相似度，用来当对照组"],
        ], columns=["模型", "它的思路", "为什么选它"]), hide_index=True, use_container_width=True)

        st.markdown("#### 三个模型的对比结果")
        st.dataframe(pd.DataFrame([
            [name,
             "%.4f ± %.4f" % (v["cv_mean"], v["cv_std"]),
             round(v["auc"], 4), round(v["acc"], 4), round(v["f1"], 4),
             round(v["recall"], 4), round(v["precision"], 4)]
            for name, v in MD["metrics"].items()
        ], columns=["模型", "5折交叉验证 AUC", "测试集 AUC", "准确率", "F1", "召回率", "精确率"]),
            hide_index=True, use_container_width=True)
        st.success("**随机森林表现最好**：测试集 AUC %.4f。"
                   "5 折交叉验证的标准差只有 %.4f，说明结果稳定，不是碰运气。"
                   % (MD["metrics"]["随机森林"]["auc"], MD["metrics"]["随机森林"]["cv_std"]))

    # ---------- 模型评估 ----------
    with tab2:
        st.subheader("为什么不能只看准确率")
        st.markdown(
            "热门游戏只占 **{:.2%}**。一个什么都不学、直接把所有游戏都猜成「不热门」的模型，"
            "准确率也有 **{:.2%}**。".format(MD["pos_rate"], MD["baseline_acc"])
        )
        l1, l2, l3 = st.columns(3)
        l1.metric("傻瓜模型（全猜不热门）· 准确率", "{:.2%}".format(MD["baseline_acc"]),
                  help="一个热门游戏都没找出来")
        l2.metric("随机森林 · 准确率", "{:.2%}".format(MD["metrics"]["随机森林"]["acc"]),
                  delta="低于基线", delta_color="inverse")
        l3.metric("随机森林 · 召回率", "{:.2%}".format(MD["metrics"]["随机森林"]["recall"]),
                  help="100 个真正的热门游戏，能找出多少个")
        st.warning(
            "**随机森林的准确率反而比「什么都不做」还低**，这不是模型差 —— "
            "它敢于预测热门，必然会犯一些错。真正该看的是 **AUC 和召回率**："
            "100 个真正的热门游戏，它能找出 {:.0f} 个。\n\n"
            "反过来看 KNN：准确率 {:.2%} 是三个模型里最高的，但召回率只有 {:.2%}，"
            "几乎找不出热门游戏。**这就是只看准确率的陷阱。**".format(
                MD["metrics"]["随机森林"]["recall"] * 100,
                MD["metrics"]["KNN"]["acc"], MD["metrics"]["KNN"]["recall"])
        )

        st.divider()
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("#### ROC 曲线")
            fig, ax = plt.subplots(figsize=(6.5, 6))
            for name, d in MD["roc"].items():
                ax.plot(d["fpr"], d["tpr"],
                        label="{}（AUC = {:.3f}）".format(name, MD["metrics"][name]["auc"]))
            ax.plot([0, 1], [0, 1], "k--", label="随机猜测（AUC = 0.5）")
            ax.set_xlabel("假正例率（把不热门的误判成热门）")
            ax.set_ylabel("真正例率（真正的热门被找出来）")
            ax.set_title("三个模型的 ROC 曲线对比")
            ax.legend(fontsize=9)
            fig.tight_layout()
            show_fig(fig)
        with cc2:
            st.markdown("#### 混淆矩阵（随机森林）")
            cm = np.array(MD["cm_rf"])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["预测-不热门", "预测-热门"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["真实-不热门", "真实-热门"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, "{:,}".format(cm[i, j]), ha="center", va="center",
                            fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "#333")
            ax.set_title("混淆矩阵")
            fig.tight_layout()
            show_fig(fig)
            st.caption("右上格是误报（模型说热门、其实不热门），左下格是漏报（真的热门没找出来）。")

        st.divider()
        st.markdown("#### 特征重要性：什么因素在决定一款游戏能不能火？")
        imp = MD["importance"]
        names = [cn(k) for k in imp.keys()][::-1]
        vals = list(imp.values())[::-1]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(names, vals, color="teal")
        for i, v in enumerate(vals):
            ax.text(v, i, "  %.4f" % v, va="center", fontsize=9)
        ax.set_xlabel("重要性")
        ax.set_title("随机森林：影响「是否成为热门」的前 12 个特征")
        fig.tight_layout()
        show_fig(fig)

        st.markdown(
            "**发行年份**（{:.4f}）和**价格**（{:.4f}）遥遥领先，"
            "而所有具体类型、是否免费、是否含多人模式的重要性都不到 0.04。\n\n"
            "**这说明对开发者来说，真正最可控、影响也最大的，是「定价」和「上市时机」——"
            "而这两项恰好是最容易调整的。**".format(imp.get("year", 0), imp.get("price_log", 0))
        )

        with st.expander("一个容易被误读的地方：「多人模式」到底有没有用？"):
            ms = MD["multi_single"]
            st.markdown(
                "单看数据，含多人模式的游戏热门率是 **{:.2%}**，纯单人是 **{:.2%}**，"
                "高出 **{:.1f} 倍**，看起来很有用。\n\n"
                "但这里藏着一个疑点 —— 多人游戏往往同时是免费或低价、发行也更早，"
                "它的优势可能来自这些因素，而不是「多人」本身。"
                "所以**控制住价格和年份，在每一个小组内部再比一次**：".format(
                    ms["r1"], ms["r0"], ms["r1"] / ms["r0"])
            )
            ctrl = pd.DataFrame(MD["multi_ctrl"],
                                columns=["价格档", "年份段", "样本数", "含多人热门率", "纯单人热门率"])
            ctrl["倍数"] = (ctrl["含多人热门率"] / ctrl["纯单人热门率"]).round(2)
            ctrl["含多人热门率"] = ctrl["含多人热门率"].apply(lambda v: "{:.1%}".format(v))
            ctrl["纯单人热门率"] = ctrl["纯单人热门率"].apply(lambda v: "{:.1%}".format(v))
            st.dataframe(ctrl, hide_index=True, use_container_width=True)
            st.markdown(
                "可以看到，优势从 1.5 倍缩到 1.0~1.4 倍，**在免费档甚至反转**（0.4~0.7 倍）"
                "—— 因为免费区里充斥着大量蹭热度的劣质多人游戏。\n\n"
                "**所以准确的表述是：多人模式和热度相关，但不能当作「做了多人就会火」的依据。**"
            )

    # ---------- 稳健性检验 ----------
    with tab3:
        st.subheader("模型学到的是真规律，还是「时间累计效应」？")
        st.markdown(
            "「发行年份」的重要性排第一，但有个疑点必须排除：**拥有量是累计值** —— "
            "2010 年发行的游戏比 2019 年发行的游戏多积累了 9 年，拥有量自然更高。\n\n"
            "所以「年份重要」有可能只是统计口径造成的假象。"
            "**检验方法**：只保留 2015 年及以后发行的游戏（大家积累时间差不多），重新建模。"
        )
        rb = MD["robust"]
        st.dataframe(pd.DataFrame([
            ["全部年份", 20511, round(MD["metrics"]["随机森林"]["auc"], 4),
             round(MD["metrics"]["随机森林"]["recall"], 4)],
            ["仅 2015 年及以后", rb["n"], round(rb["auc"], 4), round(rb["recall"], 4)],
            ["仅 2015 年及以后（去掉年份特征）", rb["n"], round(rb["auc_no_year"], 4), "—"],
        ], columns=["数据集", "样本量", "测试集 AUC", "召回率"]), hide_index=True, use_container_width=True)

        st.markdown("#### 三点结论")
        st.markdown(
            "1. **只比「同一代」的游戏，模型依然有效**（AUC {:.4f}，仍在 0.8 以上），"
            "说明它捕捉到的不只是时间累计效应。\n"
            "2. **去掉年份特征之后 AUC 掉到 {:.4f}**，说明年份在同期游戏内部仍然有区分能力。\n"
            "3. **在同期子集里，价格的重要性上升到第一位。**"
            "这进一步说明：定价才是开发者最可控、也最有效的着力点。".format(rb["auc"], rb["auc_no_year"])
        )
        st.dataframe(pd.DataFrame(
            [[cn(k), round(v, 4)] for k, v in rb["importance_top"].items()],
            columns=["2015 年后子集里最重要的特征", "重要性"]
        ), hide_index=True, use_container_width=True)
        st.info("为什么要专门做这一步？因为如果不检验，「年份重要」这个结论就无法区分"
                "是真实规律还是统计假象。**主动质疑自己的结论，比多跑一个模型更有价值。**")

    # ---------- 聚类 ----------
    with tab4:
        st.subheader("不用标签，让算法自己分组")
        st.markdown(
            "分类是**有监督**的 —— 用我们自己定义的「热门」标签去学，万一标准定得不好，学到的也是错的。\n\n"
            "聚类是**无监督**的 —— 完全不给标签，只按数据本身的相似程度分组。"
            "**如果聚类的结果和分类的结论对得上，就说明结论不是靠「自己定义的标准」硬凑出来的。**"
        )

        st.markdown("#### 怎么确定分几类？")
        st.dataframe(pd.DataFrame(
            [[d["k"], round(d["inertia"], 1), round(d["silhouette"], 4)] for d in MD["k_scan"]],
            columns=["k（簇数）", "inertia（越小越紧凑）", "轮廓系数（越接近 1 越好）"]
        ), hide_index=True, use_container_width=True)

        ks = MD["k_scan"]
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax1.plot([d["k"] for d in ks], [d["inertia"] for d in ks], marker="o", color="steelblue")
        ax1.set_xlabel("簇的数量 k")
        ax1.set_ylabel("inertia（越小越紧凑）", color="steelblue")
        ax2 = ax1.twinx()
        ax2.plot([d["k"] for d in ks], [d["silhouette"] for d in ks], marker="s", color="crimson")
        ax2.set_ylabel("轮廓系数（越大越好）", color="crimson")
        ax1.set_title("如何确定簇的数量 k")
        fig.tight_layout()
        show_fig(fig)

        st.markdown(
            "**inertia 从头到尾一直在下降**，所以它本身给不出明确答案；"
            "**轮廓系数是有高有低的**，最高在 k=2。\n\n"
            "**为什么最终取 k=3，而不是轮廓系数最高的 k=2？**"
            "因为 k=2 只是把游戏粗暴地分成「热门」和「长尾」两类，信息量有限；"
            "而 k=3 的轮廓系数仍在可用范围内（一般认为大于 0.25 就说明分群有效），"
            "并且三个簇在业务上含义清晰、互相区分。"
        )

        st.divider()
        st.markdown("#### 三个簇长什么样")
        prof = MD["cluster_profile"]
        names3 = {"0": "高人气 · 高口碑", "1": "小众 · 高口碑", "2": "低价 · 低口碑"}
        st.dataframe(pd.DataFrame([
            [k, names3[k], "{:,.0f}".format(prof[k]["n"]), "%.3f" % prof[k]["rate"],
             "{:,.0f}".format(prof[k]["pos"]), "{:,.0f}".format(prof[k]["own"]),
             "%.2f" % prof[k]["price"]]
            for k in ["0", "1", "2"]
        ], columns=["簇", "特点", "游戏数量", "平均好评率", "好评数中位数",
                    "拥有量中位数", "平均售价"]), hide_index=True, use_container_width=True)

        pc1, pc2 = st.columns([3, 2])
        with pc1:
            pca = MD["pca"]
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(pca["x"], pca["y"], c=pca["c"], cmap="viridis", s=7, alpha=0.5)
            ax.set_xlabel("主成分 1")
            ax.set_ylabel("主成分 2")
            ax.set_title("聚类结果（PCA 降到 2 维展示）")
            plt.colorbar(sc, ax=ax, label="簇编号")
            fig.tight_layout()
            show_fig(fig)
            st.caption("四个指标没法画在平面上，用 PCA 压成两维；颜色代表每款游戏属于哪一群。")
        with pc2:
            st.markdown("**每个簇的代表游戏**")
            for k in ["0", "1", "2"]:
                st.markdown("**簇 %s · %s**" % (k, names3[k]))
                for g in MD["cluster_top"][k]:
                    st.markdown("- " + g)

        st.success(
            "**聚类和分类的结论互相印证了。** 分类模型说「价格是影响能否成为热门的重要因素」，"
            "聚类则发现「低价那一簇的口碑最差」—— 有监督和无监督两条完全不同的技术路线，"
            "指向了同一个结论。这比单独一个模型的结果更有说服力。"
        )

    # ---------- 交互式预测 ----------
    st.divider()
    st.subheader("🎯 试一试：给一款游戏，预测它成为热门的概率")
    st.caption("下面用的是本项目训练好的逻辑回归模型。拖动滑块，看看概率怎么变。")

    p1, p2, p3 = st.columns(3)
    in_price = p1.slider("售价（美元）", 0.0, 60.0, 9.99, 0.01)
    in_year = p2.slider("发行年份", 1997, 2019, 2016, 1)
    in_ach = p3.checkbox("有成就系统", value=True)
    in_multi = p3.checkbox("含多人模式", value=True)

    sel = st.multiselect("游戏类型（可多选）", options=MD["top_genres"],
                         default=["Indie", "Action"], format_func=cn)

    # 用训练好的系数算概率：先标准化，再线性组合，最后过 sigmoid
    feats = MD["features"]
    x = {f: 0.0 for f in feats}
    x["price_log"] = float(np.log1p(in_price))
    x["is_free"] = 1.0 if in_price == 0 else 0.0
    x["year"] = float(in_year)
    x["has_multi"] = 1.0 if in_multi else 0.0
    x["has_ach"] = 1.0 if in_ach else 0.0
    x["genre_count"] = float(len(sel))
    for g in sel:
        if g in x:
            x[g] = 1.0

    z = MD["lr"]["intercept"]
    for i, f in enumerate(feats):
        z += MD["lr"]["coef"][i] * (x[f] - MD["lr"]["mean"][i]) / MD["lr"]["scale"][i]
    prob = 1.0 / (1.0 + np.exp(-z))

    r1, r2 = st.columns([1, 2])
    r1.metric("预测：成为热门的概率", "{:.1%}".format(prob))
    r2.progress(min(max(prob, 0.0), 1.0))
    if prob >= 0.5:
        st.success("模型判断：**大概率会成为热门**（概率 {:.1%}）".format(prob))
    elif prob >= 0.2:
        st.info("模型判断：**有一定机会**（概率 {:.1%}）".format(prob))
    else:
        st.warning("模型判断：**大概率不会成为热门**（概率 {:.1%}）".format(prob))
    st.caption("提醒：全样本中真正成为热门的游戏只占 {:.2%}，所以概率偏低是正常现象。"
               "这个模型只用了「游戏上线时就能知道」的信息，"
               "没有使用好评数、差评数等结果类字段（那会造成数据泄漏）。".format(MD["pos_rate"]))


# ============================================================
# 页面 6：关于本项目
# ============================================================
else:
    st.title("ℹ️ 关于本项目")
    st.markdown(
        "这是一份课程实训作业的交互式呈现，"
        "基于和鲸社区（Heywhale）公开数据集 **Steam游戏数据分析**，"
        "用 Python 完成从数据清洗到机器学习建模的全流程。"
    )

    st.subheader("分析流程")
    st.markdown(
        "1. **数据质量检查** —— 检查缺失值、重复行和异常的 0，"
        "其中识别出游玩时长里 77.2% 的「假 0」是最关键的一步\n"
        "2. **数据清洗** —— 去重、计算好评率、价格分档、拥有量区间转数值\n"
        "3. **探索性分析** —— 三张图分别回答价格、类型和「最受欢迎」三个问题\n"
        "4. **机器学习建模** —— 分类（逻辑回归 / 随机森林 / KNN）+ 聚类（K-Means）\n"
        "5. **稳健性检验** —— 用同期样本排除「时间累计效应」造成的假象"
    )

    st.subheader("三条主要结论")
    st.markdown(
        "- **类型不是关键**：各类型的平均好评率只差几个百分点，都紧贴整体均值\n"
        "- **便宜不等于好**：0-5 美元区间好评率最低（68.4%），10-20 美元最高（75.7%）\n"
        "- **最受欢迎很难**：只有 0.70% 的游戏同时做到「好评数过万」和「好评率 90% 以上」"
    )

    st.subheader("这个项目最值得说的部分")
    st.markdown(
        "不是模型精度，而是**贯穿全程的自我质疑**：\n\n"
        "1. 识别出常规检查发现不了的**「假 0」** —— "
        "把 0 当真实值算出的中位游玩时长是 0 分钟，排除后是 228 分钟，差了一个量级\n"
        "2. 把「有没有游玩时长数据」当作强特征时，AUC 一度达到 0.92，"
        "但判定它是**数据泄漏**（那个字段不是游戏的属性），主动删掉，AUC 回落到真实的 0.84\n"
        "3. 发现「多人模式」的结论前后不一致，于是控制价格和年份重新验证，修正了原来的说法\n"
        "4. 担心「年份重要」只是拥有量累计造成的假象，用同期样本做了**稳健性检验**\n\n"
        "**这些事情都不会让程序报任何错，只能靠对每一步的结论保持怀疑才能发现。**"
    )

    st.subheader("数据来源")
    st.markdown(
        "本数据集取自**和鲸社区（Heywhale）**，由用户「爱打字的程序员」整理发布。\n\n"
        "- 数据集页面：https://www.heywhale.com/mw/dataset/6a166f8230366a83c0b27a69\n"
        "- 原始数据：Steam 平台公开数据，整合了 SteamSpy 等第三方游戏数据统计接口\n"
        "- 数据规模：27,075 行 × 18 列，覆盖 1997–2019 年\n"
        "- 主要内容：游戏基础信息、运营属性、玩家反馈与市场表现"
    )

    st.subheader("技术栈")
    st.markdown("Python · pandas · matplotlib · scikit-learn · Streamlit")
