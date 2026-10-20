# -*- coding: utf-8 -*-
"""
关灯科技 · 拣货到位缺货预警 —— 可复现分析脚本

用法:
    python analyze.py                    # 默认读 ./pick_tasks.csv
    python analyze.py path/to/pick_tasks.csv

只用本题发放的数据，不补造数据。
"""
import sys
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "pick_tasks.csv"

# 业务口径（任务书给定）
LOSS_PER_SHORTAGE = 75      # 一次到位缺货损失 75 分钟
COST_PER_PREWARN = 12       # 提前标记缺货一次花费 12 分钟

pd.set_option("display.width", 200)


def load(path):
    df = pd.read_csv(path)
    df = df.rename(columns=lambda s: s.strip())
    return df


def add_time_features(df):
    """★ 关键：pick_time 带 +08:00，sync_time 带 Z，必须先归一化到 UTC 再相减。

    直接按字符串墙钟相减会得到约 497 分钟的负值（差一个 8 小时的时区偏移）。
    """
    df["pick_utc"] = pd.to_datetime(df["pick_time"], format="ISO8601", utc=True)
    df["sync_utc"] = pd.to_datetime(df["sync_time"], format="ISO8601", utc=True)
    # 库存数据陈旧度 = 拣货时刻 - 最后一次库存同步时刻（分钟）
    df["stock_age_min"] = (df["pick_utc"] - df["sync_utc"]).dt.total_seconds() / 60
    return df


def clean(df):
    """去完全重复 + 剔除结果未回传（pick_status 空值）。"""
    n0 = len(df)
    d = df.drop_duplicates()
    n1 = len(d)
    miss = int(d["pick_status"].isna().sum())
    c = d.dropna(subset=["pick_status"]).copy()
    c["pick_status"] = c["pick_status"].astype(int)
    print(f"[清洗] 原始 {n0} → 去重 {n1}（删 {n0-n1} 条完全重复）→ 剔未回传 {len(c)}（删 {miss} 条）")
    return c


def shortage_rate_table(df, col, bins, labels, title):
    df["_bin"] = pd.cut(df[col], bins, labels=labels)
    g = df.groupby("_bin", observed=True).agg(
        任务数=("task_id", "size"),
        缺货数=("pick_status", lambda s: int((s == 0).sum())),
        缺货率=("pick_status", lambda s: round(100 * (1 - s.mean()), 2)),
    )
    print(f"\n--- {title} ---")
    print(g.to_string())


def cost_at_threshold(df, th):
    """按陈旧度阈值 th 预警：命中=真缺货，误报=其实有货。"""
    warn = df[df["stock_age_min"] > th]
    tp = int((warn["pick_status"] == 0).sum())
    fp = int((warn["pick_status"] == 1).sum())
    base = int((df["pick_status"] == 0).sum()) * LOSS_PER_SHORTAGE
    saved = tp * LOSS_PER_SHORTAGE - (tp + fp) * COST_PER_PREWARN
    precision = 100 * tp / max(tp + fp, 1)
    return dict(阈值=f"> {th} 分钟", 预警数=len(warn), 预警占比=f"{100*len(warn)/len(df):.1f}%",
                命中=tp, 误报=fp, 精确率=f"{precision:.1f}%",
                节省分钟=saved, 节省占比=f"{100*saved/base:.1f}%")


def main():
    RAW = load(CSV)
    RAW = add_time_features(RAW)

    print("=" * 78)
    print("0. 时区陷阱（必须先处理，否则后面全是错的）")
    naive = (pd.to_datetime(RAW["pick_time"].str.replace(r"\+08:00", "", regex=True))
             - pd.to_datetime(RAW["sync_time"].str.replace("Z", "", regex=False))
             ).dt.total_seconds() / 60
    print(f"   墙钟直接相减 : 均值 {naive.mean():.1f} 分钟（全为负，差 8 小时时区）")
    print(f"   UTC 归一化后 : 均值 {RAW['stock_age_min'].mean():.1f} 分钟，"
          f"中位 {RAW['stock_age_min'].median():.0f}，最大 {RAW['stock_age_min'].max():.0f}")
    print(f"   → 两者相差 {naive.mean() - RAW['stock_age_min'].mean():.0f} 分钟 ≈ 8 小时")

    df = clean(RAW)
    print(f"   缺货 {int((df.pick_status==0).sum())} / 成功 {int((df.pick_status==1).sum())}，"
          f"整体缺货率 {100*(df.pick_status==0).mean():.2f}%")

    # 反例：错误口径
    wrong = (int((RAW.pick_status == 0).sum()) + int(RAW.pick_status.isna().sum())) / len(RAW)
    print(f"\n1. 口径错误的影响")
    print(f"   错误口径（不去重 + 未回传当失败）缺货率 {100*wrong:.2f}%")
    print(f"   正确口径（去重 + 剔除未回传）     缺货率 {100*(df.pick_status==0).mean():.2f}%")
    print(f"   → 虚高 {100*wrong - 100*(df.pick_status==0).mean():.2f} 个百分点")

    print("\n2. 库存数据陈旧度 → 缺货率（核心发现）")
    shortage_rate_table(df, "stock_age_min", [0, 5, 10, 15, 20, 30, 45, 75],
                        ["0-5", "5-10", "10-15", "15-20", "20-30", "30-45", "45-75"],
                        "按库存陈旧度分箱")

    print("\n3. 账面库存 → 缺货率（证明它不是原因）")
    shortage_rate_table(df, "stock_level", [-1, 3, 6, 10, 15, 20, 30],
                        ["0-3", "4-6", "7-10", "11-15", "16-20", "21-30"],
                        "按账面可用库存分箱")

    print("\n4. 交叉验证：陈旧度 × 账面库存")
    df["age_hi"] = df["stock_age_min"] > 20
    df["lv_bin"] = pd.cut(df["stock_level"], [-1, 3, 6, 10, 15, 20, 30],
                          labels=["0-3", "4-6", "7-10", "11-15", "16-20", "21-30"])
    piv = df.pivot_table(index="age_hi", columns="lv_bin", values="pick_status",
                         aggfunc=lambda s: round(100 * (1 - s.mean()), 2), observed=True)
    piv.index = ["陈旧<=20min", "陈旧>20min"]
    print("   缺货率(%):")
    print(piv.to_string())
    pivn = df.pivot_table(index="age_hi", columns="lv_bin", values="task_id",
                          aggfunc="size", observed=True)
    pivn.index = ["陈旧<=20min", "陈旧>20min"]
    print("   样本数:")
    print(pivn.to_string())

    print("\n5. 预警策略成本核算")
    print(f"   口径：缺货 {LOSS_PER_SHORTAGE} 分钟/次，提前标记 {COST_PER_PREWARN} 分钟/次")
    base = int((df.pick_status == 0).sum()) * LOSS_PER_SHORTAGE
    print(f"   不预警基线：{int((df.pick_status==0).sum())} 次缺货 × {LOSS_PER_SHORTAGE} = {base} 分钟")
    rows = [cost_at_threshold(df, t) for t in (10, 15, 18, 20, 22, 25, 30)]
    print(pd.DataFrame(rows).to_string(index=False))
    breakeven = COST_PER_PREWARN / (COST_PER_PREWARN + LOSS_PER_SHORTAGE) * 100
    print(f"   ★ 盈亏平衡点：精确率需 > {breakeven:.1f}%（"
          f"{COST_PER_PREWARN}/{COST_PER_PREWARN}+{LOSS_PER_SHORTAGE}）")

    print("\n6. 按仓库 / 品类")
    w = df.groupby("warehouse").agg(
        任务数=("task_id", "size"),
        缺货率=("pick_status", lambda s: round(100 * (1 - s.mean()), 2)),
        平均陈旧=("stock_age_min", "mean")).round(1)
    print(w.sort_values("缺货率", ascending=False).to_string())
    c = df.groupby("category").agg(
        任务数=("task_id", "size"),
        缺货率=("pick_status", lambda s: round(100 * (1 - s.mean()), 2)))
    print("\n" + c.sort_values("缺货率", ascending=False).to_string())


if __name__ == "__main__":
    main()
