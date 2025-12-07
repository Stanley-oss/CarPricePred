from datetime import datetime

from models.catmodel.predict_price import predict_price  # 复用你现成的预测函数


def _ask(prompt, cast=str, allow_empty=True):
    """简单的输入助手：支持留空、自动类型转换"""
    while True:
        s = input(prompt).strip()
        if s == "" and allow_empty:
            return None
        if cast is str:
            return s
        try:
            return cast(s)
        except Exception:
            print("输入格式不对，请重新输入。")


def _norm_fuel(x):
    if x is None:
        return None
    s = x.strip().lower()
    if s in ["petrol", "gasoline", "p", "汽油", "油"]:
        return "Petrol"
    if s in ["diesel", "d", "柴油"]:
        return "Diesel"
    return "Other"


def _norm_gear(x):
    if x is None:
        return None
    s = x.strip().lower()
    if s in ["a", "auto", "automatic", "at", "自动"]:
        return "Automatic"
    if s in ["m", "man", "manual", "mt", "手动"]:
        return "Manual"
    return "Unknown"


def interactive_loop():
    print("\n=== 二手车价格预测（交互式） ===")
    print("提示：直接回车 = 未知/默认\n")

    while True:
        # 基本信息
        brand = _ask("品牌（如 Toyota）：")
        model = _ask("型号（如 Corolla）：")

        year = _ask("年份（如 2019）：", float)
        age = _ask("车龄（年，可留空由程序用年份自动算）：", float)

        milage = _ask("行驶里程（km，如 45000）：", float)
        engine = _ask("排量（cc，如 1798）：", float)
        max_power = _ask("最大马力（bhp，如 138）：", float)
        seats = _ask("座椅数（如 5）：", float)

        # 枚举字段
        fuel_raw = _ask("燃料类型（Petrol/Diesel/Other，可写中文）：")
        fuel_type = _norm_fuel(fuel_raw)

        gear_raw = _ask("变速箱（Automatic/Manual，可写中文/AT/MT）：")
        transmission = _norm_gear(gear_raw)

        # 可选：挂牌日期，用于 period，更精准的市场系数
        date_raw = _ask("挂牌日期（YYYY-MM-DD，可留空，用年份代替）：")
        listing_date = None
        if date_raw:
            try:
                listing_date = datetime.fromisoformat(date_raw).date().isoformat()
            except Exception:
                print("日期格式不符合 YYYY-MM-DD，已忽略。")
                listing_date = None

        payload = {
            "brand": brand,
            "model": model,
            "year": year,
            "age": age,
            "milage": milage,
            "fuel_type": fuel_type,
            "engine": engine,
            "max_power": max_power,
            "transmission": transmission,
            "seats": seats,
            "listing_date": listing_date,
        }

        # 调用你的模型
        out = predict_price(payload)

        # 展示结果
        print("\n--- 预测结果 ---")
        print(f"点预测价格 P50：{out['p50']:,.2f}")
        print(f"价格区间     ：[{out['lo']:,.2f} , {out['hi']:,.2f}]")
        print(f"相对区间宽度 WR=(hi-lo)/P50：{out['wr']:.3f}")
        print(f"分组键       ：{out['group_key']}")
        print(f"period / bin ：{out['period']} / {out['period_bin']}")
        print(f"市场系数 M_t ：{out['market_multiplier']:.4f}")
        # 可选：提示一下原始区间有多宽
        if "wr_raw" in out and out["wr_raw"] > out["wr"]:
            print(f"(内部原始区间 WR_raw≈{out['wr_raw']:.3f}，已按 ±6万 截断展示)")

        # 是否继续
        cont = input("\n继续预测下一辆吗？(Y/n)：").strip().lower()
        if cont in ["n", "no", "q", "quit", "exit"]:
            break
        print("")


if __name__ == "__main__":
    interactive_loop()
