# predict_cli.py
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from predict_price import predict_price  # 直接复用你已有的预测函数

def _ask(prompt, cast=str, allow_empty=True):
    while True:
        s = input(prompt).strip()
        if s == "" and allow_empty:
            return None
        if cast is str:
            return s
        try:
            return cast(s)
        except Exception:
            print("输入格式不对，请重输。")

def _norm_fuel(x):
    if x is None: return None
    s = x.strip().lower()
    if s in ["petrol", "gasoline", "p", "油", "汽油"]: return "Petrol"
    if s in ["diesel", "d", "柴油"]:                   return "Diesel"
    return "Other"

def _norm_gear(x):
    if x is None: return None
    s = x.strip().lower()
    if s in ["a", "auto", "automatic", "自动", "at"]: return "Automatic"
    if s in ["m", "man", "manual", "手动", "mt"]:     return "Manual"
    return "Unknown"

def interactive_loop():
    print("\n=== 二手车价格预测（交互式）===\n（留空回车 = 未知/默认）\n")

    while True:
        brand = _ask("品牌（如 Toyota）：")
        model = _ask("型号（如 Corolla）：")

        year  = _ask("年份（如 2019）：", float)
        age   = _ask("车龄（年；可留空让程序由年份自动算）：", float)

        milage    = _ask("里程（km，如 45000）：", float)
        engine    = _ask("排量（cc，如 1798）：", float)
        max_power = _ask("最大马力（bhp，如 138）：", float)
        seats     = _ask("座椅数（如 5）：", float)

        fuel_raw  = _ask("燃料类型（Petrol/Diesel/Other；可写中文）：")
        fuel_type = _norm_fuel(fuel_raw)

        gear_raw  = _ask("变速箱（Automatic/Manual；可写中文/AT/MT）：")
        transmission = _norm_gear(gear_raw)

        date_raw  = _ask("挂牌日期（YYYY-MM-DD，可留空，留空则用年份）：")
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

        out = predict_price(payload)

        print("\n--- 预测结果 ---")
        print(f"点预测 P50：{out['p50']:,.2f}")
        print(f"可信区间：[{out['lo']:,.2f} , {out['hi']:,.2f}]")
        print(f"相对宽度 WR=(hi-lo)/p50：{out['wr']:.3f}")
        print(f"分组键：{out['group_key']}")
        print(f"period：{out['period']}   市场系数 M_t：{out['market_multiplier']:.4f}")

        cont = input("\n继续预测下一辆吗？(Y/n)：").strip().lower()
        if cont in ["n", "no", "q", "quit", "exit"]:
            break
        print("")

def main():
    parser = argparse.ArgumentParser(description="二手车价格预测（命令行/交互两用）")
    parser.add_argument("--brand")
    parser.add_argument("--model")
    parser.add_argument("--year", type=float)
    parser.add_argument("--age", type=float)
    parser.add_argument("--milage", type=float)
    parser.add_argument("--fuel_type")
    parser.add_argument("--engine", type=float)
    parser.add_argument("--max_power", type=float)
    parser.add_argument("--transmission")
    parser.add_argument("--seats", type=float)
    parser.add_argument("--listing_date")  # YYYY-MM-DD
    args = parser.parse_args()

    # 如果没传任何参数 -> 进入交互式
    if not any(vars(args).values()):
        interactive_loop()
        return

    # 否则使用命令行参数直接预测
    payload = {
        "brand": args.brand,
        "model": args.model,
        "year": args.year,
        "age": args.age,
        "milage": args.milage,
        "fuel_type": _norm_fuel(args.fuel_type) if args.fuel_type else None,
        "engine": args.engine,
        "max_power": args.max_power,
        "transmission": _norm_gear(args.transmission) if args.transmission else None,
        "seats": args.seats,
        "listing_date": args.listing_date,
    }
    out = predict_price(payload)
    print("\n--- 预测结果 ---")
    for k, v in out.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
