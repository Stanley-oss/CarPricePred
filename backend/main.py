from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.catmodel.predictor import CatboostCarPricePredictor
from models.resnet.predictor import ResnetCarPricePredictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CarPayload(BaseModel):
    brand: str
    model: str
    year: int
    age: int
    milage: int
    fuel_type: str
    engine: int
    max_power: int
    transmission: str
    seats: int
    use_resnet: bool = False


try:
    catmodel_predictor = CatboostCarPricePredictor()
    resnet_predictor = ResnetCarPricePredictor()
except Exception as err:
    raise RuntimeError(f"Failed to load models: {err}") from err


def payload_to_dict(payload: CarPayload) -> dict:
    return {
        "brand": payload.brand,
        "model": payload.model,
        "year": payload.year,
        "age": payload.age,
        "milage": payload.milage,
        "fuel_type": payload.fuel_type,
        "engine": payload.engine,
        "max_power": payload.max_power,
        "transmission": payload.transmission,
        "seats": payload.seats,
    }


@app.post("/predict")
async def predict_price(payload: CarPayload):
    features = payload_to_dict(payload)
    print("Received payload:)")
    for item in features.items():
        print(item)
    print(f"Model type: {'resnet' if payload.use_resnet else 'catboost'}")

    try:
        if payload.use_resnet:
            price = resnet_predictor.predict_price(features)
            price = float(price)
            price_usd = price / 83.4  # to USD
            print(f"Resnet price: {price} -> to USD {price_usd}")
            return {
                "model_type": "resnet",
                "price": price_usd,
            }
        else:
            result = catmodel_predictor.predict(features)
            print("--- 预测结果 ---")
            print(f"点预测价格 P50\t\t\t：{result['p50']:,.2f}")
            print(f"价格区间\t\t\t：[{result['lo']:,.2f} , {result['hi']:,.2f}]")
            print(f"相对区间宽度 WR=(hi-lo)/P50\t：{result['wr']:.3f}")
            print(f"分组键\t\t\t\t：{result['group_key']}")
            print(f"period / bin\t\t\t：{result['period']} / {result['period_bin']}")
            print(f"市场系数 M_t\t\t\t：{result['market_multiplier']:.4f}")
            if "wr_raw" in result and result["wr_raw"] > result["wr"]:
                print(f"(内部原始区间 WR_raw≈{result['wr_raw']:.3f}，已按±6万截断展示)")

            result["p50"] /= 83.4  # to USD
            result["lo"] /= 83.4
            result["hi"] /= 83.4
            result["lo_raw"] /= 83.4
            result["hi_raw"] /= 83.4

            if hasattr(result, "to_dict"):
                result = result.to_dict()

            return {
                "model_type": "catboost",
                "result": result,
            }

    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {err}",
        ) from err


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
