import os
import sys
from typing import Any


class CatboostCarPricePredictor:
    def __init__(self):
        self._model_loaded = False
        self._predictor_func = None
        self._lazy_load()

    def _lazy_load(self):
        if self._model_loaded:
            return

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)

            from models.catmodel.predict_price import predict_price

            self._predictor_func = predict_price
            self._model_loaded = True

        except ImportError as e:
            raise ImportError(f"Failed to import modules. Error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models in predict_price.py: {e}") from e

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._lazy_load()

        required_keys = ["brand", "model", "year"]
        missing_keys = [k for k in required_keys if k not in payload or payload[k] is None]

        if missing_keys:
            raise ValueError(f"Payload missing critical keys: {missing_keys}")

        try:
            result = self._predictor_func(payload)
            return result
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}") from e


if __name__ == "__main__":
    predictor = CatboostCarPricePredictor()

    payload = {
        "brand": "Toyota",
        "model": "Corolla",
        "year": 2019,
        "age": 4,
        "milage": 45000,
        "fuel_type": "Petrol",
        "engine": 1798,
        "max_power": 138,
        "transmission": "Automatic",
        "seats": 5,
    }

    try:
        out = predictor.predict(payload)
        for k, v in out.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"Error: {e}")
