from models.catmodel.predictor import CatboostCarPricePredictor
from models.resnet.predictor import ResnetCarPricePredictor


def main():
    catmodel_pred = CatboostCarPricePredictor()
    resnet_pred = ResnetCarPricePredictor()

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

    catmodel_result = catmodel_pred.predict(payload)
    resnet_result = resnet_pred.predict_price(payload)

    print(f"Catboost Predictor Result: {catmodel_result}")
    print(f"Resnet Predictor Result: {resnet_result}")


if __name__ == "__main__":
    main()
