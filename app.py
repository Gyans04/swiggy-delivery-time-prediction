from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import json
import joblib
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning

# set the output as pandas
set_config(transform_output='pandas')


class Data(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


def load_model(model_path):
    model = joblib.load(model_path)
    return model


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer


# columns to preprocess in data
num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]

nominal_cat_cols = ['weather', 'type_of_order', 'type_of_vehicle',
                     "festival", "city_type", "is_weekend", "order_time_of_day"]

ordinal_cat_cols = ["traffic", "distance_type"]

# --- Load model directly from the saved artifact ---
# We skip the MLflow model registry lookup here on purpose: the registry
# stores an *absolute path* to the artifact recorded at training time
# (e.g. C:/Users/.../mlruns/...), which breaks when the app is served from
# a different machine or a Docker container. Loading model.joblib directly
# avoids that path dependency entirely and is a common real-world pattern —
# MLflow's registry is still useful for tracking/versioning, just not as
# the runtime lookup every request depends on.
model = load_model("models/model.joblib")
print("Model loaded directly from models/model.joblib")

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ("regressor", model)
])

# create the app
app = FastAPI()


@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"


@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
    }, index=[0])

    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)
    # get the predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return predictions


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000)