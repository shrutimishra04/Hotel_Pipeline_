import joblib
import pandas as pd
from src.preprocess import clean_data, treat_outliers_iqr
from src.feature_engineering import engineer_features

model_path = r"C:\Users\Honey Upadhyay\OneDrive\Desktop\hotel_cancellation_assessment\models\hotel_cancellation_model.joblib"
model = joblib.load(model_path)

# ---- Raw Input example ----
input_data = {
    'Booking_ID': ['B11111'],
    'no_of_adults': [2],
    'no_of_children': [1],
    'no_of_weekend_nights': [1],
    'no_of_week_nights': [3],
    'type_of_meal_plan': ['Meal Plan 1'],
    'required_car_parking_space': [0],
    'room_type_reserved': ['Room_Type 1'],
    'lead_time': [90],
    'arrival_year': [2023],
    'arrival_month': [8],
    'arrival_date': [16],
    'market_segment_type': ['Online'],
    'repeated_guest': [0],
    'no_of_previous_cancellations': [0],
    'no_of_previous_bookings_not_canceled': [0],
    'avg_price_per_room': [120.0],
    'no_of_special_requests': [1],
    'booking_status': ['Not_Canceled']  # dummy target required for cleaning
}

df = pd.DataFrame(input_data)

# ---- Apply same steps as training ----
df = clean_data(df)
df = treat_outliers_iqr(df)
df = engineer_features(df)

# Remove target if exists (not needed for prediction)
if 'booking_status' in df.columns:
    df = df.drop(columns=['booking_status'])

# ---- Predict ----
prediction = model.predict(df)
print("\nPrediction:", "Cancelled ❌" if prediction[0] == 1 else "Not Cancelled ✅")


