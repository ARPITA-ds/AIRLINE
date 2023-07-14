import os
import sys
import pandas as pd
from airline.logger import logger
from airline.exception import AirlineException
from dataclasses import dataclass
from airline.utils.common import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        preprocessro_path = os.path.join("artifact/stage02_data_transformation/preprocessing", "preprocessing_obj.pkl")
        model_path = os.path.join("artifact/stage04_model_training/best_model", "best_model.pkl")

        processor = load_object(preprocessro_path)
        model = load_object(model_path)

        scaled = processor.transform(features)
        pred = model.predict(scaled)

        return pred


class CustomClass:
    def __init__(self, 
                  Age:int,
                  Flight_Distance:int, 
                  Inflight_wifi_service:int, 
                  Departure_Arrival_time_convenient:int, 
                  Ease_of_Online_booking:int,
                  Food_and_drink:int,
                  Online_boarding:int,  
                  Seat_comfort:int, 
                  Inflight_entertainment:int,
                  On_board_service:int, 
                  Leg_room_service:int,
                  Baggage_handling:int,
                  Checkin_service:int,
                  Inflight_service:int,
                  Cleanliness:int,                  
                  Gender:int,
                  Customer_Type:str,
                  Type_of_Travel:str,
                  Class:str,
                  ):
        self.Age = Age
        self.Flight_Distance = Flight_Distance
        self.Inflight_wifi_service = Inflight_wifi_service
        self.Departure_Arrival_time_convenient = Departure_Arrival_time_convenient
        self.Ease_of_Online_booking = Ease_of_Online_booking
        self.Food_and_drink = Food_and_drink
        self.Online_boarding = Online_boarding
        self.Seat_comfort = Seat_comfort
        self.Inflight_entertainment = Inflight_entertainment
        self.On_board_service = On_board_service
        self.Leg_room_service = Leg_room_service
        self.Baggage_handling = Baggage_handling
        self.Checkin_service = Checkin_service
        self.Inflight_service = Inflight_service
        self.Cleanliness = Cleanliness        
        self.Gender = Gender
        self.Customer_Type = Customer_Type
        self.Type_of_Travel = Type_of_Travel
        self.Class = Class
        
        
        
    logger.info("converting data into Dataframe")
    def get_data_DataFrame(self):
        try:
            custom_input = {
                'Age': [self.Age],
                'Flight Distance': [self.Flight_Distance],
                'Inflight wifi service':[self.Inflight_wifi_service],
                'Departure/Arrival time convenient':[self.Departure_Arrival_time_convenient],
                'Ease of Online booking':[self.Ease_of_Online_booking],                
                'Food and drink':[self.Food_and_drink],
                'Online boarding':[self.Online_boarding],
                'Seat comfort':[self.Seat_comfort],
                'Inflight entertainment':[self.Inflight_entertainment],
                'On-board service':[self.On_board_service],
                'Leg room service':[self.Leg_room_service],
                'Baggage handling':[self.Baggage_handling],
                'Checkin service':[self.Checkin_service],
                'Inflight service':[self.Inflight_service],
                'Cleanliness':[self.Cleanliness],                
                'Gender':[self.Gender],
                'Customer Type':[self.Customer_Type],
                'Type of Travel':[self.Type_of_Travel],
                "Class":[self.Class]
                


            }

            data = pd.DataFrame(custom_input)

            return data
        except Exception as e:
            raise AirlineException(e, sys)