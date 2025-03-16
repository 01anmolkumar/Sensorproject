import shutil
import os
import sys
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from flask import Request  # Corrected type hint
from src.constant import artifact_folder  # Ensure artifact_folder is properly imported
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "prediction_file.csv"
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)


class PredictionPipeline:
    def __init__(self, request: Request):  # Fixed Type Hint Issue
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def save_input_files(self) -> str:
        """
        Saves the uploaded input CSV file and returns its path.
        """
        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)

            input_csv_file.save(pred_file_path)

            logging.info(f"File {input_csv_file.filename} saved successfully at {pred_file_path}")
            return pred_file_path
        except Exception as e:
            logging.error("Error occurred while saving input file")
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        """
        Loads the model and preprocessor, transforms input features, and makes predictions.
        """
        try:
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(file_path=self.prediction_pipeline_config.preprocessor_path)

            transformed_x = preprocessor.transform(features)
            preds = model.predict(transformed_x)

            return preds
        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)

    def get_predicted_dataframe(self, input_dataframe_path: str) -> None:
        """
        Reads input CSV, makes predictions, and saves results in a new CSV file.
        """
        try:
            prediction_column_name: str = "prediction"
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)

            # Drop "Unnamed: 0" column if present
            if "Unnamed: 0" in input_dataframe.columns:
                input_dataframe = input_dataframe.drop(columns="Unnamed: 0")

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = predictions

            # Mapping numerical predictions to labels
            target_column_mapping = {0: 'bad', 1: 'good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index=False)

            logging.info(f"Predictions saved to {self.prediction_pipeline_config.prediction_file_path}")
        except Exception as e:
            logging.error("Error occurred while generating predictions")
            raise CustomException(e, sys)

    def run_pipeline(self) -> PredictionPipelineConfig:
        """
        Runs the full prediction pipeline: saving input, processing, and saving predictions.
        """
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_pipeline_config
        except Exception as e:
            logging.error("Error occurred in the prediction pipeline")
            raise CustomException(e, sys)
