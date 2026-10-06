import os
import joblib
import json
import pandas as pd
import numpy as np

class SupplyChainDecisionSystem:
    def __init__(self, model_path='models/best_model.joblib', 
                 preprocessor_path='models/preprocessor.joblib', 
                 metadata_path='models/metadata.json'):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError('Trained model artifacts not found. Run main.py first!')
            
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.threshold = self.metadata.get('optimal_threshold', 0.50)
        self.feature_names = self.preprocessor.feature_names
        
    def _assign_risk_tier(self, prob: float) -> str:
        if prob >= 0.80:
            return 'CRITICAL'
        elif prob >= 0.65:
            return 'HIGH'
        elif prob >= 0.45:
            return 'MEDIUM'
        else:
            return 'LOW'

    def predict_shipment(self, order_info: dict) -> dict:
        df_raw = pd.DataFrame([order_info])
        
        # Ensure order date
        if 'order_date' not in df_raw.columns:
            if 'order date (DateOrders)' in df_raw.columns:
                df_raw['order_date'] = pd.to_datetime(df_raw['order date (DateOrders)'])
            else:
                df_raw['order_date'] = pd.Timestamp.now()
                
        # Transform features
        trans_out = self.preprocessor.transform(df_raw, is_train=False)
        X_trans = trans_out[0] if isinstance(trans_out, tuple) else trans_out
        
        # Predict probability
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        prediction = 'DELAYED' if prob >= self.threshold else 'ON-TIME'
        
        confidence = prob if prediction == 'DELAYED' else (1.0 - prob)
        risk_level = self._assign_risk_tier(prob)
        
        shipment_id = order_info.get('Order Id', order_info.get('shipment_id', 'N/A'))
        
        return {
            'Shipment ID': shipment_id,
            'Prediction': prediction,
            'Probability': round(prob, 4),
            'Confidence': f'{confidence * 100:.1f}%',
            'Risk Level': risk_level,
            'Optimal Threshold Used': self.threshold,
            'Shipping Mode': order_info.get('Shipping Mode', 'Unknown'),
            'Destination Region': order_info.get('Order Region', 'Unknown'),
            'Destination Market': order_info.get('Market', 'Unknown')
        }
