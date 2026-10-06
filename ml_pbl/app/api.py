from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.decision_system import SupplyChainDecisionSystem
import uvicorn

app = FastAPI(
    title='Supply Chain Delay Prediction API',
    description='Production Decision Support System for Intelligent Shipment Management',
    version='1.0.0'
)

decision_system = None

@app.on_event('startup')
def load_model():
    global decision_system
    decision_system = SupplyChainDecisionSystem()

class ShipmentPayload(BaseModel):
    Shipping_Mode: str = Field(..., alias='Shipping Mode', example='Standard Class')
    Market: str = Field(..., example='Pacific Asia')
    Order_Region: str = Field(..., alias='Order Region', example='Southeast Asia')
    Customer_Segment: str = Field('Consumer', alias='Customer Segment', example='Consumer')
    Category_Name: Optional[str] = Field('Sporting Goods', alias='Category Name')
    Department_Name: Optional[str] = Field('Fitness', alias='Department Name')
    Sales: Optional[float] = Field(327.75, example=327.75)
    Order_Item_Quantity: Optional[int] = Field(1, alias='Order Item Quantity', example=1)
    Order_Item_Discount: Optional[float] = Field(13.11, alias='Order Item Discount', example=13.11)
    Benefit_per_order: Optional[float] = Field(91.25, alias='Benefit per order', example=91.25)
    Days_for_shipment_scheduled: Optional[int] = Field(4, alias='Days for shipment (scheduled)', example=4)

@app.get('/health')
def health_check():
    return {'status': 'healthy', 'model_loaded': decision_system is not None}

@app.post('/predict')
def predict_delay(shipment: Dict[str, Any]):
    if decision_system is None:
        raise HTTPException(status_code=503, detail='Model not yet loaded')
    try:
        result = decision_system.predict_shipment(shipment)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
