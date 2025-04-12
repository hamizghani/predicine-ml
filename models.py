from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Tokens(BaseModel):
    userId: int
    refresh: str
    access: str

class TransactionHistory(BaseModel):
    id: int
    createdAt: datetime
    updatedAt: datetime
    medicineId: int
    amount: int = 0
    userId: int
    price: int = 0


class GetRecommendedInput(BaseModel):
    history: list[TransactionHistory]
    users: list['User']
    userStocks: list['UserStock']
    

class DiseaseMedicineCorrelation(BaseModel):
    id: int
    diseaseId: int
    medicineId: int
    correlationPercentage: float

class DiseaseRecord(BaseModel):
    id: int
    diseaseId: int
    amount: int
    timestamp: datetime

class Disease(BaseModel):
    id: int
    name: str
    diseaseMedicineCorrelation: List[DiseaseMedicineCorrelation] = []
    diseaseRecords: List[DiseaseRecord] = []

class StockBatch(BaseModel):
    id: int
    expirationDate: datetime
    amount: int = 0
    userStockId: int

class ForecastedUserStock(BaseModel):
    id: int
    userStockId: int
    requiredStock: int
    percentage: int
    stockoutDate: datetime

class Medicine(BaseModel):
    id: int
    name: str
    description: str
    brief: str
    imageUrl: str

class UserStock(BaseModel):
    id: int
    medicineId: int
    total: int = 0
    sold: int = 0
    userId: int
    batches: List[StockBatch] = []
    forecastedUserStock: Optional[ForecastedUserStock] = None

class User(BaseModel):
    id: int
    username: str
    name: str
    passwordHash: str
    region: str
    sales: int = 0
    quantitySold: int = 0
    price: List[int] = []

# Relations
User.update_forward_refs()
UserStock.update_forward_refs()
Medicine.update_forward_refs()
Disease.update_forward_refs()
ForecastedUserStock.update_forward_refs()
TransactionHistory.update_forward_refs()
Tokens.update_forward_refs()
StockBatch.update_forward_refs()
DiseaseRecord.update_forward_refs()
DiseaseMedicineCorrelation.update_forward_refs()
