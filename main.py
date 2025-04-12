from datetime import datetime
import json
from typing import Annotated
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

from sklearn import base
import dordor
import models


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/v1/recommended')
def get_rec(baseData: Annotated[models.GetRecommendedInput, Body(embed=True)]):
    print(baseData)
    generatedData =dordor.generate_features(baseData.users, dordor.medicine, dordor.medicinedisease, dordor.disease, baseData.history)
    recommendedProducts = dordor.get_recom(generatedData)
    recommendedAmounts = dordor.calculate_recom(recommendedProducts, baseData.userStocks)
    print(recommendedProducts, recommendedAmounts)
    return JSONResponse({'data': [{'id':a, 'amount':b} for (a,b) in zip(recommendedProducts, recommendedAmounts)]})


    
# @app.get('/v1/inventory')
# def get_inventory(username: str):
#     return JSONResponse(prediction.get_inventory(username))

# @app.post('/v1/inventory/predict')
# def predict_by_inventory(predictionInput: Annotated[PredictionInternalInput, Body(embed=True)]):
#     try:
#         prediction_output = prediction.infer_stockout(predictionInput)
#         return JSONResponse({'status':"success", 'data':prediction_output})
#     except Exception as e:
#         return JSONResponse({'status':'error', 'error': str(e)}, status_code=500)



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080)
