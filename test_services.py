import sys
sys.path.insert(0, '.')
# Test imports
from services.preprocessing_service import PreprocessingService, FEATURE_ORDER
from services.prediction_service import PredictionService
import pandas as pd

print('Testing preprocessing...')
prep = PreprocessingService()
test_input = {
    'N_Days': 1500,
    'Age_Years': 50,
    'Bilirubin': 2.0,
    'Cholesterol': 300.0,
    'Albumin': 3.5,
    'Copper': 90.0,
    'Alk_Phos': 1000.0,
    'SGOT': 120.0,
    'Tryglicerides': 120.0,
    'Platelets': 250.0,
    'Prothrombin': 10.5,
    'Status': 'C',
    'Drug': 'Placebo',
    'Sex': 'F',
    'Ascites': 'N',
    'Hepatomegaly': 'N',
    'Spiders': 'N',
    'Edema': 'N',
}
processed = prep.preprocess_single(test_input)
print('Processed DataFrame:')
print(processed)
print('Age in days:', processed['Age'].values[0])
print()

print('Testing prediction...')
pred_svc = PredictionService()
result = pred_svc.predict_single(processed)
print('Predicted stage:', result['predicted_stage'])
print('Confidence:', f"{result['confidence']:.2f}%")
print('Probabilities:', result['probabilities'])
print()

print('Testing batch...')
df = pd.read_csv('data/liver_cirrhosis.csv').drop('Stage', axis=1).head(5)
processed_batch, warnings = prep.preprocess_batch(df.copy())
batch_results = pred_svc.predict_batch(processed_batch)
print('Batch results:')
print(batch_results)
print()
print('All tests PASSED!')
