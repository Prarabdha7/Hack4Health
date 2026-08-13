import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TABULAR_FEATURES, STRESS_CLASSES
from fusion_model import predict

if len(sys.argv) != 21:
    print("Usage: python src/run_fusion.py IMAGE_PATH AUDIO_PATH " + " ".join(TABULAR_FEATURES)); raise SystemExit(1)
try:
    values=[float(x) for x in sys.argv[3:]]
except ValueError: print("All tabular values must be numeric."); raise SystemExit(1)
result=predict(sys.argv[1],sys.argv[2],values)
print("\nModality probabilities")
for name,p in (("Facial",result["facial"]),("Speech",result["speech"]),("Tabular",result["tabular"]),("Fused",result["fused"])):
    print(f"\n{name}")
    for cls,v in zip(STRESS_CLASSES,p): print(f"{cls}: {v:.4f}")
print(f"\nFinal Prediction: {result['prediction']}")
