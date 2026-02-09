# main.py
from veritas_engine import VeritasEngine

def run_research_cycle():
    print("--- Starting Veritas Optimization Cycle ---")
    engine = VeritasEngine()

    engine.prepare_data()
    engine.optimize()
    
    print("--- Cycle Complete. Artifacts in /onnx_quantized ---")

if __name__ == "__main__":
    run_research_cycle()
