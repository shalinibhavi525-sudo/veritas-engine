from veritas_engine import VeritasEngine
import os

def main():
    engine = VeritasEngine()

    if not os.path.exists('train.tsv'):
        print("❌ Error: train.tsv not found. Please download the LIAR dataset.")
        return

    dataset = engine.prepare_data()

    print("--- Phase 1: Training ---")
    engine.run_training(dataset, epochs=1)

    print("--- Phase 2: Squeezing ---")
    engine.squeeze_model()

    print("--- Phase 3: Testing Veritas ---")
    sample = "Scientists discovered giants living on Mars."
    result = engine.predict(sample)
    print(f"Claim: {sample}")
    print(f"Result: {result['label']} ({result['confidence']:.2%})")
    print(f"Inference Time: {result['latency_ms']:.2f}ms")

if __name__ == "__main__":
    main()
