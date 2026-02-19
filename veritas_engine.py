import os
import time
import shutil
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import evaluate

class VeritasEngine:
    """
    VeritasEngine: A pipeline for fine-tuning and hardware-aware optimization
     of Transformer models for misinformation detection.
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trained_temp_path = "./veritas_trained_temp"
        self.onnx_path = "onnx_quantized"

    def prepare_data(self, train_path='train.tsv', val_path='valid.tsv'):
        """Remaps LIAR dataset labels to binary classification."""
        col_names = ['id', 'label_text', 'statement', 'subjects', 'speaker', 'job',  
                     'state', 'party', 'barely_true_cts', 'false_cts', 'half_true_cts', 
                     'mostly_true_cts', 'pants_fire_cts', 'context']
        
        df_train = pd.read_csv(train_path, sep='\t', header=None, names=col_names, on_bad_lines='skip', quoting=3)
        df_val = pd.read_csv(val_path, sep='\t', header=None, names=col_names, on_bad_lines='skip', quoting=3)

        label_map = {
            'true': 1, 'mostly-true': 1, 'half-true': 1,
            'barely-true': 0, 'false': 0, 'pants-fire': 0
        }

        for df in [df_train, df_val]:
            df['label'] = df['label_text'].map(label_map)
            
        df_train = df_train[['statement', 'label']].dropna()
        df_val = df_val[['statement', 'label']].dropna()

        return DatasetDict({
            'train': Dataset.from_pandas(df_train),
            'validation': Dataset.from_pandas(df_val)
        })

    def run_training(self, dataset, epochs=3):
        """Fine-tunes the baseline Transformer model."""
        def tokenize_func(examples):
            return self.tokenizer(examples["statement"], padding="max_length", truncation=True, max_length=128)

        tokenized_ds = dataset.map(tokenize_func, batched=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics
        )

        print("🚀 Starting Training...")
        trainer.train()
 
        if os.path.exists(self.trained_temp_path):
            shutil.rmtree(self.trained_temp_path)
        self.model.save_pretrained(self.trained_temp_path)
        self.tokenizer.save_pretrained(self.trained_temp_path)
        return trainer.evaluate()

    def squeeze_model(self):
        """Performs ONNX export and VNNI-aware INT8 quantization."""
        print("⚡ Starting Hardware-Aware Optimization (ONNX)...")
        model_onnx = ORTModelForSequenceClassification.from_pretrained(
            self.trained_temp_path, export=True
        )
        
        quantizer = ORTQuantizer.from_pretrained(model_onnx)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

        quantizer.quantize(
            save_dir=self.onnx_path,
            quantization_config=qconfig,
        )
        return os.path.join(self.onnx_path, "model_quantized.onnx")

    def predict(self, text):
        """Performs optimized inference using the squeezed engine."""

        ort_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_path, file_name="model_quantized.onnx"
        )
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = ort_model(**inputs)
        latency = (time.time() - start_time) * 1000

        logits = torch.from_numpy(outputs.logits) if isinstance(outputs.logits, np.ndarray) else outputs.logits
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        
        label = "RELIABLE" if pred.item() == 1 else "UNRELIABLE"
        return {"label": label, "confidence": conf.item(), "latency_ms": latency}

if __name__ == "__main__":

    engine = VeritasEngine()
    if os.path.exists('train.tsv'):
        data = engine.prepare_data()
        engine.run_training(data, epochs=1)
        engine.squeeze_model()
        print(engine.predict("The economy is showing signs of recovery."))
