import json
import tensorflow as tf
from transformers import GPT2TokenizerFast

# Hugging Face 토크나이저 로드 (Docker 빌드 시 저장된 경로 사용)
tokenizer = GPT2TokenizerFast.from_pretrained("/var/task/kogpt2-tokenizer")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 0

# SavedModel 로드
saved_model_path = "/var/task/saved_model"
model = tf.keras.models.load_model(saved_model_path)


def preprocess_text(text):
    return text.lower()


def predict_text(text):
    """욕설 여부 예측"""
    sentence = preprocess_text(text)
    inputs = tokenizer.encode_plus(
        sentence,
        max_length=1000,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    prediction = model.predict(inputs['input_ids'])[0][0]
    return prediction


def lambda_handler(event, context):
    """AWS Lambda 핸들러"""
    body = json.loads(event.get("body", "{}"))
    text = body.get("text", "")

    if not text:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No text provided."})
        }

    prediction = predict_text(text)
    result = "욕설" if prediction >= 0.975 else "일반어"

    return {
        "statusCode": 200,
        "body": json.dumps({
            "text": text,
            "prediction": result
        }, ensure_ascii=False)
    }
