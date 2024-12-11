# 1. AWS Lambda의 Amazon Linux 2 Python 3.10 이미지 사용
FROM public.ecr.aws/lambda/python:3.10

# 2. 필요한 시스템 패키지 설치
RUN yum -y install gcc gcc-c++ make python3-devel freetype-devel

# 3. Python 패키지 설치
RUN pip install --upgrade pip
RUN pip install numpy==1.22.0
RUN pip install tensorflow==2.10
RUN pip install transformers

# 4. Hugging Face 모델 및 토크나이저 미리 다운로드
RUN python3 -c "from transformers import GPT2TokenizerFast; \
    tokenizer = GPT2TokenizerFast.from_pretrained('skt/kogpt2-base-v2'); \
    tokenizer.save_pretrained('/var/task/kogpt2-tokenizer')"

# 5. Lambda 함수 핸들러 및 모델 복사
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY saved_model ${LAMBDA_TASK_ROOT}/saved_model

# 6. Lambda 핸들러 설정
CMD ["lambda_function.lambda_handler"]