# Gloss Recommender

클론하거나 pull한 뒤 저장소 루트에서 아래만 실행하면 된다. 환경 변수, 데이터 복사, 모델 수동 설치는 필요 없다.

```bash
docker compose up -d --build
```

첫 실행에서 `qwen3:14b`와 `bge-m3`를 Ollama가 자동으로 받는다. 이미 받은 적 있으면 건너뛴다. 접속 주소는 `http://localhost:8777`이다.

사전 조건은 Docker Compose와 NVIDIA Container Toolkit뿐이다. 모델은 이미지에 넣지 않고 받는 쪽 계정의 `~/.ollama`에 남기므로 `docker compose down` 후에도 다시 받지 않는다.
