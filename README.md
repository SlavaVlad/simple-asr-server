# Simple ASR Server

������� ������ ��� ��������������� ������������� ���� (ASR) �� ���� OpenAI Whisper.

## �����������

- ��������� ��������� ������� Whisper (tiny, base, small, medium, large, turbo)
- ��� ������� ������: plaintext, simple JSON, ������ JSON
- �������� speedup ��� ��������� ����� ����� ��������������
- �������������� ����������� ����� � �������������� ������
- API ����� ��� ������������
- Docker ���������

## ������� �����

### ��������� ���������

1. ���������� �����������:
```bash
pip install -r requirements.txt
```

2. ���������� � ��������� ���������� ���������:
```bash
cp .env.example .env
```

3. ��������� ������:
```bash
python app.py
```

### Docker

1. ��������� � ��������� ���������:
```bash
docker-compose up --build
```

## API

### POST /transcribe

������������� ���� �� ����������.

**���������:**
- `file` (����) - ��������� ��� �������������
- `model_name` (�����������) - ������ Whisper ��� �������������
- `output_format` - ������ ������: `plaintext`, `simple`, ��� `json`
- `speedup` - ����������� ��������� ����� (0.25-4.0)

**���������:**
- `x-api-key` - API ����

**�������:**

```bash
# ������� ��������� �����
curl -X POST "http://localhost:9854/transcribe?output_format=plaintext&speedup=1.5" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@audio.wav"

# JSON � ������ �������
curl -X POST "http://localhost:9854/transcribe?output_format=simple" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@audio.wav"

# ������ JSON ����� � �������������� ������ ������
curl -X POST "http://localhost:9854/transcribe?output_format=json&model_name=base" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@audio.wav"
```

### GET /health

�������� ��������� �������.

## ���������� ���������

��. `.env.example` ��� ������� ������ ��������� ����������:

- `HOST` - ���� ������� (�� ���������: 0.0.0.0)
- `PORT` - ���� ������� (�� ���������: 9854)
- `DEFAULT_MODEL` - ������ �� ��������� (�� ���������: turbo)
- `MODEL_DOWNLOAD_ROOT` - ����� ��� �������� �������
- `KEYS_FILE` - ���� � API �������
