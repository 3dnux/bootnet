# syntax=docker/dockerfile:1

# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Ajustes recomendados para contenedores Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Certificados para HTTPS (CoinGecko usa TLS)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiamos el script y el README (opcional, para referencia)
COPY main.py /app/
COPY README.md /app/

# Permitir pasar argumentos al script mediante docker run ... <args>
ENTRYPOINT ["python", "main.py"]
