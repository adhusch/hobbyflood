# Live-Betrieb: holt alle INTERVAL_MIN Minuten Pegel, Stationsregen und ICON-D2, rechnet die Kette und stellt gui/live.html bereit.
# Eigene Modellrechnung, keine amtliche Warnung.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Rechenkern (routing_core.c: Routing, Speicher des Zwischengebiets, Vorregenindex) einmal übersetzen; ohne ihn rechnet Python, 10-mal langsamer
RUN python -c "import routing; assert routing._load_lib(), 'C-Rechenkern nicht übersetzt'" \
    && useradd --create-home app && mkdir -p gui/archiv data/live && chown -R app:app /app
USER app

ENV INTERVAL_MIN=30 PORT=8080
EXPOSE 8080
VOLUME ["/app/data/live", "/app/gui/archiv"]
CMD ["sh", "deploy/run_live.sh"]
