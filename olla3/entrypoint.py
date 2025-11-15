from fastapi import FastAPI, Response
import uvicorn
import os
import logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=\"OLLA2 Multi-AI System\", version=\"5.3.0\")

@app.get(\"/healthz\")
async def health_check():
    \"\"\"Health check endpoint\"\"\"
    return {
        \"status\": \"healthy\", 
        \"service\": \"olla2\", 
        \"version\": \"5.3.0\"
    }

@app.get(\"/metrics\")
async def metrics_endpoint():
    \"\"\"Prometheus metrics endpoint - returns proper format\"\"\"
    try:
        # Prometheus formatında metrics döndür
        return Response(
            generate_latest(REGISTRY), 
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f\"Metrics generation failed: {e}\")
        return Response(\"# Metrics temporarily unavailable\\n\", media_type=CONTENT_TYPE_LATEST)

@app.get(\"/api/metrics/agent-stats\")  
async def agent_metrics():
    \"\"\"Custom agent metrics endpoint (JSON format)\"\"\"
    # Basit mock data - gerçek implementasyonda enhanced_metrics kullanılacak
    return {
        \"status\": \"mock_data\",
        \"message\": \"Enhanced metrics system will be integrated soon\",
        \"sample_metrics\": {
            \"active_agents\": 2,
            \"critic_success_rate\": 85.5,
            \"patch_success_rate\": 78.2
        }
    }

@app.get(\"/\")
async def root():
    return {\"message\": \"OLLA2 Multi-AI System - Metrics Ready\"}

if __name__ == \"__main__\":
    port = int(os.getenv(\"PORT\", 8000))
    logger.info(f\"Starting OLLA2 server on port {port}\")
    uvicorn.run(app, host=\"0.0.0.0\", port=port, log_level=\"info\")
