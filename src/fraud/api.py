"""Loopback FastAPI service and an offline, buildless dashboard."""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud.config import ROOT
from fraud.serving import Replay, load_release


class Transaction(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    txn_id: str = Field(min_length=1, max_length=80)
    timestamp: datetime
    customer_id: int = Field(ge=0)
    amount_inr: float = Field(gt=0, le=100_000_000)
    merchant_category: Literal['grocery', 'food', 'fuel', 'travel', 'shopping',
                               'utilities', 'electronics', 'gaming']
    city: str = Field(min_length=1, max_length=80)
    distance_from_home_km: float | None = Field(default=None, ge=0, le=50_000)
    device_type: Literal['mobile', 'desktop', 'pos_terminal', 'unknown'] | None = None
    is_new_merchant: Literal[0, 1]
    txns_last_1h: int = Field(ge=0, le=10000)
    customer_avg_amount_inr: float = Field(gt=0, le=100_000_000)

    @field_validator('merchant_category', 'device_type', mode='before')
    @classmethod
    def normalise(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator('timestamp')
    @classmethod
    def local_time(cls, value):
        if value.tzinfo is not None:
            raise ValueError('Use local time without a timezone, matching the training data.')
        return value


def create_app(release_directory=None, data_path=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = load_release(release_directory)
        app.state.replay = Replay(data_path)
        yield

    app = FastAPI(title='Fraud Lab local prediction API', version='3.0.0',
                  docs_url=None, redoc_url=None, lifespan=lifespan,
                  telemetry={'tracing': False}, max_body_size=16_384)
    web = ROOT / 'web'
    app.mount('/assets', StaticFiles(directory=web), name='assets')

    @app.get('/', include_in_schema=False)
    def dashboard():
        return FileResponse(web / 'index.html')

    @app.get('/docs', include_in_schema=False)
    def documentation():
        # Default Swagger loads a CDN; this reference works completely offline.
        return FileResponse(web / 'api.html')

    @app.get('/health')
    def health():
        predictor = app.state.predictor
        return {'status': 'ready', 'model': predictor.bundle['name'],
                'release': predictor.manifest['release'], 'threshold': predictor.bundle['threshold'],
                'data_version': predictor.bundle['data_version']}

    @app.get('/api/model')
    def model():
        return {**app.state.predictor.manifest, 'replay': app.state.replay.info()}

    @app.get('/api/samples')
    def samples(mode: Literal['natural', 'balanced', 'routine', 'fraud'] = 'balanced',
                offset: int = Query(default=0, ge=0), count: int = Query(default=1, ge=1, le=50)):
        return {'samples': [app.state.replay.sample(mode, offset + i) for i in range(count)],
                'next_offset': offset + count, 'mode': mode}

    @app.post('/predict')
    def predict(transaction: Transaction):
        # No label is accepted. A single snapshot also makes reload safe for in-flight requests.
        predictor = app.state.predictor
        return predictor.predict(transaction.model_dump())

    @app.post('/api/reload')
    def reload():
        try:
            predictor = load_release(release_directory)
        except (OSError, ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        app.state.predictor = predictor
        return health()

    return app
