"""
ASV Router + FastAPI: production intent routing endpoint.

Run: pip install fastapi uvicorn && maturin develop --release && uvicorn python_fastapi:app
Test: curl -X POST localhost:8000/route -H 'Content-Type: application/json' -d '{"query": "cancel my order"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from asv_router import Router

app = FastAPI(title="ASV Intent Router")

# Initialize router with intents
router = Router()
router.begin_batch()
router.add_intent("cancel_order", ["cancel my order", "stop my order", "I want to cancel"])
router.add_intent("track_order", ["where is my package", "track my order", "shipping status"])
router.add_intent("refund", ["I want a refund", "get my money back", "return and refund"])
router.add_intent("contact_human", ["talk to a person", "speak to agent", "human representative"])
router.end_batch()


class RouteRequest(BaseModel):
    query: str
    threshold: float = 0.3


class LearnRequest(BaseModel):
    query: str
    intent_id: str


@app.post("/route")
def route(req: RouteRequest):
    results = router.route(req.query)
    if not results:
        raise HTTPException(404, "No matching intent")
    return {"intent": results[0]["id"], "score": results[0]["score"]}


@app.post("/route_multi")
def route_multi(req: RouteRequest):
    result = router.route_multi(req.query, req.threshold)
    return result


@app.post("/learn")
def learn(req: LearnRequest):
    router.learn(req.query, req.intent_id)
    return {"status": "learned"}


@app.get("/intents")
def intents():
    return router.intent_ids()
