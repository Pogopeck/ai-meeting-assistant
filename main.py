from fastapi import FastAPI, File, UploadFile, Query
from ai_engine import transcribe_audio, extract_actions
import boto3
from datetime import datetime
import io, os

app = FastAPI(title="AI Meeting Assistant")

# DynamoDB (optional in dev)
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("meeting-logs") if os.getenv("AWS_EXECUTION_ENV") else None

@app.post("/process_audio")
async def process_audio(
    audio: UploadFile = File(...),
    meeting_id: str = Query("default", description="Team meeting ID")
):
    audio_bytes = await audio.read()
    transcript = transcribe_audio(audio_bytes)
    actions = extract_actions(transcript)

    # Save to DB if available
    if table:
        table.put_item(Item={
            "meeting_id": meeting_id,
            "timestamp": int(datetime.now().timestamp()),
            "transcript": transcript,
            "action_items": actions
        })

    return {
        "meeting_id": meeting_id,
        "transcript": transcript,
        "action_items": actions
    }

@app.get("/logs/{meeting_id}")
async def get_logs(meeting_id: str):
    if not table:
        return {"error": "DynamoDB not configured"}
    resp = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("meeting_id").eq(meeting_id)
    )
    return {"items": resp["Items"]}