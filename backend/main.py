from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient
from PIL import Image
from datetime import datetime
import os
import io


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="ARANYA-RAKSHAK AI",
    description="AI-powered forest risk intelligence and early-warning system",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROBOFLOW
# =========================================================

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

MODEL_ID = "indian-animals-detection-pgvz8/2"

ROBOFLOW_URL = "https://serverless.roboflow.com"

CLIENT = None

if ROBOFLOW_API_KEY:
    CLIENT = InferenceHTTPClient(
        api_url=ROBOFLOW_URL,
        api_key=ROBOFLOW_API_KEY
    )


# =========================================================
# SUPPORTED WILDLIFE
# =========================================================

SUPPORTED_SPECIES = [
    "tiger",
    "leopard",
    "elephant",
    "rhinoceros",
    "rhino",
    "bear"
]


# =========================================================
# FOREST ZONES
# =========================================================

ZONES = {

    "safe": {
        "risk": "LOW",
        "message": "Normal forest activity"
    },

    "caution": {
        "risk": "MEDIUM",
        "message": "Caution advised"
    },

    "restricted": {
        "risk": "HIGH",
        "message": "Human and vehicle entry restricted"
    },

    "critical": {
        "risk": "CRITICAL",
        "message": "Immediate danger zone"
    }

}


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "success": True,
        "message": "ARANYA-RAKSHAK AI backend is running"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "success": True,

        "status": "ONLINE",

        "system": "ARANYA-RAKSHAK AI",

        "wildlife_model": MODEL_ID,

        "roboflow_configured":
            ROBOFLOW_API_KEY is not None,

        "modules": [

            "wildlife_detection",

            "human_intrusion",

            "vehicle_intrusion",

            "suspicious_activity",

            "forest_zones",

            "risk_engine",

            "safari_safety",

            "electrical_hazard"

        ]

    }


# =========================================================
# WILDLIFE DETECTION
# KEEP THIS MODULE STABLE
# =========================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    if not ROBOFLOW_API_KEY:

        raise HTTPException(
            status_code=500,
            detail=(
                "Roboflow API key is not configured. "
                "Set ROBOFLOW_API_KEY in the backend terminal."
            )
        )


    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type could not be determined."
        )


    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )


    image_bytes = await file.read()


    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )


    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )


    try:

        result = CLIENT.infer(
            image,
            model_id=MODEL_ID
        )

    except Exception as e:

        error_text = str(e)

        if (
            "401" in error_text
            or "Unauthorized" in error_text
            or "api_key" in error_text.lower()
        ):

            raise HTTPException(
                status_code=502,
                detail=(
                    "Roboflow rejected the API key. "
                    "Use a Roboflow Private API Key authorized "
                    "for Serverless Inference."
                )
            )

        raise HTTPException(
            status_code=502,
            detail="Roboflow wildlife inference failed."
        )


    detections = []

    predictions = result.get(
        "predictions",
        []
    )


    for prediction in predictions:

        species = str(
            prediction.get(
                "class",
                "unknown"
            )
        ).strip()


        confidence = float(
            prediction.get(
                "confidence",
                0
            )
        )


        box = {

            "x": prediction.get("x"),

            "y": prediction.get("y"),

            "width": prediction.get("width"),

            "height": prediction.get("height")

        }


        detections.append({

            "species": species,

            "confidence":
                round(confidence, 3),

            "confidence_percent":
                round(
                    confidence * 100,
                    1
                ),

            "box": box

        })


    return {

        "success": True,

        "model": MODEL_ID,

        "detections": detections,

        "detection_count":
            len(detections)

    }


# =========================================================
# ZONES
# =========================================================

@app.get("/zones")
def get_zones():

    return {

        "success": True,

        "zones": ZONES

    }


# =========================================================
# GENERAL RISK ENGINE
# =========================================================

@app.post("/risk")
async def calculate_risk(

    wildlife: str = "none",

    human_present: bool = False,

    zone: str = "safe",

    safari_vehicle: bool = False

):

    wildlife = wildlife.lower().strip()

    zone = zone.lower().strip()


    if zone not in ZONES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid zone. "
                "Use safe, caution, restricted or critical."
            )
        )


    risk = "LOW"

    alert = "No immediate danger."

    action = "Continue normal monitoring."


    if zone == "critical":

        risk = "CRITICAL"

        alert = "Critical zone detected."

        action = (
            "Immediately alert forest response team."
        )


    elif (
        zone == "restricted"
        and human_present
    ):

        risk = "CRITICAL"

        alert = (
            "Unauthorized human detected "
            "in restricted zone."
        )

        action = (
            "Send immediate forest security alert."
        )


    elif (
        wildlife in [
            "tiger",
            "leopard"
        ]
        and human_present
    ):

        risk = "CRITICAL"

        alert = (
            f"{wildlife.title()} detected near human."
        )

        action = (
            "Issue immediate warning "
            "and alert forest team."
        )


    elif (
        wildlife in [
            "elephant",
            "rhinoceros",
            "rhino",
            "bear"
        ]
        and human_present
    ):

        risk = "HIGH"

        alert = (
            f"{wildlife.title()} detected near human."
        )

        action = (
            "Warn people and notify forest team."
        )


    elif (
        safari_vehicle
        and wildlife in SUPPORTED_SPECIES
    ):

        risk = "HIGH"

        alert = (
            f"{wildlife.title()} detected "
            "near safari vehicle."
        )

        action = (
            "Warn safari vehicle "
            "and maintain safe distance."
        )


    elif (
        wildlife != "none"
        and zone == "caution"
    ):

        risk = "MEDIUM"

        alert = (
            f"{wildlife.title()} detected "
            "in caution zone."
        )

        action = "Increase monitoring."


    elif wildlife != "none":

        risk = "LOW"

        alert = (
            f"{wildlife.title()} detected."
        )

        action = "Continue monitoring."


    return {

        "success": True,

        "risk": risk,

        "alert": alert,

        "recommended_action": action,

        "zone": zone,

        "wildlife": wildlife,

        "human_present": human_present,

        "safari_vehicle": safari_vehicle

    }


# =========================================================
# SAFARI SAFETY
# =========================================================

@app.post("/safari")
async def safari_safety(

    wildlife: str = "none",

    distance_meters: float = 999,

    human_present: bool = True

):

    wildlife = wildlife.lower().strip()


    dangerous_animals = [

        "tiger",
        "leopard",
        "elephant",
        "rhinoceros",
        "rhino",
        "bear"

    ]


    if (
        wildlife in dangerous_animals
        and distance_meters <= 100
    ):

        return {

            "success": True,

            "risk": "CRITICAL",

            "alert": (
                f"{wildlife.title()} is very close "
                "to safari vehicle."
            ),

            "action": (
                "Stop vehicle, maintain distance "
                "and alert forest team."
            )

        }


    if (
        wildlife in dangerous_animals
        and distance_meters <= 300
    ):

        return {

            "success": True,

            "risk": "HIGH",

            "alert": (
                f"{wildlife.title()} detected near safari."
            ),

            "action": (
                "Maintain safe distance "
                "and proceed carefully."
            )

        }


    if wildlife in dangerous_animals:

        return {

            "success": True,

            "risk": "MEDIUM",

            "alert": (
                f"{wildlife.title()} detected "
                "in safari area."
            ),

            "action": "Continue monitoring."

        }


    return {

        "success": True,

        "risk": "LOW",

        "alert": (
            "No dangerous wildlife detected."
        ),

        "action": (
            "Normal safari operation."
        )

    }


# =========================================================
# ELECTRICAL HAZARD
# SEPARATE MODULE
# =========================================================

@app.post("/electrical-hazard")
async def electrical_hazard(

    detected: bool = False,

    location: str = "unknown"

):

    location = location.strip()


    if detected:

        return {

            "success": True,

            "hazard": True,

            "risk": "CRITICAL",

            "location": location,

            "alert": (
                "Electrical hazard detected."
            ),

            "action": (
                "Keep people away and notify "
                "forest/electrical response team."
            )

        }


    return {

        "success": True,

        "hazard": False,

        "risk": "LOW",

        "location": location,

        "alert": (
            "No electrical hazard detected."
        ),

        "action": "Continue monitoring."

    }


# =========================================================
# HUMAN INTRUSION
# =========================================================

@app.post("/human-intrusion")
async def human_intrusion(

    human_detected: bool = False,

    zone: str = "safe"

):

    zone = zone.lower().strip()


    if zone not in ZONES:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid zone. "
                "Use safe, caution, restricted or critical."
            )

        )


    if (
        human_detected
        and zone in [
            "restricted",
            "critical"
        ]
    ):

        return {

            "success": True,

            "human_detected": True,

            "intrusion": True,

            "risk": "CRITICAL",

            "zone": zone,

            "alert": (
                "Unauthorized human detected "
                "in protected zone."
            ),

            "action": (
                "Immediately alert forest security team."
            )

        }


    if (
        human_detected
        and zone == "caution"
    ):

        return {

            "success": True,

            "human_detected": True,

            "intrusion": True,

            "risk": "HIGH",

            "zone": zone,

            "alert": (
                "Human detected in caution zone."
            ),

            "action": (
                "Warn person and increase monitoring."
            )

        }


    return {

        "success": True,

        "human_detected":
            human_detected,

        "intrusion": False,

        "risk": "LOW",

        "zone": zone,

        "alert": (
            "No unauthorized human intrusion detected."
        ),

        "action":
            "Continue monitoring."

    }


# =========================================================
# UNAUTHORIZED VEHICLE
# =========================================================

@app.post("/vehicle-intrusion")
async def vehicle_intrusion(

    vehicle_detected: bool = False,

    authorized: bool = True,

    zone: str = "safe",

    camera_id: str = "CAM-01"

):

    zone = zone.lower().strip()

    camera_id = camera_id.strip()


    if zone not in ZONES:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid zone. "
                "Use safe, caution, restricted or critical."
            )

        )


    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )


    vehicle_number = "Not available"


    # -----------------------------------------------------
    # CRITICAL
    # -----------------------------------------------------

    if (
        vehicle_detected
        and not authorized
        and zone in [
            "restricted",
            "critical"
        ]
    ):

        return {

            "success": True,

            "vehicle_detected": True,

            "authorized": False,

            "intrusion": True,

            "risk": "CRITICAL",

            "zone": zone,

            "camera_id": camera_id,

            "timestamp": timestamp,

            "vehicle_number":
                vehicle_number,

            "alert": (
                "Unauthorized vehicle detected "
                "in protected zone."
            ),

            "action": (
                "Immediately alert forest "
                "security team."
            )

        }


    # -----------------------------------------------------
    # HIGH
    # -----------------------------------------------------

    if (
        vehicle_detected
        and not authorized
        and zone == "caution"
    ):

        return {

            "success": True,

            "vehicle_detected": True,

            "authorized": False,

            "intrusion": True,

            "risk": "HIGH",

            "zone": zone,

            "camera_id": camera_id,

            "timestamp": timestamp,

            "vehicle_number":
                vehicle_number,

            "alert": (
                "Unauthorized vehicle detected "
                "in caution zone."
            ),

            "action": (
                "Track vehicle and notify "
                "forest monitoring team."
            )

        }


    # -----------------------------------------------------
    # AUTHORIZED VEHICLE
    # -----------------------------------------------------

    if (
        vehicle_detected
        and authorized
    ):

        return {

            "success": True,

            "vehicle_detected": True,

            "authorized": True,

            "intrusion": False,

            "risk": "LOW",

            "zone": zone,

            "camera_id": camera_id,

            "timestamp": timestamp,

            "vehicle_number":
                vehicle_number,

            "alert": (
                "Authorized vehicle detected."
            ),

            "action":
                "Continue monitoring."

        }


    # -----------------------------------------------------
    # NO VEHICLE
    # -----------------------------------------------------

    return {

        "success": True,

        "vehicle_detected": False,

        "authorized": authorized,

        "intrusion": False,

        "risk": "LOW",

        "zone": zone,

        "camera_id": camera_id,

        "timestamp": timestamp,

        "vehicle_number":
            vehicle_number,

        "alert": (
            "No vehicle intrusion detected."
        ),

        "action":
            "Continue monitoring."

    }


# =========================================================
# SUSPICIOUS / UNAUTHORIZED ACTIVITY
# SMUGGLING-RISK INDICATOR
# =========================================================

@app.post("/suspicious-activity")
async def suspicious_activity(

    human_detected: bool = False,

    vehicle_detected: bool = False,

    vehicle_authorized: bool = True,

    zone: str = "safe",

    camera_id: str = "CAM-01",

    wildlife_detected: str = "none",

    vehicle_number: str = "Not available"

):

    zone = zone.lower().strip()

    camera_id = camera_id.strip()

    wildlife_detected = (
        wildlife_detected
        .lower()
        .strip()
    )

    vehicle_number = (
        vehicle_number
        .strip()
    )


    if zone not in ZONES:

        raise HTTPException(

            status_code=400,

            detail=(
                "Invalid zone. "
                "Use safe, caution, restricted or critical."
            )

        )


    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )


    # -----------------------------------------------------
    # STRONGEST SECURITY INDICATOR
    # -----------------------------------------------------

    if (
        vehicle_detected
        and not vehicle_authorized
        and human_detected
        and zone in [
            "restricted",
            "critical"
        ]
    ):

        return {

            "success": True,

            "suspicious_activity": True,

            "activity_type":
                "Suspected Unauthorized Activity",

            "risk": "CRITICAL",

            "camera_id": camera_id,

            "timestamp": timestamp,

            "zone": zone,

            "human_detected": True,

            "vehicle_detected": True,

            "vehicle_authorized": False,

            "vehicle_number":
                vehicle_number,

            "wildlife_detected":
                wildlife_detected,

            "alert": (
                "Multiple indicators suggest "
                "suspected unauthorized activity "
                "in a protected zone."
            ),

            "action": (
                "Alert Forest Control Room and "
                "dispatch authorized field personnel "
                "for verification."
            )

        }


    # -----------------------------------------------------
    # UNAUTHORIZED VEHICLE
    # -----------------------------------------------------

    if (
        vehicle_detected
        and not vehicle_authorized
        and zone in [
            "restricted",
            "critical"
        ]
    ):

        return {

            "success": True,

            "suspicious_activity": True,

            "activity_type":
                "Suspected Unauthorized Vehicle Activity",

            "risk": "HIGH",

            "camera_id": camera_id,

            "timestamp": timestamp,

            "zone": zone,

            "human_detected":
                human_detected,

            "vehicle_detected": True,

            "vehicle_authorized": False,

            "vehicle_number":
                vehicle_number,

            "wildlife_detected":
                wildlife_detected,

            "alert": (
                "Unauthorized vehicle activity "
                "detected in protected zone."
            ),

            "action": (
                "Track the vehicle and notify "
                "forest monitoring team."
            )

        }


    # -----------------------------------------------------
    # HUMAN + VEHICLE IN CAUTION
    # -----------------------------------------------------

    if (
        human_detected
        and vehicle_detected
        and not vehicle_authorized
        and zone == "caution"
    ):

        return {

            "success": True,

            "suspicious_activity": True,

            "activity_type":
                "Suspicious Activity",

            "risk": "HIGH",

            "camera_id": camera_id,

            "timestamp": timestamp,

            "zone": zone,

            "human_detected": True,

            "vehicle_detected": True,

            "vehicle_authorized": False,

            "vehicle_number":
                vehicle_number,

            "wildlife_detected":
                wildlife_detected,

            "alert": (
                "Suspicious human and vehicle "
                "activity detected."
            ),

            "action": (
                "Increase monitoring and "
                "request field verification."
            )

        }


    # -----------------------------------------------------
    # NORMAL
    # -----------------------------------------------------

    return {

        "success": True,

        "suspicious_activity": False,

        "activity_type":
            "Normal Activity",

        "risk": "LOW",

        "camera_id": camera_id,

        "timestamp": timestamp,

        "zone": zone,

        "human_detected":
            human_detected,

        "vehicle_detected":
            vehicle_detected,

        "vehicle_authorized":
            vehicle_authorized,

        "vehicle_number":
            vehicle_number,

        "wildlife_detected":
            wildlife_detected,

        "alert":
            "No suspicious activity detected.",

        "action":
            "Continue normal monitoring."

    }